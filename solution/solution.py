"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Hoàn thiện tất cả TODO trong template.py:
    - QAPair / EvalResult (Task 1)
    - RAGASEvaluator: faithfulness / relevance / completeness / context recall / context precision / run_full_eval (Task 2 + 2b)
    - rerank_by_overlap (Exercise 3.5)
    - LLMJudge: score_response / detect_bias (Task 3)
    - BenchmarkRunner: run / generate_report / run_regression / identify_failures (Task 4)
    - FailureAnalyzer: categorize_failures / find_root_cause / generate_improvement_suggestions / generate_improvement_log (Task 5)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """Một cặp câu hỏi - đáp án trong Golden Dataset.

    Fields:
        question:            Câu hỏi của user.
        expected_answer:     Đáp án tham chiếu (do expert viết).
        context:             Context nguồn (chuỗi đơn; có thể rỗng).
        metadata:            Metadata tùy chọn (difficulty, category,...).
        retrieved_contexts:  Danh sách chunk theo thứ hạng retriever
                             (dùng cho context recall/precision ở Task 2b).
    """
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)
    retrieved_contexts: list = field(default_factory=list)


@dataclass
class EvalResult:
    """Kết quả đánh giá cho một QAPair.

    Fields:
        qa_pair:            QAPair gốc.
        actual_answer:      Câu trả lời thực tế của agent.
        faithfulness:       Float 0-1, mức độ grounded trong context.
        relevance:          Float 0-1, mức độ liên quan với question.
        completeness:       Float 0-1, mức độ bao phủ expected answer.
        passed:             True nếu cả 3 score >= 0.5.
        failure_type:       None nếu passed, ngược lại: "hallucination",
                            "irrelevant", "incomplete" hoặc "off_topic".
        context_precision:  Float 0-1 hoặc None — chất lượng ranking retrieval.
        context_recall:     Float 0-1 hoặc None — mức phủ của expected bởi chunks.
    """
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Trả về trung bình cộng 3 metric: (faithfulness + relevance + completeness) / 3."""
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------
# In production, replace with actual RAGAS framework:
#   from ragas import evaluate
#   from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
#
# Or DeepEval:
#   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#
# Or TruLens:
#   from trulens.core import Feedback
# ---------------------------------------------------------------------------

# Common English stopwords được loại bỏ để phản ánh từ nội dung,
# không phải filler ("is"/"a"/"the" làm phồng mọi score).
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, bỏ qua punctuation và stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


class RAGASEvaluator:
    """Đánh giá RAG pipeline bằng RAGAS-inspired heuristics (word overlap)."""

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """Đo mức grounded của answer trong context.

        faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
        Trả về 1.0 nếu answer rỗng. Clamp [0.0, 1.0].
        """
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 1.0
        context_tokens = _tokenize(context)
        if not context_tokens:
            return 0.0
        overlap = answer_tokens & context_tokens
        return max(0.0, min(1.0, len(overlap) / len(answer_tokens)))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """Đo mức liên quan của answer với question.

        relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
        Trả về 1.0 nếu question rỗng. Clamp [0.0, 1.0].
        """
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 0.0
        overlap = answer_tokens & question_tokens
        return max(0.0, min(1.0, len(overlap) / len(question_tokens)))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """Đo mức answer bao phủ expected answer.

        completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
        Trả về 1.0 nếu expected rỗng. Clamp [0.0, 1.0].
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 0.0
        overlap = answer_tokens & expected_tokens
        return max(0.0, min(1.0, len(overlap) / len(expected_tokens)))

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (đánh giá bước GET CONTEXT)
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — đo mức expected answer được phủ bởi UNION các chunk.

        union_tokens = ⋃ _tokenize(chunk) cho mỗi chunk
        recall = |expected_tokens ∩ union_tokens| / |expected_tokens|

        Recall thấp => retriever bỏ sót evidence cần thiết cho answer.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens |= _tokenize(chunk)
        if not union_tokens:
            return 0.0
        overlap = expected_tokens & union_tokens
        return max(0.0, min(1.0, len(overlap) / len(expected_tokens)))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — rank-aware Average Precision (AP@K) như RAGAS.

        1. Chunk "relevant" nếu: |chunk ∩ expected| / |expected| >= threshold
        2. Precision@k = (#relevant trong top-k) / k
        3. AP@K = (1 / #relevant) * Σ_k [ Precision@k · relevant_k ]

        Trả về 1.0 nếu expected rỗng; 0.0 nếu không có chunk hoặc không có
        chunk relevant. Reorder chunk relevant lên đầu (rerank) sẽ tăng score.
        """
        if not contexts:
            return 0.0
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0

        # Xác định chunk nào relevant
        is_relevant: list[bool] = []
        for chunk in contexts:
            chunk_tokens = _tokenize(chunk)
            if not chunk_tokens:
                is_relevant.append(False)
                continue
            overlap = chunk_tokens & expected_tokens
            coverage = len(overlap) / len(expected_tokens)
            is_relevant.append(coverage >= relevance_threshold)

        total_relevant = sum(is_relevant)
        if total_relevant == 0:
            return 0.0

        # AP@K
        running_relevant = 0
        precision_sum = 0.0
        for k, rel in enumerate(is_relevant, start=1):
            if rel:
                running_relevant += 1
                precision_at_k = running_relevant / k
                precision_sum += precision_at_k

        ap = precision_sum / total_relevant
        return max(0.0, min(1.0, ap))

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        retrieved_contexts: list[str] | None = None,
    ) -> EvalResult:
        """Chạy 3 metric answer-side, kết hợp thành EvalResult.

        passed = True nếu cả 3 score >= 0.5.
        failure_type (first match wins):
            faithfulness < 0.3 → "hallucination"
            relevance    < 0.3 → "irrelevant"
            completeness < 0.3 → "incomplete"
            nếu fail mà không match → "off_topic"
        Nếu truyền retrieved_contexts, sẽ tính thêm context_recall / context_precision.
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        passed = (
            faithfulness >= 0.5
            and relevance >= 0.5
            and completeness >= 0.5
        )

        failure_type: str | None = None
        if not passed:
            if faithfulness < 0.3:
                failure_type = "hallucination"
            elif relevance < 0.3:
                failure_type = "irrelevant"
            elif completeness < 0.3:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"

        context_precision: float | None = None
        context_recall: float | None = None
        if retrieved_contexts is not None:
            context_recall = self.evaluate_context_recall(retrieved_contexts, expected)
            context_precision = self.evaluate_context_precision(retrieved_contexts, expected)

        qa = QAPair(
            question=question,
            expected_answer=expected,
            context=context,
        )

        return EvalResult(
            qa_pair=qa,
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=context_precision,
            context_recall=context_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (Exercise 3.5 — boost Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """Reranker lexical tối giản: sắp xếp chunk theo word-overlap với query,
    chunk overlap cao nhất lên đầu. Đứng thay cho cross-encoder reranker thật.

    Reorder chunk relevant lên đầu sẽ tăng rank-aware Context Precision
    MÀ KHÔNG thay đổi tập chunk (recall giữ nguyên).
    """
    if not contexts:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return list(contexts)

    def score(chunk: str) -> int:
        return len(_tokenize(chunk) & query_tokens)

    return sorted(contexts, key=score, reverse=True)


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """Dùng LLM để chấm điểm AI response theo rubric 1–5."""

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """Chấm response bằng judge LLM.

        Quy trình:
            1. Build prompt gồm question, answer, rubric.
            2. Gọi judge_llm_fn(prompt).
            3. Parse JSON scores; nếu fail thì trả default 0.5 cho từng tiêu chí.

        Returns:
            {"scores": {criterion: float 0-1, ...}, "reasoning": str}
        """
        rubric_lines = "\n".join(
            f"- {name}: {desc}" for name, desc in rubric.items()
        )
        prompt = (
            "Bạn là LLM judge chấm điểm AI response theo rubric 1–5.\n"
            "Hãy trả về JSON với key là tên tiêu chí, value là điểm số thực trong [0, 1] "
            "(1 = tốt nhất, 0 = tệ nhất).\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Rubric:\n{rubric_lines}\n\n"
            "Chỉ trả về JSON, ví dụ: {\"accuracy\": 0.8, \"clarity\": 0.7}"
        )

        raw = self.judge_llm_fn(prompt)
        scores: dict[str, float] = {}
        reasoning = raw if isinstance(raw, str) else str(raw)

        parsed: Any = None
        # Tìm JSON block trong response
        match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                parsed = None
        else:
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                parsed = None

        if isinstance(parsed, dict):
            for name in rubric:
                if name in parsed:
                    try:
                        val = float(parsed[name])
                        scores[name] = max(0.0, min(1.0, val))
                    except (TypeError, ValueError):
                        scores[name] = 0.5
                else:
                    scores[name] = 0.5
        else:
            # Không parse được → default 0.5
            scores = {name: 0.5 for name in rubric}

        return {"scores": scores, "reasoning": reasoning}

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Phát hiện bias trong batch điểm judge.

        positional_bias: response đầu tiên trong batch thường được chấm cao hơn
                         (nếu score trung bình response đầu > score trung bình các
                          response còn lại nhiều hơn 0.05 thì flag True).
        leniency_bias:   score trung bình toàn batch > 0.8.
        severity_bias:    score trung bình toàn batch < 0.3.
        """
        if not scores_batch:
            return {
                "positional_bias": False,
                "leniency_bias": False,
                "severity_bias": False,
            }

        def avg_of(item: dict[str, Any]) -> float:
            scores = item.get("scores") or {}
            vals = [float(v) for v in scores.values() if isinstance(v, (int, float))]
            if not vals:
                return 0.0
            return sum(vals) / len(vals)

        all_scores: list[float] = [avg_of(it) for it in scores_batch]
        overall_avg = sum(all_scores) / len(all_scores)

        # Positional bias: nếu batch có >= 2, so sánh phần tử đầu với phần còn lại
        positional_bias = False
        if len(all_scores) >= 2:
            first = all_scores[0]
            rest = all_scores[1:]
            rest_avg = sum(rest) / len(rest)
            if first - rest_avg > 0.05:
                positional_bias = True

        leniency_bias = overall_avg > 0.8
        severity_bias = overall_avg < 0.3

        return {
            "positional_bias": positional_bias,
            "leniency_bias": leniency_bias,
            "severity_bias": severity_bias,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Chạy benchmark evaluation đầy đủ."""

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """Chạy tất cả QA pairs qua agent + evaluator, trả về list EvalResult."""
        results: list[EvalResult] = []
        for pair in qa_pairs:
            actual = agent_fn(pair.question)
            result = evaluator.run_full_eval(
                answer=actual,
                question=pair.question,
                context=pair.context,
                expected=pair.expected_answer,
                retrieved_contexts=pair.retrieved_contexts or None,
            )
            results.append(result)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """Tổng hợp kết quả: pass rate, avg scores, failure types."""
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_completeness": 0.0,
                "failure_types": {},
            }

        passed = sum(1 for r in results if r.passed)
        avg_faithfulness = sum(r.faithfulness for r in results) / total
        avg_relevance = sum(r.relevance for r in results) / total
        avg_completeness = sum(r.completeness for r in results) / total

        failure_types: dict[str, int] = {}
        for r in results:
            if not r.passed and r.failure_type:
                failure_types[r.failure_type] = failure_types.get(r.failure_type, 0) + 1

        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total,
            "avg_faithfulness": avg_faithfulness,
            "avg_relevance": avg_relevance,
            "avg_completeness": avg_completeness,
            "failure_types": failure_types,
        }

    def run_regression(
        self,
        new_results: list[EvalResult],
        baseline_results: list[EvalResult],
    ) -> dict[str, Any]:
        """So sánh new vs baseline; phát hiện regression khi drop > 0.05.

        Returns:
            dict với keys:
              - new_avg_faithfulness, new_avg_relevance, new_avg_completeness
              - baseline_avg_faithfulness, baseline_avg_relevance, baseline_avg_completeness
              - regressions: list[str]  — tên metric bị regression
              - passed: bool              — True nếu không có regression
        """
        def avg(results: list[EvalResult], attr: str) -> float:
            if not results:
                return 0.0
            return sum(getattr(r, attr) for r in results) / len(results)

        new_f = avg(new_results, "faithfulness")
        new_r = avg(new_results, "relevance")
        new_c = avg(new_results, "completeness")
        base_f = avg(baseline_results, "faithfulness")
        base_r = avg(baseline_results, "relevance")
        base_c = avg(baseline_results, "completeness")

        regressions: list[str] = []
        if base_f - new_f > 0.05:
            regressions.append("faithfulness")
        if base_r - new_r > 0.05:
            regressions.append("relevance")
        if base_c - new_c > 0.05:
            regressions.append("completeness")

        return {
            "new_avg_faithfulness": new_f,
            "new_avg_relevance": new_r,
            "new_avg_completeness": new_c,
            "baseline_avg_faithfulness": base_f,
            "baseline_avg_relevance": base_r,
            "baseline_avg_completeness": base_c,
            "regressions": regressions,
            "passed": len(regressions) == 0,
        }

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """Lọc EvalResults có bất kỳ score nào dưới threshold."""
        return [
            r for r in results
            if r.faithfulness < threshold
            or r.relevance < threshold
            or r.completeness < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """Phân tích failure results: cluster, root cause, improvement log."""

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """Đếm failure theo failure_type."""
        categories: dict[str, int] = {}
        for f in failures:
            ft = f.failure_type or "unknown"
            categories[ft] = categories.get(ft, 0) + 1
        return categories

    def find_root_cause(self, failure: EvalResult) -> str:
        """Gợi ý root cause dựa trên score thấp nhất."""
        scores = {
            "faithfulness": failure.faithfulness,
            "relevance": failure.relevance,
            "completeness": failure.completeness,
        }
        # Ưu tiên xét faithfulness (hallucination) trước, sau đó relevance,
        # sau đó completeness; nếu cả 3 đều thấp thì multi-issue.
        low = [name for name, val in scores.items() if val < 0.5]
        if len(low) >= 2:
            return "Multiple issues detected — review full pipeline"
        if "faithfulness" in low:
            return "Context is missing or irrelevant — improve retrieval"
        if "relevance" in low:
            return "Answer does not address the question — improve prompt clarity"
        if "completeness" in low:
            return "Answer is missing key information — increase context window or improve generation"
        # Không metric nào dưới 0.5 nhưng vẫn fail (passed=False)
        return "Multiple issues detected — review full pipeline"

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """Sinh danh sách suggestions ưu tiên dựa trên failure pattern."""
        if not failures:
            return ["No failures detected — keep monitoring in CI/CD"]

        categories = self.categorize_failures(failures)
        suggestions: list[str] = []

        # Suggest theo từng loại failure
        if categories.get("hallucination", 0) > 0:
            suggestions.append(
                "Implement hallucination checker (verify claims against retrieved context) "
                "to flag unsupported statements before returning to user"
            )
        if categories.get("irrelevant", 0) > 0:
            suggestions.append(
                "Rewrite agent prompt with explicit intent classification + few-shot examples "
                "to keep answers on-topic"
            )
        if categories.get("incomplete", 0) > 0:
            suggestions.append(
                "Increase chunk size or expand top-k in retriever to surface more evidence, "
                "and add a completeness check that lists missing sub-points"
            )
        if categories.get("off_topic", 0) > 0:
            suggestions.append(
                "Add intent detection / routing step before generation to detect off-scope "
                "queries and route to clarification or refusal"
            )
        if categories.get("refusal", 0) > 0:
            suggestions.append(
                "Loosen guardrails on in-scope queries; only refuse out-of-domain requests"
            )

        # Gợi ý chung luôn hữu ích
        suggestions.append(
            "Augment golden dataset with the new failure cases and rerun benchmark weekly"
        )

        # Nếu số lượng failure nhiều, thêm suggestion về CI/CD
        if len(failures) >= 3:
            suggestions.append(
                "Add CI/CD quality gate: block merge when pass_rate drops > 5% "
                "or any single metric falls below 0.5"
            )

        return suggestions

    def generate_improvement_log(
        self,
        failures: list,
        suggestions: list[str],
    ) -> str:
        """Xuất Markdown table theo dõi failure + improvement actions.

        Format:
        | Failure ID | Type | Root Cause | Suggested Fix | Status |
        |------------|------|------------|---------------|--------|
        | F001       | ...  | ...        | ...           | Open   |
        """
        lines: list[str] = []
        lines.append("| Failure ID | Type | Root Cause | Suggested Fix | Status |")
        lines.append("|------------|------|------------|---------------|--------|")

        if not failures:
            lines.append("| -          | -    | No failures detected | - | - |")
            return "\n".join(lines)

        for idx, f in enumerate(failures, start=1):
            failure_id = f"F{idx:03d}"
            ftype = f.failure_type or "unknown"
            # Root cause ngắn gọn dựa trên find_root_cause
            cause = self.find_root_cause(f)
            if cause == "Context is missing or irrelevant — improve retrieval":
                cause_short = "Retrieval gap"
            elif cause == "Answer does not address the question — improve prompt clarity":
                cause_short = "Prompt/Intent issue"
            elif cause == "Answer is missing key information — increase context window or improve generation":
                cause_short = "Completeness gap"
            else:
                cause_short = "Multi-issue"
            # Map suggestion: ưu tiên theo idx-1, fallback về suggestion tổng quát
            if idx - 1 < len(suggestions):
                fix = suggestions[idx - 1]
            elif suggestions:
                fix = suggestions[-1]
            else:
                fix = "Investigate and add a regression test"
            lines.append(
                f"| {failure_id} | {ftype} | {cause_short} | {fix} | Open |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    qa_pairs = [
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()
    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
