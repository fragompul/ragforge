"""RAGAS-style automated evaluation suite for production RAG pipelines.

Evaluates retrieval quality and generation quality independently across five key metrics:

1. **Context Precision** (Set & Ranked): Evaluates whether retrieved contexts are relevant,
   and whether relevant contexts are ranked higher in the top-k list (MAP@k).
2. **Context Recall**: Evaluates whether all necessary ground-truth documents were retrieved.
3. **Context F1**: Harmonic mean of context precision and recall.
4. **Faithfulness**: Measures answer grounding in retrieved contexts (anti-hallucination metric).
5. **Answer Relevancy**: Evaluates how directly the generated answer addresses the user query.
6. **Answer Semantic Similarity**: Measures semantic fidelity against a reference answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragforge.embeddings import EmbedFn, cosine_similarity, hashing_embed
from ragforge.pipeline import RagPipeline

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


@dataclass
class EvalCase:
    """A test case for evaluating pipeline retrieval and generation."""

    query: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    ground_truth_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Scored evaluation result for an individual test case."""

    query: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float | None
    context_recall: float | None
    ranked_context_precision: float | None = None
    context_f1: float | None = None
    answer_similarity: float | None = None
    retrieved_doc_ids: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def overall(self) -> float:
        """Compute the unweighted average across all defined metric scores."""
        scores: list[float] = [self.faithfulness, self.answer_relevancy]
        if self.context_precision is not None:
            scores.append(self.context_precision)
        if self.context_recall is not None:
            scores.append(self.context_recall)
        if self.answer_similarity is not None:
            scores.append(self.answer_similarity)
        return sum(scores) / len(scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": (
                round(self.context_precision, 4) if self.context_precision is not None else None
            ),
            "ranked_context_precision": (
                round(self.ranked_context_precision, 4)
                if self.ranked_context_precision is not None
                else None
            ),
            "context_recall": (
                round(self.context_recall, 4) if self.context_recall is not None else None
            ),
            "context_f1": (round(self.context_f1, 4) if self.context_f1 is not None else None),
            "answer_similarity": (
                round(self.answer_similarity, 4) if self.answer_similarity is not None else None
            ),
            "overall": round(self.overall, 4),
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "latency_ms": round(self.latency_ms, 2),
        }


class EvaluationSummary(Sequence[EvalResult]):
    """Aggregated evaluation results over a test suite, implementing the Sequence interface."""

    def __init__(self, results: list[EvalResult]) -> None:
        self.results = results

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: Any) -> Any:
        return self.results[index]

    def __iter__(self) -> Iterator[EvalResult]:
        return iter(self.results)

    @property
    def mean_faithfulness(self) -> float:
        return sum(r.faithfulness for r in self.results) / max(len(self.results), 1)

    @property
    def mean_answer_relevancy(self) -> float:
        return sum(r.answer_relevancy for r in self.results) / max(len(self.results), 1)

    @property
    def mean_context_precision(self) -> float | None:
        valid = [r.context_precision for r in self.results if r.context_precision is not None]
        return (sum(valid) / len(valid)) if valid else None

    @property
    def mean_ranked_context_precision(self) -> float | None:
        valid = [
            r.ranked_context_precision
            for r in self.results
            if r.ranked_context_precision is not None
        ]
        return (sum(valid) / len(valid)) if valid else None

    @property
    def mean_context_recall(self) -> float | None:
        valid = [r.context_recall for r in self.results if r.context_recall is not None]
        return (sum(valid) / len(valid)) if valid else None

    @property
    def mean_context_f1(self) -> float | None:
        valid = [r.context_f1 for r in self.results if r.context_f1 is not None]
        return (sum(valid) / len(valid)) if valid else None

    @property
    def mean_answer_similarity(self) -> float | None:
        valid = [r.answer_similarity for r in self.results if r.answer_similarity is not None]
        return (sum(valid) / len(valid)) if valid else None

    @property
    def mean_overall(self) -> float:
        return sum(r.overall for r in self.results) / max(len(self.results), 1)

    @property
    def mean_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results) / max(len(self.results), 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": len(self.results),
            "mean_faithfulness": round(self.mean_faithfulness, 4),
            "mean_answer_relevancy": round(self.mean_answer_relevancy, 4),
            "mean_context_precision": (
                round(self.mean_context_precision, 4)
                if self.mean_context_precision is not None
                else None
            ),
            "mean_ranked_context_precision": (
                round(self.mean_ranked_context_precision, 4)
                if self.mean_ranked_context_precision is not None
                else None
            ),
            "mean_context_recall": (
                round(self.mean_context_recall, 4) if self.mean_context_recall is not None else None
            ),
            "mean_context_f1": (
                round(self.mean_context_f1, 4) if self.mean_context_f1 is not None else None
            ),
            "mean_answer_similarity": (
                round(self.mean_answer_similarity, 4)
                if self.mean_answer_similarity is not None
                else None
            ),
            "mean_overall": round(self.mean_overall, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "results": [r.to_dict() for r in self.results],
        }

    def to_markdown_table(self) -> str:
        """Format the evaluation summary as a GitHub Flavored Markdown table."""
        lines = [
            "| Metric | Mean Score |",
            "| :--- | :--- |",
            f"| **Overall Score** | `{self.mean_overall:.4f}` |",
            f"| **Faithfulness** | `{self.mean_faithfulness:.4f}` |",
            f"| **Answer Relevancy** | `{self.mean_answer_relevancy:.4f}` |",
        ]
        if self.mean_context_precision is not None:
            lines.append(f"| **Context Precision** | `{self.mean_context_precision:.4f}` |")
        if self.mean_ranked_context_precision is not None:
            lines.append(
                f"| **Ranked Precision (MAP@k)** | `{self.mean_ranked_context_precision:.4f}` |"
            )
        if self.mean_context_recall is not None:
            lines.append(f"| **Context Recall** | `{self.mean_context_recall:.4f}` |")
        if self.mean_context_f1 is not None:
            lines.append(f"| **Context F1** | `{self.mean_context_f1:.4f}` |")
        if self.mean_answer_similarity is not None:
            lines.append(f"| **Answer Similarity** | `{self.mean_answer_similarity:.4f}` |")
        lines.append(f"| **Mean Latency (ms)** | `{self.mean_latency_ms:.2f} ms` |")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save evaluation report to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def _context_precision(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float | None:
    if not relevant_doc_ids:
        return None
    if not retrieved_doc_ids:
        return 0.0
    relevant_set = set(relevant_doc_ids)
    hits = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant_set)
    return hits / len(retrieved_doc_ids)


def _ranked_context_precision(
    retrieved_doc_ids: list[str], relevant_doc_ids: list[str]
) -> float | None:
    """Mean Average Precision (MAP@k) assessing if relevant docs appear at top ranks."""
    if not relevant_doc_ids:
        return None
    if not retrieved_doc_ids:
        return 0.0

    relevant_set = set(relevant_doc_ids)
    hits = 0
    sum_precisions = 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_set:
            hits += 1
            sum_precisions += hits / rank

    if hits == 0:
        return 0.0
    return sum_precisions / min(len(relevant_set), len(retrieved_doc_ids))


def _context_recall(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float | None:
    if not relevant_doc_ids:
        return None
    retrieved_set = set(retrieved_doc_ids)
    hits = sum(1 for doc_id in relevant_doc_ids if doc_id in retrieved_set)
    return hits / len(relevant_doc_ids)


def _context_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def _faithfulness(answer: str, contexts: list[str]) -> float:
    answer_tokens = _token_set(answer)
    if not answer_tokens:
        return 1.0
    context_tokens: set[str] = set()
    for context in contexts:
        context_tokens |= _token_set(context)
    grounded = len(answer_tokens & context_tokens)
    return grounded / len(answer_tokens)


def _answer_relevancy(query: str, answer: str, embed_fn: EmbedFn) -> float:
    return cosine_similarity(embed_fn(query), embed_fn(answer))


def _answer_similarity(answer: str, ground_truth: str, embed_fn: EmbedFn) -> float:
    return cosine_similarity(embed_fn(answer), embed_fn(ground_truth))


def evaluate_pipeline(
    pipeline: RagPipeline,
    cases: list[EvalCase],
    k: int = 5,
    candidate_pool: int = 20,
    embed_fn: EmbedFn = hashing_embed,
) -> EvaluationSummary:
    """Run an automated evaluation suite over a RagPipeline instance."""
    results: list[EvalResult] = []

    for case in cases:
        rag_answer = pipeline.answer(case.query, k=k, candidate_pool=candidate_pool)
        retrieved_doc_ids = [c.doc_id for c in rag_answer.contexts]
        contexts_text = [c.text for c in rag_answer.contexts]

        cp = _context_precision(retrieved_doc_ids, case.relevant_doc_ids)
        rcp = _ranked_context_precision(retrieved_doc_ids, case.relevant_doc_ids)
        cr = _context_recall(retrieved_doc_ids, case.relevant_doc_ids)
        f1 = _context_f1(cp, cr)

        ans_sim = (
            _answer_similarity(rag_answer.answer, case.ground_truth_answer, embed_fn)
            if case.ground_truth_answer is not None
            else None
        )

        results.append(
            EvalResult(
                query=case.query,
                answer=rag_answer.answer,
                faithfulness=_faithfulness(rag_answer.answer, contexts_text),
                answer_relevancy=_answer_relevancy(case.query, rag_answer.answer, embed_fn),
                context_precision=cp,
                ranked_context_precision=rcp,
                context_recall=cr,
                context_f1=f1,
                answer_similarity=ans_sim,
                retrieved_doc_ids=retrieved_doc_ids,
                latency_ms=rag_answer.total_latency_ms,
            )
        )

    return EvaluationSummary(results)
