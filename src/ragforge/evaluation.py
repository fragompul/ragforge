"""RAGAS-style automatic evaluation, implemented without an external
evaluation-framework dependency.

Four complementary metrics, each catching a different failure mode:

- **context_precision** -- of what was retrieved, how much was actually
  relevant? Low precision means the index is retrieving noise.
- **context_recall** -- of what's relevant, how much was retrieved? Low
  recall means the answer is missing information even if generation is
  perfect -- a retrieval/chunking problem, not a prompting problem.
- **faithfulness** -- is the generated answer grounded in the retrieved
  contexts, or does it introduce unsupported claims (hallucination)?
- **answer_relevancy** -- does the answer actually address the question?

Context precision/recall require a ground-truth set of relevant document
IDs per test case; when omitted, those two metrics are reported as
``None`` rather than a misleading zero.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ragforge.embeddings import cosine_similarity, hashing_embed
from ragforge.pipeline import RagPipeline

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


@dataclass
class EvalCase:
    query: str
    relevant_doc_ids: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    query: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float | None
    context_recall: float | None

    @property
    def overall(self) -> float:
        scores = [self.faithfulness, self.answer_relevancy]
        if self.context_precision is not None:
            scores.append(self.context_precision)
        if self.context_recall is not None:
            scores.append(self.context_recall)
        return sum(scores) / len(scores)


def _context_precision(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float | None:
    if not relevant_doc_ids:
        return None
    if not retrieved_doc_ids:
        return 0.0
    relevant_set = set(relevant_doc_ids)
    hits = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant_set)
    return hits / len(retrieved_doc_ids)


def _context_recall(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float | None:
    if not relevant_doc_ids:
        return None
    retrieved_set = set(retrieved_doc_ids)
    hits = sum(1 for doc_id in relevant_doc_ids if doc_id in retrieved_set)
    return hits / len(relevant_doc_ids)


def _faithfulness(answer: str, contexts: list[str]) -> float:
    answer_tokens = _token_set(answer)
    if not answer_tokens:
        return 1.0
    context_tokens: set[str] = set()
    for context in contexts:
        context_tokens |= _token_set(context)
    grounded = len(answer_tokens & context_tokens)
    return grounded / len(answer_tokens)


def _answer_relevancy(
    query: str, answer: str, embed_fn: Callable[[str], list[float]]
) -> float:
    return cosine_similarity(embed_fn(query), embed_fn(answer))


def evaluate_pipeline(
    pipeline: RagPipeline,
    cases: list[EvalCase],
    k: int = 5,
    embed_fn: Callable[[str], list[float]] = hashing_embed,
) -> list[EvalResult]:
    results = []
    for case in cases:
        rag_answer = pipeline.answer(case.query, k=k)
        retrieved_doc_ids = [c.doc_id for c in rag_answer.contexts]
        contexts_text = [c.text for c in rag_answer.contexts]

        results.append(
            EvalResult(
                query=case.query,
                answer=rag_answer.answer,
                faithfulness=_faithfulness(rag_answer.answer, contexts_text),
                answer_relevancy=_answer_relevancy(case.query, rag_answer.answer, embed_fn),
                context_precision=_context_precision(retrieved_doc_ids, case.relevant_doc_ids),
                context_recall=_context_recall(retrieved_doc_ids, case.relevant_doc_ids),
            )
        )
    return results
