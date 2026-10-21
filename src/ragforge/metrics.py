"""Standard ranked-retrieval metrics (nDCG, Recall, Precision, MRR, MAP).

These are the metrics used by BEIR, MTEB and TREC, implemented from their
definitions so retrieval quality can be reported on public benchmarks
(see ``scripts/eval_beir.py``) rather than only on hand-made examples.

Conventions: ``ranked`` is a list of document ids, best first; ``relevance``
maps a document id to a graded relevance (``> 0`` means relevant, higher is
better). A document that is not in ``relevance`` has relevance 0. Every
function returns a value in ``[0, 1]`` and ``0.0`` for a query with no
relevant documents.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def dcg_at_k(ranked: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """Discounted cumulative gain with the standard ``rel / log2(rank + 1)`` form."""
    _check_k(k)
    return sum(
        relevance.get(doc_id, 0.0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
        if relevance.get(doc_id, 0.0) > 0
    )


def ndcg_at_k(ranked: Sequence[str], relevance: Mapping[str, float], k: int = 10) -> float:
    """Normalized DCG: DCG of the ranking divided by DCG of the ideal ranking."""
    _check_k(k)
    ideal = sorted((r for r in relevance.values() if r > 0), reverse=True)
    ideal_dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal[:k], start=1))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(ranked, relevance, k) / ideal_dcg


def recall_at_k(ranked: Sequence[str], relevance: Mapping[str, float], k: int = 10) -> float:
    """Fraction of relevant documents found in the top ``k`` (distinct ids only)."""
    _check_k(k)
    relevant = {d for d, r in relevance.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevance: Mapping[str, float], k: int = 10) -> float:
    """Fraction of the top ``k`` positions occupied by relevant documents."""
    _check_k(k)
    if not any(r > 0 for r in relevance.values()):
        return 0.0
    hits = sum(1 for d in ranked[:k] if relevance.get(d, 0.0) > 0)
    return hits / k


def mrr_at_k(ranked: Sequence[str], relevance: Mapping[str, float], k: int = 10) -> float:
    """Reciprocal rank of the first relevant document within the top ``k``."""
    _check_k(k)
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if relevance.get(doc_id, 0.0) > 0:
            return 1.0 / rank
    return 0.0


def average_precision(ranked: Sequence[str], relevance: Mapping[str, float], k: int = 100) -> float:
    """Average precision over the top ``k`` (mean of precision at each relevant hit)."""
    _check_k(k)
    total_relevant = sum(1 for r in relevance.values() if r > 0)
    if total_relevant == 0:
        return 0.0
    hits = 0
    score = 0.0
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if relevance.get(doc_id, 0.0) > 0:
            hits += 1
            score += hits / rank
    return score / total_relevant


def mean_metric(
    runs: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    metric: str = "ndcg",
    k: int = 10,
) -> float:
    """Average ``metric`` over all queries in ``qrels`` (missing runs score 0).

    ``metric`` is one of ``ndcg``, ``recall``, ``precision``, ``mrr``, ``map``.
    """
    functions = {
        "ndcg": ndcg_at_k,
        "recall": recall_at_k,
        "precision": precision_at_k,
        "mrr": mrr_at_k,
        "map": average_precision,
    }
    if metric not in functions:
        raise ValueError(f"metric must be one of {sorted(functions)}, got {metric!r}")
    if not qrels:
        return 0.0
    fn = functions[metric]
    return sum(fn(runs.get(qid, []), rel, k) for qid, rel in qrels.items()) / len(qrels)
