"""MMR with ``relevance='fused'`` keeps the first-stage ranking and only adds diversity."""

from __future__ import annotations

import pytest

from ragforge.chunking import Chunk
from ragforge.index import ScoredChunk
from ragforge.reranking import MaxMarginalRelevanceReranker


def _cand(cid: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=cid, text=text, doc_id=cid, position=0), score=score)


def test_fused_relevance_preserves_first_stage_order_when_lambda_is_one():
    candidates = [
        _cand("lexical-win", "alpha beta", 0.9),
        _cand("middle", "gamma delta", 0.5),
        _cand("embedding-win", "completely query like text words", 0.1),
    ]
    reranker = MaxMarginalRelevanceReranker(lambda_mult=1.0, relevance="fused")
    ranked = reranker.rerank("query like text words", candidates, k=3)
    assert [r.chunk.id for r in ranked] == ["lexical-win", "middle", "embedding-win"]

    dense = MaxMarginalRelevanceReranker(lambda_mult=1.0).rerank(
        "query like text words", candidates, k=3
    )
    assert dense[0].chunk.id == "embedding-win"


def test_fused_relevance_still_penalizes_redundancy():
    candidates = [
        _cand("a", "tower in paris france landmark", 1.0),
        _cand("a-dup", "tower in paris france landmark", 0.95),
        _cand("b", "python language creator guido", 0.6),
    ]
    ranked = MaxMarginalRelevanceReranker(lambda_mult=0.5, relevance="fused").rerank(
        "tower", candidates, k=2
    )
    assert [r.chunk.id for r in ranked] == ["a", "b"]


def test_fused_relevance_handles_equal_scores():
    candidates = [_cand("x", "one", 0.3), _cand("y", "two", 0.3)]
    ranked = MaxMarginalRelevanceReranker(relevance="fused").rerank("q", candidates, k=2)
    assert len(ranked) == 2


def test_invalid_relevance_rejected():
    with pytest.raises(ValueError, match="relevance"):
        MaxMarginalRelevanceReranker(relevance="vibes")
