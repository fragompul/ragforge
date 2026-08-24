import pytest

from ragforge.chunking import Chunk
from ragforge.index import ScoredChunk
from ragforge.reranking import (
    CrossEncoderReranker,
    HeuristicReranker,
    MaxMarginalRelevanceReranker,
    NoopReranker,
)


def _scored(id_: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id=id_, text=text, doc_id=id_, position=0),
        score=score,
        provenance="test",
    )


def test_noop_reranker_preserves_order_and_truncates():
    candidates = [
        _scored("a", "text a", 0.9),
        _scored("b", "text b", 0.5),
        _scored("c", "text c", 0.1),
    ]
    result = NoopReranker().rerank("query", candidates, k=2)
    assert [c.chunk.id for c in result] == ["a", "b"]
    assert NoopReranker().rerank("query", candidates, k=0) == []


def test_heuristic_reranker_prefers_higher_token_overlap():
    candidates = [
        _scored("low", "completely unrelated content about gardening", 0.9),
        _scored("high", "python programming tutorial for beginners", 0.1),
    ]
    result = HeuristicReranker().rerank("python programming tutorial", candidates, k=2)
    assert result[0].chunk.id == "high"
    assert "rerank:jaccard" in result[0].provenance


def test_heuristic_reranker_edge_cases():
    candidates = [_scored("a", "some text", 0.5)]
    assert len(HeuristicReranker().rerank("", candidates, k=1)) == 1
    assert HeuristicReranker().rerank("query", [], k=5) == []
    assert HeuristicReranker().rerank("query", candidates, k=0) == []


def test_cross_encoder_reranker_single_fn():
    def mock_cross_encoder(query: str, text: str) -> float:
        if "target" in text:
            return 0.95
        return 0.10

    candidates = [
        _scored("doc1", "irrelevant context", 0.8),
        _scored("doc2", "this contains the target info", 0.2),
    ]
    reranker = CrossEncoderReranker(score_fn=mock_cross_encoder)
    result = reranker.rerank("query", candidates, k=1)

    assert len(result) == 1
    assert result[0].chunk.id == "doc2"
    assert result[0].score == 0.95


def test_cross_encoder_reranker_batch_fn():
    def mock_batch_fn(query: str, texts: list[str]) -> list[float]:
        return [float(len(t)) for t in texts]

    candidates = [
        _scored("short", "short text", 0.5),
        _scored("long", "much longer text content here", 0.5),
    ]
    reranker = CrossEncoderReranker(batch_score_fn=mock_batch_fn)
    result = reranker.rerank("query", candidates, k=2)

    assert result[0].chunk.id == "long"


def test_cross_encoder_reranker_validation():
    with pytest.raises(ValueError, match="Either score_fn or batch_score_fn"):
        CrossEncoderReranker()


def test_mmr_reranker_diversity_selection():
    candidates = [
        _scored("c1_dupA", "Machine learning neural networks deep learning models.", 0.9),
        _scored("c2_dupB", "Machine learning neural networks deep learning architectures.", 0.89),
        _scored("c3_distinct", "Database indexing B-tree storage engines and SQL queries.", 0.7),
    ]

    # With lambda_mult=0.5, MMR should penalize dupB due to high similarity with dupA
    mmr = MaxMarginalRelevanceReranker(lambda_mult=0.5)
    reranked = mmr.rerank("Machine learning and databases", candidates, k=2)

    selected_ids = [c.chunk.id for c in reranked]
    assert len(selected_ids) == 2
    # c3_distinct should be picked for diversity instead of the duplicate c2_dupB
    assert "c3_distinct" in selected_ids


def test_mmr_reranker_validations_and_empty():
    with pytest.raises(ValueError, match="lambda_mult must be between 0.0 and 1.0"):
        MaxMarginalRelevanceReranker(lambda_mult=1.5)

    mmr = MaxMarginalRelevanceReranker()
    assert mmr.rerank("query", [], k=5) == []
    assert mmr.rerank("query", [_scored("c1", "text", 0.5)], k=0) == []
