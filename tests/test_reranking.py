from ragforge.chunking import Chunk
from ragforge.index import ScoredChunk
from ragforge.reranking import HeuristicReranker, NoopReranker


def _scored(id_: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=id_, text=text, doc_id=id_, position=0), score=score)


def test_noop_reranker_preserves_order_and_truncates():
    candidates = [
        _scored("a", "text a", 0.9),
        _scored("b", "text b", 0.5),
        _scored("c", "text c", 0.1),
    ]
    result = NoopReranker().rerank("query", candidates, k=2)
    assert [c.chunk.id for c in result] == ["a", "b"]


def test_heuristic_reranker_prefers_higher_token_overlap():
    candidates = [
        _scored("low", "completely unrelated content about gardening", 0.9),
        _scored("high", "python programming tutorial for beginners", 0.1),
    ]
    result = HeuristicReranker().rerank("python programming tutorial", candidates, k=2)
    assert result[0].chunk.id == "high"


def test_heuristic_reranker_handles_empty_query_gracefully():
    candidates = [_scored("a", "some text", 0.5)]
    result = HeuristicReranker().rerank("", candidates, k=1)
    assert len(result) == 1
