import pytest

from ragforge.chunking import Chunk
from ragforge.index import BM25Index, HybridRetriever, ScoredChunk, VectorIndex
from ragforge.query_expansion import (
    MultiQueryRetriever,
    identity_expansion,
    llm_query_expansion_fn,
)


def _chunk(id_: str, text: str) -> Chunk:
    return Chunk(id=id_, text=text, doc_id=id_, position=0)


CHUNKS = [
    _chunk("c1", "Enterprise plans include a 99.99% uptime guarantee."),
    _chunk("c2", "You can terminate your subscription at any time from account settings."),
    _chunk("c3", "The weather in Paris is mild in autumn."),
]


def _build_hybrid_retriever() -> HybridRetriever:
    bm25 = BM25Index()
    bm25.add(CHUNKS)
    vector = VectorIndex()
    vector.add(CHUNKS)
    return HybridRetriever(bm25, vector)


def test_identity_expansion_returns_no_extra_variants():
    assert identity_expansion("anything") == []


def test_multi_query_retriever_finds_vocabulary_mismatch_via_expansion():
    retriever = _build_hybrid_retriever()

    def expand(query: str) -> list[str]:
        return ["terminate subscription cancel plan"]

    multi = MultiQueryRetriever(retriever, expand_fn=expand)

    # The literal query alone doesn't share vocabulary with c2 ("cancel my plan"
    # vs. "terminate your subscription"), but the expansion does.
    results = multi.search("cancel my plan", k=3)

    assert any(r.chunk.id == "c2" for r in results)
    assert all(r.provenance.startswith("multi_query(") for r in results)


def test_multi_query_retriever_always_includes_original_query():
    retriever = _build_hybrid_retriever()
    multi = MultiQueryRetriever(retriever, expand_fn=identity_expansion)

    results = multi.search("uptime guarantee", k=3)
    assert results[0].chunk.id == "c1"


def test_multi_query_retriever_deduplicates_identical_variants():
    retriever = _build_hybrid_retriever()
    calls: list[str] = []

    class TrackingRetriever:
        def search(self, query, k, filter_fn=None):
            calls.append(query)
            return retriever.search(query, k=k, filter_fn=filter_fn)

    multi = MultiQueryRetriever(TrackingRetriever(), expand_fn=lambda q: [q, q.strip(), q + ""])
    multi.search("uptime guarantee", k=2)

    assert calls == ["uptime guarantee"]


def test_multi_query_retriever_respects_max_expansions():
    retriever = _build_hybrid_retriever()
    calls: list[str] = []

    class TrackingRetriever:
        def search(self, query, k, filter_fn=None):
            calls.append(query)
            return retriever.search(query, k=k, filter_fn=filter_fn)

    multi = MultiQueryRetriever(
        TrackingRetriever(),
        expand_fn=lambda q: ["v1", "v2", "v3", "v4"],
        max_expansions=2,
    )
    multi.search("uptime", k=2)

    assert calls == ["uptime", "v1", "v2"]


def test_multi_query_retriever_k_zero_returns_empty():
    multi = MultiQueryRetriever(_build_hybrid_retriever())
    assert multi.search("uptime", k=0) == []


def test_multi_query_retriever_validates_parameters():
    with pytest.raises(ValueError, match="k_rrf must be positive"):
        MultiQueryRetriever(_build_hybrid_retriever(), k_rrf=0)
    with pytest.raises(ValueError, match="max_expansions must be non-negative"):
        MultiQueryRetriever(_build_hybrid_retriever(), max_expansions=-1)


def test_llm_query_expansion_fn_parses_numbered_lines():
    def fake_generate(query: str, contexts: list[str]) -> str:
        assert contexts == []
        return "1. cancel subscription\n2) terminate my plan\n- stop billing\n\n"

    expand = llm_query_expansion_fn(fake_generate, num_variants=3)
    variants = expand("cancel my plan")

    assert variants == ["cancel subscription", "terminate my plan", "stop billing"]


def test_llm_query_expansion_fn_truncates_to_num_variants():
    def fake_generate(query: str, contexts: list[str]) -> str:
        return "a\nb\nc\nd\ne"

    expand = llm_query_expansion_fn(fake_generate, num_variants=2)
    assert expand("q") == ["a", "b"]


def test_multi_query_retriever_merges_provenance_across_variants():
    class StubRetriever:
        def search(self, query, k, filter_fn=None):
            if query == "original":
                return [ScoredChunk(chunk=CHUNKS[0], score=1.0, provenance="bm25")]
            return [ScoredChunk(chunk=CHUNKS[0], score=0.9, provenance="vector")]

    multi = MultiQueryRetriever(StubRetriever(), expand_fn=lambda q: ["expanded"])
    results = multi.search("original", k=1)

    assert len(results) == 1
    assert "bm25" in results[0].provenance
    assert "vector" in results[0].provenance
