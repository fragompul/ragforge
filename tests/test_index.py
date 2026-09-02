import tempfile
from pathlib import Path

import pytest

from ragforge.chunking import Chunk
from ragforge.index import ApproxVectorIndex, BM25Index, HybridRetriever, VectorIndex


def _chunk(id_: str, text: str, metadata: dict | None = None) -> Chunk:
    return Chunk(id=id_, text=text, doc_id=id_, position=0, metadata=metadata or {})


CHUNKS = [
    _chunk("c1", "The cat sat on the mat in the living room.", {"topic": "pets", "lang": "en"}),
    _chunk(
        "c2",
        "Quarterly revenue exceeded analyst forecasts this year.",
        {"topic": "finance", "lang": "en"},
    ),
    _chunk("c3", "A dog played fetch with a ball in the park.", {"topic": "pets", "lang": "en"}),
]


def test_bm25_ranks_exact_keyword_match_highest():
    index = BM25Index()
    index.add(CHUNKS)

    results = index.search("cat mat", k=3)

    assert results[0].chunk.id == "c1"
    assert results[0].score > 0
    assert results[0].provenance == "bm25"


def test_bm25_metadata_filter():
    index = BM25Index()
    index.add(CHUNKS)

    # Filter out pets, so c1 and c3 are skipped even if query matches
    results = index.search(
        "cat dog revenue", k=3, filter_fn=lambda c: c.metadata.get("topic") == "finance"
    )

    assert len(results) == 1
    assert results[0].chunk.id == "c2"


def test_bm25_delete_and_clear():
    index = BM25Index()
    index.add(CHUNKS)

    deleted = index.delete("c1")
    assert deleted == 1
    assert len(index._chunks) == 2
    assert index.search("cat mat", k=3) == []

    index.clear()
    assert len(index._chunks) == 0


def test_bm25_serialization():
    index = BM25Index(k1=1.2, b=0.8)
    index.add(CHUNKS)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bm25.json"
        index.save(path)

        loaded = BM25Index.load(path)
        assert loaded.k1 == 1.2
        assert loaded.b == 0.8
        assert len(loaded._chunks) == len(CHUNKS)

        results = loaded.search("cat mat", k=1)
        assert results[0].chunk.id == "c1"


def test_bm25_parameter_validations():
    with pytest.raises(ValueError, match="k1 must be non-negative"):
        BM25Index(k1=-0.1)
    with pytest.raises(ValueError, match="b must be between 0.0 and 1.0"):
        BM25Index(b=1.5)


def test_vector_index_search_and_filter():
    index = VectorIndex()
    index.add(CHUNKS)

    results = index.search("the cat sat on a mat", k=3)
    assert results[0].chunk.id == "c1"

    # With filter
    filtered = index.search(
        "the cat sat on a mat",
        k=3,
        filter_fn=lambda c: c.metadata.get("topic") == "finance",
    )
    assert all(r.chunk.metadata.get("topic") == "finance" for r in filtered)


def test_vector_index_add_vectors_and_delete():
    index = VectorIndex()
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    chunks = [_chunk("v1", "t1"), _chunk("v2", "t2")]

    index.add_vectors(chunks, vecs)
    assert len(index._chunks) == 2
    assert len(index._vectors) == 2

    # Mismatch check
    with pytest.raises(ValueError, match="Mismatch"):
        index.add_vectors([_chunk("v3", "t3")], [[1.0, 0.0], [0.0, 1.0]])

    deleted = index.delete("v1")
    assert deleted == 1
    assert len(index._chunks) == 1


def test_vector_index_serialization():
    index = VectorIndex()
    index.add(CHUNKS)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "vec.json"
        index.save(path)

        loaded = VectorIndex.load(path)
        assert len(loaded._chunks) == len(CHUNKS)
        results = loaded.search("cat", k=1)
        assert len(results) == 1


def test_approx_vector_index_search_and_filter():
    index = ApproxVectorIndex(m=8, ef_construction=50)
    index.add(CHUNKS)

    results = index.search("the cat sat on a mat", k=3)
    assert results[0].chunk.id == "c1"
    assert results[0].provenance == "vector:hnsw"

    filtered = index.search(
        "the cat sat on a mat",
        k=3,
        filter_fn=lambda c: c.metadata.get("topic") == "finance",
    )
    assert all(r.chunk.metadata.get("topic") == "finance" for r in filtered)


def test_approx_vector_index_delete_and_clear():
    index = ApproxVectorIndex(m=8, ef_construction=50)
    index.add(CHUNKS)

    removed = index.delete("c1")
    assert removed == 1
    results = index.search("cat mat", k=5)
    assert all(r.chunk.id != "c1" for r in results)

    index.clear()
    assert index.search("cat", k=5) == []


def test_approx_vector_index_add_vectors_mismatch():
    index = ApproxVectorIndex()
    with pytest.raises(ValueError, match="Mismatch"):
        index.add_vectors([_chunk("v1", "t1")], [[1.0, 0.0], [0.0, 1.0]])


def test_approx_vector_index_serialization_round_trip():
    index = ApproxVectorIndex(m=8, ef_construction=50)
    index.add(CHUNKS)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "hnsw.json"
        index.save(path)

        loaded = ApproxVectorIndex.load(path)
        results = loaded.search("cat mat", k=1)
        assert results[0].chunk.id == "c1"


def test_approx_vector_index_empty_search():
    index = ApproxVectorIndex()
    assert index.search("anything", k=3) == []
    assert index.search("cat", k=0) == []


def test_hybrid_retriever_accepts_approx_vector_index():
    bm25 = BM25Index()
    bm25.add(CHUNKS)
    approx_vector = ApproxVectorIndex(m=8, ef_construction=50)
    approx_vector.add(CHUNKS)

    hybrid = HybridRetriever(bm25, approx_vector)
    results = hybrid.search("cat mat", k=2)

    assert len(results) <= 2
    assert results[0].chunk.id == "c1"


def test_hybrid_retriever_weighted_rrf_and_ranks():
    bm25 = BM25Index()
    bm25.add(CHUNKS)
    vector = VectorIndex()
    vector.add(CHUNKS)

    hybrid = HybridRetriever(bm25, vector, k_rrf=60, weight_bm25=2.0, weight_vector=1.0)

    results = hybrid.search("cat mat", k=2)

    assert len(results) <= 2
    assert results[0].chunk.id == "c1"
    assert "bm25" in results[0].provenance or "vector" in results[0].provenance
    assert "bm25" in results[0].ranks


def test_hybrid_retriever_filter_and_validations():
    bm25 = BM25Index()
    bm25.add(CHUNKS)
    vector = VectorIndex()
    vector.add(CHUNKS)

    hybrid = HybridRetriever(bm25, vector)
    results = hybrid.search(
        "cat dog revenue",
        k=5,
        filter_fn=lambda c: c.metadata.get("topic") == "finance",
    )
    assert len(results) == 1
    assert results[0].chunk.id == "c2"

    with pytest.raises(ValueError, match="k_rrf must be positive"):
        HybridRetriever(bm25, vector, k_rrf=0)
    with pytest.raises(ValueError, match="weight_bm25 must be non-negative"):
        HybridRetriever(bm25, vector, weight_bm25=-1.0)
    with pytest.raises(ValueError, match="weight_vector must be non-negative"):
        HybridRetriever(bm25, vector, weight_vector=-1.0)
