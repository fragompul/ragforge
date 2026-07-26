from ragforge.chunking import Chunk
from ragforge.index import BM25Index, HybridRetriever, VectorIndex


def _chunk(id_: str, text: str) -> Chunk:
    return Chunk(id=id_, text=text, doc_id=id_, position=0)


CHUNKS = [
    _chunk("c1", "The cat sat on the mat in the living room."),
    _chunk("c2", "Quarterly revenue exceeded analyst forecasts this year."),
    _chunk("c3", "A dog played fetch with a ball in the park."),
]


def test_bm25_ranks_exact_keyword_match_highest():
    index = BM25Index()
    index.add(CHUNKS)

    results = index.search("cat mat", k=3)

    assert results[0].chunk.id == "c1"
    assert results[0].score > 0


def test_bm25_returns_no_results_for_unmatched_query():
    index = BM25Index()
    index.add(CHUNKS)
    assert index.search("nonexistent gibberish term", k=3) == []


def test_vector_index_ranks_lexically_similar_text_highest():
    index = VectorIndex()
    index.add(CHUNKS)

    results = index.search("the cat sat on a mat", k=3)

    assert results[0].chunk.id == "c1"


def test_vector_index_empty_corpus_returns_empty():
    index = VectorIndex()
    assert index.search("anything", k=3) == []


def test_hybrid_retriever_fuses_bm25_and_vector_results():
    bm25 = BM25Index()
    bm25.add(CHUNKS)
    vector = VectorIndex()
    vector.add(CHUNKS)
    hybrid = HybridRetriever(bm25, vector)

    results = hybrid.search("cat mat", k=2)

    assert len(results) <= 2
    assert results[0].chunk.id == "c1"


def test_hybrid_retriever_deduplicates_candidates_present_in_both_lists():
    bm25 = BM25Index()
    bm25.add(CHUNKS)
    vector = VectorIndex()
    vector.add(CHUNKS)
    hybrid = HybridRetriever(bm25, vector)

    results = hybrid.search("cat mat", k=10)
    ids = [r.chunk.id for r in results]
    assert len(ids) == len(set(ids))
