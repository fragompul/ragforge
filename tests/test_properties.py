"""Property-based tests (Hypothesis) for invariants that hold across *any*
valid input, complementing the example-based unit tests elsewhere.

Example-based tests (``test_chunking.py``, ``test_index.py``, ``test_ann.py``)
pin down specific, documented behaviors on hand-picked inputs. These tests
instead assert properties that must hold universally -- e.g. "a chunk's
recorded offsets always locate its exact text in the source document" -- and
let Hypothesis search for inputs that violate them. Example counts are
capped (``max_examples``) to keep CI runtime predictable; this trades
exhaustiveness for breadth against the targeted unit tests.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from ragforge.ann import HNSWIndex
from ragforge.chunking import (
    Chunk,
    FixedSizeChunker,
    MarkdownChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
)
from ragforge.embeddings import cosine_similarity, normalize_vector
from ragforge.index import BM25Index, VectorIndex

_TEXT = st.text(min_size=0, max_size=400)
_FAST = settings(max_examples=50, deadline=None)


@_FAST
@given(text=_TEXT, chunk_size=st.integers(min_value=1, max_value=50))
def test_fixed_size_chunker_never_exceeds_word_budget(text: str, chunk_size: int) -> None:
    overlap = chunk_size - 1 if chunk_size > 1 else 0
    chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)

    for chunk in chunker.chunk(text, doc_id="d"):
        assert len(chunk.text.split()) <= chunk_size


@_FAST
@given(
    text=_TEXT,
    chunk_size=st.integers(min_value=10, max_value=200),
    overlap=st.integers(min_value=0, max_value=9),
)
def test_recursive_chunker_offsets_always_locate_exact_text(
    text: str, chunk_size: int, overlap: int
) -> None:
    chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    clean_text = text.strip()

    for chunk in chunker.chunk(text, doc_id="d"):
        assert 0 <= chunk.start_char <= chunk.end_char <= len(clean_text)
        # The offsets ragforge attaches to a chunk (for citation/highlighting)
        # must round-trip back to that chunk's exact text in the source.
        assert clean_text[chunk.start_char : chunk.end_char] == chunk.text


@_FAST
@given(text=_TEXT, max_chars=st.integers(min_value=20, max_value=300))
def test_sentence_chunker_drops_no_non_whitespace_characters(text: str, max_chars: int) -> None:
    chunker = SentenceChunker(max_chars=max_chars)
    chunks = chunker.chunk(text, doc_id="d")

    original_chars = set("".join(text.split()))
    chunked_chars = set("".join("".join(c.text.split()) for c in chunks))
    # Overlap may *repeat* characters across chunks, but every character
    # present in the source must survive somewhere in the chunked output.
    assert original_chars <= chunked_chars


@_FAST
@given(text=_TEXT)
def test_markdown_chunker_never_crashes_on_arbitrary_text(text: str) -> None:
    chunks = MarkdownChunker().chunk(text, doc_id="d")
    assert isinstance(chunks, list)
    for chunk in chunks:
        assert chunk.doc_id == "d"


@_FAST
@given(
    texts=st.lists(st.text(min_size=1, max_size=80), min_size=0, max_size=15),
    k=st.integers(min_value=1, max_value=10),
)
def test_bm25_search_respects_k_and_returns_sorted_scores(texts: list[str], k: int) -> None:
    chunks = [
        Chunk(id=f"c{i}", text=text, doc_id=f"c{i}", position=0)
        for i, text in enumerate(texts)
        if text.strip()
    ]
    index = BM25Index()
    index.add(chunks)

    results = index.search("the quick brown fox", k=k)

    assert len(results) <= k
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@_FAST
@given(
    texts=st.lists(st.text(min_size=1, max_size=80), min_size=0, max_size=15),
    k=st.integers(min_value=1, max_value=10),
)
def test_vector_index_search_respects_k_and_returns_sorted_scores(texts: list[str], k: int) -> None:
    chunks = [
        Chunk(id=f"c{i}", text=text, doc_id=f"c{i}", position=0)
        for i, text in enumerate(texts)
        if text.strip()
    ]
    index = VectorIndex()
    index.add(chunks)

    results = index.search("the quick brown fox", k=k)

    assert len(results) <= k
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@_FAST
@given(
    n=st.integers(min_value=1, max_value=40),
    dims=st.integers(min_value=2, max_value=16),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_hnsw_search_result_invariants_hold_for_random_corpora(
    n: int, dims: int, seed: int
) -> None:
    import random as _random

    rng = _random.Random(seed)
    vectors = [normalize_vector([rng.gauss(0.0, 1.0) for _ in range(dims)]) for _ in range(n)]

    index = HNSWIndex(m=4, ef_construction=30, seed=seed)
    inserted_ids = [index.add(v) for v in vectors]

    query = normalize_vector([rng.gauss(0.0, 1.0) for _ in range(dims)])
    k = min(5, n)
    results = index.search(query, k=k, ef_search=30)

    assert len(results) <= k
    result_ids = [nid for _, nid in results]
    assert len(result_ids) == len(set(result_ids)), "HNSW returned duplicate node ids"
    assert all(nid in inserted_ids for nid in result_ids)

    similarities = [sim for sim, _ in results]
    assert similarities == sorted(similarities, reverse=True)
    for sim in similarities:
        assert -1.0 - 1e-9 <= sim <= 1.0 + 1e-9


@_FAST
@given(
    st.integers(min_value=1, max_value=20).flatmap(
        lambda n: st.tuples(
            st.lists(
                st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            ),
            st.lists(
                st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            ),
        )
    )
)
def test_cosine_similarity_is_always_bounded(vectors: tuple[list[float], list[float]]) -> None:
    a, b = vectors
    similarity = cosine_similarity(a, b)
    assert -1.0 - 1e-9 <= similarity <= 1.0 + 1e-9
