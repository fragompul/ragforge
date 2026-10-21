"""Regression tests for the inverted-index BM25 implementation."""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from ragforge.chunking import Chunk
from ragforge.index import BM25Index, _tokenize


def _naive_bm25(chunks: list[Chunk], query: str, k1: float, b: float) -> list[tuple[str, float]]:
    """Reference implementation: the original O(n) scan over every chunk."""
    tokens = [_tokenize(c.text) for c in chunks]
    n = len(chunks)
    avg = sum(len(t) for t in tokens) / n
    df: Counter[str] = Counter()
    for t in tokens:
        df.update(set(t))
    out = []
    for chunk, toks in zip(chunks, tokens, strict=True):
        tf = Counter(toks)
        score = 0.0
        for term in _tokenize(query):
            if term not in tf:
                continue
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            norm = 1.0 - b + b * len(toks) / avg
            score += idf * tf[term] * (k1 + 1.0) / (tf[term] + k1 * norm)
        if score > 0:
            out.append((chunk.id, score))
    out.sort(key=lambda p: p[1], reverse=True)
    return out


def _corpus(n: int, seed: int) -> list[Chunk]:
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(40)]
    return [
        Chunk(
            id=f"c{i}",
            text=" ".join(rng.choice(vocab) for _ in range(rng.randint(3, 30))),
            doc_id=f"d{i % 7}",
            position=i,
        )
        for i in range(n)
    ]


def test_matches_naive_reference_on_random_corpus():
    chunks = _corpus(300, seed=1)
    index = BM25Index()
    index.add(chunks)
    rng = random.Random(2)

    for _ in range(25):
        query = " ".join(f"w{rng.randrange(45)}" for _ in range(rng.randint(1, 4)))
        expected = _naive_bm25(chunks, query, index.k1, index.b)[:10]
        actual = [(r.chunk.id, r.score) for r in index.search(query, k=10)]
        assert [cid for cid, _ in actual] == [cid for cid, _ in expected]
        for (_, a), (_, e) in zip(actual, expected, strict=True):
            assert a == pytest.approx(e)


def test_incremental_add_equals_single_add():
    chunks = _corpus(60, seed=3)
    one_shot = BM25Index()
    one_shot.add(chunks)
    incremental = BM25Index()
    for chunk in chunks:
        incremental.add([chunk])

    a = [(r.chunk.id, r.score) for r in one_shot.search("w1 w2 w3", k=20)]
    b = [(r.chunk.id, r.score) for r in incremental.search("w1 w2 w3", k=20)]
    assert a == b


def test_delete_rebuilds_statistics_exactly():
    chunks = _corpus(80, seed=4)
    full = BM25Index()
    full.add(chunks)
    removed = full.delete("d3")

    survivors = [c for c in chunks if c.doc_id != "d3"]
    fresh = BM25Index()
    fresh.add(survivors)

    assert removed == len(chunks) - len(survivors)
    assert [(r.chunk.id, r.score) for r in full.search("w5 w6", k=15)] == [
        (r.chunk.id, r.score) for r in fresh.search("w5 w6", k=15)
    ]
    assert full.delete("missing") == 0


def test_filter_applies_before_ranking_and_ties_are_stable():
    chunks = [
        Chunk(
            id=f"c{i}", text="alpha beta", doc_id=f"d{i}", position=0, metadata={"even": i % 2 == 0}
        )
        for i in range(6)
    ]
    index = BM25Index()
    index.add(chunks)

    results = index.search("alpha", k=10, filter_fn=lambda c: c.metadata["even"])

    assert [r.chunk.id for r in results] == ["c0", "c2", "c4"]


def test_repeated_query_terms_still_count_each_occurrence():
    index = BM25Index()
    index.add([Chunk(id="a", text="alpha beta", doc_id="a", position=0)])
    single = index.search("alpha", k=1)[0].score
    double = index.search("alpha alpha", k=1)[0].score
    assert double == pytest.approx(2 * single)


def test_english_analyzer_drops_stopwords_and_stems():
    from ragforge.index import english_analyzer

    assert english_analyzer("The Towers are standing in Paris, completed in 1889") == [
        "tower",
        "stand",
        "paris",
        "complet",
        "1889",
    ]
    assert english_analyzer("policies glasses boxes bus analysis") == [
        "policy",
        "glass",
        "box",
        "bus",
        "analysis",
    ]


def test_english_analyzer_matches_inflected_queries_and_roundtrips(tmp_path):
    chunks = [
        Chunk(id="a", text="The towers were completed in 1889", doc_id="a", position=0),
        Chunk(id="b", text="Unrelated text about rivers", doc_id="b", position=0),
    ]
    simple, english = BM25Index(), BM25Index(analyzer="english")
    simple.add(chunks)
    english.add(chunks)

    assert simple.search("tower completing", k=2) == []
    assert [r.chunk.id for r in english.search("tower completing", k=2)] == ["a"]

    path = tmp_path / "bm25.json"
    english.save(path)
    loaded = BM25Index.load(path)
    assert loaded.analyzer == "english"
    assert [r.chunk.id for r in loaded.search("towers", k=2)] == ["a"]
    assert "analyzer" not in simple.to_dict()


def test_unknown_analyzer_rejected():
    with pytest.raises(ValueError, match="analyzer"):
        BM25Index(analyzer="klingon")
