"""ApproxVectorIndex: in-graph filtering with exact fallback, and compaction."""

from __future__ import annotations

import random

from ragforge.chunking import Chunk
from ragforge.embeddings import normalize_vector
from ragforge.index import ApproxVectorIndex, VectorIndex


def _build(n: int = 600, seed: int = 0):
    rng = random.Random(seed)
    centers = [normalize_vector([rng.gauss(0, 1) for _ in range(16)]) for _ in range(6)]
    vectors = [
        normalize_vector([x + rng.gauss(0, 0.2) for x in rng.choice(centers)]) for _ in range(n)
    ]
    table = {f"t{i}": v for i, v in enumerate(vectors)}
    queries = {f"q{i}": normalize_vector([rng.gauss(0, 1) for _ in range(16)]) for i in range(15)}
    lookup = {**table, **queries}
    chunks = [
        Chunk(id=f"c{i}", text=f"t{i}", doc_id=f"d{i}", position=0, metadata={"bucket": i % 20})
        for i in range(n)
    ]
    embed = lookup.__getitem__
    ann = ApproxVectorIndex(embed_fn=embed, m=8, ef_construction=60, ef_search=40)
    exact = VectorIndex(embed_fn=embed)
    ann.add_vectors(chunks, vectors)
    exact.add_vectors(chunks, vectors)
    return ann, exact, list(queries), chunks


def test_filtered_results_match_exact_when_filter_is_moderately_selective():
    ann, exact, queries, _ = _build()
    keep = lambda c: c.metadata["bucket"] < 10  # noqa: E731  (50 % pass)
    agree = 0.0
    for q in queries:
        got = {r.chunk.id for r in ann.search(q, k=10, filter_fn=keep)}
        truth = {r.chunk.id for r in exact.search(q, k=10, filter_fn=keep)}
        assert len(got) == 10
        agree += len(got & truth) / 10
    assert agree / len(queries) > 0.9


def test_extremely_selective_filter_falls_back_to_exact_scan_and_never_under_returns():
    ann, exact, queries, _ = _build()
    keep = lambda c: c.metadata["bucket"] == 3  # noqa: E731  (5 % pass -> 30 chunks)
    ann.filter_max_visits = 30  # force the traversal to give up early
    for q in queries:
        got = ann.search(q, k=10, filter_fn=keep)
        truth = exact.search(q, k=10, filter_fn=keep)
        assert [r.chunk.id for r in got] == [r.chunk.id for r in truth]
        assert all(r.provenance == "vector:hnsw+scan" for r in got)


def test_filter_matching_fewer_than_k_returns_all_matches_and_none_if_no_match():
    ann, _, queries, _ = _build()
    one_doc = ann.search(queries[0], k=10, filter_fn=lambda c: c.doc_id == "d17")
    assert [r.chunk.doc_id for r in one_doc] == ["d17"]
    assert ann.search(queries[0], k=10, filter_fn=lambda c: False) == []


def test_unfiltered_path_still_reports_hnsw_provenance():
    ann, _, queries, _ = _build()
    assert {r.provenance for r in ann.search(queries[0], k=5)} == {"vector:hnsw"}


def test_filter_respects_deleted_documents():
    ann, _, queries, chunks = _build()
    ann.delete("d5")
    got = ann.search(queries[0], k=600, filter_fn=lambda c: True)
    assert "d5" not in {r.chunk.doc_id for r in got}


def test_compact_reclaims_tombstones_and_keeps_chunk_mapping():
    ann, exact, queries, chunks = _build(n=300)
    for i in range(0, 300, 3):
        ann.delete(f"d{i}")
        exact.delete(f"d{i}")
    assert ann.compact() == 100
    assert ann.compact() == 0
    assert ann._graph.size_including_tombstones == 200

    q = queries[0]
    got = {r.chunk.id for r in ann.search(q, k=10)}
    truth = {r.chunk.id for r in exact.search(q, k=10)}
    assert len(got & truth) >= 8
    assert all(r.chunk.doc_id in {c.doc_id for c in chunks} for r in ann.search(q, k=10))


def test_filter_max_visits_roundtrips():
    ann, _, _, _ = _build(n=60)
    ann.filter_max_visits = 123
    assert ApproxVectorIndex.from_dict(ann.to_dict()).filter_max_visits == 123
