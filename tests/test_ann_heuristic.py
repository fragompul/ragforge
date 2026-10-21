"""Tests for HNSW heuristic neighbor selection, tracing, and legacy-format loading."""

from __future__ import annotations

import random

import pytest

from ragforge.ann import HNSWIndex
from ragforge.embeddings import normalize_vector


def _clustered(n: int, dims: int, clusters: int, rng: random.Random) -> list[list[float]]:
    centers = [normalize_vector([rng.gauss(0, 1) for _ in range(dims)]) for _ in range(clusters)]
    out = []
    for _ in range(n):
        c = rng.choice(centers)
        out.append(normalize_vector([x + rng.gauss(0, 0.15) for x in c]))
    return out


def _exact_topk(query: list[float], vectors: list[list[float]], k: int) -> set[int]:
    order = sorted(
        range(len(vectors)),
        key=lambda i: -sum(a * b for a, b in zip(query, vectors[i], strict=True)),
    )
    return set(order[:k])


def _mean_recall(index: HNSWIndex, vectors, queries, ef: int) -> float:
    total = 0.0
    for q in queries:
        exact = _exact_topk(q, vectors, 10)
        approx = {nid for _, nid in index.search(q, k=10, ef_search=ef)}
        total += len(exact & approx) / 10
    return total / len(queries)


def test_m_max0_defaults_to_twice_m():
    assert HNSWIndex(m=8).m_max0 == 16


def test_m_max0_below_m_is_rejected():
    with pytest.raises(ValueError, match="m_max0"):
        HNSWIndex(m=16, m_max0=8)


def test_heuristic_selection_beats_closest_m_on_clustered_data():
    rng = random.Random(3)
    vectors = _clustered(1500, 24, 15, rng)
    queries = _clustered(25, 24, 15, rng)

    recalls = {}
    for heuristic in (False, True):
        index = HNSWIndex(m=8, ef_construction=40, heuristic=heuristic, seed=5)
        for v in vectors:
            index.add(v)
        recalls[heuristic] = _mean_recall(index, vectors, queries, ef=50)

    assert recalls[True] > recalls[False] + 0.1
    assert recalls[True] > 0.8


def test_recall_improves_with_ef_search():
    rng = random.Random(11)
    vectors = _clustered(1200, 24, 10, rng)
    queries = _clustered(20, 24, 10, rng)
    index = HNSWIndex(m=8, ef_construction=60, seed=2)
    for v in vectors:
        index.add(v)

    narrow = _mean_recall(index, vectors, queries, ef=12)
    wide = _mean_recall(index, vectors, queries, ef=150)

    assert wide > narrow
    assert wide > 0.95


def test_near_duplicate_heavy_corpus_keeps_outliers_reachable():
    """Re-ingested boilerplate (near-identical vectors) must not strand distinct chunks."""
    rng = random.Random(21)

    def jittered() -> list[float]:
        return [1.0 + rng.gauss(0, 1e-3), rng.gauss(0, 1e-3), rng.gauss(0, 1e-3)]

    index = HNSWIndex(m=4, ef_construction=20)
    dups = [index.add(jittered()) for _ in range(3)]
    other = index.add([0.0, 1.0, 0.0])
    dups += [index.add(jittered()) for _ in range(60)]

    assert index.search([0.0, 1.0, 0.0], k=1, ef_search=40)[0][1] == other
    found = {nid for _, nid in index.search([1.0, 0.0, 0.0], k=10, ef_search=40)}
    assert len(found) == 10
    assert found <= set(dups)


def test_exact_duplicates_do_not_crowd_out_a_distinct_vector():
    index = HNSWIndex(m=4, ef_construction=20)
    for _ in range(3):
        index.add([1.0, 0.0, 0.0])
    other = index.add([0.0, 1.0, 0.0])
    for _ in range(27):
        index.add([1.0, 0.0, 0.0])

    assert index.search([0.0, 1.0, 0.0], k=1, ef_search=40)[0][1] == other


def test_vectors_are_normalized_on_insert():
    index = HNSWIndex()
    node = index.add([3.0, 4.0])
    assert index.vector(node) == pytest.approx([0.6, 0.8])
    # A scaled query still has cosine similarity 1.0.
    assert index.search([30.0, 40.0], k=1)[0][0] == pytest.approx(1.0)


def test_deleted_nodes_never_returned_but_graph_stays_navigable():
    rng = random.Random(8)
    vectors = _clustered(300, 16, 6, rng)
    index = HNSWIndex(m=6, ef_construction=40)
    ids = [index.add(v) for v in vectors]
    doomed = set(ids[:100])
    for nid in doomed:
        index.mark_deleted(nid)

    for q in vectors[:100]:
        results = index.search(q, k=5, ef_search=30)
        assert len(results) == 5
        assert not doomed & {nid for _, nid in results}
    assert index.is_deleted(ids[0])
    assert len(index) == 200


def test_search_trace_reports_every_layer_and_matches_search():
    rng = random.Random(4)
    vectors = _clustered(400, 16, 8, rng)
    index = HNSWIndex(m=4, ef_construction=30, seed=9)
    for v in vectors:
        index.add(v)

    trace = index.search_trace(vectors[0], k=5, ef_search=20)

    assert trace["results"] == index.search(vectors[0], k=5, ef_search=20)
    levels = [step["level"] for step in trace["path"]]
    assert levels == list(range(index.max_level, -1, -1))
    assert trace["path"][0]["entry"] == index.entry_point
    assert all(step["visited"][0] == step["entry"] for step in trace["path"])


def test_trace_on_empty_index_is_empty():
    assert HNSWIndex().search_trace([1.0, 0.0]) == {"results": [], "path": []}


def test_introspection_helpers():
    index = HNSWIndex(m=4, ef_construction=20)
    a = index.add([1.0, 0.0])
    b = index.add([0.9, 0.1])
    assert b in index.neighbors(a, 0)
    assert index.neighbors(a, 99) == []
    assert index.level_of(a) >= 0


def test_roundtrip_preserves_heuristic_flag_and_results():
    rng = random.Random(6)
    vectors = _clustered(200, 12, 5, rng)
    index = HNSWIndex(m=6, ef_construction=30, heuristic=False)
    for v in vectors:
        index.add(v)

    clone = HNSWIndex.from_dict(index.to_dict())

    assert clone.heuristic is False
    assert clone.search(vectors[3], k=5) == index.search(vectors[3], k=5)


def test_legacy_v01_payload_loads_and_normalizes():
    legacy = {
        "m": 4,
        "m_max0": 32,
        "ef_construction": 20,
        "seed": 1,
        "next_id": 2,
        "entry_point": 0,
        "max_level": 0,
        "vectors": {"0": [2.0, 0.0], "1": [0.0, 5.0]},
        "levels": {"0": 0, "1": 0},
        "neighbors": {"0": {"0": [1]}, "1": {"0": [0]}},
        "deleted": [],
    }
    index = HNSWIndex.from_dict(legacy)
    assert index.heuristic is True
    assert index.search([0.0, 1.0], k=1)[0] == (pytest.approx(1.0), 1)
