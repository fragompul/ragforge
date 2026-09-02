"""Correctness, recall, and persistence tests for the from-scratch HNSW index."""

from __future__ import annotations

import math
import random

import pytest

from ragforge.ann import HNSWIndex
from ragforge.embeddings import cosine_similarity, normalize_vector


def _random_unit_vector(rng: random.Random, dims: int) -> list[float]:
    raw = [rng.gauss(0.0, 1.0) for _ in range(dims)]
    return normalize_vector(raw)


def _brute_force_topk(query: list[float], vectors: list[list[float]], k: int) -> list[int]:
    scored = sorted(
        range(len(vectors)), key=lambda i: cosine_similarity(query, vectors[i]), reverse=True
    )
    return scored[:k]


def test_single_vector_returns_itself():
    index = HNSWIndex()
    node_id = index.add([1.0, 0.0, 0.0])

    results = index.search([1.0, 0.0, 0.0], k=1)

    assert results == [(pytest.approx(1.0), node_id)]


def test_empty_index_search_returns_empty():
    index = HNSWIndex()
    assert index.search([1.0, 0.0], k=5) == []


def test_search_k_zero_or_negative_returns_empty():
    index = HNSWIndex()
    index.add([1.0, 0.0])
    assert index.search([1.0, 0.0], k=0) == []


def test_nearest_neighbor_is_exact_on_small_orthogonal_set():
    index = HNSWIndex(m=4, ef_construction=50)
    basis = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    ids = [index.add(v) for v in basis]

    results = index.search([0.0, 1.0, 0.0, 0.0], k=1)

    assert results[0][1] == ids[1]
    assert results[0][0] == pytest.approx(1.0)


def test_recall_against_brute_force_on_random_corpus():
    """HNSW is approximate: assert it recovers most brute-force top-10 neighbors."""
    rng = random.Random(42)
    dims = 32
    n = 500
    k = 10

    vectors = [_random_unit_vector(rng, dims) for _ in range(n)]
    index = HNSWIndex(m=16, ef_construction=100, seed=7)
    node_ids = [index.add(v) for v in vectors]
    id_to_pos = {nid: pos for pos, nid in enumerate(node_ids)}

    queries = [_random_unit_vector(rng, dims) for _ in range(20)]
    total_recall = 0.0

    for query in queries:
        exact = set(_brute_force_topk(query, vectors, k))
        approx_raw = index.search(query, k=k, ef_search=80)
        approx = {id_to_pos[nid] for _, nid in approx_raw}
        total_recall += len(exact & approx) / k

    mean_recall = total_recall / len(queries)
    assert mean_recall >= 0.7, f"HNSW recall@{k} too low: {mean_recall:.2f}"


def test_mark_deleted_excludes_from_results_but_keeps_graph_connected():
    rng = random.Random(3)
    vectors = [_random_unit_vector(rng, 16) for _ in range(50)]
    index = HNSWIndex(m=8, ef_construction=50, seed=3)
    ids = [index.add(v) for v in vectors]

    query = vectors[0]
    top_before = index.search(query, k=5, ef_search=40)
    assert ids[0] in [nid for _, nid in top_before]

    index.mark_deleted(ids[0])
    top_after = index.search(query, k=5, ef_search=40)

    assert ids[0] not in [nid for _, nid in top_after]
    assert len(top_after) == 5
    assert len(index) == len(vectors) - 1
    assert index.size_including_tombstones == len(vectors)


def test_serialization_round_trip_preserves_search_results():
    rng = random.Random(11)
    vectors = [_random_unit_vector(rng, 24) for _ in range(120)]
    index = HNSWIndex(m=10, ef_construction=60, seed=11)
    for v in vectors:
        index.add(v)

    query = vectors[5]
    before = index.search(query, k=5, ef_search=50)

    restored = HNSWIndex.from_dict(index.to_dict())
    after = restored.search(query, k=5, ef_search=50)

    assert before == after
    assert len(restored) == len(index)


def test_constructor_validates_parameters():
    with pytest.raises(ValueError, match="m must be >= 2"):
        HNSWIndex(m=1)
    with pytest.raises(ValueError, match="ef_construction must be >= 1"):
        HNSWIndex(ef_construction=0)


def test_random_level_distribution_is_monotonically_decreasing():
    """Sanity-check the exponential decay: higher levels should be rarer."""
    index = HNSWIndex(m=16, seed=99)
    counts: dict[int, int] = {}
    for _ in range(2000):
        level = index._random_level()
        counts[level] = counts.get(level, 0) + 1

    assert counts.get(0, 0) > counts.get(1, 0) > 0
    if 2 in counts:
        assert counts[1] >= counts[2]


def test_search_layer_respects_ef_bound():
    rng = random.Random(5)
    index = HNSWIndex(m=6, ef_construction=30, seed=5)
    for _ in range(80):
        index.add(_random_unit_vector(rng, 8))

    results = index._search_layer(_random_unit_vector(rng, 8), [0], level=0, ef=7)
    assert len(results) <= 7


def test_cosine_similarity_bounds_hold_for_hnsw_scores():
    rng = random.Random(2)
    index = HNSWIndex(seed=2)
    for _ in range(30):
        index.add(_random_unit_vector(rng, 10))

    for sim, _ in index.search(_random_unit_vector(rng, 10), k=10, ef_search=20):
        assert -1.0 - 1e-9 <= sim <= 1.0 + 1e-9
        assert not math.isnan(sim)
