"""HNSW robustness: reachability under duplicate floods, filtered search, compaction."""

from __future__ import annotations

import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ragforge.ann import HNSWIndex
from ragforge.embeddings import normalize_vector

BASES = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.6, 0.8, 0.0], [0.0, 0.6, 0.8]]


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    picks=st.lists(st.integers(min_value=0, max_value=len(BASES) - 1), min_size=1, max_size=120),
    jitter=st.sampled_from([0.0, 1e-9, 1e-3]),
    m=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=50),
)
def test_every_node_is_reachable_even_in_duplicate_floods(picks, jitter, m, seed):
    rng = random.Random(seed)
    index = HNSWIndex(m=m, ef_construction=12, seed=seed)
    for pick in picks:
        base = BASES[pick]
        index.add([x + rng.gauss(0, jitter) for x in base] if jitter else list(base))

    for level in range(index.max_level + 1):
        assert index.unreachable(level) == []


def test_exact_duplicate_flood_returns_ten_distinct_duplicates():
    index = HNSWIndex(m=4, ef_construction=20)
    dups = [index.add([1.0, 0.0, 0.0]) for _ in range(3)]
    other = index.add([0.0, 1.0, 0.0])
    dups += [index.add([1.0, 0.0, 0.0]) for _ in range(27)]

    assert index.search([0.0, 1.0, 0.0], k=1, ef_search=40)[0][1] == other
    found = {nid for _, nid in index.search([1.0, 0.0, 0.0], k=10, ef_search=40)}
    assert len(found) == 10 and found <= set(dups)
    assert index.unreachable() == []


def test_pinned_backbone_survives_pruning_and_serialization():
    index = HNSWIndex(m=2, ef_construction=10)
    for _ in range(40):
        index.add([1.0, 0.0])
    for level, a, b in index._pinned:
        assert b in index.neighbors(a, level) and a in index.neighbors(b, level)

    clone = HNSWIndex.from_dict(index.to_dict())
    assert clone._pinned == index._pinned and clone.unreachable() == []


def test_pinned_children_are_balanced_so_hubs_do_not_grow_unbounded():
    index = HNSWIndex(m=4, ef_construction=30)
    for _ in range(300):
        index.add([1.0, 0.0, 0.0])
    assert max(index._pinned_children.values()) <= 6  # bounded, not O(n)


def _clustered(n: int, rng: random.Random) -> list[list[float]]:
    centers = [normalize_vector([rng.gauss(0, 1) for _ in range(16)]) for _ in range(8)]
    return [
        normalize_vector([x + rng.gauss(0, 0.2) for x in rng.choice(centers)]) for _ in range(n)
    ]


def test_filtered_search_only_returns_accepted_nodes_with_good_recall():
    rng = random.Random(1)
    vectors = _clustered(800, rng)
    index = HNSWIndex(m=8, ef_construction=60, seed=3)
    ids = [index.add(v) for v in vectors]
    allowed = {nid for nid in ids if nid % 3 == 0}  # 33 % selectivity

    total = 0.0
    queries = _clustered(20, rng)
    for q in queries:
        got = index.search(q, k=10, ef_search=80, accept=allowed.__contains__)
        assert got and all(nid in allowed for _, nid in got)
        exact = sorted(
            allowed, key=lambda i: -sum(a * b for a, b in zip(q, vectors[i], strict=True))
        )
        total += len({nid for _, nid in got} & set(exact[:10])) / 10
    assert total / len(queries) > 0.9


def test_very_selective_filter_with_visit_cap_can_return_fewer_results():
    rng = random.Random(2)
    vectors = _clustered(400, rng)
    index = HNSWIndex(m=6, ef_construction=40)
    ids = [index.add(v) for v in vectors]
    rare = {ids[7]}
    uncapped = index.search(vectors[0], k=5, ef_search=20, accept=rare.__contains__)
    capped = index.search(vectors[0], k=5, ef_search=20, accept=rare.__contains__, max_visits=5)

    assert [nid for _, nid in uncapped] == [ids[7]]
    assert len(capped) <= 1


def test_filter_combines_with_tombstones():
    index = HNSWIndex(m=4, ef_construction=20)
    ids = [index.add([1.0, i * 0.01]) for i in range(30)]
    index.mark_deleted(ids[0])
    got = index.search([1.0, 0.0], k=5, ef_search=30, accept=lambda n: n % 2 == 0)
    assert got and all(nid % 2 == 0 and nid != ids[0] for _, nid in got)


def test_compact_drops_tombstones_and_renumbers_densely():
    rng = random.Random(4)
    vectors = _clustered(300, rng)
    index = HNSWIndex(m=6, ef_construction=40, seed=9)
    ids = [index.add(v) for v in vectors]
    for nid in ids[::2]:
        index.mark_deleted(nid)
    before = {tuple(index.vector(nid)) for _, nid in index.search(vectors[1], k=8, ef_search=60)}

    mapping = index.compact()

    assert index.size_including_tombstones == len(index) == 150
    assert sorted(mapping.values()) == list(range(150)) and set(mapping) == set(ids[1::2])
    assert index.unreachable() == []
    after = {tuple(index.vector(nid)) for _, nid in index.search(vectors[1], k=8, ef_search=60)}
    assert len(before & after) >= 6  # same live vectors, graph rebuilt


def test_compact_on_empty_and_fully_deleted_index():
    empty = HNSWIndex()
    assert empty.compact() == {}
    index = HNSWIndex()
    node = index.add([1.0, 0.0])
    index.mark_deleted(node)
    assert index.compact() == {} and len(index) == 0 and index.search([1.0, 0.0]) == []
    assert index.add([0.0, 1.0]) == 0


def test_unreachable_detects_a_severed_node():
    index = HNSWIndex(m=2, ef_construction=8)
    for i in range(6):
        index.add([1.0, i * 0.1])
    victim = 5
    for level in range(index.level_of(victim) + 1):
        for nb in list(index.neighbors(victim, level)):
            index._neighbors[nb][level].discard(victim)
    assert victim in index.unreachable(0)
    assert HNSWIndex().unreachable() == []


def test_legacy_payload_without_pinned_loads():
    index = HNSWIndex(m=3)
    for i in range(10):
        index.add([1.0, i * 0.1])
    payload = index.to_dict()
    payload.pop("pinned")
    assert HNSWIndex.from_dict(payload).search([1.0, 0.0], k=3)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_tiny_graphs_are_connected(n):
    index = HNSWIndex(m=2)
    for _ in range(n):
        index.add([1.0, 0.0])
    assert index.unreachable() == []
