"""Hierarchical Navigable Small World (HNSW) approximate nearest-neighbor search.

A from-scratch, dependency-free implementation of the HNSW graph proposed by
Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search
using Hierarchical Navigable Small World graphs" (2016/2018, arXiv:1603.09320).

Why this exists: ``VectorIndex`` (see :mod:`ragforge.index`) performs an exact
brute-force cosine scan in ``O(n)`` per query. That is fine up to a few tens of
thousands of vectors, but production-scale corpora (10^6+ chunks) need
sub-linear search. HNSW builds a multi-layer proximity graph where the top
layers act as express lanes (few nodes, long edges) and the bottom layer is a
dense navigable graph (many nodes, short edges) -- mirroring a skip list, but
in metric space instead of a sorted sequence. Expected query complexity is
``O(log n)`` graph hops, each doing ``O(M)`` work, against ``O(n)`` for brute
force. See ``docs/math.md`` for the complexity argument and the tradeoffs
against exactness.

This implementation trades some of the original paper's sophistication
(notably heuristic neighbor selection, "extend candidates") for readability,
while keeping the two properties that make HNSW work:

1. Layer assignment via an exponentially decaying random level (fewer nodes
   as you go up), giving logarithmic-height navigation.
2. Greedy best-first search per layer, seeded by the entry point discovered
   one layer up.

Deletions are lazy (tombstones): removing a node's edges outright can
disconnect the graph, so a deleted node is excluded from *results* but its
edges are still traversed so downstream neighbors remain reachable. This is
the same approach used by production libraries such as hnswlib.
"""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ragforge.embeddings import cosine_similarity

NodeId = int


@dataclass
class HNSWIndex:
    """A dependency-free HNSW graph over vectors, keyed by opaque integer ids.

    Args:
        m: Max bidirectional connections per node at layers >= 1. Also
            controls the level-generation distribution (``1/ln(m)``). Higher
            ``m`` improves recall at the cost of memory and build time.
        m_max0: Max connections at layer 0 (the dense base layer). The
            original paper recommends ``2 * m``.
        ef_construction: Beam width used while inserting; larger values build
            a higher-quality (higher recall) graph more slowly.
        seed: Random seed for level assignment, so builds are reproducible.
    """

    m: int = 16
    m_max0: int = 32
    ef_construction: int = 200
    seed: int = 1337

    _vectors: dict[NodeId, list[float]] = field(default_factory=dict, init=False)
    _neighbors: dict[NodeId, dict[int, set[NodeId]]] = field(default_factory=dict, init=False)
    _levels: dict[NodeId, int] = field(default_factory=dict, init=False)
    _deleted: set[NodeId] = field(default_factory=set, init=False)
    _entry_point: NodeId | None = field(default=None, init=False)
    _max_level: int = field(default=-1, init=False)
    _next_id: NodeId = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.m < 2:
            raise ValueError(f"m must be >= 2, got {self.m}")
        if self.ef_construction < 1:
            raise ValueError(f"ef_construction must be >= 1, got {self.ef_construction}")
        self._level_mult = 1.0 / math.log(self.m)
        self._rng = random.Random(self.seed)

    def __len__(self) -> int:
        return len(self._vectors) - len(self._deleted)

    @property
    def size_including_tombstones(self) -> int:
        """Total nodes ever inserted, including lazily-deleted ones."""
        return len(self._vectors)

    def _random_level(self) -> int:
        # Exponential decay: P(level >= L) = m^-L, giving O(log n) expected height.
        return int(-math.log(self._rng.random() + 1e-12) * self._level_mult)

    def _sim(self, a: Sequence[float], b: Sequence[float]) -> float:
        return cosine_similarity(a, b)

    def add(self, vector: list[float]) -> NodeId:
        """Insert a vector, returning its assigned node id."""
        node_id = self._next_id
        self._next_id += 1
        self._vectors[node_id] = vector
        level = self._random_level()
        self._levels[node_id] = level
        self._neighbors[node_id] = {lc: set() for lc in range(level + 1)}

        if self._entry_point is None:
            self._entry_point = node_id
            self._max_level = level
            return node_id

        entry = self._entry_point
        for lc in range(self._max_level, level, -1):
            layer_results = self._search_layer(vector, [entry], lc, ef=1)
            if layer_results:
                entry = layer_results[0][1]

        for lc in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(vector, [entry], lc, ef=self.ef_construction)
            max_conn = self.m_max0 if lc == 0 else self.m
            selected = [nid for _, nid in candidates[:max_conn]]

            for neighbor_id in selected:
                self._neighbors[node_id][lc].add(neighbor_id)
                self._neighbors[neighbor_id].setdefault(lc, set()).add(node_id)
                self._prune(neighbor_id, lc, max_conn)

            if candidates:
                entry = candidates[0][1]

        if level > self._max_level:
            self._max_level = level
            self._entry_point = node_id

        return node_id

    def _prune(self, node_id: NodeId, level: int, max_conn: int) -> None:
        """Keep only the ``max_conn`` closest neighbors of ``node_id`` at ``level``."""
        neighbors = self._neighbors[node_id].get(level, set())
        if len(neighbors) <= max_conn:
            return
        vec = self._vectors[node_id]
        ranked = sorted(neighbors, key=lambda nid: self._sim(vec, self._vectors[nid]), reverse=True)
        self._neighbors[node_id][level] = set(ranked[:max_conn])

    def _search_layer(
        self, query: Sequence[float], entry_points: list[NodeId], level: int, ef: int
    ) -> list[tuple[float, NodeId]]:
        """Greedy best-first beam search over a single layer.

        Returns up to ``ef`` ``(similarity, node_id)`` pairs sorted descending,
        excluding tombstoned nodes -- though tombstoned nodes are still
        traversed so the graph stays connected around them.
        """
        visited: set[NodeId] = set(entry_points)
        candidates: list[tuple[float, NodeId]] = []  # max-heap via negated similarity
        results: list[tuple[float, NodeId]] = []  # min-heap of best `ef` results so far

        for ep in entry_points:
            sim = self._sim(query, self._vectors[ep])
            heapq.heappush(candidates, (-sim, ep))
            if ep not in self._deleted:
                heapq.heappush(results, (sim, ep))

        while candidates:
            neg_sim, current = heapq.heappop(candidates)
            current_sim = -neg_sim
            if results and len(results) >= ef and current_sim < results[0][0]:
                break

            for neighbor in self._neighbors.get(current, {}).get(level, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                n_sim = self._sim(query, self._vectors[neighbor])

                worst_kept = results[0][0] if len(results) >= ef else -math.inf
                if n_sim > worst_kept or len(results) < ef:
                    heapq.heappush(candidates, (-n_sim, neighbor))
                    if neighbor not in self._deleted:
                        heapq.heappush(results, (n_sim, neighbor))
                        if len(results) > ef:
                            heapq.heappop(results)

        return sorted(results, key=lambda pair: pair[0], reverse=True)

    def search(
        self, query: list[float], k: int = 5, ef_search: int | None = None
    ) -> list[tuple[float, NodeId]]:
        """Return up to ``k`` ``(similarity, node_id)`` pairs nearest to ``query``."""
        if self._entry_point is None or k <= 0:
            return []

        ef = max(ef_search or self.ef_construction, k)
        entry = self._entry_point
        for lc in range(self._max_level, 0, -1):
            layer_results = self._search_layer(query, [entry], lc, ef=1)
            if layer_results:
                entry = layer_results[0][1]

        results = self._search_layer(query, [entry], 0, ef=ef)
        return results[:k]

    def mark_deleted(self, node_id: NodeId) -> None:
        """Tombstone a node: excluded from future results, kept for graph connectivity."""
        self._deleted.add(node_id)

    def vector(self, node_id: NodeId) -> list[float]:
        return self._vectors[node_id]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full graph state (vectors, edges, levels, tombstones)."""
        return {
            "m": self.m,
            "m_max0": self.m_max0,
            "ef_construction": self.ef_construction,
            "seed": self.seed,
            "next_id": self._next_id,
            "entry_point": self._entry_point,
            "max_level": self._max_level,
            "vectors": {str(k): v for k, v in self._vectors.items()},
            "levels": {str(k): v for k, v in self._levels.items()},
            "neighbors": {
                str(node): {str(level): sorted(ids) for level, ids in layers.items()}
                for node, layers in self._neighbors.items()
            },
            "deleted": sorted(self._deleted),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HNSWIndex:
        """Deserialize a graph previously produced by :meth:`to_dict`."""
        index = cls(
            m=data.get("m", 16),
            m_max0=data.get("m_max0", 32),
            ef_construction=data.get("ef_construction", 200),
            seed=data.get("seed", 1337),
        )
        index._next_id = data.get("next_id", 0)
        index._entry_point = data.get("entry_point")
        index._max_level = data.get("max_level", -1)
        index._vectors = {int(k): v for k, v in data.get("vectors", {}).items()}
        index._levels = {int(k): v for k, v in data.get("levels", {}).items()}
        index._neighbors = {
            int(node): {int(level): set(ids) for level, ids in layers.items()}
            for node, layers in data.get("neighbors", {}).items()
        }
        index._deleted = set(data.get("deleted", []))
        return index
