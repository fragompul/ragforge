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

Implementation notes -- the three properties that make HNSW work:

1. Layer assignment via an exponentially decaying random level (fewer nodes
   as you go up), giving logarithmic-height navigation.
2. Greedy best-first beam search per layer, seeded by the entry point found
   one layer up.
3. **Heuristic neighbor selection** (Algorithm 4 of the paper): a candidate is
   linked only if it is closer to the new node than to every neighbor already
   selected. This keeps edges *diverse* -- pointing into different directions
   and clusters -- instead of wiring a node to its ``M`` nearest (and mutually
   redundant) neighbors. Without it, clustered or duplicate-heavy data
   fragments into islands that greedy search can never leave, and recall
   collapses as the corpus grows (see ``docs/benchmarks.md``).

Vectors are L2-normalized on insertion, so cosine similarity reduces to a
single dot product in the hot loop.

Deletions are lazy (tombstones): removing a node's edges outright can
disconnect the graph, so a deleted node is excluded from *results* but its
edges are still traversed so downstream neighbors remain reachable. This is
the same approach used by production libraries such as hnswlib.
"""

from __future__ import annotations

import heapq
import math
import operator
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ragforge.embeddings import normalize_vector

NodeId = int

# Unit vectors this similar are treated as exact duplicates during neighbor selection.
_DUPLICATE_SIM = 1.0 - 1e-9


@dataclass
class HNSWIndex:
    """A dependency-free HNSW graph over vectors, keyed by opaque integer ids.

    Args:
        m: Number of bidirectional links created per inserted node. Also
            controls the level-generation distribution (``1/ln(m)``). Higher
            ``m`` improves recall at the cost of memory and build time.
        m_max0: Max connections kept at layer 0 (the dense base layer).
            Defaults to ``2 * m``, as recommended by the original paper.
        ef_construction: Beam width used while inserting; larger values build
            a higher-quality (higher recall) graph more slowly.
        seed: Random seed for level assignment, so builds are reproducible.
        heuristic: Use the paper's diversity-aware neighbor selection
            (Algorithm 4). Disabling it falls back to "link the M closest",
            which is simpler but fragments clustered data.
    """

    m: int = 16
    m_max0: int | None = None
    ef_construction: int = 200
    seed: int = 1337
    heuristic: bool = True

    _vectors: dict[NodeId, list[float]] = field(default_factory=dict, init=False)
    _neighbors: dict[NodeId, dict[int, set[NodeId]]] = field(default_factory=dict, init=False)
    _levels: dict[NodeId, int] = field(default_factory=dict, init=False)
    _deleted: set[NodeId] = field(default_factory=set, init=False)
    _entry_point: NodeId | None = field(default=None, init=False)
    _max_level: int = field(default=-1, init=False)
    _next_id: NodeId = field(default=0, init=False)
    # Reachability backbone: one pinned (never pruned) bidirectional edge between each node and
    # a parent chosen at insertion, per layer. Keys are (level, low_id, high_id).
    _pinned: set[tuple[int, int, int]] = field(default_factory=set, init=False)
    _pinned_children: dict[tuple[int, int], int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.m < 2:
            raise ValueError(f"m must be >= 2, got {self.m}")
        if self.ef_construction < 1:
            raise ValueError(f"ef_construction must be >= 1, got {self.ef_construction}")
        if self.m_max0 is None:
            self.m_max0 = 2 * self.m
        if self.m_max0 < self.m:
            raise ValueError(f"m_max0 must be >= m, got m_max0={self.m_max0} < m={self.m}")
        self._level_mult = 1.0 / math.log(self.m)
        self._rng = random.Random(self.seed)

    def __len__(self) -> int:
        return len(self._vectors) - len(self._deleted)

    @property
    def size_including_tombstones(self) -> int:
        """Total nodes ever inserted, including lazily-deleted ones."""
        return len(self._vectors)

    @property
    def entry_point(self) -> NodeId | None:
        """Id of the top-layer entry node, or ``None`` for an empty graph."""
        return self._entry_point

    @property
    def max_level(self) -> int:
        """Highest populated layer, or ``-1`` for an empty graph."""
        return self._max_level

    def level_of(self, node_id: NodeId) -> int:
        """Highest layer ``node_id`` participates in."""
        return self._levels[node_id]

    def neighbors(self, node_id: NodeId, level: int = 0) -> list[NodeId]:
        """Sorted neighbor ids of ``node_id`` at ``level`` (empty if absent there)."""
        return sorted(self._neighbors.get(node_id, {}).get(level, ()))

    def is_deleted(self, node_id: NodeId) -> bool:
        return node_id in self._deleted

    def unreachable(self, level: int = 0) -> list[NodeId]:
        """Nodes that cannot be reached from the entry point along ``level`` edges.

        A healthy graph returns ``[]`` at layer 0: every node is reachable, so a query can
        in principle find it. (Tombstoned nodes are traversed like any other.)
        """
        if self._entry_point is None:
            return []
        members = [n for n, lv in self._levels.items() if lv >= level]
        seen = {self._entry_point}
        stack = [self._entry_point]
        while stack:
            current = stack.pop()
            for nb in self._neighbors.get(current, {}).get(level, ()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return sorted(n for n in members if n not in seen)

    def compact(self) -> dict[NodeId, NodeId]:
        """Rebuild the graph without tombstones; return the ``old_id -> new_id`` mapping.

        Lazy deletion keeps dead nodes in the graph forever, which costs memory and widens
        every beam. ``compact`` re-inserts only live vectors (ids are renumbered densely, in
        their original order) and clears the tombstone set. ``O(n log n)`` -- run it when
        ``size_including_tombstones`` is much larger than ``len(index)``.
        """
        live = [
            (nid, vec) for nid, vec in sorted(self._vectors.items()) if nid not in self._deleted
        ]
        self._vectors, self._neighbors, self._levels = {}, {}, {}
        self._deleted, self._pinned, self._pinned_children = set(), set(), {}
        self._entry_point, self._max_level, self._next_id = None, -1, 0
        self._rng = random.Random(self.seed)
        mapping: dict[NodeId, NodeId] = {}
        for old_id, vec in live:
            mapping[old_id] = self.add(vec)
        return mapping

    def _random_level(self) -> int:
        # Exponential decay: P(level >= L) = m^-L, giving O(log n) expected height.
        return int(-math.log(self._rng.random() + 1e-12) * self._level_mult)

    @staticmethod
    def _sim(a: Sequence[float], b: Sequence[float]) -> float:
        # Stored vectors and queries are unit-length, so cosine == dot product.
        return float(sum(map(operator.mul, a, b)))

    def _cap(self, level: int) -> int:
        assert self.m_max0 is not None
        return self.m_max0 if level == 0 else self.m

    def add(self, vector: Sequence[float]) -> NodeId:
        """Insert a vector (L2-normalized internally), returning its node id."""
        unit = normalize_vector(vector)
        node_id = self._next_id
        self._next_id += 1
        self._vectors[node_id] = unit
        level = self._random_level()
        self._levels[node_id] = level
        self._neighbors[node_id] = {lc: set() for lc in range(level + 1)}

        if self._entry_point is None:
            self._entry_point = node_id
            self._max_level = level
            return node_id

        entry = self._entry_point
        for lc in range(self._max_level, level, -1):
            layer_results = self._search_layer(unit, [entry], lc, ef=1)
            if layer_results:
                entry = layer_results[0][1]

        for lc in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(unit, [entry], lc, ef=self.ef_construction)
            selected = self._select_neighbors(candidates, self.m)

            self._pin_parent(node_id, lc, candidates)
            for neighbor_id in selected:
                self._neighbors[node_id][lc].add(neighbor_id)
                self._neighbors[neighbor_id].setdefault(lc, set()).add(node_id)
                self._prune(neighbor_id, lc, self._cap(lc))

            if candidates:
                entry = candidates[0][1]

        if level > self._max_level:
            self._max_level = level
            self._entry_point = node_id

        return node_id

    def _select_neighbors(
        self, candidates: list[tuple[float, NodeId]], max_links: int
    ) -> list[NodeId]:
        """Pick up to ``max_links`` neighbors from ``(similarity, id)`` candidates.

        ``candidates`` must be sorted by similarity to the base node, descending.
        With ``heuristic=True`` a candidate is kept only if it is closer to the
        base node than to any already-kept neighbor (diversity); slots left
        over are back-filled with the best discarded candidates so the node
        never ends up under-connected.
        """
        if not self.heuristic or len(candidates) <= max_links:
            return [nid for _, nid in candidates[:max_links]]

        selected: list[tuple[float, NodeId]] = []
        discarded: list[NodeId] = []
        for sim_to_base, cand in candidates:
            if len(selected) >= max_links:
                break
            cand_vec = self._vectors[cand]
            if all(
                # Reject a candidate strictly closer to a kept neighbor than to the
                # base node, or an exact duplicate of one (equal-distance ties would
                # otherwise let a pile of identical vectors crowd out every
                # distinct neighbor and strand outliers with no inbound edges).
                (sim := self._sim(cand_vec, self._vectors[sel])) <= sim_to_base
                and sim < _DUPLICATE_SIM
                for _, sel in selected
            ):
                selected.append((sim_to_base, cand))
            else:
                discarded.append(cand)

        chosen = [nid for _, nid in selected]
        chosen.extend(discarded[: max_links - len(chosen)])
        return chosen

    def _prune(self, node_id: NodeId, level: int, max_conn: int) -> None:
        """Shrink ``node_id``'s neighbor set at ``level`` to at most ``max_conn``."""
        neighbors = self._neighbors[node_id].get(level, set())
        if len(neighbors) <= max_conn:
            return
        pinned = {nb for nb in neighbors if self._is_pinned(level, node_id, nb)}
        vec = self._vectors[node_id]
        ranked = sorted(
            ((self._sim(vec, self._vectors[nid]), nid) for nid in neighbors if nid not in pinned),
            key=lambda pair: (-pair[0], pair[1]),
        )
        free_slots = max(0, max_conn - len(pinned))
        self._neighbors[node_id][level] = pinned | set(self._select_neighbors(ranked, free_slots))

    @staticmethod
    def _pin_key(level: int, a: NodeId, b: NodeId) -> tuple[int, int, int]:
        return (level, a, b) if a < b else (level, b, a)

    def _is_pinned(self, level: int, a: NodeId, b: NodeId) -> bool:
        return self._pin_key(level, a, b) in self._pinned

    def _pin_parent(
        self, node_id: NodeId, level: int, candidates: list[tuple[float, NodeId]]
    ) -> None:
        """Pin a never-pruned edge between ``node_id`` and a near, lightly loaded parent.

        Pruning can drop every inbound edge of a node when many near-identical vectors
        compete for the same slots, leaving it unreachable. One pinned parent edge per node
        and layer makes the graph connected by construction (a spanning tree of pinned
        edges), while preferring parents with few pinned children keeps hub degrees bounded.
        """
        if not candidates:
            return
        parent = next(
            (nid for _, nid in candidates if self._pinned_children.get((level, nid), 0) < 2),
            min(
                (nid for _, nid in candidates),
                key=lambda n: self._pinned_children.get((level, n), 0),
            ),
        )
        self._pinned.add(self._pin_key(level, node_id, parent))
        self._pinned_children[(level, parent)] = self._pinned_children.get((level, parent), 0) + 1
        self._neighbors[node_id][level].add(parent)
        self._neighbors[parent].setdefault(level, set()).add(node_id)

    def _search_layer(
        self,
        query: Sequence[float],
        entry_points: list[NodeId],
        level: int,
        ef: int,
        visit_log: list[NodeId] | None = None,
        accept: Callable[[NodeId], bool] | None = None,
        max_visits: int | None = None,
    ) -> list[tuple[float, NodeId]]:
        """Greedy best-first beam search over a single layer.

        Returns up to ``ef`` ``(similarity, node_id)`` pairs sorted descending. Every node
        takes part in *navigation* (so the graph stays traversable around tombstones and
        filtered-out nodes), but with ``accept`` only nodes it approves can enter the result
        set, and the beam's stopping rule is evaluated over those accepted results alone.
        ``max_visits`` bounds the work when ``accept`` is very selective. If given,
        ``visit_log`` receives every node whose similarity was evaluated, in order.
        """
        visited: set[NodeId] = set(entry_points)
        candidates: list[tuple[float, NodeId]] = []  # max-heap via negated similarity
        results: list[tuple[float, NodeId]] = []  # min-heap of best `ef` results so far

        for ep in entry_points:
            sim = self._sim(query, self._vectors[ep])
            heapq.heappush(candidates, (-sim, ep))
            if accept is None or accept(ep):
                heapq.heappush(results, (sim, ep))
            if visit_log is not None:
                visit_log.append(ep)

        while candidates:
            neg_sim, current = heapq.heappop(candidates)
            if len(results) >= ef and -neg_sim < results[0][0]:
                break

            for neighbor in self._neighbors.get(current, {}).get(level, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                n_sim = self._sim(query, self._vectors[neighbor])
                if visit_log is not None:
                    visit_log.append(neighbor)

                if len(results) < ef or n_sim > results[0][0]:
                    heapq.heappush(candidates, (-n_sim, neighbor))
                    if accept is None or accept(neighbor):
                        heapq.heappush(results, (n_sim, neighbor))
                        if len(results) > ef:
                            heapq.heappop(results)

            if max_visits is not None and len(visited) >= max_visits:
                break

        return sorted(results, key=lambda pair: (-pair[0], pair[1]))

    def search(
        self,
        query: Sequence[float],
        k: int = 5,
        ef_search: int | None = None,
        accept: Callable[[NodeId], bool] | None = None,
        max_visits: int | None = None,
    ) -> list[tuple[float, NodeId]]:
        """Return up to ``k`` ``(similarity, node_id)`` pairs nearest to ``query``.

        ``accept`` restricts *results* (not navigation) to nodes it approves -- the basis of
        filtered search; ``max_visits`` caps the layer-0 work for very selective filters, in
        which case fewer than ``k`` results may come back (see
        :meth:`ragforge.index.ApproxVectorIndex.search` for the exact-scan fallback).
        """
        return self._search(query, k, ef_search, None, accept, max_visits)

    def search_trace(
        self, query: Sequence[float], k: int = 5, ef_search: int | None = None
    ) -> dict[str, Any]:
        """Like :meth:`search`, but also report the traversal for visualization.

        Returns ``{"results": [(sim, id), ...], "path": [{"level": L,
        "entry": id, "visited": [ids...]}, ...]}`` where ``path`` lists the
        layers from the top down to layer 0.
        """
        path: list[dict[str, Any]] = []
        results = self._search(query, k, ef_search, path)
        return {"results": results, "path": path}

    def _search(
        self,
        query: Sequence[float],
        k: int,
        ef_search: int | None,
        path: list[dict[str, Any]] | None,
        accept: Callable[[NodeId], bool] | None = None,
        max_visits: int | None = None,
    ) -> list[tuple[float, NodeId]]:
        if self._entry_point is None or k <= 0:
            return []

        unit = normalize_vector(query)
        ef = max(ef_search or self.ef_construction, k)
        deleted = self._deleted
        if deleted and accept is not None:
            user_accept = accept

            def layer0_accept(node: NodeId) -> bool:
                return node not in deleted and user_accept(node)
        elif deleted:

            def layer0_accept(node: NodeId) -> bool:
                return node not in deleted
        else:
            layer0_accept = accept  # type: ignore[assignment]
        entry = self._entry_point
        layer0: list[tuple[float, NodeId]] = []
        for lc in range(self._max_level, -1, -1):
            log: list[NodeId] | None = [] if path is not None else None
            layer_ef = ef if lc == 0 else 1
            layer_results = self._search_layer(
                unit,
                [entry],
                lc,
                ef=layer_ef,
                visit_log=log,
                accept=layer0_accept if lc == 0 else None,
                max_visits=max_visits if lc == 0 else None,
            )
            if path is not None and log is not None:
                path.append({"level": lc, "entry": entry, "visited": log})
            if layer_results and lc > 0:
                entry = layer_results[0][1]
            elif lc == 0:
                layer0 = layer_results

        return layer0[:k]

    def mark_deleted(self, node_id: NodeId) -> None:
        """Tombstone a node: excluded from future results, kept for graph connectivity."""
        self._deleted.add(node_id)

    def vector(self, node_id: NodeId) -> list[float]:
        """The stored (unit-length) vector for ``node_id``."""
        return self._vectors[node_id]

    def structure(self) -> dict[str, Any]:
        """The graph *without* its vectors, for storage formats that keep vectors elsewhere.

        Node ids are dense (``0..n-1``) outside of :meth:`compact`, so ``levels`` is indexed by
        node id and ``neighbors[level]`` maps node -> sorted neighbor ids.
        """
        n = self._next_id
        return {
            "params": {
                "m": self.m,
                "m_max0": self.m_max0,
                "ef_construction": self.ef_construction,
                "seed": self.seed,
                "heuristic": self.heuristic,
            },
            "next_id": n,
            "entry_point": self._entry_point,
            "max_level": self._max_level,
            "levels": [self._levels[i] for i in range(n)],
            "neighbors": {
                level: {
                    i: sorted(self._neighbors[i][level])
                    for i in range(n)
                    if level in self._neighbors[i]
                }
                for level in range(self._max_level + 1)
            },
            "deleted": sorted(self._deleted),
            "pinned": sorted(self._pinned),
        }

    @classmethod
    def from_structure(cls, data: dict[str, Any], vectors: Any) -> HNSWIndex:
        """Rebuild a graph from :meth:`structure` output, using ``vectors`` as the store.

        ``vectors`` is any ``Mapping[int, Sequence[float]]`` supporting ``[]`` assignment
        (e.g. an mmap-backed store); vectors are *not* copied or re-normalized.
        """
        params = data["params"]
        index = cls(**params)
        index._vectors = vectors
        index._next_id = data["next_id"]
        index._entry_point = data["entry_point"]
        index._max_level = data["max_level"]
        index._levels = dict(enumerate(data["levels"]))
        index._neighbors = {i: {} for i in range(index._next_id)}
        for level, table in data["neighbors"].items():
            for node, nbrs in table.items():
                index._neighbors[int(node)][int(level)] = set(nbrs)
        for i, lv in index._levels.items():
            for level in range(lv + 1):
                index._neighbors[i].setdefault(level, set())
        index._deleted = set(data["deleted"])
        index._pinned = {(int(lv), int(a), int(b)) for lv, a, b in data["pinned"]}
        for lv, a, b in index._pinned:
            for node in (a, b):
                index._pinned_children[(lv, node)] = index._pinned_children.get((lv, node), 0) + 1
        return index

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full graph state (vectors, edges, levels, tombstones)."""
        return {
            "m": self.m,
            "m_max0": self.m_max0,
            "ef_construction": self.ef_construction,
            "seed": self.seed,
            "heuristic": self.heuristic,
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
            "pinned": sorted(self._pinned),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HNSWIndex:
        """Deserialize a graph previously produced by :meth:`to_dict`.

        Graphs saved by ragforge 0.1 (unnormalized vectors, no ``heuristic``
        flag) load fine: vectors are re-normalized on the way in.
        """
        m = data.get("m", 16)
        index = cls(
            m=m,
            m_max0=data.get("m_max0", 2 * m),
            ef_construction=data.get("ef_construction", 200),
            seed=data.get("seed", 1337),
            heuristic=data.get("heuristic", True),
        )
        index._next_id = data.get("next_id", 0)
        index._entry_point = data.get("entry_point")
        index._max_level = data.get("max_level", -1)
        index._vectors = {int(k): normalize_vector(v) for k, v in data.get("vectors", {}).items()}
        index._levels = {int(k): v for k, v in data.get("levels", {}).items()}
        index._neighbors = {
            int(node): {int(level): set(ids) for level, ids in layers.items()}
            for node, layers in data.get("neighbors", {}).items()
        }
        index._deleted = set(data.get("deleted", []))
        index._pinned = {(int(lv), int(a), int(b)) for lv, a, b in data.get("pinned", [])}
        for lv, a, b in index._pinned:
            for node in (a, b):
                key = (lv, node)
                index._pinned_children[key] = index._pinned_children.get(key, 0) + 1
        return index
