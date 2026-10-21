"""N-way weighted Reciprocal Rank Fusion over any number of retrievers.

:class:`ragforge.index.HybridRetriever` fuses exactly two sources (BM25 and dense
vectors). Modern retrieval stacks add more signals -- a learned sparse retriever
(SPLADE), an entity-graph retriever, a second embedding model -- and RRF handles them
the same way: sum ``weight / (k + rank)`` over every list a document appears in.
:class:`FusionRetriever` is that generalization, with per-source provenance and ranks
so the UI and traces can show *which* retriever contributed what.

Anything with ``search(query, k, filter_fn=None) -> list[ScoredChunk]`` is a source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ragforge.chunking import Chunk
from ragforge.index import FilterFn, ScoredChunk


class Searchable(Protocol):
    def search(
        self, query: str, k: int, filter_fn: FilterFn | None = None
    ) -> list[ScoredChunk]: ...


@dataclass
class FusionSource:
    """One ranked list to fuse: a name (shown in provenance), a retriever and a weight."""

    name: str
    index: Any  # Searchable; Any keeps heterogeneous index classes assignable
    weight: float = 1.0


class FusionRetriever:
    """Weighted RRF across named sources. Weight ``0`` disables a source without removing it."""

    def __init__(self, sources: list[FusionSource], k_rrf: int = 60) -> None:
        if k_rrf <= 0:
            raise ValueError(f"k_rrf must be positive, got {k_rrf}")
        names = [s.name for s in sources]
        if len(set(names)) != len(names):
            raise ValueError(f"source names must be unique, got {names}")
        if any(s.weight < 0 for s in sources):
            raise ValueError("source weights must be non-negative")
        self.sources = sources
        self.k_rrf = k_rrf

    def weight(self, name: str) -> float:
        return next(s.weight for s in self.sources if s.name == name)

    def set_weight(self, name: str, weight: float) -> None:
        if weight < 0:
            raise ValueError("weight must be non-negative")
        for source in self.sources:
            if source.name == name:
                source.weight = weight
                return
        raise KeyError(name)

    # Convenience aliases so code written for HybridRetriever (e.g. the web bridge) works.
    @property
    def weight_bm25(self) -> float:
        return self.weight("bm25")

    @weight_bm25.setter
    def weight_bm25(self, value: float) -> None:
        self.set_weight("bm25", value)

    @property
    def weight_vector(self) -> float:
        return self.weight("vector")

    @weight_vector.setter
    def weight_vector(self, value: float) -> None:
        self.set_weight("vector", value)

    def search(
        self,
        query: str,
        k: int = 5,
        candidate_pool: int = 20,
        filter_fn: FilterFn | None = None,
    ) -> list[ScoredChunk]:
        """Fuse every active source's top ``max(k, candidate_pool)`` into one ranking."""
        if k <= 0:
            return []
        pool = max(k, candidate_pool)
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        ranks: dict[str, dict[str, int]] = {}
        for source in self.sources:
            if source.weight <= 0:
                continue
            for rank, hit in enumerate(source.index.search(query, k=pool, filter_fn=filter_fn), 1):
                cid = hit.chunk.id
                scores[cid] = scores.get(cid, 0.0) + source.weight / (self.k_rrf + rank)
                chunks[cid] = hit.chunk
                ranks.setdefault(cid, {})[source.name] = rank

        fused = [
            ScoredChunk(
                chunk=chunks[cid],
                score=score,
                provenance="+".join(ranks[cid]),
                ranks=ranks[cid],
            )
            for cid, score in scores.items()
        ]
        fused.sort(key=lambda sc: (-sc.score, sc.chunk.id))
        return fused[:k]
