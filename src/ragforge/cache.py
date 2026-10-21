"""Semantic caching: answer near-duplicate questions without re-running the pipeline.

Exact-match caches miss almost everything in natural language ("how do I
cancel?" vs. "how can I cancel my plan?"), yet generation is the slowest and
most expensive stage of RAG. A *semantic* cache embeds each query and serves a
stored answer when a new query lands within a cosine-similarity threshold of a
previous one -- the technique behind GPTCache and several commercial gateways.

Correctness matters more than hit rate, so this implementation is deliberately
conservative:

- the cache key is partitioned by retrieval parameters (``k``,
  ``candidate_pool``, and an optional ``namespace`` such as a tenant id), so an
  answer is never served across different retrieval settings or tenants;
- calls with a ``filter_fn`` bypass the cache (a callable cannot be compared or
  hashed safely) unless the caller supplies a stable ``cache_key`` for it;
- entries expire after ``ttl_seconds`` and the whole cache is dropped whenever
  the underlying corpus changes (``ingest``/``delete``/``clear`` through
  :class:`CachedPipeline`), so stale answers never outlive their sources;
- the threshold defaults high (0.92), because a wrong cache hit is worse than a
  miss.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ragforge.chunking import Chunk
from ragforge.embeddings import EmbedFn, hashing_embed, normalize_vector
from ragforge.index import FilterFn
from ragforge.pipeline import RagAnswer, RagPipeline


@dataclass
class CacheEntry:
    query: str
    vector: list[float]
    value: Any
    partition: tuple[Any, ...]
    created_at: float
    hits: int = 0


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    bypasses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "bypasses": self.bypasses,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass
class SemanticCache:
    """LRU cache keyed by query embedding similarity.

    Args:
        embed_fn: Query embedder. Use a real semantic model in production: with
            the built-in bag-of-words ``hashing_embed`` only near-identical word
            sets will match.
        threshold: Minimum cosine similarity for a hit.
        max_size: Maximum number of entries (least-recently-used evicted).
        ttl_seconds: Entry lifetime; ``None`` disables expiry.
        clock: Monotonic time source (injectable for tests).
    """

    embed_fn: EmbedFn = hashing_embed
    threshold: float = 0.92
    max_size: int = 1024
    ttl_seconds: float | None = None
    clock: Callable[[], float] = time.monotonic
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        if not -1.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [-1, 1], got {self.threshold}")
        if self.max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {self.max_size}")
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {self.ttl_seconds}")
        self._entries: OrderedDict[int, CacheEntry] = OrderedDict()
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._entries)

    def _expired(self, entry: CacheEntry) -> bool:
        return self.ttl_seconds is not None and self.clock() - entry.created_at > self.ttl_seconds

    def get(self, query: str, partition: tuple[Any, ...] = ()) -> CacheEntry | None:
        """Return the most similar live entry in ``partition`` if it clears the threshold."""
        vector = normalize_vector(self.embed_fn(query))
        best_id: int | None = None
        best_sim = -2.0
        for entry_id, entry in list(self._entries.items()):
            if self._expired(entry):
                del self._entries[entry_id]
                self.stats.expirations += 1
                continue
            if entry.partition != partition or len(entry.vector) != len(vector):
                continue
            sim = sum(a * b for a, b in zip(vector, entry.vector, strict=True))
            if sim > best_sim:
                best_id, best_sim = entry_id, sim

        if best_id is not None and best_sim >= self.threshold:
            self._entries.move_to_end(best_id)
            entry = self._entries[best_id]
            entry.hits += 1
            self.stats.hits += 1
            return entry
        self.stats.misses += 1
        return None

    def put(self, query: str, value: Any, partition: tuple[Any, ...] = ()) -> CacheEntry:
        """Store ``value`` for ``query``, evicting the least-recently-used entry if full."""
        entry = CacheEntry(
            query=query,
            vector=normalize_vector(self.embed_fn(query)),
            value=value,
            partition=partition,
            created_at=self.clock(),
        )
        self._entries[self._next_id] = entry
        self._next_id += 1
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)
            self.stats.evictions += 1
        return entry

    def clear(self) -> None:
        """Drop every entry (statistics are kept)."""
        self._entries.clear()


class CachedPipeline:
    """A :class:`RagPipeline` wrapper that serves repeated questions from a :class:`SemanticCache`.

    Unknown attributes delegate to the wrapped pipeline, so it drops in wherever a
    pipeline is expected. Mutating the corpus through the wrapper invalidates the cache.
    """

    def __init__(
        self,
        pipeline: RagPipeline,
        cache: SemanticCache | None = None,
        namespace: str = "",
    ) -> None:
        self.pipeline = pipeline
        self.cache = cache or SemanticCache(embed_fn=pipeline.embed_fn)
        self.namespace = namespace

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pipeline, name)

    def answer(
        self,
        query: str,
        k: int = 5,
        candidate_pool: int = 20,
        filter_fn: FilterFn | None = None,
        cache_key: str | None = None,
    ) -> RagAnswer:
        """Like :meth:`RagPipeline.answer`; ``cache_key`` names the filter so it can be cached."""
        if filter_fn is not None and cache_key is None:
            self.cache.stats.bypasses += 1
            return self.pipeline.answer(
                query, k=k, candidate_pool=candidate_pool, filter_fn=filter_fn
            )

        partition = (self.namespace, k, candidate_pool, cache_key)
        entry = self.cache.get(query, partition)
        if entry is not None:
            cached: RagAnswer = entry.value
            return RagAnswer(
                query=query,
                answer=cached.answer,
                contexts=cached.contexts,
                prompt=cached.prompt,
                retrieval_latency_ms=0.0,
                generation_latency_ms=0.0,
                total_latency_ms=0.0,
                metadata={**cached.metadata, "cache": "hit", "cached_query": entry.query},
            )

        result = self.pipeline.answer(
            query, k=k, candidate_pool=candidate_pool, filter_fn=filter_fn
        )
        result.metadata = {**result.metadata, "cache": "miss"}
        self.cache.put(query, result, partition)
        return result

    def ingest(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        self.cache.clear()
        return self.pipeline.ingest(doc_id, text, metadata)

    def ingest_batch(self, documents: list[dict[str, Any]]) -> list[Chunk]:
        self.cache.clear()
        return self.pipeline.ingest_batch(documents)

    def delete(self, doc_id: str) -> int:
        self.cache.clear()
        return self.pipeline.delete(doc_id)

    def clear(self) -> None:
        self.cache.clear()
        self.pipeline.clear()
