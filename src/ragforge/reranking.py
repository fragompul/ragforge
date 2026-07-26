"""Reranking: a second, more precise pass over retrieval candidates.

Retrieval (BM25/vector/hybrid) optimizes for recall over a large corpus
and is comparatively cheap; a reranker trades extra compute for a more
accurate final ordering over the small candidate set retrieval already
narrowed down -- the standard two-stage pattern used in production search
and RAG systems. `Reranker` is an interface so a real cross-encoder model
can be dropped in without changing the pipeline.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ragforge.index import ScoredChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        raise NotImplementedError


class NoopReranker(Reranker):
    """Passes candidates through unchanged, truncated to ``k``."""

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        return candidates[:k]


class HeuristicReranker(Reranker):
    """A dependency-free stand-in for a cross-encoder: re-scores each
    candidate by query/chunk token-overlap (Jaccard similarity), which
    captures exact-term precision that RRF-fused scores can blur.

    Swap in a real cross-encoder (e.g. a `sentence-transformers`
    CrossEncoder wrapped in this interface) for production-quality
    reranking; this keeps the pipeline runnable offline in tests.
    """

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        query_tokens = _token_set(query)
        rescored = []
        for candidate in candidates:
            chunk_tokens = _token_set(candidate.chunk.text)
            union = query_tokens | chunk_tokens
            overlap = len(query_tokens & chunk_tokens) / len(union) if union else 0.0
            rescored.append(ScoredChunk(chunk=candidate.chunk, score=overlap))
        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored[:k]
