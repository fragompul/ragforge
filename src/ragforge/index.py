"""Retrieval indices: sparse (BM25), dense (vector), and a hybrid combiner.

Hybrid search exists because BM25 and dense retrieval fail in
complementary ways: BM25 misses paraphrases and synonyms ("car" vs.
"automobile"); dense retrieval misses exact-match signals that matter a
lot in practice (product SKUs, error codes, proper nouns). Combining
both with Reciprocal Rank Fusion is a simple, weight-free way to get the
benefits of each without tuning a blend ratio.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from ragforge.chunking import Chunk
from ragforge.embeddings import cosine_similarity, hashing_embed

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class BM25Index:
    """A from-scratch Okapi BM25 implementation (no external search dependency)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_doc_len = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            self._chunks.append(chunk)
            self._doc_tokens.append(tokens)
            for term in set(tokens):
                self._doc_freq[term] += 1
        if self._doc_tokens:
            self._avg_doc_len = sum(len(t) for t in self._doc_tokens) / len(self._doc_tokens)

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._doc_freq.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        query_terms = _tokenize(query)
        scored: list[ScoredChunk] = []
        for chunk, tokens in zip(self._chunks, self._doc_tokens, strict=True):
            term_freqs = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for term in query_terms:
                if term not in term_freqs:
                    continue
                tf = term_freqs[term]
                idf = self._idf(term)
                length_norm = 1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1e-9)
                denom = tf + self.k1 * length_norm
                score += idf * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:k]


class VectorIndex:
    """Cosine-similarity search over embedded chunks."""

    def __init__(self, embed_fn: Callable[[str], list[float]] = hashing_embed) -> None:
        self.embed_fn = embed_fn
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks.append(chunk)
            self._vectors.append(self.embed_fn(chunk.text))

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        query_vec = self.embed_fn(query)
        scored = [
            ScoredChunk(chunk=chunk, score=cosine_similarity(query_vec, vec))
            for chunk, vec in zip(self._chunks, self._vectors, strict=True)
        ]
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:k]


class HybridRetriever:
    """Combines a sparse and a dense index via Reciprocal Rank Fusion (RRF).

    RRF avoids tuning a weighted blend of two differently-scaled score
    distributions (BM25 scores and cosine similarities aren't comparable)
    by fusing on rank position instead: ``score = sum(1 / (k_rrf + rank))``
    across both retrievers' result lists.
    """

    def __init__(self, bm25_index: BM25Index, vector_index: VectorIndex, k_rrf: int = 60) -> None:
        self.bm25_index = bm25_index
        self.vector_index = vector_index
        self.k_rrf = k_rrf

    def search(self, query: str, k: int = 5, candidate_pool: int = 20) -> list[ScoredChunk]:
        bm25_results = self.bm25_index.search(query, k=candidate_pool)
        vector_results = self.vector_index.search(query, k=candidate_pool)

        rrf_scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}
        for results in (bm25_results, vector_results):
            for rank, scored in enumerate(results):
                rrf_scores[scored.chunk.id] = rrf_scores.get(scored.chunk.id, 0.0) + 1.0 / (
                    self.k_rrf + rank + 1
                )
                chunks_by_id[scored.chunk.id] = scored.chunk

        fused = [
            ScoredChunk(chunk=chunks_by_id[cid], score=score) for cid, score in rrf_scores.items()
        ]
        fused.sort(key=lambda sc: sc.score, reverse=True)
        return fused[:k]
