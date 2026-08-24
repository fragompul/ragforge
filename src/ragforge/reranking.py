"""Reranking: precision and diversity re-scoring over candidate retrieval sets.

Retrieval (BM25, dense vector, or hybrid RRF) optimizes for high recall over large
corpora cheaply. A reranker executes a second-stage, compute-intensive pass over a small
candidate pool to achieve optimal ranking and context diversity.

Three reranker implementations are provided:
1. ``NoopReranker``: Identity pass-through.
2. ``HeuristicReranker``: Lexical token-overlap (Jaccard similarity) re-scoring.
3. ``CrossEncoderReranker``: Adapter for neural cross-encoder models (e.g. BGE-reranker,
   Cohere Rerank, sentence-transformers CrossEncoder).
4. ``MaxMarginalRelevanceReranker`` (MMR): Diversity-aware reranker that balances relevance
   against redundancy to prevent context-window saturation with duplicate information.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from ragforge.embeddings import EmbedFn, cosine_similarity, hashing_embed
from ragforge.index import ScoredChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class Reranker(ABC):
    """Abstract base class for second-stage rerankers."""

    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        """Rerank retrieval candidates and return top-k scored chunks."""
        raise NotImplementedError


class NoopReranker(Reranker):
    """Passes candidates through unchanged, truncated to ``k``."""

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if k <= 0:
            return []
        return candidates[:k]


class HeuristicReranker(Reranker):
    """Token-overlap (Jaccard similarity) reranker.

    Provides a fast, zero-dependency stand-in for cross-encoders that captures
    exact-term coverage and penalizes diluted candidate chunks.
    """

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if k <= 0 or not candidates:
            return []

        query_tokens = _token_set(query)
        rescored: list[ScoredChunk] = []

        for candidate in candidates:
            chunk_tokens = _token_set(candidate.chunk.text)
            union = query_tokens | chunk_tokens
            overlap = len(query_tokens & chunk_tokens) / len(union) if union else 0.0
            rescored.append(
                ScoredChunk(
                    chunk=candidate.chunk,
                    score=overlap,
                    provenance=f"rerank:jaccard({candidate.provenance})",
                    ranks=candidate.ranks,
                )
            )

        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored[:k]


class CrossEncoderReranker(Reranker):
    """Neural cross-encoder reranker adapter.

    Accepts any callable scoring function ``score_fn(query: str, text: str) -> float``
    or a batch scoring function. Compatible with SentenceTransformers CrossEncoder,
    Cohere Rerank API, or custom transformer scoring endpoints.
    """

    def __init__(
        self,
        score_fn: Callable[[str, str], float] | None = None,
        batch_score_fn: Callable[[str, Sequence[str]], Sequence[float]] | None = None,
    ) -> None:
        if score_fn is None and batch_score_fn is None:
            raise ValueError("Either score_fn or batch_score_fn must be provided")
        self.score_fn = score_fn
        self.batch_score_fn = batch_score_fn

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if k <= 0 or not candidates:
            return []

        rescored: list[ScoredChunk] = []

        if self.batch_score_fn is not None:
            texts = [c.chunk.text for c in candidates]
            scores = self.batch_score_fn(query, texts)
            for candidate, score in zip(candidates, scores, strict=True):
                rescored.append(
                    ScoredChunk(
                        chunk=candidate.chunk,
                        score=float(score),
                        provenance=f"rerank:cross_encoder({candidate.provenance})",
                        ranks=candidate.ranks,
                    )
                )
        elif self.score_fn is not None:
            for candidate in candidates:
                score = self.score_fn(query, candidate.chunk.text)
                rescored.append(
                    ScoredChunk(
                        chunk=candidate.chunk,
                        score=float(score),
                        provenance=f"rerank:cross_encoder({candidate.provenance})",
                        ranks=candidate.ranks,
                    )
                )

        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored[:k]


class MaxMarginalRelevanceReranker(Reranker):
    """Maximal Marginal Relevance (MMR) reranker for relevance + diversity optimization.

    MMR solves the context redundancy problem in RAG by selecting chunks that are
    highly relevant to the query while maximizing novelty relative to already-selected contexts:

    ``MMR_score = lambda_mult * sim(doc, query) - (1 - lambda_mult) * max_sim(doc, selected)``

    - ``lambda_mult = 1.0``: Pure relevance (identical to standard vector ranking).
    - ``lambda_mult = 0.0``: Pure diversity / maximal novelty.
    - Default ``lambda_mult = 0.7`` provides balanced relevance with non-redundant coverage.
    """

    def __init__(
        self,
        lambda_mult: float = 0.7,
        embed_fn: EmbedFn = hashing_embed,
    ) -> None:
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError(f"lambda_mult must be between 0.0 and 1.0, got {lambda_mult}")
        self.lambda_mult = lambda_mult
        self.embed_fn = embed_fn

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if k <= 0 or not candidates:
            return []

        target_k = min(k, len(candidates))
        query_vec = self.embed_fn(query)
        candidate_vecs = [self.embed_fn(c.chunk.text) for c in candidates]

        # Compute query similarities
        query_sims = [cosine_similarity(query_vec, v) for v in candidate_vecs]

        selected_indices: list[int] = []
        selected_scores: list[float] = []
        remaining_indices = list(range(len(candidates)))

        while len(selected_indices) < target_k and remaining_indices:
            best_idx = -1
            best_mmr = -float("inf")

            for idx in remaining_indices:
                rel_score = query_sims[idx]

                if not selected_indices:
                    max_sim_to_selected = 0.0
                else:
                    max_sim_to_selected = max(
                        cosine_similarity(candidate_vecs[idx], candidate_vecs[s_idx])
                        for s_idx in selected_indices
                    )

                mmr_score = (
                    self.lambda_mult * rel_score - (1.0 - self.lambda_mult) * max_sim_to_selected
                )

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx == -1:
                break

            selected_indices.append(best_idx)
            selected_scores.append(best_mmr)
            remaining_indices.remove(best_idx)

        return [
            ScoredChunk(
                chunk=candidates[idx].chunk,
                score=score,
                provenance=f"rerank:mmr({candidates[idx].provenance})",
                ranks=candidates[idx].ranks,
            )
            for idx, score in zip(selected_indices, selected_scores, strict=True)
        ]
