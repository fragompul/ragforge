"""Learned-sparse retrieval: an inverted index over *weighted* terms (SPLADE-style).

BM25 weights terms by counting them; learned sparse retrievers such as SPLADE use a
language model to predict, for every text, a sparse bag of weighted vocabulary terms --
including terms that are *not in the text* but are semantically implied ("automobile"
for a passage about cars). Retrieval stays a cheap inverted-index lookup, so it has
BM25's operational profile with much of a dense model's vocabulary-gap robustness, and
it fuses naturally with BM25 and dense vectors through RRF
(:class:`ragforge.fusion.FusionRetriever`).

:class:`SparseVectorIndex` is model-agnostic: it accepts any
``encode_fn(text) -> {term: weight}``. :func:`lexical_sparse_encode_fn` is a dependency-free
encoder (log-scaled term frequency) for tests and demos; :func:`splade_encode_fn` adapts a
real SPLADE model through ``sentence-transformers``.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Callable
from typing import Any

from ragforge.chunking import Chunk
from ragforge.index import FilterFn, ScoredChunk, english_analyzer

SparseEncodeFn = Callable[[str], dict[str, float]]


def lexical_sparse_encode_fn(text: str) -> dict[str, float]:
    """Dependency-free encoder: ``log(1 + tf)`` over analyzed tokens (no learned expansion)."""
    return {term: math.log1p(tf) for term, tf in Counter(english_analyzer(text)).items()}


def splade_encode_fn(
    model_name: str = "naver/splade-cocondenser-ensembledistil",
    model: Any = None,
    top_k: int = 96,
) -> SparseEncodeFn:
    """Build a ``SparseEncodeFn`` backed by a SPLADE model (``pip install ragforge[local]``).

    The model is loaded lazily on construction (on CPU: sparse tensors are not supported by
    every accelerator backend). ``top_k`` keeps only the heaviest terms, which bounds index
    size with little quality loss.
    """
    if model is not None:
        resolved = model
    else:
        try:
            from sentence_transformers import SparseEncoder
        except ImportError as exc:
            raise ImportError(
                "SPLADE encoding needs sentence-transformers >= 5. "
                "Install it with:\n    pip install 'ragforge[local]'"
            ) from exc
        resolved = SparseEncoder(model_name, device="cpu")

    def _encode(text: str) -> dict[str, float]:
        # Dense output (vocab-sized) is robust across devices; keep only the heaviest terms.
        row = resolved.encode([text], convert_to_sparse_tensor=False)[0]
        weights = row.tolist() if hasattr(row, "tolist") else list(row)
        top = heapq.nlargest(top_k, ((w, i) for i, w in enumerate(weights) if w > 0))
        tokens = resolved.tokenizer.convert_ids_to_tokens([i for _, i in top])
        return {str(token): float(w) for token, (w, _) in zip(tokens, top, strict=True)}

    return _encode


class SparseVectorIndex:
    """Inverted index over weighted terms; score = dot product of query and document weights.

    Implements the same ``add / delete / clear / search / to_dict`` surface as the other
    indexes, plus :meth:`load_dict` so it can be an *extra source* of a
    :class:`~ragforge.pipeline.RagPipeline` (persisted weights are reused on load, so a
    real model is not re-run).
    """

    def __init__(self, encode_fn: SparseEncodeFn = lexical_sparse_encode_fn) -> None:
        self.encode_fn = encode_fn
        self._chunks: list[Chunk] = []
        self._weights: list[dict[str, float]] = []
        self._postings: dict[str, dict[int, float]] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    def _append(self, chunk: Chunk, weights: dict[str, float]) -> None:
        position = len(self._chunks)
        self._chunks.append(chunk)
        self._weights.append(weights)
        for term, weight in weights.items():
            if weight > 0:
                self._postings.setdefault(term, {})[position] = weight

    def add(self, chunks: list[Chunk]) -> None:
        """Encode and index new chunks."""
        for chunk in chunks:
            self._append(chunk, self.encode_fn(chunk.index_text))

    def add_encoded(self, chunks: list[Chunk], weights: list[dict[str, float]]) -> None:
        """Index chunks with precomputed term weights (e.g. from a batch-encoded corpus)."""
        if len(chunks) != len(weights):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided with {len(weights)} encodings"
            )
        for chunk, w in zip(chunks, weights, strict=True):
            self._append(chunk, w)

    def delete(self, doc_id: str) -> int:
        kept = [
            (c, w) for c, w in zip(self._chunks, self._weights, strict=True) if c.doc_id != doc_id
        ]
        removed = len(self._chunks) - len(kept)
        if removed:
            self.clear()
            for chunk, weights in kept:
                self._append(chunk, weights)
        return removed

    def clear(self) -> None:
        self._chunks, self._weights, self._postings = [], [], {}

    def search(
        self, query: str, k: int = 5, filter_fn: FilterFn | None = None
    ) -> list[ScoredChunk]:
        if k <= 0 or not self._chunks:
            return []
        scores: dict[int, float] = {}
        for term, q_weight in self.encode_fn(query).items():
            if q_weight <= 0:
                continue
            for position, d_weight in self._postings.get(term, {}).items():
                scores[position] = scores.get(position, 0.0) + q_weight * d_weight
        if filter_fn is not None:
            scores = {p: s for p, s in scores.items() if filter_fn(self._chunks[p])}
        top = heapq.nsmallest(k, ((-s, p) for p, s in scores.items() if s > 0))
        return [
            ScoredChunk(chunk=self._chunks[p], score=-neg, provenance="sparse") for neg, p in top
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": [c.to_dict() for c in self._chunks],
            "weights": [{t: round(w, 6) for t, w in ws.items()} for ws in self._weights],
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        """Replace this index's contents with a serialized state (no encoder calls)."""
        self.clear()
        chunks = [Chunk.from_dict(c) for c in data.get("chunks", [])]
        self.add_encoded(chunks, [dict(w) for w in data.get("weights", [])])

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], encode_fn: SparseEncodeFn = lexical_sparse_encode_fn
    ) -> SparseVectorIndex:
        index = cls(encode_fn)
        index.load_dict(data)
        return index
