"""Optional NumPy-accelerated vector index (``pip install ragforge[fast]``).

ragforge's core is dependency-free on purpose, which means its default cosine
scan is a pure-Python loop. :class:`NumpyVectorIndex` keeps the exact same
semantics -- exact (non-approximate) cosine search, identical ranking and
tie-breaking -- but stores all vectors as one contiguous unit-normalized
matrix, so a query becomes a single BLAS matrix-vector product. On a laptop
that is typically one to two orders of magnitude faster than ``VectorIndex``
(see ``docs/benchmarks.md``), and it needs no graph, no tuning, and gives
100 % recall: for corpora up to a few hundred thousand chunks it is often the
right choice before reaching for HNSW.

NumPy is imported lazily when the index is constructed, so importing
``ragforge`` never requires it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragforge.chunking import Chunk
from ragforge.embeddings import EmbedFn, hashing_embed
from ragforge.index import FilterFn, ScoredChunk

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


def _import_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - exercised only without numpy
        raise ImportError(
            "NumpyVectorIndex requires NumPy. Install it with: pip install 'ragforge[fast]'"
        ) from exc
    return numpy


class NumpyVectorIndex:
    """Exact cosine-similarity search over a contiguous NumPy matrix.

    Implements the same surface as :class:`ragforge.index.VectorIndex`
    (``add``, ``add_vectors``, ``delete``, ``clear``, ``search``, persistence)
    and therefore satisfies :class:`ragforge.index.VectorSearchable`.
    """

    def __init__(self, embed_fn: EmbedFn = hashing_embed) -> None:
        self._np = _import_numpy()
        self.embed_fn = embed_fn
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # shape (n, dims), rows unit-normalized

    def __len__(self) -> int:
        return len(self._chunks)

    def _unit_rows(self, vectors: list[list[float]]) -> Any:
        np = self._np
        arr = np.asarray(vectors, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("vectors must be a list of equal-length float lists")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return np.asarray(arr / norms)

    def add(self, chunks: list[Chunk]) -> None:
        """Embed and index new chunks."""
        if chunks:
            self.add_vectors(chunks, [self.embed_fn(c.index_text) for c in chunks])

    def add_vectors(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Index chunks with precomputed embeddings."""
        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatch: {len(chunks)} chunks provided with {len(vectors)} vectors")
        if not chunks:
            return
        rows = self._unit_rows(vectors)
        if self._matrix is not None and rows.shape[1] != self._matrix.shape[1]:
            raise ValueError(
                f"Vector dimension mismatch: index has {self._matrix.shape[1]} dims, "
                f"got {rows.shape[1]}"
            )
        self._matrix = rows if self._matrix is None else self._np.vstack([self._matrix, rows])
        self._chunks.extend(chunks)

    def delete(self, doc_id: str) -> int:
        """Delete all chunks for a given document."""
        if self._matrix is None:
            return 0
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        removed = len(self._chunks) - len(keep)
        if removed:
            self._chunks = [self._chunks[i] for i in keep]
            self._matrix = self._matrix[keep] if keep else None
        return removed

    def clear(self) -> None:
        """Clear all indexed chunks and vectors."""
        self._chunks = []
        self._matrix = None

    def search(
        self, query: str, k: int = 5, filter_fn: FilterFn | None = None
    ) -> list[ScoredChunk]:
        """Exact top-``k`` cosine search with optional metadata filtering."""
        if k <= 0 or self._matrix is None:
            return []
        np = self._np

        query_vec = np.asarray(self.embed_fn(query), dtype=np.float64)
        if query_vec.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"Vector dimension mismatch: index has {self._matrix.shape[1]} dims, "
                f"query has {query_vec.shape[0]}"
            )
        norm = float(np.linalg.norm(query_vec))
        scores = self._matrix @ (query_vec / norm) if norm > 0.0 else np.zeros(len(self._chunks))
        scores = np.clip(scores, -1.0, 1.0)

        candidates = np.arange(len(self._chunks))
        if filter_fn is not None:
            mask = np.fromiter((filter_fn(c) for c in self._chunks), dtype=bool)
            candidates = candidates[mask]
        if candidates.size == 0:
            return []

        cand_scores = scores[candidates]
        if candidates.size > k:
            # Keep everything tied with the k-th best so ties break by index, like VectorIndex.
            threshold = np.partition(cand_scores, candidates.size - k)[candidates.size - k]
            keep = cand_scores >= threshold
            candidates, cand_scores = candidates[keep], cand_scores[keep]
        order = np.lexsort((candidates, -cand_scores))[:k]
        return [
            ScoredChunk(
                chunk=self._chunks[int(candidates[i])],
                score=float(cand_scores[i]),
                provenance="vector:numpy",
            )
            for i in order
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the same JSON shape as ``VectorIndex`` (vectors are unit-length)."""
        vectors = [] if self._matrix is None else self._matrix.tolist()
        return {"chunks": [c.to_dict() for c in self._chunks], "vectors": vectors}

    @classmethod
    def from_dict(cls, data: dict[str, Any], embed_fn: EmbedFn | None = None) -> NumpyVectorIndex:
        """Deserialize an index (also accepts ``VectorIndex`` payloads)."""
        index = cls(embed_fn=embed_fn or hashing_embed)
        chunks = [Chunk.from_dict(c) for c in data.get("chunks", [])]
        vectors = data.get("vectors", [])
        if vectors:
            index.add_vectors(chunks, vectors)
        else:
            index.add(chunks)
        return index

    def save(self, path: str | Path) -> None:
        """Save the index to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path, embed_fn: EmbedFn | None = None) -> NumpyVectorIndex:
        """Load an index from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, embed_fn=embed_fn)
