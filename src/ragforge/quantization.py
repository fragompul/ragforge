"""Vector quantization: 4x-32x smaller embeddings with a measured recall cost.

Embeddings dominate a RAG system's memory: 1M chunks x 384 dims x 4 bytes is 1.5 GB
before any index overhead. Quantization stores a lossy, much smaller code per vector and
searches in code space, then *rescores* a shortlist with the full-precision vectors to
win most of the accuracy back -- the standard recipe in production vector databases.

Two quantizers, both dependency-free:

``ScalarQuantizer`` (``mode="int8"``)
    Maps every dimension to one byte using a per-dimension affine range learned from
    data: 4x smaller than float32. The dot product with a float query is computed
    without decoding, as ``bias + sum(scaled_query_i * code_i)``.
``BinaryQuantizer`` (``mode="binary"``)
    Keeps one bit per dimension (is the value above that dimension's mean?): 32x
    smaller. Similarity is Hamming distance -- an XOR and a population count on a
    Python integer, one of the fastest operations the interpreter has.

:class:`QuantizedVectorIndex` is a drop-in :class:`~ragforge.index.VectorSearchable`
backend (``RagPipeline(vector_backend="int8" | "binary")``). With
``keep_full_precision=True`` it keeps the original floats for rescoring; setting it to
``False`` gives the smallest footprint (code only) at lower accuracy. See
``docs/benchmarks.md`` for measured recall/memory trade-offs on real embeddings.
"""

from __future__ import annotations

import base64
import heapq
import json
import operator
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from ragforge.chunking import Chunk
from ragforge.embeddings import EmbedFn, hashing_embed, normalize_vector
from ragforge.index import FilterFn, ScoredChunk

Mode = Literal["int8", "binary"]


class ScalarQuantizer:
    """Per-dimension 8-bit affine quantizer (``code = round((x - lo) / step)``)."""

    def __init__(self, lo: Sequence[float], hi: Sequence[float]) -> None:
        if len(lo) != len(hi) or not lo:
            raise ValueError("lo and hi must be non-empty and the same length")
        self.lo = [float(v) for v in lo]
        self.step = [
            max((float(h) - float(low)) / 255.0, 1e-12) for low, h in zip(lo, hi, strict=True)
        ]

    @property
    def dims(self) -> int:
        return len(self.lo)

    @classmethod
    def fit(cls, vectors: Sequence[Sequence[float]]) -> ScalarQuantizer:
        """Learn each dimension's range from ``vectors``."""
        if not vectors:
            raise ValueError("cannot fit a quantizer on zero vectors")
        dims = len(vectors[0])
        lo = [min(v[d] for v in vectors) for d in range(dims)]
        hi = [max(v[d] for v in vectors) for d in range(dims)]
        return cls(lo, hi)

    def encode(self, vector: Sequence[float]) -> bytes:
        if len(vector) != self.dims:
            raise ValueError(f"expected {self.dims} dims, got {len(vector)}")
        return bytes(
            min(255, max(0, round((x - low) / step)))
            for x, low, step in zip(vector, self.lo, self.step, strict=True)
        )

    def decode(self, code: bytes) -> list[float]:
        return [low + c * step for c, low, step in zip(code, self.lo, self.step, strict=True)]

    def prepare_query(self, query: Sequence[float]) -> tuple[list[float], float]:
        """Precompute ``(scaled_query, bias)`` so scoring a code is one fused dot product."""
        scaled = [q * step for q, step in zip(query, self.step, strict=True)]
        bias = sum(q * low for q, low in zip(query, self.lo, strict=True))
        return scaled, bias

    @staticmethod
    def score(prepared: tuple[list[float], float], code: bytes) -> float:
        scaled, bias = prepared
        return float(bias + sum(map(operator.mul, scaled, code)))

    def to_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "step": self.step}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScalarQuantizer:
        q = cls(
            data["lo"], [low + 255.0 * s for low, s in zip(data["lo"], data["step"], strict=True)]
        )
        q.step = list(data["step"])
        return q


class BinaryQuantizer:
    """One bit per dimension: set when the value exceeds that dimension's threshold."""

    def __init__(self, thresholds: Sequence[float]) -> None:
        if not thresholds:
            raise ValueError("thresholds must be non-empty")
        self.thresholds = [float(t) for t in thresholds]

    @property
    def dims(self) -> int:
        return len(self.thresholds)

    @classmethod
    def fit(cls, vectors: Sequence[Sequence[float]]) -> BinaryQuantizer:
        """Use each dimension's mean as its threshold (centers the bits, ~50 % ones)."""
        if not vectors:
            raise ValueError("cannot fit a quantizer on zero vectors")
        dims = len(vectors[0])
        n = len(vectors)
        return cls([sum(v[d] for v in vectors) / n for d in range(dims)])

    def encode(self, vector: Sequence[float]) -> int:
        if len(vector) != self.dims:
            raise ValueError(f"expected {self.dims} dims, got {len(vector)}")
        code = 0
        for i, (x, t) in enumerate(zip(vector, self.thresholds, strict=True)):
            if x > t:
                code |= 1 << i
        return code

    def similarity(self, a: int, b: int) -> float:
        """``1 - 2 * hamming / dims``: 1 for identical codes, -1 for complementary ones."""
        return 1.0 - 2.0 * (a ^ b).bit_count() / self.dims

    def to_dict(self) -> dict[str, Any]:
        return {"thresholds": self.thresholds}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BinaryQuantizer:
        return cls(data["thresholds"])


class QuantizedVectorIndex:
    """Quantized cosine search with optional full-precision rescoring.

    Args:
        embed_fn: Text embedder.
        mode: ``"int8"`` (4x smaller) or ``"binary"`` (32x smaller).
        rescore_factor: Shortlist ``k * rescore_factor`` candidates by code score, then
            rescore them exactly (requires ``keep_full_precision``).
        keep_full_precision: Keep the original unit vectors for rescoring (and for
            :meth:`refit`). The *code* is what is searched; the floats are a rescoring
            store that production systems place on disk, so :meth:`memory_bytes`
            reports code and rescoring store separately.

    The quantizer is fitted on the first batch of vectors added. Vectors added later reuse
    that fit (values outside the learned range are clipped); call :meth:`refit` after
    large distribution shifts.
    """

    def __init__(
        self,
        embed_fn: EmbedFn = hashing_embed,
        mode: Mode = "int8",
        rescore_factor: int = 4,
        keep_full_precision: bool = True,
    ) -> None:
        if mode not in ("int8", "binary"):
            raise ValueError(f"mode must be 'int8' or 'binary', got {mode!r}")
        if rescore_factor < 1:
            raise ValueError(f"rescore_factor must be >= 1, got {rescore_factor}")
        self.embed_fn = embed_fn
        self.mode: Mode = mode
        self.rescore_factor = rescore_factor
        self.keep_full_precision = keep_full_precision
        self._chunks: list[Chunk] = []
        self._codes: list[Any] = []
        self._full: list[list[float]] = []
        self._quantizer: ScalarQuantizer | BinaryQuantizer | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    def _make_quantizer(
        self, vectors: Sequence[Sequence[float]]
    ) -> ScalarQuantizer | BinaryQuantizer:
        return ScalarQuantizer.fit(vectors) if self.mode == "int8" else BinaryQuantizer.fit(vectors)

    def add(self, chunks: list[Chunk]) -> None:
        """Embed and index new chunks."""
        if chunks:
            self.add_vectors(chunks, [self.embed_fn(c.index_text) for c in chunks])

    def add_vectors(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Index chunks with precomputed embeddings (vectors are unit-normalized)."""
        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatch: {len(chunks)} chunks provided with {len(vectors)} vectors")
        if not chunks:
            return
        units = [normalize_vector(v) for v in vectors]
        if self._quantizer is None:
            self._quantizer = self._make_quantizer(units)
        elif len(units[0]) != self._quantizer.dims:
            raise ValueError(
                f"Vector dimension mismatch: index has {self._quantizer.dims}, got {len(units[0])}"
            )
        for vec in units:
            self._codes.append(self._quantizer.encode(vec))
            if self.keep_full_precision:
                self._full.append(vec)
        self._chunks.extend(chunks)

    def refit(self) -> None:
        """Re-learn the quantizer from the stored full-precision vectors and re-encode."""
        if not self.keep_full_precision:
            raise ValueError("refit requires keep_full_precision=True")
        if not self._full:
            return
        self._quantizer = self._make_quantizer(self._full)
        self._codes = [self._quantizer.encode(v) for v in self._full]

    def delete(self, doc_id: str) -> int:
        """Delete all chunks for a given document."""
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        removed = len(self._chunks) - len(keep)
        if removed:
            self._chunks = [self._chunks[i] for i in keep]
            self._codes = [self._codes[i] for i in keep]
            if self.keep_full_precision:
                self._full = [self._full[i] for i in keep]
        return removed

    def clear(self) -> None:
        """Remove every chunk and forget the fitted quantizer."""
        self._chunks, self._codes, self._full, self._quantizer = [], [], [], None

    def _code_scores(self, unit_query: list[float]) -> list[float]:
        quantizer = self._quantizer
        assert quantizer is not None
        if isinstance(quantizer, ScalarQuantizer):
            prepared = quantizer.prepare_query(unit_query)
            return [quantizer.score(prepared, code) for code in self._codes]
        qcode = quantizer.encode(unit_query)
        return [quantizer.similarity(qcode, code) for code in self._codes]

    def search(
        self, query: str, k: int = 5, filter_fn: FilterFn | None = None
    ) -> list[ScoredChunk]:
        """Top-``k`` by code score, rescored exactly when full-precision vectors are kept."""
        if k <= 0 or not self._chunks or self._quantizer is None:
            return []
        unit = normalize_vector(self.embed_fn(query))
        if len(unit) != self._quantizer.dims:
            raise ValueError(
                f"Vector dimension mismatch: index has {self._quantizer.dims}, "
                f"query has {len(unit)}"
            )
        scores = self._code_scores(unit)
        allowed = [i for i in range(len(scores)) if filter_fn is None or filter_fn(self._chunks[i])]
        if not allowed:
            return []

        if self.keep_full_precision:
            shortlist = heapq.nlargest(
                min(len(allowed), k * self.rescore_factor), allowed, key=lambda i: (scores[i], -i)
            )
            exact = [(float(sum(map(operator.mul, unit, self._full[i]))), i) for i in shortlist]
            top = heapq.nlargest(k, exact, key=lambda p: (p[0], -p[1]))
            tag = f"vector:{self.mode}+rescore"
        else:
            top = heapq.nlargest(k, ((scores[i], i) for i in allowed), key=lambda p: (p[0], -p[1]))
            tag = f"vector:{self.mode}"
        return [ScoredChunk(chunk=self._chunks[i], score=s, provenance=tag) for s, i in top]

    def memory_bytes(self) -> dict[str, int]:
        """Bytes used by the codes, the optional rescoring store, and a float32 baseline."""
        n = len(self._chunks)
        dims = self._quantizer.dims if self._quantizer else 0
        code = n * dims if self.mode == "int8" else n * ((dims + 7) // 8)
        return {
            "codes": code,
            "rescoring_store": n * dims * 4 if self.keep_full_precision else 0,
            "float32_baseline": n * dims * 4,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize (codes as base64 / hex; full vectors only if kept)."""
        if self.mode == "int8":
            codes = [base64.b64encode(c).decode("ascii") for c in self._codes]
        else:
            width = ((self._quantizer.dims + 7) // 8) if self._quantizer else 0
            codes = [c.to_bytes(width, "little").hex() for c in self._codes]
        return {
            "mode": self.mode,
            "rescore_factor": self.rescore_factor,
            "keep_full_precision": self.keep_full_precision,
            "quantizer": self._quantizer.to_dict() if self._quantizer else None,
            "chunks": [c.to_dict() for c in self._chunks],
            "codes": codes,
            "vectors": self._full if self.keep_full_precision else [],
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], embed_fn: EmbedFn | None = None
    ) -> QuantizedVectorIndex:
        index = cls(
            embed_fn=embed_fn or hashing_embed,
            mode=data["mode"],
            rescore_factor=data.get("rescore_factor", 4),
            keep_full_precision=data.get("keep_full_precision", True),
        )
        index._chunks = [Chunk.from_dict(c) for c in data.get("chunks", [])]
        if data.get("quantizer"):
            quant_cls = ScalarQuantizer if index.mode == "int8" else BinaryQuantizer
            index._quantizer = quant_cls.from_dict(data["quantizer"])
        if index.mode == "int8":
            index._codes = [base64.b64decode(c) for c in data.get("codes", [])]
        else:
            index._codes = [
                int.from_bytes(bytes.fromhex(c), "little") for c in data.get("codes", [])
            ]
        index._full = (
            [list(v) for v in data.get("vectors", [])] if index.keep_full_precision else []
        )
        return index

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str | Path, embed_fn: EmbedFn | None = None) -> QuantizedVectorIndex:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f), embed_fn=embed_fn)
