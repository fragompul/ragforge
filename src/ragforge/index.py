"""Retrieval indices: sparse (BM25), dense (vector), and hybrid fusion.

Hybrid search leverages the complementary strengths of lexical and semantic retrieval:
- BM25 handles exact keyword queries, proper nouns, and specific identifiers (e.g. error codes).
- Dense vector search handles semantic similarity, conceptual queries, and synonyms.
- Reciprocal Rank Fusion (RRF) combines both rankings without fragile score normalization.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ragforge.ann import HNSWIndex
from ragforge.chunking import Chunk
from ragforge.embeddings import EmbedFn, cosine_similarity, hashing_embed

_TOKEN_RE = re.compile(r"[a-z0-9]+")

FilterFn = Callable[[Chunk], bool]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredChunk:
    """A chunk paired with its relevance score and ranking metadata."""

    chunk: Chunk
    score: float
    provenance: str = ""
    ranks: dict[str, int] = field(default_factory=dict)


class BM25Index:
    """A self-contained Okapi BM25 implementation with document updates and filtering."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be between 0.0 and 1.0, got {b}")

        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_doc_len = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the BM25 index and update term statistics."""
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            self._chunks.append(chunk)
            self._doc_tokens.append(tokens)
            for term in set(tokens):
                self._doc_freq[term] += 1

        if self._doc_tokens:
            self._avg_doc_len = sum(len(t) for t in self._doc_tokens) / len(self._doc_tokens)

    def delete(self, doc_id: str) -> int:
        """Remove all chunks associated with `doc_id` and rebuild term frequencies."""
        initial_count = len(self._chunks)
        new_chunks: list[Chunk] = []
        new_doc_tokens: list[list[str]] = []
        new_doc_freq: Counter[str] = Counter()

        for chunk, tokens in zip(self._chunks, self._doc_tokens, strict=True):
            if chunk.doc_id != doc_id:
                new_chunks.append(chunk)
                new_doc_tokens.append(tokens)
                for term in set(tokens):
                    new_doc_freq[term] += 1

        self._chunks = new_chunks
        self._doc_tokens = new_doc_tokens
        self._doc_freq = new_doc_freq
        self._avg_doc_len = (
            sum(len(t) for t in self._doc_tokens) / len(self._doc_tokens)
            if self._doc_tokens
            else 0.0
        )
        return initial_count - len(self._chunks)

    def clear(self) -> None:
        """Clear all indexed chunks and reset statistics."""
        self._chunks = []
        self._doc_tokens = []
        self._doc_freq = Counter()
        self._avg_doc_len = 0.0

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        if n == 0:
            return 0.0
        df = self._doc_freq.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: FilterFn | None = None,
    ) -> list[ScoredChunk]:
        """Search the BM25 index with optional metadata filtering."""
        if k <= 0:
            return []

        query_terms = _tokenize(query)
        if not query_terms or not self._chunks:
            return []

        scored: list[ScoredChunk] = []
        for chunk, tokens in zip(self._chunks, self._doc_tokens, strict=True):
            if filter_fn is not None and not filter_fn(chunk):
                continue

            term_freqs = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for term in query_terms:
                if term not in term_freqs:
                    continue
                tf = term_freqs[term]
                idf = self._idf(term)
                length_norm = 1.0 - self.b + self.b * doc_len / max(self._avg_doc_len, 1e-9)
                denom = tf + self.k1 * length_norm
                score += idf * (tf * (self.k1 + 1.0)) / denom

            if score > 0:
                scored.append(ScoredChunk(chunk=chunk, score=score, provenance="bm25"))

        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:k]

    def to_dict(self) -> dict[str, Any]:
        """Serialize BM25 index state to a dictionary."""
        return {
            "k1": self.k1,
            "b": self.b,
            "chunks": [c.to_dict() for c in self._chunks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BM25Index:
        """Deserialize BM25 index state from a dictionary."""
        index = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        chunks = [Chunk.from_dict(c) for c in data.get("chunks", [])]
        index.add(chunks)
        return index

    def save(self, path: str | Path) -> None:
        """Save the index to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        """Load an index from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class VectorIndex:
    """Cosine-similarity search over embedded chunks with filtering and persistence."""

    def __init__(self, embed_fn: EmbedFn = hashing_embed) -> None:
        self.embed_fn = embed_fn
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk]) -> None:
        """Embed and index new chunks."""
        for chunk in chunks:
            self._chunks.append(chunk)
            self._vectors.append(self.embed_fn(chunk.text))

    def add_vectors(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Index chunks with precomputed embeddings."""
        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatch: {len(chunks)} chunks provided with {len(vectors)} vectors")
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def delete(self, doc_id: str) -> int:
        """Delete all chunks for a given document."""
        initial_count = len(self._chunks)
        new_chunks: list[Chunk] = []
        new_vectors: list[list[float]] = []

        for chunk, vec in zip(self._chunks, self._vectors, strict=True):
            if chunk.doc_id != doc_id:
                new_chunks.append(chunk)
                new_vectors.append(vec)

        self._chunks = new_chunks
        self._vectors = new_vectors
        return initial_count - len(self._chunks)

    def clear(self) -> None:
        """Clear all indexed chunks and vectors."""
        self._chunks = []
        self._vectors = []

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: FilterFn | None = None,
    ) -> list[ScoredChunk]:
        """Search the vector index with optional metadata filtering."""
        if k <= 0 or not self._chunks:
            return []

        query_vec = self.embed_fn(query)
        scored: list[ScoredChunk] = []

        for chunk, vec in zip(self._chunks, self._vectors, strict=True):
            if filter_fn is not None and not filter_fn(chunk):
                continue
            sim = cosine_similarity(query_vec, vec)
            scored.append(ScoredChunk(chunk=chunk, score=sim, provenance="vector"))

        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:k]

    def to_dict(self) -> dict[str, Any]:
        """Serialize vector index state to a dictionary."""
        return {
            "chunks": [c.to_dict() for c in self._chunks],
            "vectors": self._vectors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], embed_fn: EmbedFn | None = None) -> VectorIndex:
        """Deserialize vector index from a dictionary."""
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
    def load(cls, path: str | Path, embed_fn: EmbedFn | None = None) -> VectorIndex:
        """Load an index from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, embed_fn=embed_fn)


@runtime_checkable
class VectorSearchable(Protocol):
    """Structural type shared by ``VectorIndex`` and ``ApproxVectorIndex``.

    ``RagPipeline`` and ``HybridRetriever`` depend on this narrow protocol
    rather than a concrete class, letting callers swap brute-force cosine
    scan for approximate HNSW search (or any future backend) without
    touching orchestration, fusion, or reranking code.
    """

    def add(self, chunks: list[Chunk]) -> None: ...

    def delete(self, doc_id: str) -> int: ...

    def clear(self) -> None: ...

    def search(
        self, query: str, k: int, filter_fn: FilterFn | None = None
    ) -> list[ScoredChunk]: ...

    def to_dict(self) -> dict[str, Any]: ...


class ApproxVectorIndex:
    """Approximate dense vector index backed by the from-scratch HNSW graph.

    Drop-in alternative to :class:`VectorIndex` implementing the same public
    surface (``add``, ``add_vectors``, ``delete``, ``clear``, ``search``,
    ``to_dict``/``from_dict``, ``save``/``load``), trading exact brute-force
    cosine ranking for expected ``O(log n)`` query time via
    :class:`ragforge.ann.HNSWIndex`.

    Use this once a corpus grows large enough that linear scan becomes the
    retrieval bottleneck (see ``docs/benchmarks.md`` for measured crossover
    points); below that, ``VectorIndex`` is simpler and equally fast in
    absolute terms.

    Filtering caveat: HNSW graph traversal cannot apply ``filter_fn`` while
    walking the graph (unlike the brute-force index, which filters every
    candidate). Filtered searches instead over-fetch nearest neighbors and
    filter the result set, which can under-return results when a filter is
    very selective relative to ``ef_search`` -- a well-known limitation of
    graph-based ANN indices also present in production systems (e.g. HNSWLIB,
    Qdrant's pre-filtering mode).
    """

    def __init__(
        self,
        embed_fn: EmbedFn = hashing_embed,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> None:
        self.embed_fn = embed_fn
        self.ef_search = ef_search
        self._graph = HNSWIndex(m=m, ef_construction=ef_construction)
        self._chunks_by_node: dict[int, Chunk] = {}

    def add(self, chunks: list[Chunk]) -> None:
        """Embed and index new chunks."""
        for chunk in chunks:
            self.add_vectors([chunk], [self.embed_fn(chunk.text)])

    def add_vectors(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Index chunks with precomputed embeddings."""
        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatch: {len(chunks)} chunks provided with {len(vectors)} vectors")
        for chunk, vector in zip(chunks, vectors, strict=True):
            node_id = self._graph.add(vector)
            self._chunks_by_node[node_id] = chunk

    def delete(self, doc_id: str) -> int:
        """Tombstone all chunks for a given document (see class docstring)."""
        removed = 0
        for node_id, chunk in list(self._chunks_by_node.items()):
            if chunk.doc_id == doc_id:
                self._graph.mark_deleted(node_id)
                del self._chunks_by_node[node_id]
                removed += 1
        return removed

    def clear(self) -> None:
        """Clear all indexed chunks and rebuild an empty graph."""
        self._graph = HNSWIndex(
            m=self._graph.m,
            ef_construction=self._graph.ef_construction,
            seed=self._graph.seed,
        )
        self._chunks_by_node.clear()

    def search(
        self, query: str, k: int = 5, filter_fn: FilterFn | None = None
    ) -> list[ScoredChunk]:
        """Search the HNSW graph with optional (over-fetch based) metadata filtering."""
        if k <= 0 or not self._chunks_by_node:
            return []

        query_vec = self.embed_fn(query)
        fetch_k = (
            k if filter_fn is None else min(len(self._chunks_by_node), max(k * 10, self.ef_search))
        )
        raw = self._graph.search(query_vec, k=fetch_k, ef_search=max(self.ef_search, fetch_k))

        scored: list[ScoredChunk] = []
        for sim, node_id in raw:
            chunk = self._chunks_by_node.get(node_id)
            if chunk is None:
                continue
            if filter_fn is not None and not filter_fn(chunk):
                continue
            scored.append(ScoredChunk(chunk=chunk, score=sim, provenance="vector:hnsw"))
            if len(scored) >= k:
                break

        return scored

    def to_dict(self) -> dict[str, Any]:
        """Serialize the HNSW graph and chunk mapping to a dictionary."""
        return {
            "ef_search": self.ef_search,
            "graph": self._graph.to_dict(),
            "chunks_by_node": {str(k): c.to_dict() for k, c in self._chunks_by_node.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], embed_fn: EmbedFn | None = None) -> ApproxVectorIndex:
        """Deserialize an approximate vector index from a dictionary."""
        index = cls(embed_fn=embed_fn or hashing_embed, ef_search=data.get("ef_search", 50))
        index._graph = HNSWIndex.from_dict(data["graph"])
        index._chunks_by_node = {
            int(k): Chunk.from_dict(v) for k, v in data.get("chunks_by_node", {}).items()
        }
        return index

    def save(self, path: str | Path) -> None:
        """Save the index to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path, embed_fn: EmbedFn | None = None) -> ApproxVectorIndex:
        """Load an index from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, embed_fn=embed_fn)


class HybridRetriever:
    """Combines BM25 sparse retrieval and dense vector retrieval using Reciprocal Rank Fusion (RRF).

    RRF formula: ``RRF_Score(d) = sum_i [ weight_i / (k_rrf + rank_i(d)) ]``
    This provides robust fusion without requiring brittle score calibration or normalization.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        vector_index: VectorSearchable,
        k_rrf: int = 60,
        weight_bm25: float = 1.0,
        weight_vector: float = 1.0,
    ) -> None:
        if k_rrf <= 0:
            raise ValueError(f"k_rrf must be positive, got {k_rrf}")
        if weight_bm25 < 0:
            raise ValueError(f"weight_bm25 must be non-negative, got {weight_bm25}")
        if weight_vector < 0:
            raise ValueError(f"weight_vector must be non-negative, got {weight_vector}")

        self.bm25_index = bm25_index
        self.vector_index = vector_index
        self.k_rrf = k_rrf
        self.weight_bm25 = weight_bm25
        self.weight_vector = weight_vector

    def search(
        self,
        query: str,
        k: int = 5,
        candidate_pool: int = 20,
        filter_fn: FilterFn | None = None,
    ) -> list[ScoredChunk]:
        """Execute hybrid search using weighted Reciprocal Rank Fusion."""
        if k <= 0:
            return []

        pool = max(k, candidate_pool)
        bm25_results = self.bm25_index.search(query, k=pool, filter_fn=filter_fn)
        vector_results = self.vector_index.search(query, k=pool, filter_fn=filter_fn)

        rrf_scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}
        provenance_map: dict[str, list[str]] = {}
        rank_map: dict[str, dict[str, int]] = {}

        for weight, source_name, results in [
            (self.weight_bm25, "bm25", bm25_results),
            (self.weight_vector, "vector", vector_results),
        ]:
            if weight <= 0:
                continue
            for rank, scored in enumerate(results, start=1):
                cid = scored.chunk.id
                rrf_increment = weight / (self.k_rrf + rank)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_increment
                chunks_by_id[cid] = scored.chunk

                if cid not in provenance_map:
                    provenance_map[cid] = []
                    rank_map[cid] = {}
                provenance_map[cid].append(source_name)
                rank_map[cid][source_name] = rank

        fused: list[ScoredChunk] = []
        for cid, score in rrf_scores.items():
            sources = provenance_map.get(cid, [])
            prov = "+".join(sources) if sources else "hybrid"
            fused.append(
                ScoredChunk(
                    chunk=chunks_by_id[cid],
                    score=score,
                    provenance=prov,
                    ranks=rank_map.get(cid, {}),
                )
            )

        fused.sort(key=lambda sc: sc.score, reverse=True)
        return fused[:k]
