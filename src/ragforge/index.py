"""Retrieval indices: sparse (BM25), dense (vector), and hybrid fusion.

Hybrid search leverages the complementary strengths of lexical and semantic retrieval:
- BM25 handles exact keyword queries, proper nouns, and specific identifiers (e.g. error codes).
- Dense vector search handles semantic similarity, conceptual queries, and synonyms.
- Reciprocal Rank Fusion (RRF) combines both rankings without fragile score normalization.
"""

from __future__ import annotations

import heapq
import json
import math
import operator
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ragforge.ann import HNSWIndex
from ragforge.chunking import Chunk
from ragforge.embeddings import EmbedFn, cosine_similarity, hashing_embed, normalize_vector

_TOKEN_RE = re.compile(r"[a-z0-9]+")

FilterFn = Callable[[Chunk], bool]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


ENGLISH_STOPWORDS = frozenset(
    [
        "a", "about", "above", "after", "again", "all", "also", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him",
        "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "me", "more", "most",
        "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
        "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
        "the", "their", "them", "then", "there", "these", "they", "this", "those", "through",
        "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
        "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your",
    ]
)  # fmt: skip


def _light_stem(token: str) -> str:
    """A tiny suffix stripper (plural / -ing / -ed): conservative, dependency-free."""
    n = len(token)
    if n > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if n > 5 and token.endswith("sses"):
        return token[:-2]
    if n > 5 and token.endswith("ing"):
        return token[:-3]
    if n > 4 and token.endswith("ed"):
        return token[:-2]
    if n > 4 and token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    if n > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def english_analyzer(text: str) -> list[str]:
    """Lowercase, drop stopwords, light-stem. Numbers and identifiers pass through."""
    return [_light_stem(t) for t in _tokenize(text) if t not in ENGLISH_STOPWORDS]


_ANALYZERS: dict[str, Callable[[str], list[str]]] = {
    "simple": _tokenize,
    "english": english_analyzer,
}


@dataclass
class ScoredChunk:
    """A chunk paired with its relevance score and ranking metadata."""

    chunk: Chunk
    score: float
    provenance: str = ""
    ranks: dict[str, int] = field(default_factory=dict)


class BM25Index:
    """A self-contained Okapi BM25 implementation backed by an inverted index.

    Postings (``term -> {chunk position: term frequency}``) are maintained
    incrementally on :meth:`add`, so a query only touches chunks that contain
    at least one query term -- ``O(sum of postings lengths)`` instead of
    re-tokenizing and scanning every chunk. Document updates (``delete``) and
    metadata filtering are supported.

    ``analyzer="simple"`` (default) lowercases and splits on non-alphanumerics;
    ``analyzer="english"`` additionally drops stopwords and applies a light
    stemmer, which noticeably improves ranking on natural-language queries.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, analyzer: str = "simple") -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be between 0.0 and 1.0, got {b}")
        if analyzer not in _ANALYZERS:
            raise ValueError(f"analyzer must be one of {sorted(_ANALYZERS)}, got {analyzer!r}")

        self.k1 = k1
        self.b = b
        self.analyzer = analyzer
        self._analyze = _ANALYZERS[analyzer]
        self._chunks: list[Chunk] = []
        self._doc_lens: list[int] = []
        self._postings: dict[str, dict[int, int]] = {}
        self._total_len = 0
        self._avg_doc_len = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the BM25 index and update term statistics."""
        for chunk in chunks:
            position = len(self._chunks)
            tokens = self._analyze(chunk.index_text)
            self._chunks.append(chunk)
            self._doc_lens.append(len(tokens))
            self._total_len += len(tokens)
            for term, tf in Counter(tokens).items():
                self._postings.setdefault(term, {})[position] = tf

        self._avg_doc_len = self._total_len / len(self._chunks) if self._chunks else 0.0

    def delete(self, doc_id: str) -> int:
        """Remove all chunks associated with `doc_id` and rebuild the postings."""
        kept = [c for c in self._chunks if c.doc_id != doc_id]
        removed = len(self._chunks) - len(kept)
        if removed:
            self.clear()
            self.add(kept)
        return removed

    def clear(self) -> None:
        """Clear all indexed chunks and reset statistics."""
        self._chunks = []
        self._doc_lens = []
        self._postings = {}
        self._total_len = 0
        self._avg_doc_len = 0.0

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        if n == 0:
            return 0.0
        df = len(self._postings.get(term, ()))
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

        query_terms = self._analyze(query)
        if not query_terms or not self._chunks:
            return []

        avg_len = max(self._avg_doc_len, 1e-9)
        scores: dict[int, float] = {}
        allowed: dict[int, bool] = {}
        for term in query_terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for position, tf in postings.items():
                if filter_fn is not None:
                    if position not in allowed:
                        allowed[position] = filter_fn(self._chunks[position])
                    if not allowed[position]:
                        continue
                length_norm = 1.0 - self.b + self.b * self._doc_lens[position] / avg_len
                denom = tf + self.k1 * length_norm
                scores[position] = scores.get(position, 0.0) + idf * (tf * (self.k1 + 1.0)) / denom

        # Ties resolve to the earlier-indexed chunk, keeping rankings deterministic.
        ranked = heapq.nsmallest(k, ((-score, pos) for pos, score in scores.items() if score > 0))
        return [
            ScoredChunk(chunk=self._chunks[pos], score=-neg_score, provenance="bm25")
            for neg_score, pos in ranked
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize BM25 index state to a dictionary."""
        payload: dict[str, Any] = {
            "k1": self.k1,
            "b": self.b,
            "chunks": [c.to_dict() for c in self._chunks],
        }
        if self.analyzer != "simple":
            payload["analyzer"] = self.analyzer
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BM25Index:
        """Deserialize BM25 index state from a dictionary."""
        index = cls(
            k1=data.get("k1", 1.5), b=data.get("b", 0.75), analyzer=data.get("analyzer", "simple")
        )
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
            self._vectors.append(self.embed_fn(chunk.index_text))

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

    Filtered search: ``filter_fn`` is applied *inside* the graph traversal -- every node
    still helps navigation, but only chunks that pass the filter can become results, and
    the beam is widened until it holds ``ef_search`` accepted candidates. When the filter
    is so selective that the (visit-capped) traversal cannot find ``k`` matches, the index
    falls back to an exact scan over the chunks that pass, so a filtered query returns
    the true top-``k`` matches whenever that many exist (and never silently under-returns,
    which the earlier over-fetch-then-filter approach could).
    """

    def __init__(
        self,
        embed_fn: EmbedFn = hashing_embed,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        filter_max_visits: int = 2000,
    ) -> None:
        self.embed_fn = embed_fn
        self.ef_search = ef_search
        self.filter_max_visits = filter_max_visits
        self._graph = HNSWIndex(m=m, ef_construction=ef_construction)
        self._chunks_by_node: dict[int, Chunk] = {}

    def add(self, chunks: list[Chunk]) -> None:
        """Embed and index new chunks."""
        for chunk in chunks:
            self.add_vectors([chunk], [self.embed_fn(chunk.index_text)])

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
        """Search the HNSW graph; ``filter_fn`` restricts results (see class docstring)."""
        if k <= 0 or not self._chunks_by_node:
            return []

        query_vec = self.embed_fn(query)
        ef = max(self.ef_search, k)
        if filter_fn is None:
            raw = self._graph.search(query_vec, k=k, ef_search=ef)
            return [
                ScoredChunk(chunk=self._chunks_by_node[nid], score=sim, provenance="vector:hnsw")
                for sim, nid in raw
                if nid in self._chunks_by_node
            ]

        verdicts: dict[int, bool] = {}

        def accept(node_id: int) -> bool:
            if node_id not in verdicts:
                chunk = self._chunks_by_node.get(node_id)
                verdicts[node_id] = chunk is not None and filter_fn(chunk)
            return verdicts[node_id]

        raw = self._graph.search(
            query_vec, k=k, ef_search=ef, accept=accept, max_visits=self.filter_max_visits
        )
        provenance = "vector:hnsw"
        if len(raw) < k:
            # Selective filter: finish with an exact scan over the chunks that pass it.
            unit = normalize_vector(query_vec)
            raw = heapq.nlargest(
                k,
                (
                    (float(sum(map(operator.mul, unit, self._graph.vector(nid)))), nid)
                    for nid in self._chunks_by_node
                    if accept(nid)
                ),
                key=lambda pair: (pair[0], -pair[1]),
            )
            provenance = "vector:hnsw+scan"
        return [
            ScoredChunk(chunk=self._chunks_by_node[nid], score=sim, provenance=provenance)
            for sim, nid in raw
        ]

    def compact(self) -> int:
        """Rebuild the graph without tombstones (see :meth:`HNSWIndex.compact`).

        Returns the number of dead nodes reclaimed.
        """
        reclaimed = self._graph.size_including_tombstones - len(self._graph)
        if reclaimed:
            mapping = self._graph.compact()
            self._chunks_by_node = {
                mapping[old]: chunk for old, chunk in self._chunks_by_node.items() if old in mapping
            }
        return reclaimed

    def to_dict(self) -> dict[str, Any]:
        """Serialize the HNSW graph and chunk mapping to a dictionary."""
        return {
            "ef_search": self.ef_search,
            "filter_max_visits": self.filter_max_visits,
            "graph": self._graph.to_dict(),
            "chunks_by_node": {str(k): c.to_dict() for k, c in self._chunks_by_node.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], embed_fn: EmbedFn | None = None) -> ApproxVectorIndex:
        """Deserialize an approximate vector index from a dictionary."""
        index = cls(
            embed_fn=embed_fn or hashing_embed,
            ef_search=data.get("ef_search", 50),
            filter_max_visits=data.get("filter_max_visits", 2000),
        )
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
