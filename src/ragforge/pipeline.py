"""End-to-end RAG pipeline orchestration with hybrid retrieval, reranking, and generation.

Ties chunking, sparse BM25 indexing, dense vector indexing, Reciprocal Rank Fusion,
custom reranking, metadata filtering, prompt formatting, and LLM generation into a
cohesive, production-ready pipeline.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragforge.chunking import Chunk, Chunker, FixedSizeChunker
from ragforge.embeddings import EmbedFn, hashing_embed
from ragforge.index import (
    ApproxVectorIndex,
    BM25Index,
    FilterFn,
    HybridRetriever,
    VectorIndex,
    VectorSearchable,
)
from ragforge.query_expansion import MultiQueryRetriever, QueryExpansionFn
from ragforge.reranking import NoopReranker, Reranker

GenerateFn = Callable[[str, list[str]], str]
PromptFormatter = Callable[[str, list[str]], str]


@dataclass
class RetrievedChunk:
    """A retrieved context chunk enriched with ranking, provenance, and metadata."""

    text: str
    score: float
    doc_id: str
    chunk_id: str = ""
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: str = ""
    ranks: dict[str, int] = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "position": self.position,
            "metadata": dict(self.metadata),
            "provenance": self.provenance,
            "ranks": dict(self.ranks),
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class RagAnswer:
    """The complete response from the RAG pipeline."""

    query: str
    answer: str
    contexts: list[RetrievedChunk]
    prompt: str = ""
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "contexts": [c.to_dict() for c in self.contexts],
            "prompt": self.prompt,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "metadata": dict(self.metadata),
        }


def default_prompt_formatter(query: str, contexts: list[str]) -> str:
    """Format retrieval contexts and user query into a standard prompt."""
    if not contexts:
        return f"Question: {query}\n\nNo context provided."

    formatted_contexts = "\n\n".join(f"[Context {i + 1}]:\n{ctx}" for i, ctx in enumerate(contexts))
    return (
        "Answer the question truthfully and accurately based strictly on the provided contexts.\n\n"
        f"--- CONTEXTS ---\n{formatted_contexts}\n\n"
        f"--- QUESTION ---\n{query}\n\n"
        "--- ANSWER ---"
    )


def _default_generate(query: str, contexts: list[str]) -> str:
    """Extractive fallback generator for zero-dependency offline operation.

    Returns the top retrieved context verbatim, or a graceful message when no
    contexts were retrieved. Real deployments pass an LLM call as ``generate_fn``.
    """
    return contexts[0] if contexts else "I don't have enough information to answer that."


class RagPipeline:
    """Production RAG orchestrator combining multi-strategy chunking, hybrid search,
    reranking, metadata filtering, prompt formatting, and generation.
    """

    def __init__(
        self,
        chunker: Chunker | None = None,
        embed_fn: EmbedFn = hashing_embed,
        reranker: Reranker | None = None,
        generate_fn: GenerateFn | None = None,
        prompt_formatter: PromptFormatter | None = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        k_rrf: int = 60,
        weight_bm25: float = 1.0,
        weight_vector: float = 1.0,
        use_ann: bool = False,
        ann_params: dict[str, Any] | None = None,
        query_expansion_fn: QueryExpansionFn | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.chunker: Chunker = chunker or FixedSizeChunker()
        self.embed_fn = embed_fn
        self.bm25_index = BM25Index(k1=bm25_k1, b=bm25_b)
        self.vector_index: VectorSearchable = (
            ApproxVectorIndex(embed_fn=embed_fn, **(ann_params or {}))
            if use_ann
            else VectorIndex(embed_fn=embed_fn)
        )
        self.k_rrf = k_rrf
        self.weight_bm25 = weight_bm25
        self.weight_vector = weight_vector
        base_retriever = HybridRetriever(
            self.bm25_index,
            self.vector_index,
            k_rrf=k_rrf,
            weight_bm25=weight_bm25,
            weight_vector=weight_vector,
        )
        self.query_expansion_fn = query_expansion_fn
        self.retriever: HybridRetriever | MultiQueryRetriever = (
            MultiQueryRetriever(base_retriever, expand_fn=query_expansion_fn, k_rrf=k_rrf)
            if query_expansion_fn is not None
            else base_retriever
        )
        self.reranker = reranker or NoopReranker()
        self.generate_fn = generate_fn or _default_generate
        self.prompt_formatter = prompt_formatter or default_prompt_formatter

    @property
    def document_count(self) -> int:
        """Total count of unique documents currently indexed."""
        return len({c.doc_id for c in self.bm25_index._chunks})

    @property
    def chunk_count(self) -> int:
        """Total count of chunks currently indexed."""
        return len(self.bm25_index._chunks)

    def ingest(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Chunk and index a single document into sparse and dense indices."""
        chunks = self.chunker.chunk(text, doc_id=doc_id, metadata=metadata)
        if chunks:
            self.bm25_index.add(chunks)
            self.vector_index.add(chunks)
        return chunks

    def ingest_batch(self, documents: list[dict[str, Any]]) -> list[Chunk]:
        """Ingest a batch of documents.

        Each document dictionary must contain:
        - ``id`` or ``doc_id``: Unique document identifier.
        - ``text``: Document text content.
        - ``metadata`` (optional): Document metadata dictionary.
        """
        all_chunks: list[Chunk] = []
        for doc in documents:
            doc_id = str(doc.get("id") or doc.get("doc_id") or "")
            if not doc_id:
                raise ValueError("Each document in batch must contain 'id' or 'doc_id'")
            text = str(doc.get("text", ""))
            meta = doc.get("metadata")
            chunks = self.ingest(doc_id=doc_id, text=text, metadata=meta)
            all_chunks.extend(chunks)
        return all_chunks

    def delete(self, doc_id: str) -> int:
        """Remove a document and its chunks from all underlying indices."""
        bm25_removed = self.bm25_index.delete(doc_id)
        self.vector_index.delete(doc_id)
        return bm25_removed

    def clear(self) -> None:
        """Clear all indexed documents and reset indices."""
        self.bm25_index.clear()
        self.vector_index.clear()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        candidate_pool: int = 20,
        filter_fn: FilterFn | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve, fuse, and rerank candidate contexts for a given query."""
        if k <= 0:
            return []

        pool = max(k, candidate_pool)
        candidates = self.retriever.search(query, k=pool, candidate_pool=pool, filter_fn=filter_fn)
        reranked = self.reranker.rerank(query, candidates, k=k)

        return [
            RetrievedChunk(
                chunk_id=sc.chunk.id,
                doc_id=sc.chunk.doc_id,
                text=sc.chunk.text,
                score=sc.score,
                position=sc.chunk.position,
                metadata=sc.chunk.metadata,
                provenance=sc.provenance,
                ranks=sc.ranks,
                start_char=sc.chunk.start_char,
                end_char=sc.chunk.end_char,
            )
            for sc in reranked
        ]

    def answer(
        self,
        query: str,
        k: int = 5,
        candidate_pool: int = 20,
        filter_fn: FilterFn | None = None,
    ) -> RagAnswer:
        """Execute end-to-end RAG: retrieve contexts, build prompt, generate answer."""
        start_time = time.perf_counter()

        retrieval_start = time.perf_counter()
        retrieved = self.retrieve(query, k=k, candidate_pool=candidate_pool, filter_fn=filter_fn)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        contexts_text = [r.text for r in retrieved]
        prompt = self.prompt_formatter(query, contexts_text)

        generation_start = time.perf_counter()
        answer_text = self.generate_fn(query, contexts_text)
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000.0

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RagAnswer(
            query=query,
            answer=answer_text,
            contexts=retrieved,
            prompt=prompt,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            metadata={
                "retrieved_count": len(retrieved),
                "k": k,
                "candidate_pool": candidate_pool,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize pipeline configuration and index state to dictionary."""
        return {
            "bm25_index": self.bm25_index.to_dict(),
            "vector_index": self.vector_index.to_dict(),
            "vector_backend": "hnsw"
            if isinstance(self.vector_index, ApproxVectorIndex)
            else "brute_force",
            "k_rrf": self.k_rrf,
            "weight_bm25": self.weight_bm25,
            "weight_vector": self.weight_vector,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        chunker: Chunker | None = None,
        embed_fn: EmbedFn | None = None,
        reranker: Reranker | None = None,
        generate_fn: GenerateFn | None = None,
        query_expansion_fn: QueryExpansionFn | None = None,
    ) -> RagPipeline:
        """Reconstruct pipeline from a serialized state dictionary.

        ``query_expansion_fn`` is not serialized (it is an arbitrary callable,
        like ``embed_fn``/``generate_fn``) -- pass it again here to re-enable
        multi-query retrieval on a loaded pipeline.
        """
        embed = embed_fn or hashing_embed
        use_ann = data.get("vector_backend") == "hnsw"
        pipeline = cls(
            chunker=chunker,
            embed_fn=embed,
            reranker=reranker,
            generate_fn=generate_fn,
            k_rrf=data.get("k_rrf", 60),
            weight_bm25=data.get("weight_bm25", 1.0),
            weight_vector=data.get("weight_vector", 1.0),
            use_ann=use_ann,
            query_expansion_fn=query_expansion_fn,
        )
        if "bm25_index" in data:
            pipeline.bm25_index = BM25Index.from_dict(data["bm25_index"])
        if "vector_index" in data:
            backend_cls = ApproxVectorIndex if use_ann else VectorIndex
            pipeline.vector_index = backend_cls.from_dict(data["vector_index"], embed_fn=embed)

        base_retriever = HybridRetriever(
            pipeline.bm25_index,
            pipeline.vector_index,
            k_rrf=pipeline.k_rrf,
            weight_bm25=pipeline.weight_bm25,
            weight_vector=pipeline.weight_vector,
        )
        pipeline.retriever = (
            MultiQueryRetriever(base_retriever, expand_fn=query_expansion_fn, k_rrf=pipeline.k_rrf)
            if query_expansion_fn is not None
            else base_retriever
        )
        return pipeline

    def save(self, path: str | Path) -> None:
        """Save the pipeline index state to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        chunker: Chunker | None = None,
        embed_fn: EmbedFn | None = None,
        reranker: Reranker | None = None,
        generate_fn: GenerateFn | None = None,
        query_expansion_fn: QueryExpansionFn | None = None,
    ) -> RagPipeline:
        """Load a pipeline index state from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(
            data,
            chunker=chunker,
            embed_fn=embed_fn,
            reranker=reranker,
            generate_fn=generate_fn,
            query_expansion_fn=query_expansion_fn,
        )
