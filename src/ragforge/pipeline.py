"""End-to-end RAG pipeline orchestration with hybrid retrieval, reranking, and generation.

Ties chunking, sparse BM25 indexing, dense vector indexing, Reciprocal Rank Fusion,
custom reranking, metadata filtering, prompt formatting, and LLM generation into a
cohesive, production-ready pipeline.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragforge.chunking import Chunk, Chunker, FixedSizeChunker
from ragforge.embeddings import EmbedFn, hashing_embed
from ragforge.fast import NumpyVectorIndex
from ragforge.fusion import FusionRetriever, FusionSource
from ragforge.guardrails import Guardrails
from ragforge.index import (
    ApproxVectorIndex,
    BM25Index,
    FilterFn,
    HybridRetriever,
    VectorIndex,
    VectorSearchable,
)
from ragforge.quantization import QuantizedVectorIndex
from ragforge.query_expansion import MultiQueryRetriever, QueryExpansionFn
from ragforge.reranking import NoopReranker, Reranker
from ragforge.routing import QueryRouter
from ragforge.telemetry import Tracer

GenerateFn = Callable[[str, list[str]], str]
PromptFormatter = Callable[[str, list[str]], str]
StreamGenerateFn = Callable[[str, list[str]], Iterator[str]]

VECTOR_BACKENDS = ("brute_force", "hnsw", "numpy", "int8", "binary")

ContextFn = Callable[[str, Chunk], str]


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
    guardrails: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
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
        if self.guardrails:
            payload["guardrails"] = dict(self.guardrails)
        return payload


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
        guardrails: Guardrails | None = None,
        stream_generate_fn: StreamGenerateFn | None = None,
        vector_backend: str | None = None,
        bm25_analyzer: str = "simple",
        extra_indexes: dict[str, Any] | None = None,
        extra_weights: dict[str, float] | None = None,
        contextualizer: ContextFn | None = None,
        router: QueryRouter | None = None,
    ) -> None:
        if vector_backend is None:
            vector_backend = "hnsw" if use_ann else "brute_force"
        elif vector_backend not in VECTOR_BACKENDS:
            raise ValueError(
                f"vector_backend must be one of {VECTOR_BACKENDS}, got {vector_backend!r}"
            )
        elif use_ann and vector_backend != "hnsw":
            raise ValueError(f"use_ann=True conflicts with vector_backend={vector_backend!r}")
        self.vector_backend = vector_backend
        self.chunker: Chunker = chunker or FixedSizeChunker()
        self.embed_fn = embed_fn
        self.bm25_index = BM25Index(k1=bm25_k1, b=bm25_b, analyzer=bm25_analyzer)
        self.vector_index: VectorSearchable = self._make_vector_index(
            vector_backend, embed_fn, ann_params
        )
        self.k_rrf = k_rrf
        self.weight_bm25 = weight_bm25
        self.weight_vector = weight_vector
        self.extra_indexes: dict[str, Any] = dict(extra_indexes or {})
        self.extra_weights: dict[str, float] = dict.fromkeys(self.extra_indexes, 1.0)
        self.extra_weights.update(extra_weights or {})
        self.contextualizer = contextualizer
        self.router = router
        self.query_expansion_fn = query_expansion_fn
        self.retriever: HybridRetriever | FusionRetriever | MultiQueryRetriever
        self._build_retriever()
        self.reranker = reranker or NoopReranker()
        self.generate_fn = generate_fn or _default_generate
        self.prompt_formatter = prompt_formatter or default_prompt_formatter
        self.tracer = tracer or Tracer()
        self.guardrails = guardrails
        self.stream_generate_fn = stream_generate_fn

    def _build_retriever(self) -> None:
        """(Re)assemble the fusion stage from the current indexes and weights."""
        base: HybridRetriever | FusionRetriever
        if self.extra_indexes:
            sources = [
                FusionSource("bm25", self.bm25_index, self.weight_bm25),
                FusionSource("vector", self.vector_index, self.weight_vector),
                *(
                    FusionSource(name, index, self.extra_weights.get(name, 1.0))
                    for name, index in self.extra_indexes.items()
                ),
            ]
            base = FusionRetriever(sources, k_rrf=self.k_rrf)
        else:
            base = HybridRetriever(
                self.bm25_index,
                self.vector_index,
                k_rrf=self.k_rrf,
                weight_bm25=self.weight_bm25,
                weight_vector=self.weight_vector,
            )
        self._base_retriever = base
        self.retriever = (
            MultiQueryRetriever(base, expand_fn=self.query_expansion_fn, k_rrf=self.k_rrf)
            if self.query_expansion_fn is not None
            else base
        )

    def _apply_weights(self, weight_bm25: float, weight_vector: float) -> None:
        base = self._base_retriever
        if isinstance(base, FusionRetriever):
            base.set_weight("bm25", weight_bm25)
            base.set_weight("vector", weight_vector)
        else:
            base.weight_bm25, base.weight_vector = weight_bm25, weight_vector

    @staticmethod
    def _make_vector_index(
        backend: str, embed_fn: EmbedFn, ann_params: dict[str, Any] | None = None
    ) -> VectorSearchable:
        if backend == "hnsw":
            return ApproxVectorIndex(embed_fn=embed_fn, **(ann_params or {}))
        if backend == "numpy":
            return NumpyVectorIndex(embed_fn=embed_fn)
        if backend in ("int8", "binary"):
            return QuantizedVectorIndex(embed_fn=embed_fn, mode=backend)  # type: ignore[arg-type]
        return VectorIndex(embed_fn=embed_fn)

    @property
    def document_count(self) -> int:
        """Total count of unique documents currently indexed."""
        return len({c.doc_id for c in self.bm25_index._chunks})

    @property
    def chunks(self) -> list[Chunk]:
        """A snapshot of every indexed chunk, in ingestion order."""
        return list(self.bm25_index._chunks)

    @property
    def chunk_count(self) -> int:
        """Total count of chunks currently indexed."""
        return len(self.bm25_index._chunks)

    def ingest(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Chunk and index a single document into sparse and dense indices."""
        chunks = self.chunker.chunk(text, doc_id=doc_id, metadata=metadata)
        if chunks and self.contextualizer is not None:
            for chunk in chunks:
                chunk.context = self.contextualizer(text, chunk)
        if chunks:
            self.bm25_index.add(chunks)
            self.vector_index.add(chunks)
            for index in self.extra_indexes.values():
                index.add(chunks)
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
        for index in self.extra_indexes.values():
            index.delete(doc_id)
        return bm25_removed

    def clear(self) -> None:
        """Clear all indexed documents and reset indices."""
        self.bm25_index.clear()
        self.vector_index.clear()
        for index in self.extra_indexes.values():
            index.clear()

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
        with self.tracer.span("retrieve", query=query, k=k, candidate_pool=pool):
            with self.tracer.span("fusion_search") as fusion_span:
                if self.router is not None:
                    decision = self.router.route(query, self.bm25_index)
                    self._apply_weights(
                        self.weight_bm25 * decision.weight_bm25,
                        self.weight_vector * decision.weight_vector,
                    )
                    fusion_span.attributes["route"] = decision.reason
                candidates = self.retriever.search(
                    query, k=pool, candidate_pool=pool, filter_fn=filter_fn
                )
            with self.tracer.span(
                "rerank", reranker=type(self.reranker).__name__, candidates=len(candidates)
            ):
                reranked = self.reranker.rerank(query, candidates, k=k)

        results = [
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
        return self._apply_guardrails(results) if self.guardrails is not None else results

    def _apply_guardrails(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Screen retrieved contexts: redact PII, neutralize or drop injections."""
        assert self.guardrails is not None
        safe: list[RetrievedChunk] = []
        dropped = flagged = 0
        with self.tracer.span("guardrails", contexts=len(results)) as span:
            for chunk in results:
                screened = self.guardrails.screen(chunk.text)
                if screened.blocked:
                    dropped += 1
                    continue
                if screened.findings:
                    flagged += 1
                    chunk.guardrails = screened.to_dict()
                chunk.text = screened.text
                safe.append(chunk)
            span.attributes.update(flagged=flagged, dropped=dropped)
        return safe

    def answer(
        self,
        query: str,
        k: int = 5,
        candidate_pool: int = 20,
        filter_fn: FilterFn | None = None,
    ) -> RagAnswer:
        """Execute end-to-end RAG: retrieve contexts, build prompt, generate answer."""
        start_time = time.perf_counter()

        with self.tracer.span("answer", query=query, k=k):
            retrieval_start = time.perf_counter()
            retrieved = self.retrieve(
                query, k=k, candidate_pool=candidate_pool, filter_fn=filter_fn
            )
            retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

            contexts_text = [r.text for r in retrieved]
            prompt = self.prompt_formatter(query, contexts_text)

            generation_start = time.perf_counter()
            with self.tracer.span("generate", contexts=len(contexts_text)):
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

    def answer_stream(
        self,
        query: str,
        k: int = 5,
        candidate_pool: int = 20,
        filter_fn: FilterFn | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream a RAG answer as events, so UIs can render progressively.

        Yields, in order: one ``{"type": "contexts", "contexts": [...]}`` event as soon as
        retrieval finishes, then ``{"type": "token", "text": str}`` events, and finally
        ``{"type": "done", "answer": RagAnswer}`` carrying the complete answer and latencies.
        Uses ``stream_generate_fn`` when configured; otherwise the regular ``generate_fn``
        output is re-emitted word by word.
        """
        start_time = time.perf_counter()
        retrieved = self.retrieve(query, k=k, candidate_pool=candidate_pool, filter_fn=filter_fn)
        retrieval_latency_ms = (time.perf_counter() - start_time) * 1000.0
        yield {"type": "contexts", "contexts": [r.to_dict() for r in retrieved]}

        contexts_text = [r.text for r in retrieved]
        prompt = self.prompt_formatter(query, contexts_text)
        generation_start = time.perf_counter()
        pieces: list[str] = []
        source: Iterator[str] = (
            self.stream_generate_fn(query, contexts_text)
            if self.stream_generate_fn is not None
            else iter(re.findall(r"\S+\s*", self.generate_fn(query, contexts_text)))
        )
        for piece in source:
            pieces.append(piece)
            yield {"type": "token", "text": piece}
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000.0

        yield {
            "type": "done",
            "answer": RagAnswer(
                query=query,
                answer="".join(pieces),
                contexts=retrieved,
                prompt=prompt,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=(time.perf_counter() - start_time) * 1000.0,
                metadata={
                    "retrieved_count": len(retrieved),
                    "k": k,
                    "candidate_pool": candidate_pool,
                    "streamed": True,
                },
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize pipeline configuration and index state to dictionary."""
        return {
            "bm25_index": self.bm25_index.to_dict(),
            "vector_index": self.vector_index.to_dict(),
            "vector_backend": self.vector_backend,
            **(
                {"extras": {n: i.to_dict() for n, i in self.extra_indexes.items()}}
                if self.extra_indexes
                else {}
            ),
            **({"extra_weights": dict(self.extra_weights)} if self.extra_indexes else {}),
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
        tracer: Tracer | None = None,
        guardrails: Guardrails | None = None,
        stream_generate_fn: StreamGenerateFn | None = None,
        extra_indexes: dict[str, Any] | None = None,
        contextualizer: ContextFn | None = None,
        router: QueryRouter | None = None,
    ) -> RagPipeline:
        """Reconstruct pipeline from a serialized state dictionary.

        ``query_expansion_fn`` and ``tracer`` are not serialized (both are
        arbitrary live objects, like ``embed_fn``/``generate_fn``) -- pass
        them again here to re-enable multi-query retrieval or tracing on a
        loaded pipeline.
        """
        embed = embed_fn or hashing_embed
        backend = data.get("vector_backend", "brute_force")
        if backend not in VECTOR_BACKENDS:
            raise ValueError(f"Unknown vector_backend {backend!r} in serialized pipeline")
        pipeline = cls(
            chunker=chunker,
            embed_fn=embed,
            reranker=reranker,
            generate_fn=generate_fn,
            k_rrf=data.get("k_rrf", 60),
            weight_bm25=data.get("weight_bm25", 1.0),
            weight_vector=data.get("weight_vector", 1.0),
            vector_backend=backend,
            query_expansion_fn=query_expansion_fn,
            tracer=tracer,
            guardrails=guardrails,
            stream_generate_fn=stream_generate_fn,
            extra_indexes=extra_indexes,
            extra_weights=data.get("extra_weights"),
            contextualizer=contextualizer,
            router=router,
        )
        for name, state in data.get("extras", {}).items():
            if name not in pipeline.extra_indexes:
                raise ValueError(
                    f"Serialized extra index {name!r} needs a configured instance: pass "
                    f"extra_indexes={{{name!r}: <index>}} to load() / from_dict()"
                )
            pipeline.extra_indexes[name].load_dict(state)
        if "bm25_index" in data:
            pipeline.bm25_index = BM25Index.from_dict(data["bm25_index"])
        if "vector_index" in data:
            backend_cls: Any = {
                "hnsw": ApproxVectorIndex,
                "numpy": NumpyVectorIndex,
                "brute_force": VectorIndex,
                "int8": QuantizedVectorIndex,
                "binary": QuantizedVectorIndex,
            }[backend]
            pipeline.vector_index = backend_cls.from_dict(data["vector_index"], embed_fn=embed)

        pipeline._build_retriever()
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
        tracer: Tracer | None = None,
        guardrails: Guardrails | None = None,
        stream_generate_fn: StreamGenerateFn | None = None,
        extra_indexes: dict[str, Any] | None = None,
        contextualizer: ContextFn | None = None,
        router: QueryRouter | None = None,
    ) -> RagPipeline:
        """Load a pipeline index state from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(
            data,
            chunker=chunker,
            embed_fn=embed_fn,
            reranker=reranker,
            tracer=tracer,
            generate_fn=generate_fn,
            query_expansion_fn=query_expansion_fn,
            guardrails=guardrails,
            stream_generate_fn=stream_generate_fn,
            extra_indexes=extra_indexes,
            contextualizer=contextualizer,
            router=router,
        )
