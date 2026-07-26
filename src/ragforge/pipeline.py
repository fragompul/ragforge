"""Ties chunking, hybrid retrieval, reranking, and generation into one
end-to-end pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ragforge.chunking import Chunk
from ragforge.embeddings import hashing_embed
from ragforge.index import BM25Index, HybridRetriever, VectorIndex
from ragforge.reranking import NoopReranker, Reranker


class ChunkerProtocol(Protocol):
    def chunk(self, text: str, doc_id: str) -> list[Chunk]: ...


@dataclass
class RetrievedChunk:
    text: str
    score: float
    doc_id: str


@dataclass
class RagAnswer:
    query: str
    answer: str
    contexts: list[RetrievedChunk]


def _default_generate(query: str, contexts: list[str]) -> str:
    """Extractive fallback generator: returns the top context verbatim.

    Real deployments pass a real LLM call as ``generate_fn``; this default
    keeps the pipeline runnable end-to-end offline for tests and examples.
    """

    return contexts[0] if contexts else "I don't have enough information to answer that."


class RagPipeline:
    def __init__(
        self,
        chunker: ChunkerProtocol,
        embed_fn: Callable[[str], list[float]] = hashing_embed,
        reranker: Reranker | None = None,
        generate_fn: Callable[[str, list[str]], str] | None = None,
    ) -> None:
        self.chunker = chunker
        self.bm25_index = BM25Index()
        self.vector_index = VectorIndex(embed_fn=embed_fn)
        self.retriever = HybridRetriever(self.bm25_index, self.vector_index)
        self.reranker = reranker or NoopReranker()
        self.generate_fn = generate_fn or _default_generate

    def ingest(self, doc_id: str, text: str) -> list[Chunk]:
        chunks = self.chunker.chunk(text, doc_id)
        self.bm25_index.add(chunks)
        self.vector_index.add(chunks)
        return chunks

    def retrieve(self, query: str, k: int = 5, candidate_pool: int = 20) -> list[RetrievedChunk]:
        candidates = self.retriever.search(query, k=candidate_pool, candidate_pool=candidate_pool)
        reranked = self.reranker.rerank(query, candidates, k=k)
        return [
            RetrievedChunk(text=sc.chunk.text, score=sc.score, doc_id=sc.chunk.doc_id)
            for sc in reranked
        ]

    def answer(self, query: str, k: int = 5) -> RagAnswer:
        retrieved = self.retrieve(query, k=k)
        answer_text = self.generate_fn(query, [r.text for r in retrieved])
        return RagAnswer(query=query, answer=answer_text, contexts=retrieved)
