"""Async API: serve many concurrent requests without blocking the event loop on the LLM.

A RAG request is two very different kinds of work: **retrieval** (CPU-bound, microseconds
to milliseconds, touches shared mutable state -- indexes, the tracer's span stack) and
**generation** (network-bound, hundreds of milliseconds to seconds, embarrassingly
parallel). :class:`AsyncRagPipeline` exploits that split:

* retrieval runs in a worker thread (``asyncio.to_thread``) behind a lock, so the event
  loop is never blocked and the non-thread-safe tracer stays consistent;
* generation is awaited *outside* the lock, so dozens of LLM calls overlap -- supply an
  ``agenerate_fn`` (a coroutine function) such as one built on the async client of your
  provider, or leave it ``None`` to run the pipeline's synchronous ``generate_fn`` in a thread.

It wraps an ordinary :class:`~ragforge.pipeline.RagPipeline`; no logic is duplicated.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, TypeVar

from ragforge.chunking import Chunk
from ragforge.index import FilterFn
from ragforge.pipeline import RagAnswer, RagPipeline, RetrievedChunk

AsyncGenerateFn = Callable[[str, list[str]], Awaitable[str]]
AsyncStreamFn = Callable[[str, list[str]], AsyncIterator[str]]
T = TypeVar("T")
_SENTINEL: Any = object()


async def _aiter_sync(iterator: Iterator[T]) -> AsyncIterator[T]:
    """Advance a blocking iterator in a worker thread, one item at a time."""
    while True:
        item = await asyncio.to_thread(next, iterator, _SENTINEL)
        if item is _SENTINEL:
            return
        yield item


class AsyncRagPipeline:
    """Awaitable facade over a :class:`RagPipeline`.

    Args:
        pipeline: The wrapped pipeline (all its configuration -- guardrails, reranker,
            router, tracer -- applies unchanged).
        agenerate_fn: ``async (query, contexts) -> str``; defaults to running
            ``pipeline.generate_fn`` in a thread.
        astream_fn: ``async generator (query, contexts) -> tokens``; defaults to the
            pipeline's synchronous streaming (or word-by-word re-emission).
        max_concurrent_generations: Upper bound on in-flight generation calls.
    """

    def __init__(
        self,
        pipeline: RagPipeline,
        agenerate_fn: AsyncGenerateFn | None = None,
        astream_fn: AsyncStreamFn | None = None,
        max_concurrent_generations: int = 32,
    ) -> None:
        if max_concurrent_generations < 1:
            raise ValueError("max_concurrent_generations must be >= 1")
        self.pipeline = pipeline
        self.agenerate_fn = agenerate_fn
        self.astream_fn = astream_fn
        self._index_lock = asyncio.Lock()
        self._generation_slots = asyncio.Semaphore(max_concurrent_generations)

    async def _locked(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        async with self._index_lock:
            return await asyncio.to_thread(fn, *args, **kwargs)

    # -- corpus --------------------------------------------------------------------

    async def aingest(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> list[Chunk]:
        return await self._locked(self.pipeline.ingest, doc_id, text, metadata)

    async def adelete(self, doc_id: str) -> int:
        return await self._locked(self.pipeline.delete, doc_id)

    # -- retrieval & answers -------------------------------------------------------

    async def aretrieve(
        self, query: str, k: int = 5, candidate_pool: int = 20, filter_fn: FilterFn | None = None
    ) -> list[RetrievedChunk]:
        return await self._locked(
            self.pipeline.retrieve, query, k=k, candidate_pool=candidate_pool, filter_fn=filter_fn
        )

    async def _generate(self, query: str, texts: list[str]) -> str:
        async with self._generation_slots:
            if self.agenerate_fn is not None:
                return await self.agenerate_fn(query, texts)
            return await asyncio.to_thread(self.pipeline.generate_fn, query, texts)

    async def aanswer(
        self, query: str, k: int = 5, candidate_pool: int = 20, filter_fn: FilterFn | None = None
    ) -> RagAnswer:
        """Like :meth:`RagPipeline.answer`; generation overlaps with other requests."""
        started = time.perf_counter()
        retrieved = await self.aretrieve(query, k, candidate_pool, filter_fn)
        retrieval_ms = (time.perf_counter() - started) * 1000.0
        texts = [r.text for r in retrieved]
        prompt = self.pipeline.prompt_formatter(query, texts)
        generated = time.perf_counter()
        answer = await self._generate(query, texts)
        return RagAnswer(
            query=query,
            answer=answer,
            contexts=retrieved,
            prompt=prompt,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=(time.perf_counter() - generated) * 1000.0,
            total_latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={"retrieved_count": len(retrieved), "k": k, "candidate_pool": candidate_pool},
        )

    async def aanswer_stream(
        self, query: str, k: int = 5, candidate_pool: int = 20, filter_fn: FilterFn | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Async counterpart of :meth:`RagPipeline.answer_stream` (same event dictionaries)."""
        started = time.perf_counter()
        retrieved = await self.aretrieve(query, k, candidate_pool, filter_fn)
        retrieval_ms = (time.perf_counter() - started) * 1000.0
        yield {"type": "contexts", "contexts": [r.to_dict() for r in retrieved]}

        texts = [r.text for r in retrieved]
        prompt = self.pipeline.prompt_formatter(query, texts)
        generated = time.perf_counter()
        pieces: list[str] = []
        async with self._generation_slots:
            source: AsyncIterator[str]
            if self.astream_fn is not None:
                source = self.astream_fn(query, texts)
            elif self.agenerate_fn is not None:
                text = await self.agenerate_fn(query, texts)
                source = _aiter_sync(iter(re.findall(r"\S+\s*", text)))
            elif self.pipeline.stream_generate_fn is not None:
                source = _aiter_sync(self.pipeline.stream_generate_fn(query, texts))
            else:
                text = await asyncio.to_thread(self.pipeline.generate_fn, query, texts)
                source = _aiter_sync(iter(re.findall(r"\S+\s*", text)))
            async for piece in source:
                pieces.append(piece)
                yield {"type": "token", "text": piece}
        yield {
            "type": "done",
            "answer": RagAnswer(
                query=query,
                answer="".join(pieces),
                contexts=retrieved,
                prompt=prompt,
                retrieval_latency_ms=retrieval_ms,
                generation_latency_ms=(time.perf_counter() - generated) * 1000.0,
                total_latency_ms=(time.perf_counter() - started) * 1000.0,
                metadata={"retrieved_count": len(retrieved), "k": k, "streamed": True},
            ),
        }
