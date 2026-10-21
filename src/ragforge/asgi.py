"""A dependency-free ASGI 3 application exposing a pipeline over HTTP.

The stdlib server in :mod:`ragforge.server` is great for a demo; production deployments
run an async server (uvicorn, hypercorn, granian) behind a load balancer. This module is a
**pure ASGI callable** -- no framework -- so ``ragforge`` stays dependency-free while you
choose the server::

    RAGFORGE_INDEX=kb.json uvicorn ragforge.asgi:app            # env-configured
    app = create_app(pipeline)                                   # programmatic

Endpoints (same contract as ``ragforge serve``):

    GET  /health        -> {"status": "ok", "document_count": N, "chunk_count": M}
    POST /query         -> RagAnswer JSON        (body: {"query", "k", "candidate_pool"})
    POST /query/stream  -> Server-Sent Events: contexts, token..., done | error

Generation overlaps across requests (see :mod:`ragforge.aio`); request bodies are capped at
1 MB and ``k`` at 100. Authentication, TLS and rate limiting belong to the server/proxy in
front -- pass ``api_key`` to additionally require ``Authorization: Bearer <key>``.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from ragforge.aio import AsyncRagPipeline
from ragforge.pipeline import RagPipeline

logger = logging.getLogger("ragforge.asgi")

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

MAX_BODY_BYTES = 1_000_000
MAX_K = 100
MAX_CANDIDATE_POOL = 1000


class RagASGIApp:
    """ASGI application serving an :class:`AsyncRagPipeline`."""

    def __init__(
        self,
        pipeline: RagPipeline | AsyncRagPipeline,
        api_key: str | None = None,
    ) -> None:
        self.apipeline = (
            pipeline if isinstance(pipeline, AsyncRagPipeline) else AsyncRagPipeline(pipeline)
        )
        self.api_key = api_key

    # -- helpers -------------------------------------------------------------------

    @staticmethod
    async def _json(send: Send, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    async def _read_body(receive: Receive) -> bytes | None:
        """Read the request body; ``None`` if it exceeds the limit."""
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return b""
            body = message.get("body", b"")
            size += len(body)
            if size > MAX_BODY_BYTES:
                return None
            chunks.append(body)
            if not message.get("more_body", False):
                return b"".join(chunks)

    def _authorized(self, scope: Scope) -> bool:
        if self.api_key is None:
            return True
        for name, value in scope.get("headers", []):
            if name.lower() == b"authorization":
                expected = f"Bearer {self.api_key}".encode()
                return hmac.compare_digest(value, expected)
        return False

    # -- ASGI entry point -----------------------------------------------------------

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return

        method, path = scope["method"], scope["path"]
        if not self._authorized(scope):
            await self._json(send, 401, {"error": "Missing or invalid bearer token"})
            return
        if method == "GET" and path == "/health":
            pipeline = self.apipeline.pipeline
            await self._json(
                send,
                200,
                {
                    "status": "ok",
                    "document_count": pipeline.document_count,
                    "chunk_count": pipeline.chunk_count,
                },
            )
            return
        if method == "POST" and path in ("/query", "/query/stream"):
            await self._query(path, receive, send)
            return
        status = 405 if path in ("/health", "/query", "/query/stream") else 404
        await self._json(send, status, {"error": f"No route for {method} {path}"})

    async def _query(self, path: str, receive: Receive, send: Send) -> None:
        raw = await self._read_body(receive)
        if raw is None:
            await self._json(send, 413, {"error": f"Request body exceeds {MAX_BODY_BYTES} bytes"})
            return
        try:
            payload = json.loads(raw or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            await self._json(send, 400, {"error": "Request body must be valid JSON"})
            return
        if not isinstance(payload, dict):
            await self._json(send, 400, {"error": "Request body must be a JSON object"})
            return
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            await self._json(send, 400, {"error": "'query' must be a non-empty string"})
            return
        try:
            k = int(payload.get("k", 5))
            pool = int(payload.get("candidate_pool", 20))
        except (TypeError, ValueError):
            await self._json(send, 400, {"error": "'k' and 'candidate_pool' must be integers"})
            return
        if not 1 <= k <= MAX_K or not 1 <= pool <= MAX_CANDIDATE_POOL:
            await self._json(
                send,
                400,
                {
                    "error": f"'k' must be in [1, {MAX_K}], "
                    f"'candidate_pool' in [1, {MAX_CANDIDATE_POOL}]"
                },
            )
            return

        if path == "/query":
            try:
                answer = await self.apipeline.aanswer(query, k=k, candidate_pool=pool)
            except Exception as exc:
                logger.exception("Error answering %r", query)
                await self._json(send, 500, {"error": str(exc)})
                return
            await self._json(send, 200, answer.to_dict())
            return

        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream"),
                    (b"cache-control", b"no-cache"),
                ],
            }
        )

        async def emit(event: str, data: dict[str, Any]) -> None:
            chunk = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
            await send({"type": "http.response.body", "body": chunk, "more_body": True})

        try:
            async for event in self.apipeline.aanswer_stream(query, k=k, candidate_pool=pool):
                if event["type"] == "done":
                    await emit("done", event["answer"].to_dict())
                else:
                    await emit(event["type"], event)
        except Exception as exc:
            logger.exception("Error streaming %r", query)
            await emit("error", {"error": str(exc)})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


def create_app(pipeline: RagPipeline | AsyncRagPipeline, api_key: str | None = None) -> RagASGIApp:
    """Build the ASGI application for ``pipeline``."""
    return RagASGIApp(pipeline, api_key=api_key)


_ENV_APP: RagASGIApp | None = None


async def app(scope: Scope, receive: Receive, send: Send) -> None:
    """Env-configured ASGI entry point for ``uvicorn ragforge.asgi:app``.

    Reads ``RAGFORGE_INDEX`` (path to a saved index; required), ``RAGFORGE_API_KEY``
    (optional bearer token) and ``RAGFORGE_GUARDRAILS`` (``flag|sanitize|drop``).
    """
    global _ENV_APP
    if _ENV_APP is None:
        index = os.environ.get("RAGFORGE_INDEX")
        if not index:
            raise RuntimeError("Set RAGFORGE_INDEX to the path of a saved ragforge index")
        guardrails = None
        mode = os.environ.get("RAGFORGE_GUARDRAILS")
        if mode:
            from ragforge.guardrails import Guardrails

            guardrails = Guardrails(action=mode)  # type: ignore[arg-type]
        _ENV_APP = RagASGIApp(
            RagPipeline.load(index, guardrails=guardrails),
            api_key=os.environ.get("RAGFORGE_API_KEY"),
        )
    await _ENV_APP(scope, receive, send)
