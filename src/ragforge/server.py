"""Minimal, dependency-free HTTP serving for a pre-built ``RagPipeline``.

Loading a pipeline once and answering queries in-process (as shown
throughout the README) works for scripts and notebooks, but a real
deployment usually puts RAG behind an HTTP endpoint other services call.
Reaching for a full web framework (FastAPI, Flask) is the normal choice and
a perfectly fine one -- but ragforge's whole design goal is a zero
required-dependency core (see ``docs/architecture.md``), so this module
implements a small JSON API entirely on ``http.server`` from the standard
library, matching the "Zero-Downtime Index Persistence" pattern: build and
save an index offline, then serve queries against it without recomputing
anything at startup.

Endpoints:
    GET  /health   -> {"status": "ok", "document_count": N, "chunk_count": M}
    POST /query    -> body {"query": str, "k": int=5, "candidate_pool": int=20}
                       response: the same JSON shape as ``RagAnswer.to_dict()``

This is intentionally not a general-purpose web framework: no routing DSL,
no middleware, no async request handling (``ThreadingHTTPServer`` handles
concurrency by running each request in its own thread instead). For
anything beyond serving a single pipeline's query endpoint behind a load
balancer, wrap ``RagPipeline`` in FastAPI/Flask directly -- the pipeline
itself has no server-framework dependency either way.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ragforge.pipeline import RagPipeline

logger = logging.getLogger("ragforge.server")


def _make_handler(pipeline: RagPipeline) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ragforge/1.0"

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "document_count": pipeline.document_count,
                        "chunk_count": pipeline.chunk_count,
                    },
                )
            else:
                self._send_json(404, {"error": f"Unknown path: {self.path}"})

        def do_POST(self) -> None:
            if self.path != "/query":
                self._send_json(404, {"error": f"Unknown path: {self.path}"})
                return

            content_length = int(self.headers.get("Content-Length") or 0)
            raw_body = self.rfile.read(content_length) if content_length else b"{}"
            try:
                payload = json.loads(raw_body or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Request body must be valid JSON"})
                return

            query = payload.get("query")
            if not isinstance(query, str) or not query.strip():
                self._send_json(400, {"error": "'query' must be a non-empty string"})
                return

            k = int(payload.get("k", 5))
            candidate_pool = int(payload.get("candidate_pool", 20))

            try:
                answer = pipeline.answer(query, k=k, candidate_pool=candidate_pool)
            except Exception as exc:
                logger.exception("Error answering query %r", query)
                self._send_json(500, {"error": str(exc)})
                return

            self._send_json(200, answer.to_dict())

        def log_message(self, format_str: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), format_str % args)

    return Handler


def serve(pipeline: RagPipeline, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Build and bind a threaded HTTP server exposing ``pipeline``.

    Returns the (already bound, not yet running) server so callers control
    its lifecycle, e.g. in tests:

        server = serve(pipeline, port=0)  # port=0: let the OS pick a free port
        threading.Thread(target=server.serve_forever, daemon=True).start()
        ...
        server.shutdown()
        server.server_close()

    For the common "run until Ctrl+C" case, use
    :func:`serve_forever_blocking` instead (used by ``ragforge serve``).
    """
    handler_cls = _make_handler(pipeline)
    return ThreadingHTTPServer((host, port), handler_cls)


def serve_forever_blocking(
    pipeline: RagPipeline, host: str = "127.0.0.1", port: int = 8000
) -> None:
    """Start serving ``pipeline`` over HTTP and block until interrupted."""
    server = serve(pipeline, host=host, port=port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
