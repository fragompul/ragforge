"""Async facade and the pure-ASGI application, driven without any ASGI server."""

from __future__ import annotations

import asyncio
import json

import pytest

from ragforge import RagPipeline
from ragforge.aio import AsyncRagPipeline
from ragforge.asgi import MAX_BODY_BYTES, RagASGIApp, create_app
from ragforge.chunking import RecursiveCharacterChunker


def _pipeline(**kw) -> RagPipeline:
    p = RagPipeline(chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=0), **kw)
    p.ingest("eiffel", "The Eiffel Tower is in Paris and was completed in 1889.")
    p.ingest("python", "Python was created by Guido van Rossum in 1991.")
    return p


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------- AsyncRagPipeline


def test_aanswer_matches_sync_answer_and_runs_in_threads():
    pipeline = _pipeline()
    apipe = AsyncRagPipeline(pipeline)
    sync = pipeline.answer("When was the Eiffel Tower completed?", k=2)
    got = run(apipe.aanswer("When was the Eiffel Tower completed?", k=2))
    assert got.answer == sync.answer and [c.doc_id for c in got.contexts] == [
        c.doc_id for c in sync.contexts
    ]
    assert got.total_latency_ms >= got.retrieval_latency_ms >= 0


def test_generation_overlaps_across_concurrent_requests():
    in_flight = peak = 0

    async def slow_generate(query, contexts):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return f"answer to {query}"

    apipe = AsyncRagPipeline(_pipeline(), agenerate_fn=slow_generate)

    async def main():
        return await asyncio.gather(*(apipe.aanswer(f"question {i}") for i in range(8)))

    results = run(main())
    assert [r.answer for r in results] == [f"answer to question {i}" for i in range(8)]
    assert peak >= 4  # LLM calls overlapped instead of running one after another


def test_generation_concurrency_is_bounded():
    peak = in_flight = 0

    async def gen(query, contexts):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return "x"

    apipe = AsyncRagPipeline(_pipeline(), agenerate_fn=gen, max_concurrent_generations=2)
    run(_gather(apipe, 10))
    assert peak == 2
    with pytest.raises(ValueError):
        AsyncRagPipeline(_pipeline(), max_concurrent_generations=0)


async def _gather(apipe, n):
    await asyncio.gather(*(apipe.aanswer("tower") for _ in range(n)))


def test_concurrent_retrieval_keeps_the_tracer_consistent():
    from ragforge.telemetry import InMemoryExporter, Tracer

    exporter = InMemoryExporter()
    apipe = AsyncRagPipeline(_pipeline(tracer=Tracer(exporters=[exporter])))

    async def main():
        await asyncio.gather(*(apipe.aretrieve("Eiffel Tower") for _ in range(20)))

    run(main())
    roots = [s for s in exporter.spans if s.name == "retrieve"]
    assert len(roots) == 20 and all(r.parent_id is None for r in roots)


def test_aingest_and_adelete():
    apipe = AsyncRagPipeline(_pipeline())

    async def main():
        await apipe.aingest("louvre", "The Louvre is a museum in Paris.")
        found = [c.doc_id for c in await apipe.aretrieve("Louvre museum", k=1)]
        removed = await apipe.adelete("louvre")
        return found, removed

    assert run(main()) == (["louvre"], 1)


@pytest.mark.parametrize("variant", ["default", "agenerate", "astream", "sync_stream"])
def test_aanswer_stream_variants(variant):
    async def agen(q, c):
        return "alpha beta gamma"

    async def astream(q, c):
        for piece in ["al", "pha ", "beta"]:
            yield piece

    kwargs = {
        "default": {},
        "agenerate": {"agenerate_fn": agen},
        "astream": {"astream_fn": astream},
        "sync_stream": {},
    }[variant]
    pipeline = _pipeline(
        generate_fn=lambda q, c: "alpha beta gamma",
        **(
            {"stream_generate_fn": lambda q, c: iter(["al", "pha ", "beta"])}
            if variant == "sync_stream"
            else {}
        ),
    )
    apipe = AsyncRagPipeline(pipeline, **kwargs)

    async def main():
        return [e async for e in apipe.aanswer_stream("tower", k=1)]

    events = run(main())
    assert events[0]["type"] == "contexts" and events[-1]["type"] == "done"
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert text == events[-1]["answer"].answer
    assert text in ("alpha beta gamma", "alpha beta")


# --------------------------------------------------------------------------- ASGI app


class Client:
    """Minimal ASGI test client: sends one request, returns (status, headers, body chunks)."""

    def __init__(self, app) -> None:
        self.app = app

    async def request(self, method, path, body=b"", headers=None):
        scope = {"type": "http", "method": method, "path": path, "headers": headers or []}
        sent = [{"type": "http.request", "body": body, "more_body": False}]
        out = []

        async def receive():
            return sent.pop(0) if sent else {"type": "http.disconnect"}

        async def send(message):
            out.append(message)

        await self.app(scope, receive, send)
        start = out[0]
        return start["status"], dict(start["headers"]), [m["body"] for m in out[1:]]


def call(app, method, path, payload=None, raw=None, headers=None):
    body = (
        raw if raw is not None else (json.dumps(payload).encode() if payload is not None else b"")
    )
    return run(Client(app).request(method, path, body, headers))


def _app(**kw):
    return create_app(_pipeline(generate_fn=lambda q, c: "The tower was completed in 1889."), **kw)


def test_health_and_query_contract():
    app = _app()
    status, _, body = call(app, "GET", "/health")
    assert status == 200 and json.loads(body[0]) == {
        "status": "ok",
        "document_count": 2,
        "chunk_count": 2,
    }
    status, headers, body = call(app, "POST", "/query", {"query": "Eiffel Tower", "k": 1})
    data = json.loads(body[0])
    assert status == 200 and headers[b"content-type"] == b"application/json"
    assert data["contexts"][0]["doc_id"] == "eiffel" and data["answer"].startswith("The tower")


def test_streaming_endpoint_emits_sse_events():
    status, headers, chunks = call(
        _app(), "POST", "/query/stream", {"query": "Eiffel Tower", "k": 1}
    )
    assert status == 200 and headers[b"content-type"] == b"text/event-stream"
    assert chunks[-1] == b""  # stream terminator
    events = []
    for block in b"".join(chunks).decode().strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    names = [n for n, _ in events]
    assert names[0] == "contexts" and names[-1] == "done" and "token" in names
    assert "".join(d["text"] for n, d in events if n == "token") == events[-1][1]["answer"]


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ({}, "query"),
        ({"query": "  "}, "query"),
        ({"query": "x", "k": "many"}, "integers"),
        ({"query": "x", "k": 0}, "must be in"),
        ({"query": "x", "k": 10_000}, "must be in"),
        ({"query": "x", "candidate_pool": 0}, "must be in"),
    ],
)
def test_validation_errors_return_400(payload, needle):
    status, _, body = call(_app(), "POST", "/query", payload)
    assert status == 400 and needle in json.loads(body[0])["error"]


def test_malformed_and_oversized_bodies():
    app = _app()
    assert call(app, "POST", "/query", raw=b"{nope")[0] == 400
    assert call(app, "POST", "/query", raw=b"[1,2]")[0] == 400
    assert call(app, "POST", "/query", raw=b"\xff\xfe")[0] == 400
    assert call(app, "POST", "/query", raw=b"x" * (MAX_BODY_BYTES + 1))[0] == 413


def test_routing_errors_and_methods():
    app = _app()
    assert call(app, "GET", "/nope")[0] == 404
    assert call(app, "GET", "/query")[0] == 405
    assert call(app, "POST", "/health")[0] == 405


def test_bearer_token_is_enforced_in_constant_time_comparison():
    app = _app(api_key="s3cret")
    assert call(app, "GET", "/health")[0] == 401
    assert call(app, "GET", "/health", headers=[(b"authorization", b"Bearer wrong")])[0] == 401
    assert call(app, "GET", "/health", headers=[(b"Authorization", b"Bearer s3cret")])[0] == 200


def test_errors_inside_generation_become_500_or_sse_error_events():
    def boom(q, c):
        raise RuntimeError("model unavailable")

    app = create_app(_pipeline(generate_fn=boom))
    status, _, body = call(app, "POST", "/query", {"query": "x"})
    assert status == 500 and "model unavailable" in json.loads(body[0])["error"]
    status, _, chunks = call(app, "POST", "/query/stream", {"query": "x"})
    assert status == 200 and b"event: error" in b"".join(chunks)


def test_lifespan_protocol():
    app = _app()

    async def main():
        incoming = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
        out = []

        async def receive():
            return incoming.pop(0)

        async def send(m):
            out.append(m["type"])

        await app({"type": "lifespan"}, receive, send)
        return out

    assert run(main()) == ["lifespan.startup.complete", "lifespan.shutdown.complete"]


def test_non_http_scopes_are_ignored():
    async def main():
        await _app()({"type": "websocket"}, None, None)

    run(main())


def test_env_configured_entry_point(tmp_path, monkeypatch):
    import ragforge.asgi as asgi_module

    path = tmp_path / "idx.json"
    _pipeline().save(path)
    monkeypatch.setattr(asgi_module, "_ENV_APP", None)
    monkeypatch.setenv("RAGFORGE_INDEX", str(path))
    monkeypatch.setenv("RAGFORGE_API_KEY", "k")
    monkeypatch.setenv("RAGFORGE_GUARDRAILS", "flag")

    assert call(asgi_module.app, "GET", "/health")[0] == 401
    status, _, body = call(
        asgi_module.app, "GET", "/health", headers=[(b"authorization", b"Bearer k")]
    )
    assert status == 200 and json.loads(body[0])["chunk_count"] == 2
    assert asgi_module._ENV_APP.apipeline.pipeline.guardrails.action == "flag"

    monkeypatch.setattr(asgi_module, "_ENV_APP", None)
    monkeypatch.delenv("RAGFORGE_INDEX")
    with pytest.raises(RuntimeError, match="RAGFORGE_INDEX"):
        call(asgi_module.app, "GET", "/health")


def test_accepts_an_async_pipeline_directly():
    apipe = AsyncRagPipeline(_pipeline())
    assert RagASGIApp(apipe).apipeline is apipe
