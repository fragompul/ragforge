"""Server hardening, SSE streaming, and agentic mode."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from ragforge.pipeline import RagPipeline
from ragforge.server import MAX_BODY_BYTES, serve


@pytest.fixture
def base_url() -> Iterator[str]:
    pipeline = RagPipeline(generate_fn=lambda q, ctx: "The tower was completed in 1889.")
    pipeline.ingest("doc1", "The Eiffel Tower is in Paris, completed in 1889.")
    server = serve(pipeline, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(url: str, body: bytes, headers: dict | None = None):
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", **(headers or {})}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def _parse_sse(raw: bytes) -> list[tuple[str, dict]]:
    events = []
    for block in raw.decode().strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_stream_endpoint_emits_contexts_tokens_done(base_url):
    status, headers, raw = _request(
        f"{base_url}/query/stream", json.dumps({"query": "Eiffel Tower", "k": 1}).encode()
    )

    assert status == 200 and headers["Content-Type"] == "text/event-stream"
    events = _parse_sse(raw)
    names = [name for name, _ in events]
    assert names[0] == "contexts" and names[-1] == "done"
    assert set(names[1:-1]) == {"token"} and len(names) > 4
    assert "".join(d["text"] for n, d in events if n == "token") == events[-1][1]["answer"]


def test_stream_endpoint_validates_like_query(base_url):
    status, _, raw = _request(f"{base_url}/query/stream", b'{"k": 1}')
    assert status == 400 and "query" in json.loads(raw)["error"]


def test_agentic_mode_returns_step_trail(base_url):
    status, _, raw = _request(
        f"{base_url}/query", json.dumps({"query": "Eiffel Tower", "agentic": True}).encode()
    )
    data = json.loads(raw)
    assert status == 200 and data["abstained"] is False
    assert [s["kind"] for s in data["steps"]][:2] == ["retrieve", "grade"]
    assert data["answer"]["answer"].startswith("The tower")


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "x", "k": "many"},
        {"query": "x", "k": 0},
        {"query": "x", "k": 10_000},
        {"query": "x", "candidate_pool": 0},
        {"query": "x", "k": None},
    ],
)
def test_invalid_numeric_parameters_return_400_not_500(base_url, payload):
    status, _, raw = _request(f"{base_url}/query", json.dumps(payload).encode())
    assert status == 400 and "error" in json.loads(raw)


def test_non_object_json_body_is_rejected(base_url):
    status, _, raw = _request(f"{base_url}/query", b"[1, 2, 3]")
    assert status == 400 and "object" in json.loads(raw)["error"]


def test_oversized_body_is_rejected_with_413(base_url):
    req = urllib.request.Request(
        f"{base_url}/query",
        data=b"{}",
        headers={"Content-Type": "application/json", "Content-Length": str(MAX_BODY_BYTES + 1)},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 413
