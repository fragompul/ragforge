"""Integration tests for the stdlib-only HTTP server.

These spin up a real ``ThreadingHTTPServer`` on an OS-assigned free port
(``port=0``) and issue real HTTP requests via ``urllib`` -- no mocking of
``http.server`` -- since the point of this module is that it behaves like a
real, if minimal, HTTP API.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from ragforge.pipeline import RagPipeline
from ragforge.server import serve


@pytest.fixture
def base_url() -> Iterator[str]:
    pipeline = RagPipeline()
    pipeline.ingest("doc1", "The Eiffel Tower is in Paris, completed in 1889.")

    server = serve(pipeline, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _post(url: str, payload: dict) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_health_endpoint_reports_index_stats(base_url: str):
    with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
        data = json.loads(response.read())

    assert data == {"status": "ok", "document_count": 1, "chunk_count": 1}


def test_query_endpoint_returns_rag_answer_shape(base_url: str):
    status, data = _post(f"{base_url}/query", {"query": "Where is the Eiffel Tower?", "k": 1})

    assert status == 200
    assert data["query"] == "Where is the Eiffel Tower?"
    assert len(data["contexts"]) == 1
    assert data["contexts"][0]["doc_id"] == "doc1"
    assert "total_latency_ms" in data


def test_query_endpoint_rejects_missing_query_field(base_url: str):
    status, data = _post(f"{base_url}/query", {})
    assert status == 400
    assert "query" in data["error"]


def test_query_endpoint_rejects_blank_query(base_url: str):
    status, data = _post(f"{base_url}/query", {"query": "   "})
    assert status == 400


def test_query_endpoint_rejects_invalid_json_body(base_url: str):
    request = urllib.request.Request(f"{base_url}/query", data=b"not valid json", method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(request, timeout=5)
    assert exc_info.value.code == 400


def test_query_endpoint_defaults_k_when_omitted(base_url: str):
    status, data = _post(f"{base_url}/query", {"query": "Eiffel Tower"})
    assert status == 200
    assert len(data["contexts"]) >= 1


def test_unknown_get_path_returns_404(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base_url}/does-not-exist", timeout=5)
    assert exc_info.value.code == 404


def test_unknown_post_path_returns_404(base_url: str):
    status, _ = _post(f"{base_url}/does-not-exist", {"query": "x"})
    assert status == 404
