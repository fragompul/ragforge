"""Tests for optional real-embedding-provider adapters.

None of these tests require the actual third-party SDKs to be installed:
each provider's client library is stubbed via ``sys.modules`` injection (or,
for Ollama, by patching ``urllib.request.urlopen``), keeping this test
module runnable in the same zero-dependency CI environment as the rest of
the suite.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from ragforge.embeddings_providers import (
    cached_embed_fn,
    cohere_embed_fn,
    ollama_embed_fn,
    openai_embed_fn,
    sentence_transformers_embed_fn,
)


def test_openai_embed_fn_calls_client_with_expected_arguments():
    calls: list[tuple[list[str], str]] = []

    class FakeResponse:
        data = [types.SimpleNamespace(embedding=[0.1, 0.2, 0.3])]

    class FakeClient:
        class embeddings:  # noqa: N801 - mirrors the openai SDK's attribute shape
            @staticmethod
            def create(input: list[str], model: str) -> FakeResponse:
                calls.append((input, model))
                return FakeResponse()

    embed = openai_embed_fn(model="text-embedding-3-small", client=FakeClient())
    vector = embed("hello world")

    assert vector == [0.1, 0.2, 0.3]
    assert calls == [(["hello world"], "text-embedding-3-small")]


def test_openai_embed_fn_raises_helpful_error_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match="pip install ragforge\\[openai\\]"):
        openai_embed_fn()


def test_cohere_embed_fn_calls_client_with_expected_arguments():
    calls: list[dict[str, object]] = []

    class FakeResponse:
        embeddings = [[0.4, 0.5]]

    class FakeClient:
        def embed(self, texts: list[str], model: str, input_type: str) -> FakeResponse:
            calls.append({"texts": texts, "model": model, "input_type": input_type})
            return FakeResponse()

    embed = cohere_embed_fn(client=FakeClient())
    vector = embed("hi there")

    assert vector == [0.4, 0.5]
    assert calls == [
        {"texts": ["hi there"], "model": "embed-english-v3.0", "input_type": "search_document"}
    ]


def test_cohere_embed_fn_raises_helpful_error_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "cohere", None)
    with pytest.raises(ImportError, match="pip install ragforge\\[cohere\\]"):
        cohere_embed_fn()


def test_sentence_transformers_embed_fn_calls_model_and_normalizes():
    class FakeArray:
        def tolist(self) -> list[float]:
            return [0.6, 0.7]

    class FakeModel:
        def encode(self, text: str, normalize_embeddings: bool) -> FakeArray:
            assert text == "some text"
            assert normalize_embeddings is True
            return FakeArray()

    embed = sentence_transformers_embed_fn(model=FakeModel())
    assert embed("some text") == [0.6, 0.7]


def test_sentence_transformers_embed_fn_raises_helpful_error_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="pip install ragforge\\[local\\]"):
        sentence_transformers_embed_fn()


def test_ollama_embed_fn_posts_to_local_server(monkeypatch):
    captured: dict[str, object] = {}

    class FakeHTTPResponse:
        def __enter__(self) -> FakeHTTPResponse:
            return self

        def __exit__(self, *exc_info: object) -> bool:
            return False

        def read(self) -> bytes:
            return json.dumps({"embedding": [0.8, 0.9]}).encode("utf-8")

    def fake_urlopen(request: object, timeout: int = 30) -> FakeHTTPResponse:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["data"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return FakeHTTPResponse()

    import ragforge.embeddings_providers as providers_module

    monkeypatch.setattr(providers_module.urllib.request, "urlopen", fake_urlopen)

    embed = ollama_embed_fn(model="nomic-embed-text", base_url="http://localhost:11434/")
    vector = embed("hello")

    assert vector == [0.8, 0.9]
    assert captured["url"] == "http://localhost:11434/api/embeddings"
    assert captured["data"] == {"model": "nomic-embed-text", "prompt": "hello"}


def test_cached_embed_fn_avoids_repeat_calls_for_identical_text():
    calls: list[str] = []

    def slow_embed(text: str) -> list[float]:
        calls.append(text)
        return [float(len(text))]

    cached = cached_embed_fn(slow_embed, max_size=10)

    assert cached("hello") == [5.0]
    assert cached("hello") == [5.0]
    assert cached("hi") == [2.0]

    assert calls == ["hello", "hi"]


def test_cached_embed_fn_returns_independent_list_copies():
    def embed(text: str) -> list[float]:
        return [1.0, 2.0]

    cached = cached_embed_fn(embed)
    first = cached("x")
    first.append(999.0)

    assert cached("x") == [1.0, 2.0]
