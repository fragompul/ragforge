"""Tests for LLM provider adapters; vendor SDKs are replaced with in-memory fakes."""

from __future__ import annotations

import io
import json
import sys
import types

import pytest

from ragforge import llm_providers as lp


class _Block:
    def __init__(self, text: str, type: str = "text") -> None:
        self.text, self.type = text, type


def test_anthropic_complete_sends_expected_request_and_joins_text_blocks():
    calls: list[dict] = []

    class Messages:
        @staticmethod
        def create(**kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(
                content=[_Block("Hel"), _Block("", "tool_use"), _Block("lo")]
            )

    complete = lp.anthropic_complete_fn(
        model="m", system="be brief", max_tokens=5, client=types.SimpleNamespace(messages=Messages)
    )

    assert complete("hi") == "Hello"
    assert calls == [
        {
            "model": "m",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "hi"}],
            "system": "be brief",
        }
    ]


def test_anthropic_complete_omits_system_when_unset():
    seen: dict = {}

    class Messages:
        @staticmethod
        def create(**kwargs):
            seen.update(kwargs)
            return types.SimpleNamespace(content=[_Block("ok")])

    lp.anthropic_complete_fn(client=types.SimpleNamespace(messages=Messages))("x")
    assert "system" not in seen


def test_anthropic_generate_formats_prompt_with_contexts():
    prompts: list[str] = []

    class Messages:
        @staticmethod
        def create(**kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            return types.SimpleNamespace(content=[_Block("answer")])

    generate = lp.anthropic_generate_fn(client=types.SimpleNamespace(messages=Messages))

    assert generate("Q?", ["ctx one"]) == "answer"
    assert "ctx one" in prompts[0] and "Q?" in prompts[0]


def test_anthropic_stream_yields_text_deltas():
    class Stream:
        text_stream = iter(["a", "b", "c"])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Messages:
        @staticmethod
        def stream(**kwargs):
            assert kwargs["messages"][0]["role"] == "user"
            return Stream()

    stream = lp.anthropic_stream_fn(client=types.SimpleNamespace(messages=Messages))
    assert "".join(stream("Q", ["c"])) == "abc"


def test_openai_complete_and_stream():
    class Completions:
        @staticmethod
        def create(**kwargs):
            if kwargs.get("stream"):
                return iter(
                    [
                        types.SimpleNamespace(
                            choices=[
                                types.SimpleNamespace(delta=types.SimpleNamespace(content="x"))
                            ]
                        ),
                        types.SimpleNamespace(choices=[]),
                        types.SimpleNamespace(
                            choices=[
                                types.SimpleNamespace(delta=types.SimpleNamespace(content=None))
                            ]
                        ),
                        types.SimpleNamespace(
                            choices=[
                                types.SimpleNamespace(delta=types.SimpleNamespace(content="y"))
                            ]
                        ),
                    ]
                )
            msg = types.SimpleNamespace(
                content=None if kwargs["messages"][-1]["content"] == "empty" else "done"
            )
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions))

    assert lp.openai_complete_fn(client=client, system="s")("p") == "done"
    assert lp.openai_complete_fn(client=client)("empty") == ""
    assert lp.openai_generate_fn(client=client)("Q", ["c"]) == "done"
    assert "".join(lp.openai_stream_fn(client=client)("Q", ["c"])) == "xy"


def test_ollama_complete_posts_json(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return io.BytesIO(json.dumps({"response": "pong"}).encode())

    monkeypatch.setattr(lp.urllib.request, "urlopen", fake_urlopen)
    complete = lp.ollama_complete_fn(model="llama", base_url="http://h:1/", timeout=3)

    assert complete("ping") == "pong"
    assert captured == {
        "url": "http://h:1/api/generate",
        "body": {"model": "llama", "prompt": "ping", "stream": False},
        "timeout": 3,
    }
    assert lp.ollama_generate_fn(base_url="http://h:1")("Q", ["c"]) == "pong"


def test_generate_from_complete_uses_custom_formatter():
    generate = lp.generate_from_complete(lambda p: p.upper(), lambda q, c: f"{q}|{','.join(c)}")
    assert generate("q", ["a", "b"]) == "Q|A,B"


@pytest.mark.parametrize(
    ("factory", "module", "extra"),
    [
        (lp.anthropic_complete_fn, "anthropic", "anthropic"),
        (lp.anthropic_stream_fn, "anthropic", "anthropic"),
        (lp.openai_complete_fn, "openai", "openai"),
        (lp.openai_stream_fn, "openai", "openai"),
    ],
)
def test_missing_sdk_raises_helpful_error(monkeypatch, factory, module, extra):
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match=f"pip install ragforge\\[{extra}\\]"):
        factory()


def test_default_clients_are_built_from_sdk(monkeypatch):
    made = []
    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda: (
        made.append("a")
        or types.SimpleNamespace(  # type: ignore[attr-defined]
            messages=types.SimpleNamespace(
                create=lambda **k: types.SimpleNamespace(content=[_Block("z")])
            )
        )
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    assert lp.anthropic_complete_fn()("x") == "z" and made == ["a"]

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda: (
        made.append("o")
        or types.SimpleNamespace(  # type: ignore[attr-defined]
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **k: types.SimpleNamespace(
                        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="w"))]
                    )
                )
            )
        )
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    assert lp.openai_complete_fn()("x") == "w" and made == ["a", "o"]
