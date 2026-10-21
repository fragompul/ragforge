"""Optional adapters bridging LLM providers to ragforge's generation hooks.

Three callable shapes appear throughout ragforge, and every adapter here
returns one of them -- so a provider is always a one-line swap:

- ``CompleteFn``  = ``prompt -> text``. Used by LLM-as-judge
  (:mod:`ragforge.judge`), agentic grading/rewriting (:mod:`ragforge.agent`),
  and multi-query expansion.
- ``GenerateFn``  = ``(query, contexts) -> text``. Used by ``RagPipeline``.
- ``StreamGenerateFn`` = ``(query, contexts) -> Iterator[str]``. Used by
  ``RagPipeline.answer_stream`` for token-by-token output.

As with :mod:`ragforge.embeddings_providers`, client libraries are imported
lazily inside the factory and a missing package raises an ``ImportError``
naming the exact install command::

    pip install ragforge[anthropic]    # anthropic_*  (Claude)
    pip install ragforge[openai]       # openai_*
    # ollama_* needs no client library: it speaks plain HTTP.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from ragforge.pipeline import PromptFormatter, default_prompt_formatter

CompleteFn = Callable[[str], str]
GenerateFn = Callable[[str, list[str]], str]
StreamGenerateFn = Callable[[str, list[str]], Iterator[str]]

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant answering from retrieved context. Use only the provided "
    "context; if it does not contain the answer, say you don't know. Cite sources as [n]."
)

# Latest Claude Sonnet at the time of writing; override with ``model=``.
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"


def _missing(package: str, extra: str) -> ImportError:
    return ImportError(
        f"'{package}' is required for this LLM adapter but is not installed. "
        f"Install it with:\n    pip install ragforge[{extra}]\n"
        f"or directly:\n    pip install {package}"
    )


def generate_from_complete(
    complete_fn: CompleteFn, prompt_formatter: PromptFormatter = default_prompt_formatter
) -> GenerateFn:
    """Lift a ``prompt -> text`` function into a pipeline ``GenerateFn``."""

    def _generate(query: str, contexts: list[str]) -> str:
        return complete_fn(prompt_formatter(query, contexts))

    return _generate


def _anthropic_client(client: Any) -> Any:
    if client is not None:
        return client
    try:
        import anthropic
    except ImportError as exc:
        raise _missing("anthropic", "anthropic") from exc
    return anthropic.Anthropic()


def _anthropic_text(response: Any) -> str:
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")


def anthropic_complete_fn(
    model: str = DEFAULT_CLAUDE_MODEL,
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    client: Any = None,
) -> CompleteFn:
    """``prompt -> text`` backed by the Anthropic Messages API (needs ``ANTHROPIC_API_KEY``)."""
    resolved = _anthropic_client(client)

    def _complete(prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        return _anthropic_text(resolved.messages.create(**kwargs))

    return _complete


def anthropic_generate_fn(
    model: str = DEFAULT_CLAUDE_MODEL,
    system: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 1024,
    prompt_formatter: PromptFormatter = default_prompt_formatter,
    client: Any = None,
) -> GenerateFn:
    """Pipeline ``GenerateFn`` backed by Claude."""
    return generate_from_complete(
        anthropic_complete_fn(model=model, system=system, max_tokens=max_tokens, client=client),
        prompt_formatter,
    )


def anthropic_stream_fn(
    model: str = DEFAULT_CLAUDE_MODEL,
    system: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 1024,
    prompt_formatter: PromptFormatter = default_prompt_formatter,
    client: Any = None,
) -> StreamGenerateFn:
    """Streaming ``(query, contexts) -> Iterator[str]`` backed by Claude."""
    resolved = _anthropic_client(client)

    def _stream(query: str, contexts: list[str]) -> Iterator[str]:
        with resolved.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt_formatter(query, contexts)}],
        ) as stream:
            yield from stream.text_stream

    return _stream


def _openai_client(client: Any) -> Any:
    if client is not None:
        return client
    try:
        import openai
    except ImportError as exc:
        raise _missing("openai", "openai") from exc
    return openai.OpenAI()


def openai_complete_fn(
    model: str = "gpt-4o-mini",
    system: str | None = None,
    temperature: float = 0.0,
    client: Any = None,
) -> CompleteFn:
    """``prompt -> text`` backed by the OpenAI Chat Completions API."""
    resolved = _openai_client(client)

    def _complete(prompt: str) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = resolved.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        return str(response.choices[0].message.content or "")

    return _complete


def openai_generate_fn(
    model: str = "gpt-4o-mini",
    system: str = DEFAULT_SYSTEM_PROMPT,
    prompt_formatter: PromptFormatter = default_prompt_formatter,
    client: Any = None,
) -> GenerateFn:
    """Pipeline ``GenerateFn`` backed by OpenAI."""
    return generate_from_complete(
        openai_complete_fn(model=model, system=system, client=client), prompt_formatter
    )


def openai_stream_fn(
    model: str = "gpt-4o-mini",
    system: str = DEFAULT_SYSTEM_PROMPT,
    prompt_formatter: PromptFormatter = default_prompt_formatter,
    client: Any = None,
) -> StreamGenerateFn:
    """Streaming generation backed by OpenAI."""
    resolved = _openai_client(client)

    def _stream(query: str, contexts: list[str]) -> Iterator[str]:
        response = resolved.chat.completions.create(
            model=model,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt_formatter(query, contexts)},
            ],
        )
        for event in response:
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                yield str(delta)

    return _stream


def ollama_complete_fn(
    model: str = "llama3.2", base_url: str = "http://localhost:11434", timeout: float = 120.0
) -> CompleteFn:
    """``prompt -> text`` against a local Ollama server (plain HTTP, no client library)."""
    endpoint = f"{base_url.rstrip('/')}/api/generate"

    def _complete(prompt: str) -> str:
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            endpoint, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return str(json.loads(response.read().decode("utf-8"))["response"])

    return _complete


def ollama_generate_fn(
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    prompt_formatter: PromptFormatter = default_prompt_formatter,
) -> GenerateFn:
    """Pipeline ``GenerateFn`` backed by a local Ollama model."""
    return generate_from_complete(ollama_complete_fn(model, base_url), prompt_formatter)
