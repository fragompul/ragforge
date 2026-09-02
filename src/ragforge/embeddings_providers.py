"""Optional adapters bridging real embedding providers to ragforge's ``EmbedFn``.

``ragforge`` core has zero required dependencies (see ``hashing_embed`` in
:mod:`ragforge.embeddings`), so it installs instantly and runs fully offline
in tests, examples, and CI. Production deployments, however, want real
semantic embeddings from a hosted API or a local model.

Every adapter here imports its client library lazily, *inside* the factory
function, and raises a clear ``ImportError`` naming the exact install command
if the dependency is missing -- so importing this module, or importing
``ragforge`` itself, never requires any of these packages to be present.
Only calling the specific factory you need pulls in that provider's library.

    pip install ragforge[openai]     # openai_embed_fn
    pip install ragforge[cohere]     # cohere_embed_fn
    pip install ragforge[local]      # sentence_transformers_embed_fn
    # ollama_embed_fn needs no client library: it speaks plain HTTP.

Each factory returns a plain ``Callable[[str], list[float]]`` matching
``ragforge.embeddings.EmbedFn``, so it drops straight into
``RagPipeline(embed_fn=...)`` or ``VectorIndex(embed_fn=...)`` with no other
code changes -- mirroring the "Production Integration" pattern shown in the
README.
"""

from __future__ import annotations

import functools
import json
import urllib.request
from typing import Any

from ragforge.embeddings import EmbedFn


def _missing_dependency(package: str, extra: str) -> ImportError:
    return ImportError(
        f"'{package}' is required for this embedding adapter but is not installed. "
        f"Install it with:\n    pip install ragforge[{extra}]\n"
        f"or directly:\n    pip install {package}"
    )


def openai_embed_fn(model: str = "text-embedding-3-small", client: Any = None) -> EmbedFn:
    """Build an ``EmbedFn`` backed by the OpenAI embeddings API.

    Requires ``pip install ragforge[openai]`` and a configured
    ``OPENAI_API_KEY`` (or pass a pre-configured ``openai.OpenAI`` instance
    via ``client``, e.g. for a custom base URL or Azure OpenAI).
    """
    if client is not None:
        resolved_client = client
    else:
        try:
            import openai
        except ImportError as exc:
            raise _missing_dependency("openai", "openai") from exc
        resolved_client = openai.OpenAI()

    def _embed(text: str) -> list[float]:
        response = resolved_client.embeddings.create(input=[text], model=model)
        return list(response.data[0].embedding)

    return _embed


def cohere_embed_fn(
    model: str = "embed-english-v3.0",
    input_type: str = "search_document",
    client: Any = None,
) -> EmbedFn:
    """Build an ``EmbedFn`` backed by the Cohere embeddings API.

    Requires ``pip install ragforge[cohere]`` and a configured
    ``CO_API_KEY`` (or pass a pre-configured ``cohere.Client`` via
    ``client``). ``input_type`` should be ``"search_query"`` for the query
    side of retrieval and ``"search_document"`` for ingested chunks --
    Cohere's v3 embedding models use asymmetric encoders for the two roles.
    """
    if client is not None:
        resolved_client = client
    else:
        try:
            import cohere
        except ImportError as exc:
            raise _missing_dependency("cohere", "cohere") from exc
        resolved_client = cohere.Client()

    def _embed(text: str) -> list[float]:
        response = resolved_client.embed(texts=[text], model=model, input_type=input_type)
        return list(response.embeddings[0])

    return _embed


def sentence_transformers_embed_fn(
    model_name: str = "all-MiniLM-L6-v2", model: Any = None
) -> EmbedFn:
    """Build an ``EmbedFn`` backed by a local ``sentence-transformers`` model.

    Requires ``pip install ragforge[local]``. Runs fully offline once the
    model weights are downloaded, at the cost of local compute -- a good fit
    for air-gapped or cost-sensitive deployments where calling a hosted
    embeddings API per chunk is undesirable.
    """
    if model is not None:
        resolved_model = model
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise _missing_dependency("sentence-transformers", "local") from exc
        resolved_model = SentenceTransformer(model_name)

    def _embed(text: str) -> list[float]:
        vector = resolved_model.encode(text, normalize_embeddings=True)
        return list(vector.tolist())

    return _embed


def ollama_embed_fn(
    model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"
) -> EmbedFn:
    """Build an ``EmbedFn`` backed by a local Ollama server.

    No client library required: this talks Ollama's HTTP embeddings API
    (``POST /api/embeddings``) directly with the standard library, so it
    works the moment ``ollama serve`` is running with an embedding model
    pulled (e.g. ``ollama pull nomic-embed-text``).
    """
    endpoint = f"{base_url.rstrip('/')}/api/embeddings"

    def _embed(text: str) -> list[float]:
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        request = urllib.request.Request(
            endpoint, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        return list(data["embedding"])

    return _embed


def cached_embed_fn(embed_fn: EmbedFn, max_size: int = 4096) -> EmbedFn:
    """Wrap any ``EmbedFn`` with an LRU cache keyed on exact text match.

    Real embedding calls (hosted API round-trips, or local model inference)
    are the dominant cost in most RAG pipelines. Overlapping chunk windows
    during re-ingestion and repeated queries during evaluation sweeps or
    interactive use frequently re-embed identical strings; caching turns
    those into free dictionary lookups.
    """

    @functools.lru_cache(maxsize=max_size)
    def _cached(text: str) -> tuple[float, ...]:
        return tuple(embed_fn(text))

    def _embed(text: str) -> list[float]:
        return list(_cached(text))

    return _embed
