"""A dependency-free default embedding function.

Real deployments should pass a real embedding model (OpenAI, Voyage,
sentence-transformers, ...) as the ``embed_fn`` argument accepted
throughout this package. ``hashing_embed`` exists so the whole pipeline
-- chunking, hybrid retrieval, reranking, evaluation -- can be exercised
offline and deterministically in tests and examples.
"""

from __future__ import annotations


def hashing_embed(text: str, dims: int = 128) -> list[float]:
    """Deterministic bag-of-words hashing embedding."""

    vector = [0.0] * dims
    for token in text.lower().split():
        vector[hash(token) % dims] += 1.0
    return vector


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
