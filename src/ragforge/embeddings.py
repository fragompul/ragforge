"""Deterministic embeddings and vector math utilities.

Real deployments should pass a real embedding model (e.g., OpenAI text-embedding-3,
Voyage AI, Cohere, or local sentence-transformers) as the ``embed_fn`` argument
accepted throughout ragforge.

``hashing_embed`` provides a zero-dependency, cross-process deterministic
bag-of-words embedding so the full pipeline (chunking, hybrid retrieval,
reranking, evaluation) can be exercised offline and deterministically in
tests, examples, and continuous integration.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Callable, Sequence

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

EmbedFn = Callable[[str], list[float]]
BatchEmbedFn = Callable[[Sequence[str]], list[list[float]]]


def hashing_embed(text: str, dims: int = 128) -> list[float]:
    """Deterministic bag-of-words hashing embedding using CRC32.

    Unlike Python's built-in `hash()`, CRC32 guarantees identical embeddings
    across different Python processes, operating systems, and interpreter restarts.
    """
    if dims <= 0:
        raise ValueError(f"dims must be positive, got {dims}")

    vector = [0.0] * dims
    tokens = _TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return vector

    for token in tokens:
        token_hash = zlib.crc32(token.encode("utf-8"))
        vector[token_hash % dims] += 1.0

    return normalize_vector(vector)


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """Normalize a vector to unit Euclidean length (L2 norm)."""
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0.0:
        return [0.0] * len(vector)
    return [float(x / norm) for x in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1.0, 1.0]. Returns 0.0 if either norm is zero.

    Raises:
        ValueError: If vectors have mismatched non-zero dimensions.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: len(a)={len(a)} vs len(b)={len(b)}")

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = float(dot / (norm_a * norm_b))
    return max(-1.0, min(1.0, sim))
