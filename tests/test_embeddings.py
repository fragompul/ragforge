import pytest

from ragforge.embeddings import (
    cosine_similarity,
    hashing_embed,
    normalize_vector,
)


def test_hashing_embed_deterministic():
    text = "Machine learning and distributed retrieval systems"
    vec1 = hashing_embed(text, dims=64)
    vec2 = hashing_embed(text, dims=64)

    assert len(vec1) == 64
    assert vec1 == vec2
    # Verify non-zero norm
    assert sum(x * x for x in vec1) > 0.0


def test_hashing_embed_empty_text():
    vec = hashing_embed("", dims=32)
    assert len(vec) == 32
    assert all(x == 0.0 for x in vec)


def test_hashing_embed_invalid_dims():
    with pytest.raises(ValueError, match="dims must be positive"):
        hashing_embed("test", dims=0)
    with pytest.raises(ValueError, match="dims must be positive"):
        hashing_embed("test", dims=-10)


def test_normalize_vector():
    vec = [3.0, 4.0]
    normed = normalize_vector(vec)
    assert pytest.approx(normed[0]) == 0.6
    assert pytest.approx(normed[1]) == 0.8
    assert pytest.approx(sum(x * x for x in normed)) == 1.0


def test_normalize_vector_zero_vector():
    vec = [0.0, 0.0, 0.0]
    assert normalize_vector(vec) == [0.0, 0.0, 0.0]


def test_cosine_similarity_bounds_and_values():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(a, b)) == 1.0

    c = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(a, c)) == 0.0

    d = [-1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(a, d)) == -1.0


def test_cosine_similarity_zero_norm():
    a = [0.0, 0.0]
    b = [1.0, 2.0]
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_dimension_mismatch():
    a = [1.0, 2.0]
    b = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        cosine_similarity(a, b)
