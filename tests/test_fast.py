"""NumpyVectorIndex must be a drop-in, ranking-identical replacement for VectorIndex."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("numpy")

from ragforge.chunking import Chunk  # noqa: E402
from ragforge.embeddings import hashing_embed  # noqa: E402
from ragforge.fast import NumpyVectorIndex  # noqa: E402
from ragforge.index import VectorIndex, VectorSearchable  # noqa: E402

WORDS = ["alpha", "beta", "gamma", "delta", "kappa", "sigma", "omega", "zeta", "theta", "iota"]


def _chunks(n: int, seed: int = 0) -> list[Chunk]:
    rng = random.Random(seed)
    return [
        Chunk(
            id=f"c{i}",
            text=" ".join(rng.choices(WORDS, k=rng.randint(2, 9))),
            doc_id=f"d{i % 5}",
            position=i,
            metadata={"parity": i % 2},
        )
        for i in range(n)
    ]


def test_satisfies_vector_searchable_protocol():
    assert isinstance(NumpyVectorIndex(), VectorSearchable)


def test_matches_brute_force_ranking_exactly():
    chunks = _chunks(200)
    exact, fast = VectorIndex(), NumpyVectorIndex()
    exact.add(chunks)
    fast.add(chunks)

    for query in ["alpha beta", "omega zeta theta", "kappa", "iota iota sigma"]:
        a = exact.search(query, k=10)
        b = fast.search(query, k=10)
        assert [r.score for r in b] == pytest.approx([r.score for r in a])
        # Exactly-tied chunks may swap places (BLAS vs. Python summation order differ in the
        # last bits), so compare ids only where the score is unambiguous.
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            neighbours = [r.score for j, r in enumerate(a) if j != i]
            if all(abs(x.score - s) > 1e-9 for s in neighbours):
                assert x.chunk.id == y.chunk.id
        assert all(r.provenance == "vector:numpy" for r in b)


def test_filtering_matches_brute_force():
    chunks = _chunks(120, seed=2)
    exact, fast = VectorIndex(), NumpyVectorIndex()
    exact.add(chunks)
    fast.add(chunks)

    def only_odd(chunk: Chunk) -> bool:
        return chunk.metadata["parity"] == 1

    a = exact.search("alpha gamma", k=7, filter_fn=only_odd)
    b = fast.search("alpha gamma", k=7, filter_fn=only_odd)
    assert [r.score for r in b] == pytest.approx([r.score for r in a])
    assert all(r.chunk.metadata["parity"] == 1 for r in b)
    assert fast.search("alpha", k=3, filter_fn=lambda c: False) == []


def test_delete_clear_and_len():
    index = NumpyVectorIndex()
    index.add(_chunks(20))
    assert len(index) == 20
    assert index.delete("d1") == 4
    assert len(index) == 16
    assert all(r.chunk.doc_id != "d1" for r in index.search("alpha", k=16))
    assert index.delete("nope") == 0
    index.clear()
    assert len(index) == 0
    assert index.search("alpha") == []
    assert index.delete("d0") == 0


def test_delete_everything_leaves_usable_index():
    index = NumpyVectorIndex()
    index.add([Chunk(id="a", text="alpha", doc_id="x", position=0)])
    index.delete("x")
    assert index.search("alpha") == []
    index.add([Chunk(id="b", text="beta", doc_id="y", position=0)])
    assert index.search("beta")[0].chunk.id == "b"


def test_ties_break_by_insertion_order():
    chunks = [Chunk(id=f"t{i}", text="same words", doc_id="d", position=i) for i in range(8)]
    index = NumpyVectorIndex()
    index.add(chunks)
    assert [r.chunk.id for r in index.search("same words", k=3)] == ["t0", "t1", "t2"]


def test_zero_query_vector_scores_zero():
    index = NumpyVectorIndex()
    index.add(_chunks(5))
    results = index.search("!!! ???", k=3)  # no tokens -> zero vector
    assert [r.score for r in results] == [0.0, 0.0, 0.0]


def test_dimension_mismatch_raises():
    index = NumpyVectorIndex()
    chunk = Chunk(id="a", text="x", doc_id="d", position=0)
    index.add_vectors([chunk], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="dimension"):
        index.add_vectors([chunk], [[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="dimension"):
        index.search("x")  # hashing_embed -> 128 dims vs 2


def test_add_vectors_length_mismatch_raises():
    with pytest.raises(ValueError, match="Mismatch"):
        NumpyVectorIndex().add_vectors([Chunk(id="a", text="x", doc_id="d", position=0)], [])


def test_ragged_vectors_rejected():
    chunk = Chunk(id="a", text="x", doc_id="d", position=0)
    with pytest.raises((ValueError, TypeError)):
        NumpyVectorIndex().add_vectors([chunk, chunk], [[1.0, 0.0], [1.0]])


def test_roundtrip_and_cross_compatibility(tmp_path):
    chunks = _chunks(30, seed=4)
    original = NumpyVectorIndex()
    original.add(chunks)
    path = tmp_path / "idx.json"
    original.save(path)

    loaded = NumpyVectorIndex.load(path)
    from_brute = NumpyVectorIndex.from_dict(
        (lambda v: (v.add(chunks), v.to_dict())[1])(VectorIndex())
    )

    for idx in (loaded, from_brute):
        assert [r.chunk.id for r in idx.search("alpha beta", k=5)] == [
            r.chunk.id for r in original.search("alpha beta", k=5)
        ]


def test_from_dict_without_vectors_reembeds():
    data = {"chunks": [c.to_dict() for c in _chunks(4)]}
    index = NumpyVectorIndex.from_dict(data, embed_fn=lambda t: hashing_embed(t, dims=16))
    assert len(index) == 4


def test_missing_numpy_raises_actionable_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(ImportError, match="ragforge\\[fast\\]"):
        NumpyVectorIndex()
