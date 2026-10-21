"""Quantizers and the quantized index: correctness, recall, memory, persistence."""

from __future__ import annotations

import random

import pytest

from ragforge.chunking import Chunk
from ragforge.embeddings import normalize_vector
from ragforge.index import VectorIndex, VectorSearchable
from ragforge.quantization import BinaryQuantizer, QuantizedVectorIndex, ScalarQuantizer


def _vectors(n: int, dims: int = 48, seed: int = 0) -> list[list[float]]:
    rng = random.Random(seed)
    centers = [normalize_vector([rng.gauss(0, 1) for _ in range(dims)]) for _ in range(10)]
    return [
        normalize_vector([x + rng.gauss(0, 0.35) for x in rng.choice(centers)]) for _ in range(n)
    ]


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(id=f"c{i}", text=f"t{i}", doc_id=f"d{i % 7}", position=0, metadata={"p": i % 2})
        for i in range(n)
    ]


def _lookup(vectors, queries):
    table = {f"t{i}": v for i, v in enumerate(vectors)} | {
        f"q{i}": v for i, v in enumerate(queries)
    }
    return table.__getitem__


def test_scalar_roundtrip_error_is_at_most_half_a_step():
    vectors = _vectors(100)
    q = ScalarQuantizer.fit(vectors)
    for v in vectors[:20]:
        decoded = q.decode(q.encode(v))
        assert all(abs(a - b) <= s / 2 + 1e-9 for a, b, s in zip(v, decoded, q.step, strict=True))


def test_scalar_score_equals_dot_with_decoded_vector():
    vectors = _vectors(50)
    q = ScalarQuantizer.fit(vectors)
    query = vectors[3]
    prepared = q.prepare_query(query)
    for v in vectors[:10]:
        code = q.encode(v)
        expected = sum(a * b for a, b in zip(query, q.decode(code), strict=True))
        assert q.score(prepared, code) == pytest.approx(expected)


def test_scalar_clips_out_of_range_values_and_validates():
    q = ScalarQuantizer([0.0, 0.0], [1.0, 1.0])
    assert q.encode([5.0, -5.0]) == bytes([255, 0])
    with pytest.raises(ValueError, match="expected 2 dims"):
        q.encode([1.0])
    with pytest.raises(ValueError):
        ScalarQuantizer([], [])
    with pytest.raises(ValueError):
        ScalarQuantizer.fit([])


def test_binary_similarity_properties():
    q = BinaryQuantizer([0.0, 0.0, 0.0, 0.0])
    a = q.encode([1, 1, -1, -1])
    assert a == 0b0011
    assert q.similarity(a, a) == 1.0
    assert q.similarity(a, q.encode([-1, -1, 1, 1])) == -1.0
    assert q.similarity(a, q.encode([1, -1, -1, -1])) == 0.5
    with pytest.raises(ValueError):
        q.encode([1.0])
    with pytest.raises(ValueError):
        BinaryQuantizer([])
    with pytest.raises(ValueError):
        BinaryQuantizer.fit([])


def test_binary_fit_centers_on_dimension_means():
    q = BinaryQuantizer.fit([[1.0, 10.0], [3.0, 30.0]])
    assert q.thresholds == [2.0, 20.0]


def _recall(index_factory, rescore: bool, n=800, queries=20):
    vectors = _vectors(n, seed=1)
    qvecs = _vectors(queries, seed=2)
    embed = _lookup(vectors, qvecs)
    exact = VectorIndex(embed_fn=embed)
    quant = index_factory(embed)
    chunks = _chunks(n)
    exact.add_vectors(chunks, vectors)
    quant.add_vectors(chunks, vectors)
    total = 0.0
    for i in range(queries):
        truth = {r.chunk.id for r in exact.search(f"q{i}", k=10)}
        got = {r.chunk.id for r in quant.search(f"q{i}", k=10)}
        total += len(truth & got) / 10
    return total / queries


def test_int8_with_rescoring_is_nearly_exact():
    recall = _recall(lambda e: QuantizedVectorIndex(embed_fn=e, mode="int8"), True)
    assert recall >= 0.98


def test_binary_with_rescoring_beats_binary_alone():
    with_rescore = _recall(lambda e: QuantizedVectorIndex(embed_fn=e, mode="binary"), True)
    codes_only = _recall(
        lambda e: QuantizedVectorIndex(embed_fn=e, mode="binary", keep_full_precision=False), False
    )
    assert with_rescore > codes_only
    assert with_rescore >= 0.4 and codes_only >= 0.15  # noisy synthetic clusters: hard case


def test_rescore_factor_trades_recall_for_work():
    low = _recall(lambda e: QuantizedVectorIndex(embed_fn=e, mode="binary", rescore_factor=1), True)
    high = _recall(
        lambda e: QuantizedVectorIndex(embed_fn=e, mode="binary", rescore_factor=8), True
    )
    assert high >= low


def test_memory_accounting():
    vectors = _vectors(100, dims=64)
    chunks = _chunks(100)
    for mode, per_vec in (("int8", 64), ("binary", 8)):
        index = QuantizedVectorIndex(mode=mode, keep_full_precision=False)  # type: ignore[arg-type]
        index.add_vectors(chunks, vectors)
        mem = index.memory_bytes()
        assert mem["codes"] == 100 * per_vec and mem["rescoring_store"] == 0
        assert mem["float32_baseline"] == 100 * 64 * 4
    kept = QuantizedVectorIndex(mode="int8")
    kept.add_vectors(chunks, vectors)
    assert kept.memory_bytes()["rescoring_store"] == 100 * 64 * 4
    assert QuantizedVectorIndex().memory_bytes()["codes"] == 0


def test_satisfies_protocol_provenance_and_filtering():
    vectors = _vectors(60)
    qvecs = _vectors(3, seed=5)
    embed = _lookup(vectors, qvecs)
    index = QuantizedVectorIndex(embed_fn=embed, mode="int8")
    assert isinstance(index, VectorSearchable)
    index.add_vectors(_chunks(60), vectors)
    hits = index.search("q0", k=5, filter_fn=lambda c: c.metadata["p"] == 1)
    assert len(hits) == 5 and all(h.chunk.metadata["p"] == 1 for h in hits)
    assert {h.provenance for h in hits} == {"vector:int8+rescore"}
    assert index.search("q0", k=5, filter_fn=lambda c: False) == []
    assert index.search("q0", k=0) == []
    codes_only = QuantizedVectorIndex(embed_fn=embed, mode="binary", keep_full_precision=False)
    codes_only.add_vectors(_chunks(60), vectors)
    assert codes_only.search("q0", k=3)[0].provenance == "vector:binary"


def test_delete_clear_refit_and_dimension_errors():
    vectors = _vectors(40, dims=16)
    index = QuantizedVectorIndex(mode="int8")
    index.add_vectors(_chunks(40), vectors)
    assert index.delete("d1") > 0 and len(index) < 40
    assert index.delete("missing") == 0
    index.refit()
    assert len(index._codes) == len(index)
    with pytest.raises(ValueError, match="dimension"):
        index.add_vectors(_chunks(1), [[1.0, 0.0]])
    with pytest.raises(ValueError, match="dimension"):
        index.search("some text")  # hashing_embed -> 128 dims vs 16
    index.clear()
    assert len(index) == 0 and index.search("x") == []
    with pytest.raises(ValueError, match="keep_full_precision"):
        QuantizedVectorIndex(keep_full_precision=False).refit()
    QuantizedVectorIndex().refit()  # empty: no-op


@pytest.mark.parametrize("kwargs", [{"mode": "fp4"}, {"rescore_factor": 0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        QuantizedVectorIndex(**kwargs)


def test_length_mismatch_and_empty_add():
    index = QuantizedVectorIndex()
    with pytest.raises(ValueError, match="Mismatch"):
        index.add_vectors(_chunks(2), [[1.0, 0.0]])
    index.add_vectors([], [])
    index.add([])
    assert len(index) == 0


@pytest.mark.parametrize("mode", ["int8", "binary"])
@pytest.mark.parametrize("keep", [True, False])
def test_persistence_roundtrip_preserves_results(tmp_path, mode, keep):
    vectors = _vectors(80, dims=32)
    qvecs = _vectors(3, dims=32, seed=9)
    embed = _lookup(vectors, qvecs)
    index = QuantizedVectorIndex(embed_fn=embed, mode=mode, keep_full_precision=keep)
    index.add_vectors(_chunks(80), vectors)
    path = tmp_path / "q.json"
    index.save(path)

    loaded = QuantizedVectorIndex.load(path, embed_fn=embed)
    assert loaded.mode == mode and loaded.keep_full_precision == keep
    assert [r.chunk.id for r in loaded.search("q1", k=8)] == [
        r.chunk.id for r in index.search("q1", k=8)
    ]
    empty = QuantizedVectorIndex.from_dict(QuantizedVectorIndex(mode=mode).to_dict())
    assert len(empty) == 0
