"""On-disk index: mmap-backed retrieval parity, incremental updates, crash safety, compaction."""

from __future__ import annotations

import json
import os
import random

import pytest

from ragforge import RagPipeline
from ragforge.ann import HNSWIndex
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.storage import (
    DiskIndex,
    StorageError,
    VectorStore,
    _read_graph,
    _write_graph,
)

TOPICS = ["engine", "forest", "market", "orbit", "protein", "quartz", "river", "signal"]


def _docs(n: int, seed: int = 0) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    return [
        (f"doc{i}", " ".join(rng.choices(TOPICS, k=12)) + f" unique{i} marker{i % 5}.")
        for i in range(n)
    ]


def _chunker():
    return RecursiveCharacterChunker(chunk_size=400, chunk_overlap=0)


BACKENDS = ["hnsw", "brute_force", "numpy"]


def _backend(name):
    if name == "numpy":
        pytest.importorskip("numpy")
    return name


def _ids(pipeline: RagPipeline, query: str, k: int = 5) -> list[str]:
    return [c.chunk_id for c in pipeline.retrieve(query, k=k)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_reopened_index_answers_like_the_in_memory_pipeline(tmp_path, backend):
    backend = _backend(backend)
    docs = _docs(60)
    memory = RagPipeline(chunker=_chunker(), vector_backend=backend)
    with DiskIndex.create(tmp_path / "ix", vector_backend=backend, chunker=_chunker()) as disk:
        for doc_id, text in docs:
            memory.ingest(doc_id, text)
            disk.ingest(doc_id, text)

    with DiskIndex.open(tmp_path / "ix", chunker=_chunker()) as reopened:
        assert reopened.pipeline.chunk_count == memory.chunk_count == 60
        for query in ["unique7", "unique31", "unique52"]:
            got, want = _ids(reopened.pipeline, query), _ids(memory, query)
            assert got[0] == want[0]
            assert len(set(got) & set(want)) >= 3  # float32 storage may reorder near-ties


def test_files_on_disk_have_the_documented_layout(tmp_path):
    with DiskIndex.create(tmp_path / "ix", vector_backend="hnsw", chunker=_chunker()) as disk:
        for doc_id, text in _docs(20):
            disk.ingest(doc_id, text)
    names = {p.name for p in (tmp_path / "ix").iterdir()}
    assert {
        "manifest.json",
        "chunks.jsonl",
        "vectors.f32",
        "dead_rows.json",
        "graph.json",
        "graph.bin",
    } <= names
    manifest = json.loads((tmp_path / "ix" / "manifest.json").read_text())
    assert (
        manifest["format"] == "ragforge-dir/1"
        and manifest["rows"] == 20
        and manifest["dims"] == 128
    )
    assert (tmp_path / "ix" / "vectors.f32").stat().st_size == 20 * 128 * 4


@pytest.mark.parametrize("backend", BACKENDS)
def test_incremental_ingest_across_sessions(tmp_path, backend):
    backend = _backend(backend)
    docs = _docs(40)
    with DiskIndex.create(tmp_path / "ix", vector_backend=backend, chunker=_chunker()) as disk:
        for doc_id, text in docs[:25]:
            disk.ingest(doc_id, text)
    with DiskIndex.open(tmp_path / "ix", chunker=_chunker()) as disk:
        for doc_id, text in docs[25:]:
            disk.ingest(doc_id, text)
        # visible immediately, before the commit
        assert disk.pipeline.retrieve("unique39", k=1)[0].doc_id == "doc39"
        disk.commit()
    with DiskIndex.open(tmp_path / "ix", chunker=_chunker(), readonly=True) as disk:
        assert disk.pipeline.chunk_count == 40
        assert disk.pipeline.retrieve("unique39", k=1)[0].doc_id == "doc39"
        assert disk.pipeline.retrieve("unique3", k=1)[0].doc_id == "doc3"


@pytest.mark.parametrize("backend", BACKENDS)
def test_delete_then_compact_reclaims_space_and_stays_consistent(tmp_path, backend):
    backend = _backend(backend)
    with DiskIndex.create(tmp_path / "ix", vector_backend=backend, chunker=_chunker()) as disk:
        for doc_id, text in _docs(50):
            disk.ingest(doc_id, text)
        disk.commit()
        size_before = (tmp_path / "ix" / "vectors.f32").stat().st_size
        for i in range(0, 50, 2):
            assert disk.delete(f"doc{i}") == 1
        assert disk.delete("doc0") == 0
        disk.commit()

        assert disk.stats()["dead_rows"] == 25
        assert "doc4" not in {c.doc_id for c in disk.pipeline.retrieve("unique4", k=10)}
        assert disk.compact() == 25 and disk.compact() == 0
        assert (tmp_path / "ix" / "vectors.f32").stat().st_size == size_before // 2
        assert disk.stats()["dead_rows"] == 0 and disk.pipeline.chunk_count == 25
        assert disk.pipeline.retrieve("unique5", k=1)[0].doc_id == "doc5"

        disk.ingest("fresh", "brand new zebra marker text")  # ids stay aligned after compaction
        assert disk.pipeline.retrieve("zebra", k=1)[0].doc_id == "fresh"

    with DiskIndex.open(tmp_path / "ix", chunker=_chunker()) as again:
        assert again.pipeline.chunk_count == 26
        assert again.pipeline.retrieve("zebra", k=1)[0].doc_id == "fresh"
        assert again.pipeline.retrieve("unique7", k=1)[0].doc_id == "doc7"


def test_reingesting_a_deleted_document_id_works(tmp_path):
    with DiskIndex.create(
        tmp_path / "ix", vector_backend="brute_force", chunker=_chunker()
    ) as disk:
        disk.ingest("a", "first version about apples")
        disk.commit()
        disk.delete("a")
        disk.ingest("a", "second version about oranges")
    with DiskIndex.open(tmp_path / "ix", chunker=_chunker()) as disk:
        hit = disk.pipeline.retrieve("oranges", k=3)
        assert [h.text for h in hit] == ["second version about oranges"]
        assert disk.pipeline.retrieve("apples", k=3)[0].text.startswith(
            "second"
        )  # old text is gone


def test_uncommitted_tail_after_a_crash_is_ignored_and_overwritten(tmp_path):
    with DiskIndex.create(tmp_path / "ix", vector_backend="hnsw", chunker=_chunker()) as disk:
        for doc_id, text in _docs(10):
            disk.ingest(doc_id, text)
    ix = tmp_path / "ix"
    with open(ix / "vectors.f32", "ab") as f:
        f.write(b"\x00" * 128 * 4 * 3)  # three half-written rows
    with open(ix / "chunks.jsonl", "a") as f:
        f.write(
            json.dumps(
                {"row": 10, "chunk": {"id": "ghost", "text": "x", "doc_id": "g", "position": 0}}
            )
            + "\n"
        )

    with DiskIndex.open(ix, chunker=_chunker()) as disk:
        assert disk.pipeline.chunk_count == 10 and "ghost" not in disk._row_of_chunk
        disk.ingest("late", "late arrival text zebra")
        disk.commit()
    assert (ix / "vectors.f32").stat().st_size == 11 * 128 * 4
    with DiskIndex.open(ix, chunker=_chunker()) as disk:
        assert disk.pipeline.chunk_count == 11
        assert disk.pipeline.retrieve("zebra", k=1)[0].doc_id == "late"


def test_corrupt_or_foreign_directories_fail_clearly(tmp_path):
    with pytest.raises(StorageError, match="No index"):
        DiskIndex.open(tmp_path / "nothing")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "manifest.json").write_text(json.dumps({"format": "other/9"}))
    with pytest.raises(StorageError, match="format"):
        DiskIndex.open(bad)

    with DiskIndex.create(
        tmp_path / "ix", vector_backend="brute_force", chunker=_chunker()
    ) as disk:
        disk.ingest("a", "some text")
    with open(tmp_path / "ix" / "vectors.f32", "r+b") as f:
        f.truncate(10)
    with pytest.raises(StorageError, match="shorter"):
        DiskIndex.open(tmp_path / "ix")
    with pytest.raises(StorageError, match="already contains"):
        DiskIndex.create(tmp_path / "ix")
    with pytest.raises(ValueError, match="vector_backend"):
        DiskIndex.create(tmp_path / "x2", vector_backend="int8")


def test_single_writer_lock_and_readonly_mode(tmp_path):
    first = DiskIndex.create(tmp_path / "ix", vector_backend="brute_force", chunker=_chunker())
    first.ingest("a", "alpha text")
    first.commit()
    try:
        with pytest.raises(StorageError, match="another process"):
            DiskIndex.open(tmp_path / "ix")
        reader = DiskIndex.open(tmp_path / "ix", readonly=True, chunker=_chunker())
        assert reader.readonly and reader.pipeline.chunk_count == 1
        for action in (
            lambda: reader.ingest("b", "x"),
            lambda: reader.delete("a"),
            reader.commit,
            reader.compact,
        ):
            with pytest.raises(StorageError, match="readonly"):
                action()
        reader.close()
    finally:
        first.close()
    DiskIndex.open(tmp_path / "ix").close()  # lock released


def test_dimension_mismatch_and_embedder_fixed_dims(tmp_path):
    with DiskIndex.create(
        tmp_path / "ix", vector_backend="brute_force", embed_fn=lambda t: [1.0, 0.0]
    ) as disk:
        disk.ingest("a", "text")
        assert disk._dims == 2
        with pytest.raises(ValueError, match="dimension"):
            disk._append([disk.pipeline.bm25_index._chunks[0]], [[1.0, 0.0, 0.0]])


def test_from_pipeline_converts_an_existing_index(tmp_path):
    source = RagPipeline(chunker=_chunker(), vector_backend="hnsw")
    for doc_id, text in _docs(30):
        source.ingest(doc_id, text)
    DiskIndex.from_pipeline(source, tmp_path / "ix").close()
    with DiskIndex.open(tmp_path / "ix", chunker=_chunker()) as disk:
        assert disk.pipeline.chunk_count == 30
        assert _ids(disk.pipeline, "unique12 engine", 3) == _ids(source, "unique12 engine", 3)
    with pytest.raises(StorageError, match="supports"):
        DiskIndex.from_pipeline(RagPipeline(vector_backend="int8"), tmp_path / "q")


def test_mmap_can_be_disabled_and_context_survives(tmp_path):
    ctx = lambda doc, chunk: "handbook context"  # noqa: E731
    with DiskIndex.create(
        tmp_path / "ix", vector_backend="brute_force", chunker=_chunker(), contextualizer=ctx
    ) as disk:
        disk.ingest("a", "plain body text")
    with DiskIndex.open(tmp_path / "ix", chunker=_chunker(), mmap=False) as disk:
        assert disk._store._map is None
        assert disk.pipeline.bm25_index._chunks[0].context == "handbook context"
        assert disk.pipeline.retrieve("handbook", k=1)[0].text == "plain body text"


def test_empty_index_roundtrip_and_stats(tmp_path):
    DiskIndex.create(tmp_path / "ix", chunker=_chunker()).close()
    with DiskIndex.open(tmp_path / "ix", chunker=_chunker()) as disk:
        assert disk.pipeline.chunk_count == 0 and disk.pipeline.retrieve("x") == []
        stats = disk.stats()
        assert stats["rows"] == 0 and "manifest.json" in stats["bytes"]


def test_vector_store_semantics(tmp_path):
    path = tmp_path / "v.f32"
    import array

    path.write_bytes(array.array("f", [1, 2, 3, 4, 5, 6]).tobytes())
    store = VectorStore(path, dims=3, rows=2)
    assert list(store[1]) == [4.0, 5.0, 6.0] and len(store) == 2
    store[2] = [7.0, 8.0, 9.0]
    assert list(store) == [0, 1, 2] and len(store) == 3 and list(store[2]) == [7.0, 8.0, 9.0]
    with pytest.raises(KeyError):
        store[5]
    store.close()
    store.close()  # idempotent
    empty = VectorStore(tmp_path / "none.f32", dims=3, rows=0)
    assert len(empty) == 0 and store.committed_rows == 2


def test_graph_binary_format_roundtrips_exactly(tmp_path):
    rng = random.Random(3)
    graph = HNSWIndex(m=4, ef_construction=20, seed=5)
    vectors = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(120)]
    for v in vectors:
        graph.add(v)
    graph.mark_deleted(7)

    _write_graph(tmp_path, graph)
    store = {i: graph.vector(i) for i in range(120)}
    loaded = _read_graph(tmp_path, store)  # type: ignore[arg-type]

    assert loaded.structure() == graph.structure()
    q = vectors[3]
    assert loaded.search(q, k=10, ef_search=40) == graph.search(q, k=10, ef_search=40)
    assert loaded.unreachable() == []


def test_hnsw_structure_roundtrip_in_memory():
    graph = HNSWIndex(m=3, ef_construction=10)
    for i in range(30):
        graph.add([1.0, i * 0.05, 0.0])
    clone = HNSWIndex.from_structure(graph.structure(), dict(graph._vectors))
    assert clone.structure() == graph.structure()


def test_os_level_atomicity_leaves_no_temp_files(tmp_path):
    with DiskIndex.create(tmp_path / "ix", vector_backend="hnsw", chunker=_chunker()) as disk:
        for doc_id, text in _docs(15):
            disk.ingest(doc_id, text)
        disk.commit()
    assert not [n for n in os.listdir(tmp_path / "ix") if n.endswith((".tmp", ".new"))]
