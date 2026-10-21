"""On-disk, memory-mapped, incrementally updatable index (``ragforge-dir/1``).

The JSON persistence used by ``RagPipeline.save`` is simple and portable, but it loads
*everything* into Python objects: a float in a list costs 32 bytes versus 4 on disk, and
startup parses the whole file. :class:`DiskIndex` stores an index as a directory::

    manifest.json    format version, config, row count (the commit record)
    chunks.jsonl     one chunk per line, append-only
    vectors.f32      little-endian float32 rows -- memory-mapped, not loaded
    dead_rows.json   rows of deleted documents (tombstones)
    graph.json/.bin  HNSW structure (CSR adjacency), when the backend is "hnsw"

and serves queries straight from the mapping: vectors are paged in by the OS on demand, so
opening a large index is near-instant and resident memory stays far below the file size.

Properties:

* **Incremental**: :meth:`DiskIndex.ingest` appends rows; :meth:`DiskIndex.delete`
  tombstones them; :meth:`DiskIndex.compact` rewrites the files without dead rows (and
  rebuilds the HNSW graph without tombstones).
* **Crash-safe commits**: data files are appended first, then the manifest (which records
  the committed row count) is replaced atomically. On open, anything past the committed
  count -- the tail of an interrupted append -- is ignored and later overwritten.
* **Single writer**: a writer takes an advisory ``flock`` on ``LOCK`` (POSIX); readers do not.

Backends: ``brute_force``, ``numpy`` and ``hnsw``. BM25 postings are rebuilt from
``chunks.jsonl`` on open (a linear pass over the text); the vectors, which dominate size
and cost, are not rebuilt.
"""

from __future__ import annotations

import json
import mmap
import os
import sys
from array import array
from collections.abc import Iterator, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from ragforge._version import __version__
from ragforge.ann import HNSWIndex
from ragforge.chunking import Chunk
from ragforge.embeddings import EmbedFn, hashing_embed, normalize_vector
from ragforge.index import ApproxVectorIndex
from ragforge.pipeline import RagPipeline

FORMAT = "ragforge-dir/1"
_BACKENDS = ("brute_force", "numpy", "hnsw")
_LITTLE = sys.byteorder == "little"

try:  # advisory write lock (POSIX)
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


class StorageError(RuntimeError):
    """The index directory is missing, malformed, locked, or inconsistent."""


# --------------------------------------------------------------------------- vectors


class VectorStore(MutableMapping[int, Sequence[float]]):
    """Row-id -> vector mapping over a memory-mapped float32 file plus an in-memory tail.

    Reads of committed rows are zero-copy ``memoryview`` slices of the mapping; rows added
    since the last commit live in an overlay dict until :meth:`DiskIndex.commit` flushes them.
    """

    def __init__(self, path: Path, dims: int, rows: int, use_mmap: bool = True) -> None:
        self.path, self.dims = path, dims
        self._overlay: dict[int, Sequence[float]] = {}
        self._rows = 0
        self._file: Any = None
        self._map: mmap.mmap | None = None
        self._view: Any = None
        self._open(rows, use_mmap)

    def _open(self, rows: int, use_mmap: bool) -> None:
        self.close()
        self._rows = rows
        if rows == 0:
            return
        need = rows * self.dims * 4
        if self.path.stat().st_size < need:
            raise StorageError(f"{self.path.name} is shorter than the manifest's {rows} rows")
        self._file = open(self.path, "rb")  # noqa: SIM115 - kept open for the mapping's lifetime
        if use_mmap:
            self._map = mmap.mmap(self._file.fileno(), need, access=mmap.ACCESS_READ)
            self._view = memoryview(self._map)
        else:
            self._view = memoryview(self._file.read(need))
        if not _LITTLE:  # pragma: no cover - big-endian hosts
            swapped = array("f")
            swapped.frombytes(bytes(self._view))
            swapped.byteswap()
            self._view = memoryview(swapped.tobytes())
        self._view = self._view.cast("f")

    @property
    def committed_rows(self) -> int:
        return self._rows

    def reopen(self, rows: int, use_mmap: bool = True) -> None:
        """Re-map after the file grew; clears the overlay (its rows are now on disk)."""
        self._overlay.clear()
        self._open(rows, use_mmap)

    def close(self) -> None:
        """Unmap the file. Slices still referenced elsewhere keep the mapping alive until freed."""
        view, mapping, file = self._view, self._map, self._file
        self._view = self._map = self._file = None
        if view is not None:
            try:
                view.release()
            except BufferError:  # a caller still holds a row slice; the GC unmaps it later
                return
        if mapping is not None:
            try:
                mapping.close()
            except BufferError:  # pragma: no cover - same situation, via the raw map
                return
        if file is not None:
            file.close()

    def __getitem__(self, row: int) -> Sequence[float]:
        if row in self._overlay:
            return self._overlay[row]
        if 0 <= row < self._rows and self._view is not None:
            return self._view[row * self.dims : (row + 1) * self.dims]  # type: ignore[no-any-return]
        raise KeyError(row)

    def __setitem__(self, row: int, vector: Sequence[float]) -> None:
        self._overlay[row] = vector

    def __delitem__(self, row: int) -> None:  # pragma: no cover - rows are never removed
        raise TypeError("vector rows are append-only; use DiskIndex.delete / compact")

    def __iter__(self) -> Iterator[int]:
        yield from range(self._rows)
        yield from sorted(r for r in self._overlay if r >= self._rows)

    def __len__(self) -> int:
        return self._rows + sum(1 for r in self._overlay if r >= self._rows)


# ---------------------------------------------------------------------------- graph


def _write_graph(directory: Path, graph: HNSWIndex) -> None:
    """Persist the HNSW structure: JSON header + CSR adjacency arrays in a binary file."""
    s = graph.structure()
    arrays: list[array[Any]] = []
    sections: list[dict[str, int]] = []
    offset = 0

    def put(arr: array[Any]) -> tuple[int, int]:
        nonlocal offset
        if not _LITTLE:  # pragma: no cover
            arr.byteswap()
        arrays.append(arr)
        start, offset = offset, offset + len(arr) * arr.itemsize
        return start, len(arr)

    n = s["next_id"]
    levels = array("b", s["levels"])
    lv_start, lv_len = put(levels)
    for level in range(s["max_level"] + 1):
        table = s["neighbors"][level]
        offsets, targets = array("q", [0]), array("i")
        for node in range(n):
            targets.extend(table.get(node, ()))
            offsets.append(len(targets))
        o_start, o_len = put(offsets)
        t_start, t_len = put(targets)
        sections.append(
            {"offsets": o_start, "offsets_len": o_len, "targets": t_start, "targets_len": t_len}
        )
    pinned = array("q", [x for triple in s["pinned"] for x in triple])
    p_start, p_len = put(pinned)

    header = {
        "params": s["params"],
        "next_id": n,
        "entry_point": s["entry_point"],
        "max_level": s["max_level"],
        "deleted": s["deleted"],
        "levels": {"offset": lv_start, "len": lv_len},
        "sections": sections,
        "pinned": {"offset": p_start, "len": p_len},
    }
    tmp_bin, tmp_json = directory / "graph.bin.tmp", directory / "graph.json.tmp"
    with open(tmp_bin, "wb") as f:
        for arr in arrays:
            arr.tofile(f)
        f.flush()
        os.fsync(f.fileno())
    tmp_json.write_text(json.dumps(header), encoding="utf-8")
    os.replace(tmp_bin, directory / "graph.bin")
    os.replace(tmp_json, directory / "graph.json")


def _read_graph(directory: Path, store: VectorStore) -> HNSWIndex:
    header = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
    blob = (directory / "graph.bin").read_bytes()

    def get(typecode: str, offset: int, length: int) -> array[Any]:
        arr: array[Any] = array(typecode)
        arr.frombytes(blob[offset : offset + length * arr.itemsize])
        if not _LITTLE:  # pragma: no cover
            arr.byteswap()
        return arr

    n = header["next_id"]
    levels = list(get("b", header["levels"]["offset"], header["levels"]["len"]))
    neighbors: dict[int, dict[int, list[int]]] = {}
    for level, sec in enumerate(header["sections"]):
        offsets = get("q", sec["offsets"], sec["offsets_len"])
        targets = get("i", sec["targets"], sec["targets_len"])
        neighbors[level] = {
            node: list(targets[offsets[node] : offsets[node + 1]])
            for node in range(n)
            if levels[node] >= level
        }
    flat = get("q", header["pinned"]["offset"], header["pinned"]["len"])
    pinned = [tuple(flat[i : i + 3]) for i in range(0, len(flat), 3)]
    return HNSWIndex.from_structure(
        {
            "params": header["params"],
            "next_id": n,
            "entry_point": header["entry_point"],
            "max_level": header["max_level"],
            "levels": levels,
            "neighbors": neighbors,
            "deleted": header["deleted"],
            "pinned": pinned,
        },
        store,
    )


# ------------------------------------------------------------------------- DiskIndex


def _pack(vector: Sequence[float]) -> array[float]:
    block = array("f", vector)
    if not _LITTLE:  # pragma: no cover
        block.byteswap()
    return block


class DiskIndex:
    """A directory-backed, memory-mapped index wrapped around a :class:`RagPipeline`.

    Create one with :meth:`create` (or :meth:`from_pipeline` to convert an existing
    pipeline), reopen it with :meth:`open`, and use :attr:`pipeline` for retrieval exactly
    like an in-memory pipeline. Mutate through :meth:`ingest` / :meth:`delete` so the files
    stay in sync, then :meth:`commit` (or use ``with DiskIndex.open(...) as idx``).
    """

    def __init__(
        self,
        path: Path,
        pipeline: RagPipeline,
        store: VectorStore | None,
        manifest: dict[str, Any],
        dead: set[int],
        row_of_chunk: dict[str, int],
        lock_fd: int | None,
        use_mmap: bool,
        embed_fn: EmbedFn,
        pipeline_kwargs: dict[str, Any],
    ) -> None:
        self.path, self.pipeline = path, pipeline
        self._store, self._manifest = store, manifest
        self._rows: int = manifest["rows"]
        self._dims: int | None = manifest["dims"]
        self._dead, self._row_of_chunk = dead, row_of_chunk
        self._lock_fd, self._use_mmap = lock_fd, use_mmap
        self._embed_fn, self._kwargs = embed_fn, pipeline_kwargs
        self._pending: list[tuple[Chunk, int, list[float]]] = []

    # -- construction ---------------------------------------------------------------

    @staticmethod
    def _config_of(pipeline: RagPipeline) -> dict[str, Any]:
        graph = getattr(pipeline.vector_index, "_graph", None)
        return {
            "vector_backend": pipeline.vector_backend,
            "k_rrf": pipeline.k_rrf,
            "weight_bm25": pipeline.weight_bm25,
            "weight_vector": pipeline.weight_vector,
            "bm25": {
                "k1": pipeline.bm25_index.k1,
                "b": pipeline.bm25_index.b,
                "analyzer": pipeline.bm25_index.analyzer,
            },
            "ann": {
                "m": getattr(graph, "m", 16),
                "ef_construction": getattr(graph, "ef_construction", 200),
                "ef_search": getattr(pipeline.vector_index, "ef_search", 50),
            },
        }

    @classmethod
    def create(
        cls,
        path: str | Path,
        vector_backend: str = "hnsw",
        embed_fn: EmbedFn = hashing_embed,
        **pipeline_kwargs: Any,
    ) -> DiskIndex:
        """Create an empty index directory (fails if it already holds an index)."""
        if vector_backend not in _BACKENDS:
            raise ValueError(f"vector_backend must be one of {_BACKENDS}, got {vector_backend!r}")
        directory = Path(path)
        if (directory / "manifest.json").exists():
            raise StorageError(f"{directory} already contains an index; use DiskIndex.open")
        directory.mkdir(parents=True, exist_ok=True)
        pipeline = RagPipeline(embed_fn=embed_fn, vector_backend=vector_backend, **pipeline_kwargs)
        manifest = {
            "format": FORMAT,
            "ragforge_version": __version__,
            "dims": None,
            "rows": 0,
            "config": cls._config_of(pipeline),
        }
        lock_fd = cls._acquire_lock(directory)
        (directory / "chunks.jsonl").write_bytes(b"")
        (directory / "vectors.f32").write_bytes(b"")
        (directory / "dead_rows.json").write_text("[]", encoding="utf-8")
        cls._write_manifest(directory, manifest)
        kwargs = {k: v for k, v in pipeline_kwargs.items() if k != "ann_params"}
        return cls(directory, pipeline, None, manifest, set(), {}, lock_fd, True, embed_fn, kwargs)

    @classmethod
    def from_pipeline(cls, pipeline: RagPipeline, path: str | Path) -> DiskIndex:
        """Persist an in-memory pipeline (brute_force / numpy / hnsw backend) as a directory."""
        backend = pipeline.vector_backend
        if backend not in _BACKENDS:
            raise StorageError(
                f"DiskIndex supports {_BACKENDS}; got {backend!r} (use RagPipeline.save for JSON)"
            )
        graph = getattr(pipeline.vector_index, "_graph", None)
        disk = cls.create(
            path,
            vector_backend=backend,
            embed_fn=pipeline.embed_fn,
            chunker=pipeline.chunker,
            bm25_analyzer=pipeline.bm25_index.analyzer,
            k_rrf=pipeline.k_rrf,
            weight_bm25=pipeline.weight_bm25,
            weight_vector=pipeline.weight_vector,
            ann_params=(
                {"m": graph.m, "ef_construction": graph.ef_construction}
                if graph is not None
                else None
            ),
        )
        grouped: dict[str, list[Chunk]] = {}
        for chunk in pipeline.bm25_index._chunks:
            grouped.setdefault(chunk.doc_id, []).append(chunk)
        for chunks in grouped.values():
            disk._append(chunks, [pipeline.embed_fn(c.index_text) for c in chunks])
        disk.commit()
        return disk

    @classmethod
    def open(
        cls,
        path: str | Path,
        embed_fn: EmbedFn | None = None,
        readonly: bool = False,
        mmap: bool = True,
        **pipeline_kwargs: Any,
    ) -> DiskIndex:
        """Open an existing index directory; ``mmap=False`` reads vectors into memory instead."""
        directory = Path(path)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise StorageError(f"No index found at {directory} (missing manifest.json)")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT:
            raise StorageError(f"Unsupported index format {manifest.get('format')!r}")
        lock_fd = None if readonly else cls._acquire_lock(directory)
        try:
            return cls._load(
                directory, manifest, embed_fn or hashing_embed, lock_fd, mmap, pipeline_kwargs
            )
        except BaseException:
            cls._release_lock(lock_fd)
            raise

    @classmethod
    def _load(
        cls,
        directory: Path,
        manifest: dict[str, Any],
        embed_fn: EmbedFn,
        lock_fd: int | None,
        use_mmap: bool,
        pipeline_kwargs: dict[str, Any],
    ) -> DiskIndex:
        cfg, rows, dims = manifest["config"], manifest["rows"], manifest["dims"]
        backend = cfg["vector_backend"]
        dead = set(json.loads((directory / "dead_rows.json").read_text(encoding="utf-8")))

        chunks_by_row: dict[int, Chunk] = {}
        with open(directory / "chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["row"] >= rows:  # tail of an interrupted append: not committed
                    break
                chunks_by_row[record["row"]] = Chunk.from_dict(record["chunk"])
        if len(chunks_by_row) != rows:
            raise StorageError(
                f"chunks.jsonl has {len(chunks_by_row)} committed rows, manifest says {rows}"
            )

        store = VectorStore(directory / "vectors.f32", dims or 0, rows, use_mmap) if rows else None
        pipeline = RagPipeline(
            embed_fn=embed_fn,
            vector_backend=backend,
            k_rrf=cfg["k_rrf"],
            weight_bm25=cfg["weight_bm25"],
            weight_vector=cfg["weight_vector"],
            bm25_analyzer=cfg["bm25"]["analyzer"],
            bm25_k1=cfg["bm25"]["k1"],
            bm25_b=cfg["bm25"]["b"],
            ann_params=cfg["ann"] if backend == "hnsw" else None,
            **pipeline_kwargs,
        )
        live_rows = [r for r in range(rows) if r not in dead]
        live_chunks = [chunks_by_row[r] for r in live_rows]
        pipeline.bm25_index.add(live_chunks)
        if store is not None:
            cls._attach_vectors(
                pipeline, directory, store, backend, live_rows, live_chunks, dead, rows, dims
            )
        row_of_chunk = {chunks_by_row[r].id: r for r in range(rows)}
        return cls(
            directory, pipeline, store, manifest, dead, row_of_chunk, lock_fd, use_mmap, embed_fn,
            pipeline_kwargs,
        )  # fmt: skip

    @staticmethod
    def _attach_vectors(
        pipeline: RagPipeline,
        directory: Path,
        store: VectorStore,
        backend: str,
        live_rows: list[int],
        live_chunks: list[Chunk],
        dead: set[int],
        rows: int,
        dims: int,
    ) -> None:
        index: Any = pipeline.vector_index
        if backend == "hnsw":
            if (directory / "graph.json").exists():
                index._graph = _read_graph(directory, store)
                index._chunks_by_node = dict(zip(live_rows, live_chunks, strict=True))
            elif dead:
                raise StorageError("graph.json is missing but tombstones exist; cannot rebuild")
            else:  # first open after compaction (or a deleted graph): rebuild from the vectors
                index.add_vectors(live_chunks, [list(store[r]) for r in live_rows])
                index._graph._vectors = store
                _write_graph(directory, index._graph)
        elif backend == "numpy":
            import numpy as np

            matrix = np.memmap(directory / "vectors.f32", dtype="<f4", mode="r", shape=(rows, dims))
            index._chunks = live_chunks
            index._matrix = matrix if not dead else matrix[live_rows]
        else:
            index._chunks = live_chunks
            index._vectors = [store[r] for r in live_rows]

    # -- locking & manifest ----------------------------------------------------------

    @staticmethod
    def _acquire_lock(directory: Path) -> int | None:
        if fcntl is None:  # pragma: no cover
            return None
        fd = os.open(directory / "LOCK", os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise StorageError(f"{directory} is open for writing by another process") from exc
        return fd

    @staticmethod
    def _release_lock(fd: int | None) -> None:
        if fd is not None and fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @staticmethod
    def _write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
        tmp = directory / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, directory / "manifest.json")

    # -- mutation --------------------------------------------------------------------

    @property
    def readonly(self) -> bool:
        return self._lock_fd is None and fcntl is not None

    def _require_writable(self) -> None:
        if self.readonly:
            raise StorageError("index was opened readonly")

    def _append(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        units = [normalize_vector(v) for v in vectors]
        if self._dims is None:
            self._dims = len(units[0])
        elif len(units[0]) != self._dims:
            raise ValueError(
                f"Vector dimension mismatch: index has {self._dims}, got {len(units[0])}"
            )
        first = self._rows + len(self._pending)
        self.pipeline.bm25_index.add(chunks)
        index: Any = self.pipeline.vector_index
        index.add_vectors(chunks, units)  # HNSW also stores each row in the (overlay) VectorStore
        for offset, (chunk, vec) in enumerate(zip(chunks, units, strict=True)):
            self._pending.append((chunk, first + offset, vec))
            self._row_of_chunk[chunk.id] = first + offset

    def ingest(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Chunk, embed and index a document; it becomes durable at the next :meth:`commit`."""
        self._require_writable()
        chunks = self.pipeline.chunker.chunk(text, doc_id=doc_id, metadata=metadata)
        if chunks and self.pipeline.contextualizer is not None:
            for chunk in chunks:
                chunk.context = self.pipeline.contextualizer(text, chunk)
        self._append(chunks, [self._embed_fn(c.index_text) for c in chunks])
        return chunks

    def delete(self, doc_id: str) -> int:
        """Delete a document everywhere (rows become tombstones until :meth:`compact`)."""
        self._require_writable()
        rows = {
            self._row_of_chunk[c.id] for c in self.pipeline.bm25_index._chunks if c.doc_id == doc_id
        }
        count = self.pipeline.delete(doc_id)
        self._dead |= rows
        return count

    def commit(self) -> None:
        """Make appended rows, tombstones and the HNSW graph durable (atomic manifest swap)."""
        self._require_writable()
        if self._pending:
            dims = self._dims or 0
            with open(self.path / "vectors.f32", "r+b") as vf:
                vf.truncate(self._rows * dims * 4)  # drop any uncommitted tail
                vf.seek(0, os.SEEK_END)
                for _, _, vec in self._pending:
                    _pack(vec).tofile(vf)
                vf.flush()
                os.fsync(vf.fileno())
            self._truncate_chunks_to(self._rows)
            with open(self.path / "chunks.jsonl", "a", encoding="utf-8") as cf:
                for chunk, row, _ in self._pending:
                    cf.write(json.dumps({"row": row, "chunk": chunk.to_dict()}) + "\n")
                cf.flush()
                os.fsync(cf.fileno())
            self._rows += len(self._pending)
            self._pending = []
            self._remap_vectors()
        (self.path / "dead_rows.json").write_text(json.dumps(sorted(self._dead)), encoding="utf-8")
        index = self.pipeline.vector_index
        if isinstance(index, ApproxVectorIndex) and self._rows:
            _write_graph(self.path, index._graph)
        self._manifest.update(
            rows=self._rows,
            dims=self._dims,
            ragforge_version=__version__,
            config=self._config_of(self.pipeline),
        )
        self._write_manifest(self.path, self._manifest)

    def _remap_vectors(self) -> None:
        """Re-map ``vectors.f32`` after it grew so new rows are served from the mapping."""
        dims = self._dims or 0
        if self._store is None:
            self._store = VectorStore(self.path / "vectors.f32", dims, self._rows, self._use_mmap)
        else:
            self._store.reopen(self._rows, self._use_mmap)
        index: Any = self.pipeline.vector_index
        if isinstance(index, ApproxVectorIndex):
            index._graph._vectors = self._store  # type: ignore[assignment]
        elif self.pipeline.vector_backend == "numpy":
            import numpy as np

            matrix = np.memmap(
                self.path / "vectors.f32", dtype="<f4", mode="r", shape=(self._rows, dims)
            )
            live = [r for r in range(self._rows) if r not in self._dead]
            index._matrix = matrix if len(live) == self._rows else matrix[live]

    def _truncate_chunks_to(self, rows: int) -> None:
        """Cut ``chunks.jsonl`` back to the committed rows (removes an interrupted tail)."""
        keep = 0
        with open(self.path / "chunks.jsonl", "rb") as f:
            for line_no, line in enumerate(f):
                if line_no >= rows:
                    break
                keep += len(line)
        with open(self.path / "chunks.jsonl", "r+b") as f:
            f.truncate(keep)

    def compact(self) -> int:
        """Rewrite the files without tombstoned rows; returns how many rows were reclaimed."""
        self._require_writable()
        self.commit()
        reclaimed = len(self._dead)
        if not reclaimed:
            return 0
        assert self._store is not None
        live = sorted(
            ((self._row_of_chunk[c.id], c) for c in self.pipeline.bm25_index._chunks),
            key=lambda p: p[0],
        )
        tmp_vec, tmp_chunks = self.path / "vectors.f32.new", self.path / "chunks.jsonl.new"
        with open(tmp_vec, "wb") as vf, open(tmp_chunks, "w", encoding="utf-8") as cf:
            for new_row, (old_row, chunk) in enumerate(live):
                _pack(self._store[old_row]).tofile(vf)
                cf.write(json.dumps({"row": new_row, "chunk": chunk.to_dict()}) + "\n")
        self._store.close()
        os.replace(tmp_vec, self.path / "vectors.f32")
        os.replace(tmp_chunks, self.path / "chunks.jsonl")
        (self.path / "dead_rows.json").write_text("[]", encoding="utf-8")
        (self.path / "graph.json").unlink(missing_ok=True)
        (self.path / "graph.bin").unlink(missing_ok=True)
        manifest = dict(self._manifest, rows=len(live))
        self._write_manifest(self.path, manifest)
        fresh = DiskIndex._load(
            self.path, manifest, self._embed_fn, self._lock_fd, self._use_mmap, self._kwargs
        )
        self.__dict__.update(fresh.__dict__)
        return reclaimed

    # -- lifecycle -------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Row accounting and file sizes in bytes."""
        return {
            "rows": self._rows + len(self._pending),
            "dead_rows": len(self._dead),
            "live_chunks": self.pipeline.chunk_count,
            "dims": self._dims,
            "backend": self.pipeline.vector_backend,
            "bytes": {p.name: p.stat().st_size for p in self.path.iterdir() if p.is_file()},
        }

    def close(self) -> None:
        """Commit pending writes (if writable), unmap the vectors and release the lock."""
        if self._lock_fd is not None and (self._pending or self._dead):
            self.commit()
        if self._store is not None:
            self._store.close()
        self._release_lock(self._lock_fd)
        self._lock_fd = None

    def __enter__(self) -> DiskIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
