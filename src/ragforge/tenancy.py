"""Multi-tenancy: strict data isolation between customers sharing one deployment.

A RAG service that serves several customers must guarantee that tenant A's question can
never retrieve tenant B's documents -- through *any* path: retrieval, answers, streaming,
the agent, the MCP tools. Two isolation models are provided, with different trade-offs:

:class:`MultiTenantRag` -- **index per tenant** (strongest isolation)
    Each tenant gets its own :class:`~ragforge.pipeline.RagPipeline` (separate BM25 and
    vector indexes), created lazily from a factory and persisted as
    ``<root>/<tenant>.json``. There is no shared structure to leak through, per-tenant
    quotas are trivial, and **erasure** (GDPR art. 17) is deleting one object and one file.
    Costs: memory per tenant, and one embedding of the corpus per tenant.

:class:`ScopedPipeline` -- **shared index, enforced filter** (densest packing)
    A view over one shared pipeline that (a) namespaces every document id with the tenant,
    (b) stamps the tenant into chunk metadata, and (c) wraps every ``filter_fn`` so a tenant
    filter always applies and cannot be removed by the caller. It quacks like a pipeline
    (``retrieve``, ``answer``, ``answer_stream``, ``bm25_index``, ``vector_index``, ...), so
    :class:`~ragforge.agent.AgenticRag` and :class:`~ragforge.mcp_server.MCPServer` accept
    it unchanged and inherit the isolation.

Tenant ids are validated against a strict pattern (no path separators, no leading dot) so
they are safe to use in file names.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from ragforge.chunking import Chunk
from ragforge.index import FilterFn, ScoredChunk
from ragforge.pipeline import RagAnswer, RagPipeline, RetrievedChunk
from ragforge.telemetry import Tracer

TENANT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_NAMESPACE_SEP = "::"
TENANT_KEY = "_tenant"


class TenantError(Exception):
    """Invalid tenant id, unknown tenant, or exceeded quota."""


def validate_tenant_id(tenant_id: str) -> str:
    """Return ``tenant_id`` if it is a safe identifier, else raise :class:`TenantError`."""
    if not isinstance(tenant_id, str) or not TENANT_ID.match(tenant_id) or ".." in tenant_id:
        raise TenantError(
            f"Invalid tenant id {tenant_id!r}: use 1-64 of A-Z a-z 0-9 _ . - (no leading symbol)"
        )
    return tenant_id


class MultiTenantRag:
    """Registry of per-tenant pipelines with quotas, persistence and erasure.

    Args:
        factory: ``tenant_id -> RagPipeline`` used for tenants without saved state. Give each
            tenant its own pipeline (do not return a shared instance).
        root: Directory for ``<tenant>.json`` snapshots; ``None`` keeps tenants in memory.
        max_tenants: Maximum number of live tenants.
        max_chunks_per_tenant: Quota enforced on :meth:`ingest` (``None`` = unlimited).
    """

    def __init__(
        self,
        factory: Callable[[str], RagPipeline],
        root: str | Path | None = None,
        max_tenants: int = 1000,
        max_chunks_per_tenant: int | None = None,
    ) -> None:
        self.factory = factory
        self.root = Path(root) if root is not None else None
        self.max_tenants = max_tenants
        self.max_chunks_per_tenant = max_chunks_per_tenant
        self._tenants: dict[str, RagPipeline] = {}
        self._lock = threading.RLock()
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, tenant_id: str) -> Path | None:
        return None if self.root is None else self.root / f"{validate_tenant_id(tenant_id)}.json"

    @property
    def tenants(self) -> list[str]:
        """Ids of tenants currently loaded or persisted."""
        with self._lock:
            ids = set(self._tenants)
            if self.root is not None:
                ids |= {p.stem for p in self.root.glob("*.json") if TENANT_ID.match(p.stem)}
            return sorted(ids)

    def get(self, tenant_id: str, create: bool = True) -> RagPipeline:
        """The tenant's pipeline: loaded from memory, from disk, or created via the factory."""
        validate_tenant_id(tenant_id)
        with self._lock:
            pipeline = self._tenants.get(tenant_id)
            if pipeline is not None:
                return pipeline
            path = self._path(tenant_id)
            exists = path is not None and path.exists()
            if not exists and not create:
                raise TenantError(f"Unknown tenant {tenant_id!r}")
            if len(self._tenants) >= self.max_tenants:
                raise TenantError(f"Tenant limit reached ({self.max_tenants})")
            fresh = self.factory(tenant_id)
            if exists:
                assert path is not None
                fresh = RagPipeline.load(
                    path,
                    chunker=fresh.chunker,
                    embed_fn=fresh.embed_fn,
                    reranker=fresh.reranker,
                    generate_fn=fresh.generate_fn,
                    guardrails=fresh.guardrails,
                    tracer=fresh.tracer,
                )
            self._tenants[tenant_id] = fresh
            return fresh

    def ingest(
        self, tenant_id: str, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> list[Chunk]:
        """Ingest into one tenant's index, enforcing the chunk quota."""
        pipeline = self.get(tenant_id)
        with self._lock:
            if self.max_chunks_per_tenant is not None:
                prospective = len(pipeline.chunker.chunk(text, doc_id=doc_id, metadata=metadata))
                if pipeline.chunk_count + prospective > self.max_chunks_per_tenant:
                    raise TenantError(
                        f"Tenant {tenant_id!r} would exceed its quota of "
                        f"{self.max_chunks_per_tenant} chunks"
                    )
            return pipeline.ingest(doc_id, text, metadata)

    def retrieve(self, tenant_id: str, query: str, **kwargs: Any) -> list[RetrievedChunk]:
        return self.get(tenant_id, create=False).retrieve(query, **kwargs)

    def answer(self, tenant_id: str, query: str, **kwargs: Any) -> RagAnswer:
        return self.get(tenant_id, create=False).answer(query, **kwargs)

    def save(self, tenant_id: str) -> Path:
        """Snapshot one tenant to ``<root>/<tenant>.json`` (atomic replace)."""
        path = self._path(tenant_id)
        if path is None:
            raise TenantError("MultiTenantRag was created without a root directory")
        pipeline = self.get(tenant_id, create=False)
        tmp = path.with_suffix(".json.tmp")
        pipeline.save(tmp)
        os.replace(tmp, path)
        return path

    def save_all(self) -> list[Path]:
        return [self.save(t) for t in list(self._tenants)]

    def evict(self, tenant_id: str, save: bool = True) -> None:
        """Drop a tenant from memory (snapshotting first when a root is configured)."""
        with self._lock:
            if tenant_id in self._tenants and save and self.root is not None:
                self.save(tenant_id)
            self._tenants.pop(tenant_id, None)

    def erase(self, tenant_id: str) -> bool:
        """Irreversibly delete a tenant's data (memory and snapshot). Returns whether it existed."""
        with self._lock:
            existed = self._tenants.pop(validate_tenant_id(tenant_id), None) is not None
            path = self._path(tenant_id)
            if path is not None and path.exists():
                path.unlink()
                existed = True
            return existed

    def stats(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {
                t: {"documents": p.document_count, "chunks": p.chunk_count}
                for t, p in sorted(self._tenants.items())
            }


class _ScopedIndex:
    """BM25 / vector index facade that always applies the tenant filter."""

    def __init__(self, scoped: ScopedPipeline, index: Any) -> None:
        self._scoped, self._index = scoped, index

    def search(
        self, query: str, k: int = 5, filter_fn: FilterFn | None = None
    ) -> list[ScoredChunk]:
        hits = self._index.search(query, k=k, filter_fn=self._scoped._wrap(filter_fn))
        return [replace(h, chunk=self._scoped._public_chunk(h.chunk)) for h in hits]


class ScopedPipeline:
    """A tenant-scoped view over a shared :class:`RagPipeline` (see module docstring)."""

    def __init__(self, pipeline: RagPipeline, tenant_id: str) -> None:
        self._pipeline = pipeline
        self.tenant_id = validate_tenant_id(tenant_id)
        self._prefix = f"{self.tenant_id}{_NAMESPACE_SEP}"
        self.bm25_index = _ScopedIndex(self, pipeline.bm25_index)
        self.vector_index = _ScopedIndex(self, pipeline.vector_index)

    # -- pipeline-compatible surface ---------------------------------------------------

    @property
    def tracer(self) -> Tracer:
        return self._pipeline.tracer

    @property
    def prompt_formatter(self) -> Any:
        return self._pipeline.prompt_formatter

    @property
    def generate_fn(self) -> Any:
        return self._pipeline.generate_fn

    @property
    def vector_backend(self) -> str:
        return self._pipeline.vector_backend

    def _owned(self, chunk: Chunk) -> bool:
        return chunk.metadata.get(TENANT_KEY) == self.tenant_id and chunk.doc_id.startswith(
            self._prefix
        )

    def _wrap(self, user_filter: FilterFn | None) -> FilterFn:
        if user_filter is None:
            return self._owned
        return lambda chunk: self._owned(chunk) and user_filter(self._public_chunk(chunk))

    def _public_chunk(self, chunk: Chunk) -> Chunk:
        metadata = {k: v for k, v in chunk.metadata.items() if k != TENANT_KEY}
        return replace(chunk, doc_id=chunk.doc_id.removeprefix(self._prefix), metadata=metadata)

    def _public(self, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        out = []
        for item in items:
            out.append(
                replace(
                    item,
                    doc_id=item.doc_id.removeprefix(self._prefix),
                    metadata={k: v for k, v in item.metadata.items() if k != TENANT_KEY},
                )
            )
        return out

    @property
    def chunks(self) -> list[Chunk]:
        """This tenant's chunks (public ids, tenant stamp removed)."""
        return [self._public_chunk(c) for c in self._pipeline.chunks if self._owned(c)]

    @property
    def document_count(self) -> int:
        return len({c.doc_id for c in self._pipeline.chunks if self._owned(c)})

    @property
    def chunk_count(self) -> int:
        return sum(1 for c in self._pipeline.chunks if self._owned(c))

    def ingest(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        stamped = {**(metadata or {}), TENANT_KEY: self.tenant_id}
        chunks = self._pipeline.ingest(self._prefix + doc_id, text, stamped)
        return [self._public_chunk(c) for c in chunks]

    def delete(self, doc_id: str) -> int:
        """Delete this tenant's document ``doc_id`` (never another tenant's)."""
        return self._pipeline.delete(self._prefix + doc_id)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        candidate_pool: int = 20,
        filter_fn: FilterFn | None = None,
    ) -> list[RetrievedChunk]:
        return self._public(
            self._pipeline.retrieve(
                query, k=k, candidate_pool=candidate_pool, filter_fn=self._wrap(filter_fn)
            )
        )

    def answer(
        self,
        query: str,
        k: int = 5,
        candidate_pool: int = 20,
        filter_fn: FilterFn | None = None,
    ) -> RagAnswer:
        result = self._pipeline.answer(
            query, k=k, candidate_pool=candidate_pool, filter_fn=self._wrap(filter_fn)
        )
        result.contexts = self._public(result.contexts)
        return result

    def answer_stream(
        self,
        query: str,
        k: int = 5,
        candidate_pool: int = 20,
        filter_fn: FilterFn | None = None,
    ) -> Iterator[dict[str, Any]]:
        for event in self._pipeline.answer_stream(
            query, k=k, candidate_pool=candidate_pool, filter_fn=self._wrap(filter_fn)
        ):
            if event["type"] == "contexts":
                for ctx in event["contexts"]:
                    ctx["doc_id"] = ctx["doc_id"].removeprefix(self._prefix)
                    ctx["metadata"] = {k: v for k, v in ctx["metadata"].items() if k != TENANT_KEY}
            elif event["type"] == "done":
                event["answer"].contexts = self._public(event["answer"].contexts)
            yield event
