"""Tenant isolation through every access path: retrieval, answers, streams, agents, MCP, files."""

from __future__ import annotations

import pytest

from ragforge import AgenticRag, RagPipeline
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.mcp_server import MCPServer
from ragforge.tenancy import MultiTenantRag, ScopedPipeline, TenantError, validate_tenant_id


def _factory(tenant_id: str) -> RagPipeline:
    return RagPipeline(chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=0))


# --------------------------------------------------------------------- id validation


@pytest.mark.parametrize("good", ["acme", "A", "tenant-1", "t_2.eu", "9lives", "a" * 64])
def test_valid_tenant_ids(good):
    assert validate_tenant_id(good) == good


@pytest.mark.parametrize(
    "bad",
    [
        "",
        ".hidden",
        "-x",
        "a/b",
        "a\\b",
        "../etc",
        "a..b",
        "a b",
        "x" * 65,
        "ünïcode",
        "a\x00b",
        None,
        5,
    ],
)
def test_invalid_tenant_ids_are_rejected(bad):
    with pytest.raises(TenantError):
        validate_tenant_id(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------- index per tenant


def test_tenants_have_fully_separate_indexes():
    rag = MultiTenantRag(_factory)
    rag.ingest("acme", "plan", "ACME secret roadmap: launch the zeppelin in March.")
    rag.ingest("globex", "plan", "Globex public handbook about bicycles.")

    assert [c.doc_id for c in rag.retrieve("acme", "zeppelin roadmap", k=3)] == ["plan"]
    assert all("zeppelin" not in c.text for c in rag.retrieve("globex", "zeppelin roadmap", k=3))
    assert "zeppelin" not in rag.answer("globex", "zeppelin roadmap").answer
    assert rag.tenants == ["acme", "globex"]
    assert rag.stats() == {
        "acme": {"documents": 1, "chunks": 1},
        "globex": {"documents": 1, "chunks": 1},
    }


def test_unknown_tenants_are_not_created_by_reads():
    rag = MultiTenantRag(_factory)
    with pytest.raises(TenantError, match="Unknown tenant"):
        rag.retrieve("ghost", "anything")
    assert rag.tenants == []


def test_quotas_for_tenants_and_chunks():
    rag = MultiTenantRag(_factory, max_tenants=2, max_chunks_per_tenant=2)
    rag.ingest("a", "d1", "one")
    rag.ingest("a", "d2", "two")
    with pytest.raises(TenantError, match="quota"):
        rag.ingest("a", "d3", "three")
    assert rag.get("a").chunk_count == 2  # the refused document left no trace
    rag.ingest("b", "d1", "x")
    with pytest.raises(TenantError, match="limit"):
        rag.ingest("c", "d1", "x")


def test_persistence_evict_reload_and_erasure(tmp_path):
    rag = MultiTenantRag(_factory, root=tmp_path)
    rag.ingest("acme", "d", "persistent zeppelin fact")
    rag.ingest("globex", "d", "other tenant bicycles")
    saved = rag.save("acme")
    assert saved == tmp_path / "acme.json" and saved.exists()
    rag.save_all()

    rag.evict("acme")
    assert "acme" not in rag._tenants and "acme" in rag.tenants  # still known via its snapshot
    assert rag.retrieve("acme", "zeppelin", k=1)[0].doc_id == "d"  # lazily reloaded

    assert rag.erase("acme") is True
    assert not (tmp_path / "acme.json").exists() and "acme" not in rag.tenants
    assert rag.erase("acme") is False
    with pytest.raises(TenantError):
        rag.retrieve("acme", "zeppelin")
    assert rag.retrieve("globex", "bicycles", k=1)  # neighbours untouched


def test_traversal_ids_cannot_escape_the_root(tmp_path):
    rag = MultiTenantRag(_factory, root=tmp_path / "root")
    for bad in ["../evil", "a/b", ".."]:
        with pytest.raises(TenantError):
            rag.ingest(bad, "d", "x")
        with pytest.raises(TenantError):
            rag.erase(bad)
    assert not (tmp_path / "evil.json").exists()


def test_save_requires_a_root_and_evict_without_root():
    rag = MultiTenantRag(_factory)
    rag.ingest("a", "d", "x")
    with pytest.raises(TenantError, match="root"):
        rag.save("a")
    rag.evict("a")
    assert rag.tenants == []


# ---------------------------------------------------------------- shared index, scoped


def _shared() -> tuple[RagPipeline, ScopedPipeline, ScopedPipeline]:
    shared = RagPipeline(
        chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=0),
        generate_fn=lambda q, ctx: ctx[0] if ctx else "no info",
    )
    a, b = ScopedPipeline(shared, "acme"), ScopedPipeline(shared, "globex")
    a.ingest("plan", "ACME secret roadmap: launch the zeppelin in March.", {"team": "x"})
    b.ingest("plan", "Globex handbook about bicycles and zeppelin-free commuting.")
    return shared, a, b


def test_scoped_views_isolate_retrieval_and_hide_internal_prefixes():
    _, a, b = _shared()
    hits_a = a.retrieve("secret roadmap zeppelin", k=5)
    assert [h.doc_id for h in hits_a] == ["plan"] and "ACME" in hits_a[0].text
    assert hits_a[0].metadata == {"team": "x"}  # tenant stamp is internal
    assert all("ACME" not in h.text for h in b.retrieve("secret roadmap zeppelin", k=5))
    assert (a.document_count, a.chunk_count, b.document_count) == (1, 1, 1)


def test_scoped_answer_stream_and_agent_cannot_cross_tenants():
    _, a, b = _shared()
    assert "ACME" in a.answer("secret roadmap", k=3).answer
    assert "ACME" not in b.answer("secret roadmap", k=3).answer

    events = list(b.answer_stream("secret roadmap zeppelin", k=3))
    assert all("ACME" not in str(e) for e in events)
    done = events[-1]["answer"]
    assert all(c.doc_id == "plan" for c in done.contexts)

    result = AgenticRag(b).run("secret roadmap ACME zeppelin", k=3)
    assert all("ACME" not in c.text for c in result.answer.contexts)


def test_scoped_raw_indexes_are_filtered_too():
    _, a, b = _shared()
    for scoped, other_marker in ((a, "Globex"), (b, "ACME")):
        for index in (scoped.bm25_index, scoped.vector_index):
            assert all(other_marker not in h.chunk.text for h in index.search("zeppelin", k=10))


def test_user_filters_compose_with_the_tenant_filter_and_cannot_remove_it():
    shared, a, _ = _shared()
    permissive = a.retrieve("zeppelin", k=10, filter_fn=lambda c: True)
    assert all("ACME" in h.text for h in permissive)
    team_x = a.retrieve("zeppelin", k=10, filter_fn=lambda c: c.metadata.get("team") == "x")
    assert len(team_x) == 1
    assert a.retrieve("zeppelin", k=10, filter_fn=lambda c: False) == []


def test_delete_is_scoped_by_namespaced_document_ids():
    shared, a, b = _shared()
    assert a.delete("plan") == 1 and a.delete("plan") == 0
    assert b.retrieve("bicycles", k=1)  # globex's document with the same public id survives
    assert shared.document_count == 1


def test_scoped_view_rejects_bad_tenant_ids_and_ignores_forged_metadata():
    shared = RagPipeline()
    with pytest.raises(TenantError):
        ScopedPipeline(shared, "../x")
    a = ScopedPipeline(shared, "acme")
    mallory = ScopedPipeline(shared, "mallory")
    # A caller cannot smuggle chunks into another tenant by forging the stamp or the id prefix.
    mallory.ingest("doc", "forged zeppelin", {"_tenant": "acme"})
    mallory.ingest("acme::evil", "zeppelin from mallory")
    assert a.retrieve("zeppelin", k=5) == []
    assert {h.doc_id for h in mallory.retrieve("zeppelin", k=5)} == {"doc", "acme::evil"}


def test_scoped_pipeline_works_behind_the_mcp_server():
    _, a, b = _shared()
    server = MCPServer(b, allow_ingest=True)

    def call(name, **args):
        reply = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": args},
            }
        )
        return reply["result"]["structuredContent"]

    for mode in ("hybrid", "bm25", "vector"):
        results = call("search_knowledge_base", query="secret roadmap zeppelin", mode=mode)[
            "results"
        ]
        assert all("ACME" not in r["text"] for r in results)
    assert "ACME" not in str(call("ask_knowledge_base", question="secret roadmap"))
    assert call("knowledge_base_stats")["documents"] == 1
    assert (
        call("ingest_document", doc_id="plan", text="Replaced globex doc")["chunks_replaced"] == 1
    )
    assert a.retrieve("secret roadmap", k=1)[0].text.startswith("ACME")  # acme's "plan" intact


def test_mcp_resources_are_tenant_scoped():
    _, a, b = _shared()
    b.ingest("handbook2", "Second globex document")
    server = MCPServer(b)
    listing = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "resources/list"})
    assert [r["name"] for r in listing["result"]["resources"]] == ["handbook2", "plan"]
    assert all("ACME" not in r["description"] for r in listing["result"]["resources"])
    read = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "resources/read",
            "params": {"uri": "ragforge://doc/plan"},
        }
    )
    assert "ACME" not in read["result"]["contents"][0]["text"]
