"""MCP server tests: JSON-RPC conformance, tools, safety defaults, and a real stdio round-trip."""

from __future__ import annotations

import io
import json

import pytest

from ragforge import RagPipeline
from ragforge._version import __version__
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.mcp_server import MAX_K, MCPServer, serve_stdio


def _pipeline() -> RagPipeline:
    pipeline = RagPipeline(chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=0))
    pipeline.ingest("eiffel", "The Eiffel Tower is in Paris. It was completed in 1889.")
    pipeline.ingest("python", "Python was created by Guido van Rossum in 1991.")
    return pipeline


def rpc(server: MCPServer, method: str, params=None, msg_id=1):
    message = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        message["params"] = params
    return server.handle_message(message)


def call(server: MCPServer, name: str, arguments=None):
    response = rpc(server, "tools/call", {"name": name, "arguments": arguments or {}})
    return response["result"]


def test_initialize_negotiates_version_and_advertises_capabilities():
    result = rpc(MCPServer(_pipeline()), "initialize", {"protocolVersion": "2024-11-05"})["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert result["capabilities"] == {
        "tools": {"listChanged": False},
        "resources": {"subscribe": False, "listChanged": False},
        "prompts": {"listChanged": False},
    }
    assert result["serverInfo"] == {"name": "ragforge", "version": __version__}

    unknown = rpc(MCPServer(_pipeline()), "initialize", {"protocolVersion": "1999-01-01"})
    assert unknown["result"]["protocolVersion"] == "2025-06-18"


def test_notifications_get_no_response_and_ping_works():
    server = MCPServer(_pipeline())
    assert server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert rpc(server, "ping")["result"] == {}


def test_tools_list_hides_ingest_by_default():
    names = [t["name"] for t in rpc(MCPServer(_pipeline()), "tools/list")["result"]["tools"]]
    assert names == ["search_knowledge_base", "ask_knowledge_base", "knowledge_base_stats"]
    with_ingest = rpc(MCPServer(_pipeline(), allow_ingest=True), "tools/list")["result"]["tools"]
    assert "ingest_document" in [t["name"] for t in with_ingest]
    assert all("inputSchema" in t and t["description"] for t in with_ingest)


def test_search_returns_ranked_passages_with_offsets():
    result = call(
        MCPServer(_pipeline()), "search_knowledge_base", {"query": "Eiffel Tower", "k": 1}
    )

    assert result["isError"] is False
    payload = result["structuredContent"]
    assert payload["count"] == 1 and payload["results"][0]["doc_id"] == "eiffel"
    assert json.loads(result["content"][0]["text"]) == payload


@pytest.mark.parametrize("mode", ["bm25", "vector"])
def test_search_modes(mode):
    payload = call(
        MCPServer(_pipeline()), "search_knowledge_base", {"query": "Guido", "mode": mode}
    )["structuredContent"]
    assert payload["mode"] == mode and payload["results"][0]["doc_id"] == "python"


def test_ask_returns_citations_and_groundedness():
    payload = call(
        MCPServer(_pipeline()), "ask_knowledge_base", {"question": "Who created Python?"}
    )["structuredContent"]
    assert payload["groundedness"] == 1.0
    assert payload["sentences"][0]["citations"][0]["doc_id"] == "python"
    assert payload["sources"][0]["doc_id"] == "python"


def test_stats_and_ingest_roundtrip():
    server = MCPServer(_pipeline(), allow_ingest=True)
    assert call(server, "knowledge_base_stats")["structuredContent"] == {
        "documents": 2,
        "chunks": 2,
        "vector_backend": "brute_force",
        "ingest_enabled": True,
    }
    added = call(
        server, "ingest_document", {"doc_id": "rust", "text": "Rust is a systems language."}
    )
    assert added["structuredContent"] == {"doc_id": "rust", "chunks_added": 1, "chunks_replaced": 0}
    replaced = call(
        server, "ingest_document", {"doc_id": "rust", "text": "Rust has a borrow checker."}
    )
    assert replaced["structuredContent"]["chunks_replaced"] == 1
    hit = call(server, "search_knowledge_base", {"query": "borrow checker", "k": 1})
    assert hit["structuredContent"]["results"][0]["doc_id"] == "rust"


def test_ingest_is_rejected_when_not_enabled():
    server = MCPServer(_pipeline())
    response = rpc(
        server, "tools/call", {"name": "ingest_document", "arguments": {"doc_id": "x", "text": "y"}}
    )
    assert response["error"]["code"] == -32602
    assert server.pipeline.document_count == 2


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("search_knowledge_base", {}),
        ("search_knowledge_base", {"query": "   "}),
        ("search_knowledge_base", {"query": "x", "k": 0}),
        ("search_knowledge_base", {"query": "x", "k": MAX_K + 1}),
        ("search_knowledge_base", {"query": "x", "k": True}),
        ("search_knowledge_base", {"query": "x", "mode": "psychic"}),
        ("ask_knowledge_base", {"question": 5}),
    ],
)
def test_bad_tool_arguments_are_reported_as_tool_errors(name, arguments):
    result = call(MCPServer(_pipeline()), name, arguments)
    assert result["isError"] is True and result["content"][0]["text"]


def test_ingest_validation():
    server = MCPServer(_pipeline(), allow_ingest=True)
    assert call(server, "ingest_document", {"doc_id": "a"})["isError"] is True
    bad_meta = call(server, "ingest_document", {"doc_id": "a", "text": "t", "metadata": [1]})
    assert bad_meta["isError"] is True


def test_unexpected_tool_exception_becomes_error_result(monkeypatch):
    server = MCPServer(_pipeline())
    monkeypatch.setattr(server.pipeline, "retrieve", lambda *a, **k: 1 / 0)
    result = call(server, "search_knowledge_base", {"query": "x"})
    assert result["isError"] is True and "Tool failed" in result["content"][0]["text"]


def test_protocol_errors():
    server = MCPServer(_pipeline())
    assert rpc(server, "no/such/method")["error"]["code"] == -32601
    assert rpc(server, "tools/call", {"arguments": {}})["error"]["code"] == -32602
    assert rpc(server, "tools/call", {"name": "nope"})["error"]["code"] == -32602
    assert (
        rpc(server, "tools/call", {"name": "knowledge_base_stats", "arguments": []})["error"][
            "code"
        ]
        == -32602
    )
    assert server.handle_message([1])["error"]["code"] == -32600
    assert (
        server.handle_message({"jsonrpc": "1.0", "id": 1, "method": "ping"})["error"]["code"]
        == -32600
    )
    assert server.handle_message({"jsonrpc": "2.0", "id": 9})["error"]["code"] == -32600
    assert server.handle_message({"jsonrpc": "2.0", "id": 1, "result": {}}) is None
    assert (
        server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": 3})["error"][
            "code"
        ]
        == -32602
    )
    assert server.handle_message({"jsonrpc": "2.0", "method": "unknown/note"}) is None


def test_parse_error_response_has_null_id():
    response = json.loads(MCPServer(_pipeline()).handle_line("{not json"))
    assert response["error"]["code"] == -32700 and response["id"] is None


def test_resources_list_read_and_templates():
    server = MCPServer(_pipeline())
    listed = rpc(server, "resources/list")["result"]
    assert [r["name"] for r in listed["resources"]] == ["eiffel", "python"]
    first = listed["resources"][0]
    assert first["uri"] == "ragforge://doc/eiffel" and first["mimeType"] == "text/plain"
    assert first["size"] == len(b"The Eiffel Tower is in Paris. It was completed in 1889.")
    assert "nextCursor" not in listed

    contents = rpc(server, "resources/read", {"uri": first["uri"]})["result"]["contents"]
    assert contents == [
        {
            "uri": "ragforge://doc/eiffel",
            "mimeType": "text/plain",
            "text": "The Eiffel Tower is in Paris. It was completed in 1889.",
        }
    ]
    templates = rpc(server, "resources/templates/list")["result"]["resourceTemplates"]
    assert templates[0]["uriTemplate"] == "ragforge://doc/{doc_id}"


def test_resource_errors_follow_the_spec():
    server = MCPServer(_pipeline())
    missing = rpc(server, "resources/read", {"uri": "ragforge://doc/nope"})["error"]
    assert missing["code"] == -32002 and missing["data"] == {"uri": "ragforge://doc/nope"}
    assert rpc(server, "resources/read", {"uri": "https://elsewhere/x"})["error"]["code"] == -32002
    assert rpc(server, "resources/read", {})["error"]["code"] == -32602
    assert rpc(server, "resources/list", {"cursor": "!!!"})["error"]["code"] == -32602
    note = {"jsonrpc": "2.0", "method": "resources/read", "params": {"uri": "ragforge://doc/x"}}
    assert server.handle_message(note) is None


def test_document_ids_are_url_encoded_and_overlap_is_deduplicated():
    pipeline = RagPipeline(chunker=RecursiveCharacterChunker(chunk_size=40, chunk_overlap=10))
    text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron."
    pipeline.ingest("my doc/1", text)
    server = MCPServer(pipeline)
    uri = rpc(server, "resources/list")["result"]["resources"][0]["uri"]
    assert uri == "ragforge://doc/my%20doc%2F1"
    body = rpc(server, "resources/read", {"uri": uri})["result"]["contents"][0]["text"]
    assert body.replace("\n", " ").split() == text.split()  # overlap removed, order kept


def test_resource_listing_is_paginated():
    pipeline = RagPipeline()
    for i in range(130):
        pipeline.ingest(f"d{i:03}", f"document number {i}")
    server = MCPServer(pipeline)
    page1 = rpc(server, "resources/list")["result"]
    assert len(page1["resources"]) == 100 and "nextCursor" in page1
    page2 = rpc(server, "resources/list", {"cursor": page1["nextCursor"]})["result"]
    assert len(page2["resources"]) == 30 and "nextCursor" not in page2
    assert not {r["name"] for r in page1["resources"]} & {r["name"] for r in page2["resources"]}


def test_prompts_list_and_get():
    server = MCPServer(_pipeline())
    prompts = rpc(server, "prompts/list")["result"]["prompts"]
    assert {p["name"] for p in prompts} == {"answer_with_citations", "summarize_document"}
    assert all(p["arguments"][0]["required"] for p in prompts)

    ask = rpc(
        server, "prompts/get", {"name": "answer_with_citations", "arguments": {"question": "Why?"}}
    )
    message = ask["result"]["messages"][0]
    assert message["role"] == "user" and "Question: Why?" in message["content"]["text"]
    assert "ask_knowledge_base" in message["content"]["text"]

    summary = rpc(
        server, "prompts/get", {"name": "summarize_document", "arguments": {"doc_id": "python"}}
    )
    first, second = summary["result"]["messages"]
    assert first["content"]["type"] == "resource"
    assert first["content"]["resource"]["text"].startswith("Python was created")
    assert second["content"]["type"] == "text"


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("answer_with_citations", {}),
        ("answer_with_citations", {"question": "  "}),
        ("summarize_document", {}),
        ("summarize_document", {"doc_id": "ghost"}),
        ("nope", {}),
        ("answer_with_citations", [1]),
    ],
)
def test_prompt_errors(name, arguments):
    response = rpc(MCPServer(_pipeline()), "prompts/get", {"name": name, "arguments": arguments})
    assert response["error"]["code"] == -32602


def test_serve_stdio_processes_stream_and_skips_blank_lines_and_notifications():
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        "",
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ]
    out = io.StringIO()
    serve_stdio(_pipeline(), stdin=io.StringIO("\n".join(lines) + "\n"), stdout=out)

    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, 2]
