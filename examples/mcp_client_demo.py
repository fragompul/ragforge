"""Talk to a ragforge MCP server exactly like an AI assistant would.

Builds an index, launches ``ragforge mcp`` as a subprocess, and drives it over
stdio with raw JSON-RPC: initialize -> tools/list -> tools/call. This is the
whole wire protocol Claude Desktop / Claude Code use under the hood.

Run with: python examples/mcp_client_demo.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from ragforge import RagPipeline
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.sample import SAMPLE_DOCUMENTS


def main() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "kb.json"
        pipeline = RagPipeline(
            chunker=RecursiveCharacterChunker(chunk_size=500, chunk_overlap=0),
            bm25_analyzer="english",
        )
        pipeline.ingest_batch(SAMPLE_DOCUMENTS)
        pipeline.save(index)

        server = subprocess.Popen(
            [sys.executable, "-m", "ragforge.cli", "mcp", "--index", str(index)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        assert server.stdin is not None and server.stdout is not None

        def rpc(method: str, params: dict | None = None, notify: bool = False) -> dict:
            message: dict = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                message["params"] = params
            if not notify:
                message["id"] = rpc.counter  # type: ignore[attr-defined]
                rpc.counter += 1  # type: ignore[attr-defined]
            server.stdin.write(json.dumps(message) + "\n")
            server.stdin.flush()
            return {} if notify else json.loads(server.stdout.readline())

        rpc.counter = 1  # type: ignore[attr-defined]
        init = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
        rpc("notifications/initialized", notify=True)
        tools = rpc("tools/list")["result"]["tools"]
        answer = rpc(
            "tools/call",
            {
                "name": "ask_knowledge_base",
                "arguments": {"question": "What does error ERR-4012 mean?"},
            },
        )["result"]["structuredContent"]
        server.stdin.close()
        server.wait(timeout=10)

    print("server:", init["result"]["serverInfo"])
    print("tools :", [t["name"] for t in tools])
    print("answer:", answer["answer"])
    print("cited :", answer["sentences"][0]["citations"][0]["doc_id"])
    return {
        "server": init["result"]["serverInfo"]["name"],
        "tools": [t["name"] for t in tools],
        "answer": answer["answer"],
        "cited_doc": answer["sentences"][0]["citations"][0]["doc_id"],
    }


if __name__ == "__main__":
    main()
