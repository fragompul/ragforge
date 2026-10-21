"""A zero-dependency Model Context Protocol (MCP) server for a ragforge index.

`MCP <https://modelcontextprotocol.io>`_ is the open standard that lets AI
assistants (Claude Desktop, Claude Code, IDE agents, ...) call external tools.
This module exposes a :class:`~ragforge.pipeline.RagPipeline` as MCP tools over
the **stdio transport** -- newline-delimited JSON-RPC 2.0 -- implemented from
scratch with the standard library, in keeping with ragforge's zero-dependency
core.

Tools:

``search_knowledge_base``
    Hybrid retrieval returning ranked passages with scores, provenance, and
    character offsets. Optional ``mode`` (``hybrid``/``bm25``/``vector``) lets an
    agent pick the retrieval strategy.
``ask_knowledge_base``
    Full RAG answer with sentence-level **verifiable citations** and a
    groundedness score (:mod:`ragforge.citations`).
``knowledge_base_stats``
    Document/chunk counts and backend configuration.
``ingest_document``
    Add a document. **Disabled unless the server is started with
    ``allow_ingest=True``** (``ragforge mcp --allow-ingest``): a read-only
    knowledge base is the safe default for a tool an LLM can call.

Register it with Claude Code::

    claude mcp add ragforge -- ragforge mcp --index rag_index.json

or with any MCP client via ``{"command": "ragforge", "args": ["mcp", "--index", ...]}``.

Resources and prompts:

``resources/list`` / ``resources/read`` / ``resources/templates/list``
    Every indexed document is readable as ``ragforge://doc/<doc_id>`` (``text/plain``,
    reassembled from its chunks), so a client can attach whole documents to a
    conversation, not just search hits.
``prompts/list`` / ``prompts/get``
    Two reusable prompt templates: ``answer_with_citations`` and ``summarize_document``.

Protocol notes: logs go to stderr only (stdout is the protocol channel);
notifications get no response; malformed or unknown requests receive standard
JSON-RPC error objects instead of crashing the server; tool failures are
reported as ``isError`` results so the model can see and react to them.
"""

from __future__ import annotations

import base64
import json
import logging
import sys
from typing import IO, Any
from urllib.parse import quote, unquote

from ragforge._version import __version__
from ragforge.citations import attribute
from ragforge.pipeline import RagPipeline

logger = logging.getLogger("ragforge.mcp")

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
MAX_K = 50
PAGE_SIZE = 100
RESOURCE_PREFIX = "ragforge://doc/"
RESOURCE_NOT_FOUND = -32002

# JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


_PROMPTS: list[dict[str, Any]] = [
    {
        "name": "answer_with_citations",
        "title": "Answer from the knowledge base, with citations",
        "description": "Ask a question and require an answer grounded in the knowledge base.",
        "arguments": [
            {"name": "question", "description": "The question to answer", "required": True}
        ],
    },
    {
        "name": "summarize_document",
        "title": "Summarize a document",
        "description": "Summarize one indexed document, attaching its full text as a resource.",
        "arguments": [{"name": "doc_id", "description": "Id of the document", "required": True}],
    },
]


class _ToolError(Exception):
    """Raised by tool handlers for user-correctable problems (reported as isError)."""


def _tool_definitions(allow_ingest: bool) -> list[dict[str, Any]]:
    k_schema = {"type": "integer", "minimum": 1, "maximum": MAX_K, "default": 5}
    tools: list[dict[str, Any]] = [
        {
            "name": "search_knowledge_base",
            "description": (
                "Search the knowledge base with hybrid retrieval (BM25 + vector, fused with "
                "Reciprocal Rank Fusion). Returns ranked passages with source document ids, "
                "scores, and character offsets."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query"},
                    "k": {**k_schema, "description": "Number of passages to return"},
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "bm25", "vector"],
                        "default": "hybrid",
                    },
                },
                "required": ["query"],
            },
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "ask_knowledge_base",
            "description": (
                "Answer a question from the knowledge base and return the answer together with "
                "sentence-level citations (document id + character offsets) and a groundedness "
                "score. Sentences without supporting evidence are flagged as unverified."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "k": {**k_schema, "description": "Number of passages to ground the answer in"},
                },
                "required": ["question"],
            },
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "knowledge_base_stats",
            "description": "Report how many documents and chunks are indexed and the backend used.",
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
    ]
    if allow_ingest:
        tools.append(
            {
                "name": "ingest_document",
                "description": "Add (or replace) a document in the in-memory knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "text": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["doc_id", "text"],
                },
                "annotations": {"readOnlyHint": False, "destructiveHint": True},
            }
        )
    return tools


class MCPServer:
    """Transport-agnostic MCP request handler around a :class:`RagPipeline`."""

    def __init__(
        self, pipeline: RagPipeline, allow_ingest: bool = False, name: str = "ragforge"
    ) -> None:
        self.pipeline = pipeline
        self.allow_ingest = allow_ingest
        self.name = name

    # -- JSON-RPC plumbing -------------------------------------------------

    def handle_line(self, line: str) -> str | None:
        """Process one raw JSON-RPC message; returns the response line, or ``None``."""
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return self._dump(self._error(None, PARSE_ERROR, "Parse error"))
        response = self.handle_message(message)
        return None if response is None else self._dump(response)

    def handle_message(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return self._error(None, INVALID_REQUEST, "Expected a JSON-RPC 2.0 request object")
        method = message.get("method")
        msg_id = message.get("id")
        is_notification = "id" not in message
        if not isinstance(method, str):
            # A response from the client (or garbage); nothing to do.
            return None if "result" in message or "error" in message else self._error(
                msg_id, INVALID_REQUEST, "Missing method"
            )  # fmt: skip

        params = message.get("params") or {}
        if not isinstance(params, dict):
            return None if is_notification else self._error(
                msg_id, INVALID_PARAMS, "params must be an object"
            )  # fmt: skip

        handler = {
            "initialize": self._initialize,
            "ping": lambda _p: {},
            "tools/list": lambda _p: {"tools": _tool_definitions(self.allow_ingest)},
            "tools/call": self._call_tool,
            "resources/list": self._list_resources,
            "resources/templates/list": lambda _p: {
                "resourceTemplates": [
                    {
                        "uriTemplate": f"{RESOURCE_PREFIX}{{doc_id}}",
                        "name": "document",
                        "title": "Indexed document",
                        "description": "Full text of an indexed document by id",
                        "mimeType": "text/plain",
                    }
                ]
            },
            "resources/read": self._read_resource,
            "prompts/list": lambda _p: {"prompts": _PROMPTS},
            "prompts/get": self._get_prompt,
        }.get(method)

        if handler is None:
            if is_notification:  # e.g. notifications/initialized, notifications/cancelled
                return None
            return self._error(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        try:
            result = handler(params)
        except _InvalidParams as exc:
            return None if is_notification else self._error(msg_id, INVALID_PARAMS, str(exc))
        except _NotFound as exc:
            if is_notification:
                return None
            return self._error(msg_id, RESOURCE_NOT_FOUND, "Resource not found", {"uri": str(exc)})
        except Exception as exc:
            logger.exception("Unhandled error in %s", method)
            return None if is_notification else self._error(msg_id, INTERNAL_ERROR, str(exc))
        return None if is_notification else {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": msg_id, "error": error}

    @staticmethod
    def _dump(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    # -- MCP methods -------------------------------------------------------

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else (
            SUPPORTED_PROTOCOL_VERSIONS[0]
        )  # fmt: skip
        return {
            "protocolVersion": version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {"name": self.name, "version": __version__},
            "instructions": (
                "Search and question-answering over a ragforge knowledge base. Prefer "
                "ask_knowledge_base when you need a cited answer, and search_knowledge_base "
                "when you want raw passages."
            ),
        }

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(name, str):
            raise _InvalidParams("tools/call requires a string 'name'")
        if not isinstance(arguments, dict):
            raise _InvalidParams("'arguments' must be an object")
        known = {t["name"] for t in _tool_definitions(self.allow_ingest)}
        if name not in known:
            raise _InvalidParams(f"Unknown tool: {name}")
        try:
            payload = getattr(self, f"_tool_{name}")(arguments)
        except _ToolError as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return {"content": [{"type": "text", "text": f"Tool failed: {exc}"}], "isError": True}
        return {
            "content": [
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}
            ],
            "structuredContent": payload,
            "isError": False,
        }

    # -- resources & prompts ------------------------------------------------------

    def _documents(self) -> dict[str, list[Any]]:
        docs: dict[str, list[Any]] = {}
        for chunk in self.pipeline.chunks:
            docs.setdefault(chunk.doc_id, []).append(chunk)
        return docs

    @staticmethod
    def _document_text(chunks: list[Any]) -> str:
        """Reassemble a document from its chunks, de-duplicating overlap when offsets allow."""
        ordered = sorted(chunks, key=lambda c: (c.start_char, c.position))
        if len({c.start_char for c in ordered}) < len(ordered):  # offsets unavailable
            return "\n\n".join(c.text for c in sorted(chunks, key=lambda c: c.position))
        parts: list[str] = []
        cursor = 0
        for chunk in ordered:
            if chunk.start_char >= cursor:
                parts.append(("\n" if parts and chunk.start_char > cursor else "") + chunk.text)
                cursor = chunk.start_char + len(chunk.text)
            elif chunk.start_char + len(chunk.text) > cursor:
                parts.append(chunk.text[cursor - chunk.start_char :])
                cursor = chunk.start_char + len(chunk.text)
        return "".join(parts)

    def _list_resources(self, params: dict[str, Any]) -> dict[str, Any]:
        cursor = params.get("cursor")
        try:
            offset = int(base64.b64decode(cursor).decode()) if cursor else 0
        except (ValueError, TypeError) as exc:
            raise _InvalidParams("Invalid cursor") from exc
        docs = self._documents()
        ids = sorted(docs)[offset : offset + PAGE_SIZE]
        resources = []
        for doc_id in ids:
            text = self._document_text(docs[doc_id])
            resources.append(
                {
                    "uri": RESOURCE_PREFIX + quote(doc_id, safe=""),
                    "name": doc_id,
                    "description": text[:160],
                    "mimeType": "text/plain",
                    "size": len(text.encode("utf-8")),
                }
            )
        result: dict[str, Any] = {"resources": resources}
        if offset + PAGE_SIZE < len(docs):
            result["nextCursor"] = base64.b64encode(str(offset + PAGE_SIZE).encode()).decode()
        return result

    def _read_resource(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri")
        if not isinstance(uri, str):
            raise _InvalidParams("resources/read requires a string 'uri'")
        if not uri.startswith(RESOURCE_PREFIX):
            raise _NotFound(uri)
        doc_id = unquote(uri[len(RESOURCE_PREFIX) :])
        chunks = self._documents().get(doc_id)
        if not chunks:
            raise _NotFound(uri)
        return {
            "contents": [
                {"uri": uri, "mimeType": "text/plain", "text": self._document_text(chunks)}
            ]
        }

    def _get_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise _InvalidParams("'arguments' must be an object")
        if name == "answer_with_citations":
            question = arguments.get("question")
            if not isinstance(question, str) or not question.strip():
                raise _InvalidParams("Missing required argument: question")
            text = (
                "Answer the question using only the knowledge base. First call "
                "`ask_knowledge_base` (or `search_knowledge_base` for raw passages). Quote the "
                "supporting sentences, cite each claim with its document id and character range, "
                "and say plainly if the knowledge base does not contain the answer.\n\n"
                f"Question: {question}"
            )
            return {
                "description": "Grounded, cited answer",
                "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
            }
        if name == "summarize_document":
            doc_id = arguments.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                raise _InvalidParams("Missing required argument: doc_id")
            chunks = self._documents().get(doc_id)
            if not chunks:
                raise _InvalidParams(f"Unknown document: {doc_id}")
            uri = RESOURCE_PREFIX + quote(doc_id, safe="")
            return {
                "description": f"Summary of {doc_id}",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "resource",
                            "resource": {
                                "uri": uri,
                                "mimeType": "text/plain",
                                "text": self._document_text(chunks),
                            },
                        },
                    },
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": "Summarize the document above in five bullet points.",
                        },
                    },
                ],
            }
        raise _InvalidParams(f"Unknown prompt: {name}")

    # -- tools -------------------------------------------------------------

    @staticmethod
    def _k(arguments: dict[str, Any]) -> int:
        k = arguments.get("k", 5)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
            raise _ToolError(f"'k' must be an integer between 1 and {MAX_K}")
        return k

    @staticmethod
    def _text(arguments: dict[str, Any], key: str) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or not value.strip():
            raise _ToolError(f"'{key}' must be a non-empty string")
        return value

    def _tool_search_knowledge_base(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = self._text(arguments, "query")
        k = self._k(arguments)
        mode = arguments.get("mode", "hybrid")
        if mode == "hybrid":
            results = [c.to_dict() for c in self.pipeline.retrieve(query, k=k)]
        elif mode in ("bm25", "vector"):
            index = self.pipeline.bm25_index if mode == "bm25" else self.pipeline.vector_index
            results = [
                {
                    "chunk_id": r.chunk.id,
                    "doc_id": r.chunk.doc_id,
                    "text": r.chunk.text,
                    "score": r.score,
                    "provenance": r.provenance,
                    "start_char": r.chunk.start_char,
                    "end_char": r.chunk.end_char,
                    "metadata": dict(r.chunk.metadata),
                }
                for r in index.search(query, k=k)
            ]
        else:
            raise _ToolError("'mode' must be one of: hybrid, bm25, vector")
        return {"query": query, "mode": mode, "count": len(results), "results": results}

    def _tool_ask_knowledge_base(self, arguments: dict[str, Any]) -> dict[str, Any]:
        question = self._text(arguments, "question")
        response = self.pipeline.answer(question, k=self._k(arguments))
        cited = attribute(response.answer, response.contexts)
        return {
            "question": question,
            "answer": response.answer,
            "groundedness": round(cited.groundedness, 4),
            "sentences": [s.to_dict() for s in cited.sentences],
            "sources": [
                {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": c.score}
                for c in response.contexts
            ],
            "latency_ms": round(response.total_latency_ms, 3),
        }

    def _tool_knowledge_base_stats(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "documents": self.pipeline.document_count,
            "chunks": self.pipeline.chunk_count,
            "vector_backend": self.pipeline.vector_backend,
            "ingest_enabled": self.allow_ingest,
        }

    def _tool_ingest_document(self, arguments: dict[str, Any]) -> dict[str, Any]:
        doc_id = self._text(arguments, "doc_id")
        text = self._text(arguments, "text")
        metadata = arguments.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise _ToolError("'metadata' must be an object")
        replaced = self.pipeline.delete(doc_id)
        chunks = self.pipeline.ingest(doc_id, text, metadata)
        return {"doc_id": doc_id, "chunks_added": len(chunks), "chunks_replaced": replaced}


class _InvalidParams(Exception):
    pass


class _NotFound(Exception):
    pass


def serve_stdio(
    pipeline: RagPipeline,
    allow_ingest: bool = False,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
) -> None:
    """Run the MCP server on stdin/stdout until the client closes the stream."""
    server = MCPServer(pipeline, allow_ingest=allow_ingest)
    source = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else sys.stdout
    for line in source:
        line = line.strip()
        if not line:
            continue
        response = server.handle_line(line)
        if response is not None:
            sink.write(response + "\n")
            sink.flush()
