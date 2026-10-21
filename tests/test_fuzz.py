"""Property-based fuzzing of every component that parses untrusted input or persists state."""

from __future__ import annotations

import asyncio
import json
import string

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from ragforge import RagPipeline
from ragforge.asgi import RagASGIApp
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.citations import attribute
from ragforge.graph import extract_entities
from ragforge.guardrails import Guardrails, InjectionDetector, PIIRedactor
from ragforge.judge import JudgeParseError, parse_verdict
from ragforge.mcp_server import MCPServer
from ragforge.storage import DiskIndex

FAST = settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=20),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=12,
)


# ------------------------------------------------------------------------------- MCP


def _server() -> MCPServer:
    pipeline = RagPipeline(chunker=RecursiveCharacterChunker(chunk_size=200, chunk_overlap=0))
    pipeline.ingest("a", "Alpha beta gamma. Delta epsilon zeta.")
    return MCPServer(pipeline, allow_ingest=True)


@FAST
@given(line=st.text(max_size=300))
def test_mcp_survives_arbitrary_lines(line):
    reply = _server().handle_line(line)
    assert reply is None or json.loads(reply)["jsonrpc"] == "2.0"


METHODS = st.sampled_from(
    ["initialize", "ping", "tools/list", "tools/call", "resources/list", "resources/read",
     "resources/templates/list", "prompts/list", "prompts/get", "unknown/method", "notifications/x"]
)  # fmt: skip


@FAST
@given(
    method=METHODS,
    params=json_values,
    msg_id=st.none() | st.integers() | st.text(max_size=5),
    notify=st.booleans(),
)
def test_mcp_always_answers_well_formed_requests_with_result_or_error(
    method, params, msg_id, notify
):
    message: dict = {"jsonrpc": "2.0", "method": method, "params": params}
    if not notify:
        message["id"] = msg_id
    reply = _server().handle_message(message)
    if reply is None:
        return
    assert reply["jsonrpc"] == "2.0" and ("result" in reply) != ("error" in reply)
    if "error" in reply:
        assert isinstance(reply["error"]["code"], int) and isinstance(
            reply["error"]["message"], str
        )
    json.dumps(reply)  # always serializable


@FAST
@given(message=json_values)
def test_mcp_rejects_arbitrary_json_without_crashing(message):
    reply = _server().handle_message(message)
    assert reply is None or reply["jsonrpc"] == "2.0"


@FAST
@given(name=st.sampled_from(["search_knowledge_base", "ask_knowledge_base", "ingest_document",
                              "knowledge_base_stats", "nope"]), arguments=json_values)  # fmt: skip
def test_mcp_tool_calls_with_junk_arguments_never_raise(name, arguments):
    reply = _server().handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    assert reply is not None
    assert "error" in reply or isinstance(reply["result"]["isError"], bool)


# ------------------------------------------------------------------------------ judge


@FAST
@given(raw=st.text(max_size=400))
def test_judge_parser_returns_a_verdict_or_a_parse_error(raw):
    try:
        verdict = parse_verdict("faithfulness", raw)
    except JudgeParseError:
        return
    assert 1 <= verdict.raw_score <= 5 and 0.0 <= verdict.score <= 1.0


@FAST
@given(score=json_values, reasoning=json_values)
def test_judge_parser_tolerates_arbitrary_json_fields(score, reasoning):
    raw = json.dumps({"reasoning": reasoning, "score": score})
    try:
        verdict = parse_verdict("correctness", raw)
    except JudgeParseError:
        return
    assert 0.0 <= verdict.score <= 1.0


# -------------------------------------------------------------------------- guardrails

UNICODE = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=300) | st.text(
    alphabet=string.printable + "​‮", max_size=300
)


@FAST
@given(text=UNICODE)
def test_guardrail_findings_have_valid_offsets_and_bounded_risk(text):
    detector, redactor = InjectionDetector(), PIIRedactor()
    findings = detector.scan(text) + redactor.scan(text)
    for f in findings:
        assert 0 <= f.start <= f.end <= len(text)
    assert 0.0 <= detector.risk(findings) <= 1.0
    redacted, pii = redactor.redact(text)
    assert all(f.kind == "pii" for f in pii)
    assert len(redacted) >= 0


@FAST
@given(text=UNICODE, action=st.sampled_from(["flag", "sanitize", "drop"]))
def test_guardrails_screen_never_raises_and_sanitized_text_is_clean_of_hidden_chars(text, action):
    result = Guardrails(action=action).screen(text)
    assert 0.0 <= result.risk <= 1.0
    if action == "flag" and not result.blocked:
        assert "​" not in result.text or "​" in text
    if action == "sanitize" and result.risk >= 0.5:
        assert not any(c in result.text for c in "​‌‍⁠﻿")


# ---------------------------------------------------------- citations, entities, chunking


@FAST
@given(answer=st.text(max_size=300), context=st.text(max_size=400))
def test_attribution_is_total_and_bounded(answer, context):
    class Ctx:
        text, chunk_id, doc_id, start_char = context, "c", "d", 0

    cited = attribute(answer, [Ctx()])
    assert 0.0 <= cited.groundedness <= 1.0
    for sentence in cited.sentences:
        assert 0.0 <= sentence.support <= 1.0
        for citation in sentence.citations:
            assert context[citation.start_char : citation.end_char] == citation.quote


@FAST
@given(text=UNICODE)
def test_entity_extraction_only_returns_substrings_of_the_text(text):
    for entity in extract_entities(text):
        assert (
            entity
            and entity in text.replace("’s", "'s")
            or entity.replace("’", "'") in text.replace("’", "'")
        )


# ------------------------------------------------------------------------------ ASGI


def _run_request(app, method, path, body):
    async def main():
        scope = {"type": "http", "method": method, "path": path, "headers": []}
        sent = [{"type": "http.request", "body": body, "more_body": False}]
        out = []

        async def receive():
            return sent.pop(0) if sent else {"type": "http.disconnect"}

        async def send(message):
            out.append(message)

        await app(scope, receive, send)
        return out[0]["status"]

    return asyncio.run(main())


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(body=st.binary(max_size=200) | json_values.map(lambda v: json.dumps(v).encode()),
       path=st.sampled_from(["/query", "/query/stream", "/health", "/other"]),
       method=st.sampled_from(["GET", "POST", "PUT"]))  # fmt: skip
def test_asgi_app_always_returns_a_sane_status(body, path, method):
    pipeline = RagPipeline()
    pipeline.ingest("a", "alpha beta")
    status = _run_request(RagASGIApp(pipeline), method, path, body)
    assert status in (200, 400, 404, 405, 413, 500)


# ------------------------------------------------------------- storage state machine

DOC_IDS = ["a", "b", "c"]


class DiskIndexMachine(RuleBasedStateMachine):
    """Random ingest / delete / commit / reopen / compact sequences against a dict model."""

    def __init__(self) -> None:
        super().__init__()
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.path = f"{self.tmp.name}/ix"
        self.model: dict[str, str] = {}
        self.disk = DiskIndex.create(
            self.path,
            vector_backend="brute_force",
            chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=0),
        )
        self.counter = 0

    def teardown(self) -> None:
        self.disk.close()
        self.tmp.cleanup()

    @rule(doc=st.sampled_from(DOC_IDS))
    def ingest(self, doc):
        if doc in self.model:  # ingest replaces: delete first, like the MCP tool
            self.disk.delete(doc)
        self.counter += 1
        token = f"tok{self.counter}"
        self.disk.ingest(doc, f"document {doc} about {token}")
        self.model[doc] = token

    @rule(doc=st.sampled_from(DOC_IDS))
    def delete(self, doc):
        removed = self.disk.delete(doc)
        assert (removed > 0) == (doc in self.model)
        self.model.pop(doc, None)

    @rule()
    def commit(self):
        self.disk.commit()

    @rule()
    def reopen(self):
        self.disk.close()
        self.disk = DiskIndex.open(
            self.path, chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=0)
        )

    @rule()
    def compact(self):
        self.disk.compact()

    @invariant()
    def index_matches_the_model(self):
        pipeline = self.disk.pipeline
        assert pipeline.document_count == len(self.model)
        for doc, token in self.model.items():
            hits = pipeline.retrieve(token, k=1)
            assert hits and hits[0].doc_id == doc


TestDiskIndexStateMachine = DiskIndexMachine.TestCase
TestDiskIndexStateMachine.settings = settings(
    max_examples=25,
    stateful_step_count=14,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
