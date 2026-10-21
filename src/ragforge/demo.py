"""``ragforge demo``: a guided, offline tour of the library in your terminal.

Runs entirely on the built-in sample handbook (:mod:`ragforge.sample`) with
deterministic hashing embeddings -- no network, no API keys. Each section shows
one capability with real output from the real code paths:

1. hybrid retrieval vs. its two components (BM25, vector), judged against known
   relevant documents;
2. guardrails neutralizing a poisoned document and redacting PII;
3. verifiable citations with a groundedness score;
4. an agentic loop (decompose -> retrieve -> grade -> rewrite -> abstain);
5. the per-stage latency trace.

Color is used only on a TTY (or with ``FORCE_COLOR``) and honors ``NO_COLOR``.
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import TextIO

from ragforge._version import __version__
from ragforge.agent import AgenticRag, heuristic_decompose_fn
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.citations import attribute
from ragforge.guardrails import Guardrails
from ragforge.index import ScoredChunk
from ragforge.pipeline import RagPipeline
from ragforge.reranking import MaxMarginalRelevanceReranker
from ragforge.sample import SAMPLE_DOCUMENTS, SAMPLE_EVAL_CASES
from ragforge.telemetry import InMemoryExporter, Tracer

_ANSI = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "ember": "\x1b[38;5;208m",
    "gold": "\x1b[38;5;220m",
    "cyan": "\x1b[38;5;45m",
    "green": "\x1b[38;5;78m",
    "red": "\x1b[38;5;203m",
    "grey": "\x1b[38;5;245m",
}

BANNER = r"""
  ____            _____
 |  _ \ __ _  __ |  ___|__  _ __ __ _  ___
 | |_) / _` |/ _` | |_ / _ \| '__/ _` |/ _ \
 |  _ < (_| | (_| |  _| (_) | | | (_| |  __/
 |_| \_\__,_|\__, |_|  \___/|_|  \__, |\___|
             |___/               |___/
"""  # noqa: W291


class _Painter:
    def __init__(self, stream: TextIO, color: bool) -> None:
        self.stream = stream
        self.color = color

    def c(self, text: str, *styles: str) -> str:
        if not self.color:
            return text
        return "".join(_ANSI[s] for s in styles) + text + _ANSI["reset"]

    def line(self, text: str = "") -> None:
        self.stream.write(text + "\n")

    def section(self, number: int, title: str) -> None:
        self.line()
        self.line(self.c(f"  {number}  ", "bold", "ember") + self.c(title, "bold"))
        self.line(self.c("  " + "─" * 66, "grey"))


def _use_color(stream: TextIO, color: bool | None) -> bool:
    if color is not None:
        return color
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _hit_cells(hits: list[ScoredChunk], relevant: set[str], p: _Painter) -> list[tuple[str, int]]:
    """``(styled text, visible width)`` per hit so columns align despite ANSI codes."""
    cells = []
    for rank, hit in enumerate(hits, start=1):
        good = hit.chunk.doc_id in relevant
        name = hit.chunk.doc_id[:20]
        plain = f"{'✓' if good else '·'} {rank}. {name}"
        mark = p.c("✓", "green", "bold") if good else p.c("·", "grey")
        cells.append((f"{mark} {rank}. {p.c(name, 'green') if good else name}", len(plain)))
    return cells


def run_demo(stream: TextIO | None = None, color: bool | None = None) -> None:
    """Print the guided tour to ``stream`` (default: stdout)."""
    out = stream or sys.stdout
    p = _Painter(out, _use_color(out, color))

    for i, row in enumerate(BANNER.strip("\n").splitlines()):
        p.line(p.c(row, "ember" if i < 3 else "gold", "bold"))
    p.line(
        p.c(f"  hybrid RAG toolkit · v{__version__} · zero dependencies · fully offline", "grey")
    )

    exporter = InMemoryExporter()
    pipeline = RagPipeline(
        chunker=RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50),
        bm25_analyzer="english",
        reranker=MaxMarginalRelevanceReranker(lambda_mult=0.8, relevance="fused"),
        tracer=Tracer(exporters=[exporter]),
        guardrails=Guardrails(action="sanitize"),
    )
    chunks = pipeline.ingest_batch(SAMPLE_DOCUMENTS)
    p.line()
    p.line(
        f"  Indexed {p.c(str(pipeline.document_count), 'bold')} documents "
        f"→ {p.c(str(len(chunks)), 'bold')} chunks (BM25 inverted index + vector index)"
    )

    _section_hybrid(p, pipeline)
    _section_guardrails(p, pipeline)
    pipeline.guardrails = None
    _section_citations(p, pipeline)
    _section_agent(p, pipeline)
    pipeline.guardrails = Guardrails()
    _section_trace(p, pipeline, exporter)

    p.line()
    p.line(p.c('  Next: ragforge ingest ./docs · ragforge query "..." --agentic --cite', "grey"))
    p.line()


_DEMO_QUERIES = (
    "How do I cancel my plan?",
    "What does error ERR-4012 mean?",
    "How does reciprocal rank fusion combine rankings?",
)


def _section_hybrid(p: _Painter, pipeline: RagPipeline) -> None:
    p.section(1, "Hybrid retrieval vs. its parts   (✓ = known relevant document)")
    width = 30
    for case in (c for c in SAMPLE_EVAL_CASES if c["query"] in _DEMO_QUERIES):
        query, relevant = case["query"], set(case["relevant_doc_ids"])
        p.line()
        p.line("  " + p.c(f"“{query}”", "bold", "gold"))
        columns = [
            ("BM25", "cyan", pipeline.bm25_index.search(query, k=3)),
            ("Vector", "cyan", pipeline.vector_index.search(query, k=3)),
            ("Hybrid · RRF", "ember", pipeline.retriever.search(query, k=3)),
        ]
        p.line("  " + "".join(p.c(n.ljust(width), "bold", tone) for n, tone, _ in columns))
        grid = [_hit_cells(hits, relevant, p) for _, _, hits in columns]
        for row in range(3):
            parts = []
            for column in grid:
                text, visible = column[row] if row < len(column) else ("", 0)
                parts.append(text + " " * (width - visible))
            p.line("  " + "".join(parts).rstrip())


def _section_guardrails(p: _Painter, pipeline: RagPipeline) -> None:
    p.section(2, "Guardrails: poisoned document & PII")
    for query in ("What is the standard shipping time?", "Who was charged twice and refunded?"):
        hits = pipeline.retrieve(query, k=4)
        top = next((h for h in hits if h.guardrails), hits[0])
        p.line()
        p.line("  " + p.c(f"“{query}”", "bold", "gold") + p.c(f"  → {top.doc_id}", "grey"))
        for finding in top.guardrails.get("findings", [])[:4]:
            tag = "INJECTION" if finding["kind"] == "injection" else "PII"
            p.line(
                f"    {p.c(tag, 'red', 'bold')} {p.c(finding['rule'], 'bold')} "
                f"{p.c(finding['excerpt'], 'grey')}"
            )
        for row in textwrap.wrap(top.text, 70):
            p.line("    " + p.c(row, "dim"))


def _section_citations(p: _Painter, pipeline: RagPipeline) -> None:
    p.section(3, "Verifiable citations & groundedness")
    question = "Why does BM25 fail on synonyms?"
    answer = pipeline.answer(question, k=3)
    first = answer.contexts[0].text.split(". ")[0] if answer.contexts else ""
    cited = attribute(f"{first}. It was invented by a chatbot in 2031.", answer.contexts)
    p.line()
    p.line("  " + p.c(f"“{question}”", "bold", "gold"))
    for sentence in cited.sentences:
        if sentence.supported and sentence.citations:
            cite = sentence.citations[0]
            p.line(f"    {p.c('✓ supported', 'green', 'bold')}   {sentence.text[:56]}…")
            where = f"[{cite.marker}] {cite.doc_id} chars {cite.start_char}–{cite.end_char}"
            p.line(p.c(f"                  ↳ {where}", "grey"))
        else:
            p.line(f"    {p.c('✗ unverified', 'red', 'bold')}  {sentence.text}")
    p.line(f"    groundedness = {p.c(f'{cited.groundedness:.2f}', 'bold')}")


def _section_agent(p: _Painter, pipeline: RagPipeline) -> None:
    p.section(4, "Agentic RAG: decompose → retrieve → grade → rewrite → abstain")
    agent = AgenticRag(pipeline, decompose_fn=heuristic_decompose_fn)
    tones = {"abstain": "red", "answer": "green", "rewrite": "ember", "stop": "grey"}
    for question in (
        "What does error ERR-4012 mean? How does reciprocal rank fusion work?",
        "What is the capital of Atlantis and its quantum zeppelin fleet?",
    ):
        result = agent.run(question, k=2)
        p.line()
        p.line("  " + p.c(f"“{question}”", "bold", "gold"))
        for step in result.steps:
            score = "" if step.score is None else f" {step.score:.2f}"
            label = step.kind.upper().ljust(10)
            tone = tones.get(step.kind, "cyan")
            p.line(
                f"    {p.c(label, tone, 'bold')}{step.query[:44]}{p.c(score, 'gold')}"
                + (p.c(f"  {step.detail}", "grey") if step.kind == "stop" else "")
            )
        verdict = "abstained" if result.abstained else "answered"
        p.line(
            f"    → {p.c(verdict, 'red' if result.abstained else 'green', 'bold')} "
            f"after {result.retrievals} retrievals (sufficiency {result.sufficiency:.2f})"
        )


def _section_trace(p: _Painter, pipeline: RagPipeline, exporter: InMemoryExporter) -> None:
    p.section(5, "Per-stage trace")
    exporter.spans.clear()
    pipeline.answer("How can I stop duplicate passages in my top results?", k=3)
    longest = max((s.duration_ms for s in exporter.spans), default=1.0) or 1.0
    for span in exporter.spans:
        bar = "█" * max(1, round(28 * span.duration_ms / longest))
        indent = "  " if span.parent_id else ""
        name = (indent + span.name).ljust(18)
        p.line(f"    {name}{p.c(bar, 'ember')} {span.duration_ms:.2f} ms")
