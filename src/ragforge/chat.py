"""``ragforge chat``: an interactive terminal session over an index.

A small REPL for exploring a knowledge base the way an end user would: ask a question,
watch the answer stream in, see which passages it came from, and flip retrieval options on
the fly. Everything is injectable (input function, output stream, color), so the whole loop
is unit-tested without a terminal.

Commands (type ``/help`` in the session)::

    /k N                      passages per answer
    /mode hybrid|bm25|vector  retrieval strategy
    /cite on|off              attribute sentences to sources (groundedness)
    /agent on|off             agentic retrieve/grade/rewrite loop
    /trace on|off             per-stage latency spans
    /sources                  show the passages behind the last answer
    /clear | /quit
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

from ragforge.agent import AgenticRag
from ragforge.citations import attribute
from ragforge.pipeline import RagAnswer, RagPipeline
from ragforge.telemetry import InMemoryExporter

HELP = """\
  /k N                      passages per answer (now {k})
  /mode hybrid|bm25|vector  retrieval strategy (now {mode})
  /cite on|off              sentence-level citations + groundedness (now {cite})
  /agent on|off             agentic retrieve/grade/rewrite loop (now {agent})
  /trace on|off             per-stage latency spans (now {trace})
  /sources                  passages behind the last answer
  /clear                    forget the last answer
  /quit                     leave (Ctrl-D also works)"""

_STYLE = {"dim": "\x1b[2m", "bold": "\x1b[1m", "ember": "\x1b[38;5;208m", "cyan": "\x1b[38;5;45m",
          "green": "\x1b[38;5;78m", "red": "\x1b[38;5;203m", "reset": "\x1b[0m"}  # fmt: skip


@dataclass
class ChatState:
    k: int = 3
    mode: str = "hybrid"
    cite: bool = True
    agent: bool = False
    trace: bool = False


class _Out:
    def __init__(self, stream: TextIO, color: bool) -> None:
        self.stream, self.color = stream, color

    def s(self, text: str, *styles: str) -> str:
        return "".join(_STYLE[x] for x in styles) + text + _STYLE["reset"] if self.color else text

    def write(self, text: str = "", end: str = "\n") -> None:
        self.stream.write(text + end)
        self.stream.flush()


def _set_mode(pipeline: RagPipeline, mode: str) -> None:
    pipeline._apply_weights(
        0.0 if mode == "vector" else pipeline.weight_bm25,
        0.0 if mode == "bm25" else pipeline.weight_vector,
    )


def _toggle(value: str) -> bool | None:
    return {"on": True, "off": False, "true": True, "false": False}.get(value.lower())


def run_chat(
    pipeline: RagPipeline,
    input_fn: Callable[[str], str] | None = None,
    stdout: TextIO | None = None,
    color: bool | None = None,
    state: ChatState | None = None,
) -> int:
    """Run the interactive loop until ``/quit`` or end of input. Returns an exit code."""
    read = input_fn or input  # resolved at call time so tests/embedders can patch input()
    stream = stdout or sys.stdout
    if color is None:
        color = not os.environ.get("NO_COLOR") and bool(getattr(stream, "isatty", lambda: False)())
    out = _Out(stream, color)
    st = state or ChatState()
    exporter = InMemoryExporter()
    pipeline.tracer.exporters.append(exporter)
    last: RagAnswer | None = None

    out.write(
        out.s("ragforge chat", "bold", "ember")
        + out.s(
            f"  ·  {pipeline.document_count} documents, {pipeline.chunk_count} chunks  ·  /help",
            "dim",
        )
    )
    try:
        while True:
            try:
                line = read(out.s("› ", "ember", "bold") if color else "> ").strip()
            except (EOFError, KeyboardInterrupt):
                out.write()
                return 0
            if not line:
                continue
            if line.startswith("/"):
                command, _, arg = line[1:].partition(" ")
                arg = arg.strip()
                if command in ("quit", "exit", "q"):
                    return 0
                if command == "help":
                    out.write(HELP.format(**vars(st)))
                elif command == "k" and arg.isdigit() and 1 <= int(arg) <= 50:
                    st.k = int(arg)
                    out.write(out.s(f"k = {st.k}", "dim"))
                elif command == "mode" and arg in ("hybrid", "bm25", "vector"):
                    st.mode = arg
                    _set_mode(pipeline, arg)
                    out.write(out.s(f"mode = {arg}", "dim"))
                elif command in ("cite", "agent", "trace") and _toggle(arg) is not None:
                    setattr(st, command, _toggle(arg))
                    out.write(out.s(f"{command} = {'on' if _toggle(arg) else 'off'}", "dim"))
                elif command == "sources":
                    _print_sources(out, last)
                elif command == "clear":
                    last = None
                    out.write(out.s("cleared", "dim"))
                else:
                    out.write(out.s("unknown or invalid command; try /help", "red"))
                continue

            exporter.spans.clear()
            last = _ask(pipeline, line, st, out)
            if st.trace:
                for span in exporter.spans:
                    indent = "  " if span.parent_id else ""
                    out.write(out.s(f"  {indent}{span.name:<16}{span.duration_ms:7.2f} ms", "dim"))
    finally:
        pipeline.tracer.exporters.remove(exporter)


def _ask(pipeline: RagPipeline, question: str, st: ChatState, out: _Out) -> RagAnswer:
    answer: RagAnswer
    if st.agent:
        result = AgenticRag(pipeline).run(question, k=st.k)
        for step in result.steps:
            score = "" if step.score is None else f" ({step.score:.2f})"
            out.write(out.s(f"  · {step.kind:<9}{step.query[:60]}{score}", "dim"))
        answer = result.answer
        out.write(out.s(answer.answer, "red" if result.abstained else "reset"))
    else:
        answer = _stream(pipeline, question, st, out)
    if st.cite and answer.contexts and answer.answer:
        cited = attribute(answer.answer, answer.contexts)
        verdict = (
            "green" if cited.groundedness >= 0.99 else "red" if cited.groundedness == 0 else "ember"
        )
        out.write(out.s(f"  groundedness {cited.groundedness:.0%}", verdict))
        for sentence in cited.unsupported:
            out.write(out.s(f"  ✗ unverified: {sentence.text[:80]}", "red"))
    _print_sources(out, answer, compact=True)
    return answer


def _stream(pipeline: RagPipeline, question: str, st: ChatState, out: _Out) -> RagAnswer:
    final: RagAnswer | None = None
    for event in pipeline.answer_stream(question, k=st.k):
        if event["type"] == "token":
            out.write(event["text"], end="")
        elif event["type"] == "done":
            final = event["answer"]
    out.write()
    assert final is not None
    return final


def _print_sources(out: _Out, answer: RagAnswer | None, compact: bool = False) -> None:
    if answer is None or not answer.contexts:
        if not compact:
            out.write(out.s("no sources yet", "dim"))
        return
    for i, ctx in enumerate(answer.contexts, 1):
        where = f"chars {ctx.start_char}–{ctx.end_char}"
        out.write(out.s(f"  [{i}] {ctx.doc_id} · {where} · score {ctx.score:.3f}", "cyan"))
        if not compact:
            out.write(out.s(f"      {ctx.text[:200]}", "dim"))
