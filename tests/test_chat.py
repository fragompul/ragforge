"""The chat REPL runs end to end with scripted input and captured output."""

from __future__ import annotations

import io

from ragforge import RagPipeline
from ragforge.chat import ChatState, run_chat
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.extractive import extractive_generate_fn


def _pipeline() -> RagPipeline:
    p = RagPipeline(
        chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=0),
        bm25_analyzer="english",
        generate_fn=extractive_generate_fn(),
    )
    p.ingest("eiffel", "The Eiffel Tower is in Paris. It was completed in 1889.")
    p.ingest("python", "Python was created by Guido van Rossum in 1991.")
    return p


def session(lines, pipeline=None, state=None):
    pipeline = pipeline or _pipeline()
    feed = iter(lines)

    def input_fn(prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    out = io.StringIO()
    code = run_chat(pipeline, input_fn=input_fn, stdout=out, color=False, state=state)
    return code, out.getvalue(), pipeline


def test_question_streams_an_answer_with_groundedness_and_sources():
    code, text, _ = session(["When was the Eiffel Tower completed?"])
    assert code == 0
    assert "ragforge chat" in text and "2 documents" in text
    assert "completed in 1889" in text
    assert "groundedness 100%" in text
    assert "[1] eiffel" in text and "chars" in text


def test_commands_change_state_and_help_shows_current_values():
    state = ChatState()
    _, text, pipeline = session(
        ["/k 2", "/mode bm25", "/cite off", "/agent on", "/trace on", "/help"], state=state
    )
    assert (state.k, state.mode, state.cite, state.agent, state.trace) == (
        2,
        "bm25",
        False,
        True,
        True,
    )
    assert "k = 2" in text and "mode = bm25" in text and "cite = off" in text
    assert "(now 2)" in text and "(now bm25)" in text
    base = pipeline._base_retriever
    assert (base.weight_bm25, base.weight_vector) == (1.0, 0.0)


def test_mode_switch_back_restores_configured_weights():
    pipeline = _pipeline()
    session(["/mode vector", "/mode hybrid"], pipeline=pipeline)
    base = pipeline._base_retriever
    assert (base.weight_bm25, base.weight_vector) == (1.0, 1.0)


def test_agent_mode_prints_decision_steps_and_abstains_when_unsupported():
    _, text, _ = session(
        ["/agent on", "When was the Eiffel Tower completed?", "capital of Atlantis zeppelins"]
    )
    assert "· retrieve" in text and "· grade" in text and "· answer" in text
    assert "· abstain" in text and "don't have enough information" in text


def test_trace_prints_span_durations():
    _, text, pipeline = session(["/trace on", "Eiffel Tower"])
    assert "fusion_search" in text and " ms" in text
    assert pipeline.tracer.exporters == []  # the REPL's exporter is removed on exit


def test_sources_and_clear_commands():
    _, text, _ = session(["/sources", "Guido van Rossum", "/sources", "/clear", "/sources"])
    assert text.count("no sources yet") == 2
    assert "[1] python" in text and "      Python was created by Guido van Rossum" in text


def test_invalid_commands_and_blank_lines_do_not_crash():
    _, text, _ = session(["", "/k 999", "/mode nonsense", "/cite maybe", "/nope", "/quit"])
    assert text.count("unknown or invalid command") == 4


def test_quit_aliases_and_eof_exit_cleanly():
    for lines in (["/quit"], ["/exit"], ["/q"], []):
        code, _, _ = session(lines)
        assert code == 0


def test_unverified_sentences_are_flagged():
    pipeline = _pipeline()
    pipeline.generate_fn = lambda q, ctx: (
        "The tower was completed in 1889. It was designed by aliens."
    )
    _, text, _ = session(["Eiffel tower"], pipeline=pipeline)
    assert "unverified: It was designed by aliens." in text
    assert "groundedness 50%" in text


def test_color_output_uses_ansi_only_when_requested():
    out = io.StringIO()
    run_chat(
        _pipeline(), input_fn=lambda p: (_ for _ in ()).throw(EOFError), stdout=out, color=True
    )
    assert "\x1b[" in out.getvalue()
