"""Tests for the agentic retrieve/grade/rewrite loop."""

from __future__ import annotations

import json

import pytest

from ragforge import RagPipeline
from ragforge.agent import (
    ABSTAIN_MESSAGE,
    AgenticRag,
    Grade,
    heuristic_decompose_fn,
    lexical_grade_fn,
    llm_decompose_fn,
    llm_grade_fn,
    llm_rewrite_fn,
    relax_rewrite_fn,
)
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.telemetry import InMemoryExporter, Tracer


def _pipeline(**kwargs) -> RagPipeline:
    pipeline = RagPipeline(
        chunker=RecursiveCharacterChunker(chunk_size=200, chunk_overlap=0), **kwargs
    )
    pipeline.ingest("eiffel", "The Eiffel Tower is in Paris. It was completed in 1889.")
    pipeline.ingest("python", "Python was created by Guido van Rossum and released in 1991.")
    return pipeline


def test_lexical_grade_covers_stems_and_reports_missing_terms():
    contexts = ["The Eiffel Tower stands in Paris."]
    full = lexical_grade_fn("Eiffel towers Paris", contexts)
    partial = lexical_grade_fn("Eiffel Tower architect Gustave", contexts)

    assert full.score == 1.0 and full.missing == []
    assert partial.score == pytest.approx(0.5)
    assert partial.missing == ["architect", "gustave"]
    assert lexical_grade_fn("the of and", contexts).score == 1.0


def test_relax_rewrite_drops_missing_terms_only():
    grade = Grade(0.5, ["architect", "Gustave"])
    assert relax_rewrite_fn("Eiffel Tower architect Gustave?", grade, []) == "Eiffel Tower"
    # When only stopwords would remain there is nothing left to search for.
    assert relax_rewrite_fn("what is the zeppelin", Grade(0.0, ["zeppelin"]), []) == ""


def test_heuristic_decompose_splits_compound_questions():
    parts = heuristic_decompose_fn("Who created Python? When was the Eiffel Tower completed?")
    assert parts == ["Who created Python", "When was the Eiffel Tower completed"]
    assert heuristic_decompose_fn("Who created Python") == ["Who created Python"]
    split_and = heuristic_decompose_fn("Who created Python and when was the Eiffel Tower completed")
    assert len(split_and) == 2


def test_sufficient_first_try_makes_a_single_retrieval():
    result = AgenticRag(_pipeline()).run("When was the Eiffel Tower completed?", k=1)

    assert result.retrievals == 1 and not result.abstained
    assert [s.kind for s in result.steps] == ["retrieve", "grade", "answer"]
    assert result.answer.contexts[0].doc_id == "eiffel"
    assert result.sufficiency == 1.0
    assert result.answer.metadata["agentic"] is True


def test_rewrite_recovers_when_first_query_has_unmatched_terms():
    result = AgenticRag(_pipeline(), abstain_below=0.2).run(
        "Eiffel Tower completed zeppelins quasars", k=1
    )

    kinds = [s.kind for s in result.steps]
    assert "rewrite" in kinds and result.retrievals >= 2
    rewrite = next(s for s in result.steps if s.kind == "rewrite")
    assert "zeppelins" not in rewrite.query


def test_abstains_when_nothing_relevant_exists():
    result = AgenticRag(_pipeline()).run("zeppelins quasars nebulae", k=2)

    assert result.abstained
    assert result.answer.answer == ABSTAIN_MESSAGE
    assert result.steps[-1].kind == "abstain"
    assert result.answer.prompt == ""


def test_loop_is_bounded_and_never_repeats_a_query():
    seen: list[str] = []

    def stubborn_rewrite(question, grade, previous):
        seen.append(question)
        return question  # tries to repeat itself

    agent = AgenticRag(
        _pipeline(),
        grade_fn=lambda q, c: Grade(0.1, ["x"]),
        rewrite_fn=stubborn_rewrite,
        max_iterations=5,
    )
    result = agent.run("Eiffel tower", k=1)

    assert result.retrievals == 1
    assert [s.kind for s in result.steps if s.kind == "stop"] == ["stop"]
    assert "no new query" in next(s for s in result.steps if s.kind == "stop").detail


def test_iteration_budget_exhaustion_is_recorded():
    counter = {"n": 0}

    def fresh_rewrite(question, grade, previous):
        counter["n"] += 1
        return f"eiffel tower attempt {counter['n']}"

    agent = AgenticRag(
        _pipeline(),
        grade_fn=lambda q, c: Grade(0.4, []),
        rewrite_fn=fresh_rewrite,
        max_iterations=3,
        abstain_below=0.0,
    )
    result = agent.run("Eiffel tower", k=1)

    assert result.retrievals == 3
    assert result.steps[-2].kind == "stop" and "budget" in result.steps[-2].detail
    assert not result.abstained


def test_decomposition_covers_every_sub_question():
    agent = AgenticRag(_pipeline(), decompose_fn=heuristic_decompose_fn)
    result = agent.run("Who created Python? When was the Eiffel Tower completed?", k=2)

    assert len(result.sub_questions) == 2
    assert {c.doc_id for c in result.answer.contexts} == {"python", "eiffel"}
    assert result.steps[0].kind == "decompose"
    assert result.retrievals == 2


def test_custom_generate_fn_receives_contexts_and_tracer_gets_spans():
    exporter = InMemoryExporter()
    pipeline = _pipeline(
        generate_fn=lambda q, ctx: f"{len(ctx)} contexts", tracer=Tracer(exporters=[exporter])
    )
    result = AgenticRag(pipeline).run("When was the Eiffel Tower completed?", k=2)

    assert result.answer.answer.endswith("contexts")
    names = {s.name for s in exporter.spans}
    assert {"agent", "agent.retrieve", "agent.generate"} <= names


def test_result_serializes_to_json():
    result = AgenticRag(_pipeline()).run("When was the Eiffel Tower completed?", k=1)
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["steps"][0]["kind"] == "retrieve" and payload["abstained"] is False


@pytest.mark.parametrize(
    "kwargs", [{"max_iterations": 0}, {"min_sufficiency": 1.5}, {"abstain_below": -0.1}]
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        AgenticRag(_pipeline(), **kwargs)


def test_llm_grade_parses_json_and_fails_soft():
    grader = llm_grade_fn(lambda p: 'Sure {"score": 0.8, "missing": ["date"], "reason": "ok"}')
    grade = grader("q", ["ctx"])
    assert (grade.score, grade.missing, grade.reason) == (0.8, ["date"], "ok")

    clamped = llm_grade_fn(lambda p: '{"score": 7}')("q", ["ctx"])
    assert clamped.score == 1.0

    fallback = llm_grade_fn(lambda p: "gibberish")("eiffel tower", ["The Eiffel Tower"])
    assert fallback.score == 1.0  # lexical fallback


def test_llm_rewrite_takes_first_line_and_strips_quotes():
    rewrite = llm_rewrite_fn(lambda p: '"eiffel tower construction date"\nbecause...')
    assert rewrite("q", Grade(0.2, ["date"]), ["q"]) == "eiffel tower construction date"
    assert llm_rewrite_fn(lambda p: "   ")("q", Grade(0.2), []) == ""


def test_llm_decompose_parses_list_and_caps_parts():
    decompose = llm_decompose_fn(lambda p: 'Here: ["a b", "c d", "e f"]', max_parts=2)
    assert decompose("q") == ["a b", "c d"]
    assert llm_decompose_fn(lambda p: "nope")("original") == ["original"]
    assert llm_decompose_fn(lambda p: "[1, ")("original") == ["original"]
