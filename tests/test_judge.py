"""Tests for the LLM-as-judge module (prompting, parsing, aggregation)."""

from __future__ import annotations

import json

import pytest

from ragforge import RagPipeline
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.evaluation import EvalCase
from ragforge.judge import (
    JudgeParseError,
    build_judge_prompt,
    evaluate_with_judge,
    judge,
    judge_answer_relevancy,
    judge_context_relevancy,
    judge_correctness,
    judge_faithfulness,
    lexical_judge_fn,
    parse_verdict,
)


def _json(score, reasoning="because"):
    return json.dumps({"reasoning": reasoning, "score": score})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (_json(5), 5),
        (_json(1), 1),
        (f"Sure!\n```json\n{_json(4)}\n```", 4),
        (f"Here you go: {_json(3)} hope it helps", 3),
        (_json(9), 5),  # clamped
        (_json(0), 1),  # clamped
        (_json(3.6), 4),
        ('{"reasoning": "x", "score": "2"}', 2),
        ("I'd say score: 4 overall", 4),
    ],
)
def test_parse_verdict_accepts_messy_outputs(raw, expected):
    assert parse_verdict("faithfulness", raw).raw_score == expected


def test_verdict_normalizes_to_unit_interval():
    assert parse_verdict("faithfulness", _json(1)).score == 0.0
    assert parse_verdict("faithfulness", _json(3)).score == 0.5
    assert parse_verdict("faithfulness", _json(5)).score == 1.0
    assert parse_verdict("faithfulness", _json(5, "good")).reasoning == "good"


@pytest.mark.parametrize("raw", ["", "no number here", '{"reasoning": "x"}', '{"score": "high"}'])
def test_unparseable_output_raises(raw):
    with pytest.raises(JudgeParseError):
        parse_verdict("faithfulness", raw)


def test_prompt_wraps_data_and_demands_json():
    prompt = build_judge_prompt(
        "faithfulness", question="Q?", answer="A.", contexts=["c1", "c2"], reference="R"
    )
    for fragment in [
        "<question>",
        "<answer>",
        "<context>",
        "[2] c2",
        "<reference>",
        "DATA",
        "JSON",
    ]:
        assert fragment in prompt
    assert "FAITHFULNESS" in prompt


def test_unknown_rubric_rejected():
    with pytest.raises(ValueError, match="Unknown rubric"):
        build_judge_prompt("vibes")  # type: ignore[arg-type]


def test_convenience_wrappers_route_to_the_right_rubric():
    seen: list[str] = []

    def fake(prompt: str) -> str:
        seen.append(prompt)
        return _json(4)

    judge_faithfulness("q", "a", ["c"], fake)
    judge_answer_relevancy("q", "a", fake)
    judge_correctness("q", "a", "ref", fake)
    judge_context_relevancy("q", ["c"], fake)
    markers = ["FAITHFULNESS", "ANSWER RELEVANCY", "CORRECTNESS", "CONTEXT RELEVANCY"]
    assert [m in p for m, p in zip(markers, seen, strict=True)] == [True] * 4
    assert judge("correctness", fake, answer="a", reference="r").rubric == "correctness"


def test_lexical_judge_scores_overlap_monotonically():
    ctx = ["The Eiffel Tower is in Paris and was completed in 1889."]
    good = judge_faithfulness("q", "The Eiffel Tower was completed in 1889.", ctx, lexical_judge_fn)
    bad = judge_faithfulness("q", "Bananas grow in tropical climates.", ctx, lexical_judge_fn)
    assert good.score > bad.score
    assert good.raw_score == 5 and bad.raw_score == 1
    assert judge_correctness("q", "Paris", "Paris", lexical_judge_fn).score == 1.0
    assert judge_answer_relevancy("tower height", "tower height is big", lexical_judge_fn).score > 0
    assert judge_context_relevancy("eiffel tower", ctx, lexical_judge_fn).score > 0


def _pipeline() -> RagPipeline:
    pipeline = RagPipeline(chunker=RecursiveCharacterChunker(chunk_size=200, chunk_overlap=0))
    pipeline.ingest("eiffel", "The Eiffel Tower is in Paris. It was completed in 1889.")
    pipeline.ingest("python", "Python was created by Guido van Rossum in 1991.")
    return pipeline


CASES = [
    EvalCase(
        query="When was the Eiffel Tower completed?",
        relevant_doc_ids=["eiffel"],
        ground_truth_answer="Completed in 1889.",
    ),
    EvalCase(query="Who created Python?", relevant_doc_ids=["python"]),
]


def test_evaluate_with_judge_aggregates_and_skips_correctness_without_reference():
    summary = evaluate_with_judge(
        _pipeline(),
        CASES,
        lexical_judge_fn,
        k=1,
        rubrics=("faithfulness", "answer_relevancy", "correctness"),
    )

    assert set(summary.cases[0].verdicts) == {"faithfulness", "answer_relevancy", "correctness"}
    assert set(summary.cases[1].verdicts) == {"faithfulness", "answer_relevancy"}
    assert summary.mean("faithfulness") == 1.0  # extractive answers are fully grounded
    assert summary.mean("nonexistent") is None
    assert "| **faithfulness** | `1.0000` |" in summary.to_markdown_table()
    assert json.loads(json.dumps(summary.to_dict()))["means"]["faithfulness"] == 1.0


def test_on_error_raise_vs_skip():
    broken = lambda prompt: "I refuse to grade"  # noqa: E731

    with pytest.raises(JudgeParseError):
        evaluate_with_judge(_pipeline(), CASES, broken, k=1)

    summary = evaluate_with_judge(_pipeline(), CASES, broken, k=1, on_error="skip")
    assert summary.means == {}
    assert set(summary.cases[0].errors) == {"faithfulness", "answer_relevancy"}
