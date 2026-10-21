"""Tests for the keyless extractive answerer."""

from __future__ import annotations

import pytest

from ragforge import RagPipeline
from ragforge.citations import attribute
from ragforge.extractive import NO_ANSWER, extractive_generate_fn

CONTEXTS = [
    "The Eiffel Tower is in Paris. It was completed in 1889. Visitors queue for hours.",
    "Python was created by Guido van Rossum. It first appeared in 1991.",
]


def test_picks_the_sentence_that_covers_the_rarest_question_terms():
    answer = extractive_generate_fn(max_sentences=1)("completed 1889", CONTEXTS)
    assert answer == "It was completed in 1889."


def test_two_sentences_cover_a_two_part_question():
    answer = extractive_generate_fn(max_sentences=2)(
        "Where is the Eiffel Tower and when was it completed?", CONTEXTS
    )
    assert answer == "The Eiffel Tower is in Paris. It was completed in 1889."


def test_returns_sentences_in_reading_order():
    answer = extractive_generate_fn(max_sentences=2)(
        "Who created Python and when did it first appear?", CONTEXTS
    )
    assert answer == "Python was created by Guido van Rossum. It first appeared in 1991."


def test_rare_terms_outweigh_common_ones():
    contexts = ["Paris is nice. Paris has a tower. The tower is tall. Paris Paris Paris."]
    answer = extractive_generate_fn(max_sentences=1)("tower Paris", contexts)
    assert answer == "Paris has a tower."


def test_no_overlap_or_empty_input_yields_no_answer():
    generate = extractive_generate_fn()
    assert generate("quantum zeppelins", CONTEXTS) == NO_ANSWER
    assert generate("the of and", CONTEXTS) == NO_ANSWER
    assert generate("Eiffel", []) == NO_ANSWER
    assert generate("Eiffel", ["   "]) == NO_ANSWER


def test_min_score_filters_weak_matches():
    strict = extractive_generate_fn(min_score=0.9)
    assert strict("Who created Python in 1991 at Google?", CONTEXTS) == NO_ANSWER


def test_answers_are_verbatim_so_citations_are_fully_grounded():
    pipeline = RagPipeline(generate_fn=extractive_generate_fn())
    pipeline.ingest("eiffel", CONTEXTS[0])
    pipeline.ingest("python", CONTEXTS[1])
    response = pipeline.answer("When was the Eiffel Tower completed?", k=2)
    cited = attribute(response.answer, response.contexts)
    assert cited.groundedness == 1.0 and cited.sentences[0].citations[0].doc_id == "eiffel"


@pytest.mark.parametrize("kwargs", [{"max_sentences": 0}, {"min_score": 2.0}, {"min_score": -1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        extractive_generate_fn(**kwargs)
