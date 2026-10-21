"""Tests for sentence-level answer attribution and groundedness scoring."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ragforge.citations import attribute
from ragforge.embeddings import hashing_embed


@dataclass
class Ctx:
    text: str
    chunk_id: str
    doc_id: str
    start_char: int = 0


EIFFEL = Ctx(
    "The Eiffel Tower is in Paris. It was completed in 1889 and stands 330 meters tall.",
    "c1",
    "eiffel",
    start_char=100,
)
PYTHON = Ctx("Python was created by Guido van Rossum. It first appeared in 1991.", "c2", "python")
CONTEXTS = [EIFFEL, PYTHON]


def test_supported_sentence_cites_exact_offsets_in_source_document():
    cited = attribute("The tower was completed in 1889.", CONTEXTS)

    sentence = cited.sentences[0]
    assert sentence.supported
    citation = sentence.citations[0]
    assert (citation.marker, citation.doc_id, citation.chunk_id) == (1, "eiffel", "c1")
    assert citation.quote == "It was completed in 1889 and stands 330 meters tall."
    # Offsets are relative to the ORIGINAL document: chunk start (100) + offset in chunk.
    assert citation.start_char == 100 + EIFFEL.text.index("It was completed")
    assert citation.end_char == citation.start_char + len(citation.quote)


def test_hallucinated_sentence_is_flagged_and_lowers_groundedness():
    answer = "The tower was completed in 1889. It was designed by Leonardo da Vinci."
    cited = attribute(answer, CONTEXTS)

    assert [s.supported for s in cited.sentences] == [True, False]
    assert cited.sentences[1].citations == []
    assert cited.groundedness == pytest.approx(0.5)
    assert cited.unsupported == [cited.sentences[1]]


def test_multi_source_answer_cites_each_source():
    cited = attribute(
        "Python was created by Guido van Rossum. The Eiffel Tower stands 330 meters tall.",
        CONTEXTS,
    )
    assert [s.citations[0].marker for s in cited.sentences] == [2, 1]
    assert cited.groundedness == 1.0


def test_explicit_markers_are_verified_not_trusted():
    cited = attribute(
        "The tower was completed in 1889 [2]. Python first appeared in 1991 [2].", CONTEXTS
    )

    wrong, right = cited.sentences
    assert wrong.claimed_markers == [2] and wrong.invalid_markers == [2]
    assert right.claimed_markers == [2] and right.invalid_markers == []


def test_empty_answer_and_empty_contexts():
    assert attribute("", CONTEXTS).groundedness == 0.0
    cited = attribute("Something unsupported.", [])
    assert cited.groundedness == 0.0 and not cited.sentences[0].supported


def test_stopword_only_sentence_is_never_supported():
    assert not attribute("It is what it is.", CONTEXTS).sentences[0].supported


def test_threshold_controls_strictness():
    answer = "The tower was completed in 1889 by Gustave Eiffel."
    assert attribute(answer, CONTEXTS, threshold=0.3).sentences[0].supported
    assert not attribute(answer, CONTEXTS, threshold=0.95).sentences[0].supported


def test_max_citations_limits_and_orders_by_marker():
    twin = Ctx("The Eiffel Tower is in Paris.", "c3", "twin")
    cited = attribute("The Eiffel Tower is in Paris.", [EIFFEL, twin, PYTHON], max_citations=2)
    assert [c.marker for c in cited.sentences[0].citations] == [1, 2]
    one = attribute("The Eiffel Tower is in Paris.", [EIFFEL, twin], max_citations=1)
    assert len(one.sentences[0].citations) == 1


def test_embedding_blend_is_bounded_and_weight_zero_equals_lexical():
    answer = "The tower was completed in 1889."
    lexical = attribute(answer, CONTEXTS).sentences[0].support
    blended = attribute(answer, CONTEXTS, embed_fn=hashing_embed).sentences[0].support
    neutral = attribute(answer, CONTEXTS, embed_fn=hashing_embed, semantic_weight=0.0)

    assert 0.0 <= blended <= 1.0
    assert neutral.sentences[0].support == pytest.approx(lexical)


def test_markdown_rendering_lists_sources_and_marks_unverified():
    cited = attribute("The tower was completed in 1889. It was painted pink.", CONTEXTS)
    md = cited.to_markdown()
    assert "The tower was completed in 1889. [1]" in md
    assert "It was painted pink. (unverified)" in md
    assert "[1] eiffel (" in md


def test_to_dict_roundtrips_to_json_types():
    import json

    payload = attribute("The tower was completed in 1889.", CONTEXTS).to_dict()
    assert json.loads(json.dumps(payload))["groundedness"] == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [{"threshold": 1.5}, {"max_citations": 0}, {"semantic_weight": -0.1}],
)
def test_invalid_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        attribute("x", CONTEXTS, **kwargs)
