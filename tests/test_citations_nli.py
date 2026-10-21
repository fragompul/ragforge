"""NLI-backed attribution: paraphrase support and contradiction detection."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ragforge.citations import attribute, nli_cross_encoder_fn


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
)


WRONG_DATE = "It was completed in 1925 and stands 330 meters tall."


def fake_nli(premise: str, hypothesis: str) -> float:
    """A deterministic stand-in: knows one paraphrase and one contradiction."""
    if "330 meters" in premise and "metres high" in hypothesis:
        return 0.97  # paraphrase with almost no shared vocabulary beyond numbers
    if "completed in 1889" in premise and "1925" in hypothesis:
        return 0.02  # contradiction
    return 0.1


def test_lexical_only_is_blind_to_paraphrase_and_fooled_by_contradiction():
    paraphrase = attribute("The tower is 330 metres high.", [EIFFEL], threshold=0.6)
    contradiction = attribute(WRONG_DATE, [EIFFEL], threshold=0.6)
    assert not paraphrase.sentences[0].supported
    assert contradiction.sentences[0].supported  # word overlap cannot tell 1889 from 1925


def test_entailment_fn_supports_paraphrase_and_rejects_contradiction():
    paraphrase = attribute("The tower is 330 metres high.", [EIFFEL], entailment_fn=fake_nli)
    contradiction = attribute(WRONG_DATE, [EIFFEL], threshold=0.6, entailment_fn=fake_nli)

    assert paraphrase.sentences[0].supported
    assert paraphrase.sentences[0].citations[0].quote.startswith("It was completed in 1889")
    assert not contradiction.sentences[0].supported
    assert contradiction.groundedness == 0.0


def test_support_score_blends_lexical_and_entailment():
    answer = "It was completed in 1889."
    pure_nli = attribute(answer, [EIFFEL], entailment_fn=lambda p, h: 0.5, nli_weight=1.0)
    pure_lex = attribute(answer, [EIFFEL], entailment_fn=lambda p, h: 0.5, nli_weight=0.0)
    assert pure_nli.sentences[0].support == pytest.approx(0.5)
    assert pure_lex.sentences[0].support == pytest.approx(1.0)


def test_only_candidate_spans_are_sent_to_the_model():
    seen = []

    def spy(premise, hypothesis):
        seen.append(premise)
        return 0.9

    long_context = Ctx(" ".join(f"Sentence number {i} about nothing." for i in range(30)), "c", "d")
    attribute("A claim about nothing.", [long_context], entailment_fn=spy, nli_candidates=2)
    assert len(seen) == 2
    seen.clear()
    attribute("A claim about nothing.", [long_context], entailment_fn=spy, nli_candidates=5)
    assert len(seen) == 5


def test_entailment_output_is_clamped_and_stopword_sentences_are_skipped():
    out = attribute("Totally fine claim here.", [EIFFEL], entailment_fn=lambda p, h: 7.0)
    assert out.sentences[0].support <= 1.0
    calls = []
    attribute("It is what it is.", [EIFFEL], entailment_fn=lambda p, h: calls.append(1) or 1.0)
    assert calls == []  # no content terms -> nothing to verify


def test_entailment_replaces_the_embedding_blend():
    from ragforge.embeddings import hashing_embed

    boom = lambda text: (_ for _ in ()).throw(AssertionError("embed_fn must not be used"))  # noqa: E731
    out = attribute(
        "The tower is 330 metres high.", [EIFFEL], embed_fn=boom, entailment_fn=fake_nli
    )
    assert out.sentences[0].supported
    assert attribute("x", [EIFFEL], embed_fn=hashing_embed) is not None


@pytest.mark.parametrize(
    "kwargs", [{"nli_weight": 1.5}, {"nli_weight": -0.1}, {"nli_candidates": 0}]
)
def test_invalid_nli_arguments(kwargs):
    with pytest.raises(ValueError):
        attribute("x", [EIFFEL], entailment_fn=fake_nli, **kwargs)


def test_nli_cross_encoder_adapter_reads_labels_and_softmaxes():
    import types

    import numpy as np

    class Stub:
        config = types.SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})

        def predict(self, pairs, apply_softmax):
            assert apply_softmax is True and pairs == [("p", "h")]
            return np.array([[0.1, 0.8, 0.1]])

    assert nli_cross_encoder_fn(model=Stub())("p", "h") == pytest.approx(0.8)

    class DifferentOrder(Stub):
        config = types.SimpleNamespace(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})

        def predict(self, pairs, apply_softmax):
            return np.array([[0.7, 0.2, 0.1]])

    assert nli_cross_encoder_fn(model=DifferentOrder())("p", "h") == pytest.approx(0.7)

    class NoConfig:
        def predict(self, pairs, apply_softmax):
            return np.array([[0.0, 0.9, 0.1]])

    assert nli_cross_encoder_fn(model=NoConfig())("p", "h") == pytest.approx(0.9)


def test_nli_adapter_reports_missing_dependency(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="ragforge\\[local\\]"):
        nli_cross_encoder_fn()
