"""Extractive answering: a keyless, deterministic ``GenerateFn``.

Without an LLM, the pipeline's built-in fallback returns the whole top chunk.
:func:`extractive_generate_fn` does better: it picks the sentences that best
cover the question's terms, weighting rarer terms more, and returns them in
reading order. Every returned sentence is copied verbatim from a retrieved
context, so the answer is -- by construction -- perfectly faithful, and
:func:`ragforge.citations.attribute` can trace it to exact character offsets.

It is a baseline for demos, tests, and offline use, not a substitute for a
generative model: it cannot synthesize across passages or paraphrase.
"""

from __future__ import annotations

import math
from collections import Counter

from ragforge.citations import _sentences
from ragforge.index import english_analyzer
from ragforge.pipeline import GenerateFn

NO_ANSWER = "I don't have enough information to answer that."


def extractive_generate_fn(max_sentences: int = 2, min_score: float = 0.0) -> GenerateFn:
    """Build a ``GenerateFn`` that answers with the best-matching context sentences.

    Args:
        max_sentences: Maximum sentences in the answer.
        min_score: Minimum IDF-weighted query coverage in ``[0, 1]`` a sentence needs.
    """
    if max_sentences < 1:
        raise ValueError(f"max_sentences must be >= 1, got {max_sentences}")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"min_score must be in [0, 1], got {min_score}")

    def _generate(query: str, contexts: list[str]) -> str:
        query_terms = set(english_analyzer(query))
        if not query_terms or not contexts:
            return NO_ANSWER

        sentences: list[tuple[int, int, str]] = []  # (context idx, sentence idx, text)
        for ci, context in enumerate(contexts):
            for si, (start, end) in enumerate(_sentences(context)):
                sentences.append((ci, si, context[start:end]))
        if not sentences:
            return NO_ANSWER

        analyzed = [set(english_analyzer(text)) for _, _, text in sentences]
        doc_freq = Counter(term for terms in analyzed for term in terms)
        n = len(sentences)
        idf = {t: math.log(1 + n / (1 + doc_freq.get(t, 0))) for t in query_terms}
        total = sum(idf.values())

        scored = []
        for position, terms in enumerate(analyzed):
            coverage = sum(idf[t] for t in query_terms & terms) / total
            if coverage > min_score:
                scored.append((coverage, -position, position))
        if not scored:
            return NO_ANSWER

        best = sorted(scored, reverse=True)[:max_sentences]
        chosen = sorted(position for _, _, position in best)
        return " ".join(sentences[p][2] for p in chosen)

    return _generate
