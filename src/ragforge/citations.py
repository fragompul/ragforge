"""Verifiable citations: tie every sentence of an answer back to source spans.

"Trust me" answers are the main reason RAG gets rejected in regulated
settings. This module post-processes any answer -- extractive, LLM-written, or
already containing ``[1]``-style markers -- and produces an auditable
attribution:

- each answer sentence is matched to the best-supporting sentence of the
  retrieved chunks, with **character offsets into the original document**
  (``start_char``/``end_char`` of the chunk + offset of the matched sentence);
- each sentence gets a **support score** in ``[0, 1]`` (fraction of its content
  words covered by the cited span; optionally blended with embedding
  similarity), and is flagged ``supported`` or not;
- the answer gets a **groundedness** score = share of sentences supported,
  which doubles as a cheap, deterministic hallucination signal;
- explicit markers written by an LLM (``[1]``, ``[2]``) are *verified* rather
  than trusted: a marker pointing at a chunk that does not support the
  sentence is reported in ``invalid_markers``.

Pure lexical attribution is a conservative baseline (it cannot see through
paraphrase). Two optional upgrades:

* ``embed_fn`` blends embedding similarity into the support score;
* ``entailment_fn`` -- a natural-language-inference model, see
  :func:`nli_cross_encoder_fn` -- asks the *right* question: does the cited span
  **entail** the answer sentence? It recognizes paraphrase ("330 metres high" ~ "330
  meters tall") and, crucially, **contradiction** ("finished in 1925" vs. "completed in
  1889"), which word overlap scores as well supported. Lexical overlap still proposes the
  candidate spans, so the model runs on a handful of sentence pairs, not the whole context.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ragforge.embeddings import EmbedFn, cosine_similarity

_WORD = re.compile(r"[a-z0-9]+")
_MARKER = re.compile(r"\[(\d{1,3})\]")
_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "more",
        "most",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "she",
        "should",
        "so",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "up",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "about",
        "after",
        "all",
        "also",
        "any",
        "being",
        "both",
        "each",
        "few",
        "other",
        "some",
        "very",
    ]
)


class ContextLike(Protocol):
    """Anything shaped like :class:`ragforge.pipeline.RetrievedChunk`."""

    text: str
    chunk_id: str
    doc_id: str
    start_char: int


def _content_tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS]


def _sentences(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` offsets of non-empty sentences in ``text``."""
    spans = []
    for m in _SENTENCE.finditer(text):
        chunk = m.group()
        stripped = chunk.strip()
        if not stripped:
            continue
        start = m.start() + (len(chunk) - len(chunk.lstrip()))
        spans.append((start, start + len(stripped)))
    return spans


@dataclass(frozen=True)
class Citation:
    """A source span supporting one answer sentence."""

    marker: int  # 1-based position of the chunk in the provided contexts
    chunk_id: str
    doc_id: str
    start_char: int  # offsets in the ORIGINAL document
    end_char: int
    quote: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "quote": self.quote,
            "score": round(self.score, 4),
        }


@dataclass
class AttributedSentence:
    """One answer sentence with its supporting citations."""

    text: str
    start: int
    end: int
    support: float
    supported: bool
    citations: list[Citation] = field(default_factory=list)
    claimed_markers: list[int] = field(default_factory=list)
    invalid_markers: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "support": round(self.support, 4),
            "supported": self.supported,
            "citations": [c.to_dict() for c in self.citations],
            "claimed_markers": list(self.claimed_markers),
            "invalid_markers": list(self.invalid_markers),
        }


@dataclass
class CitedAnswer:
    """An answer with sentence-level attribution and a groundedness score."""

    answer: str
    sentences: list[AttributedSentence]
    groundedness: float

    @property
    def unsupported(self) -> list[AttributedSentence]:
        return [s for s in self.sentences if not s.supported]

    def to_markdown(self) -> str:
        """Answer with ``[n]`` markers appended per sentence, plus a source list."""
        parts: list[str] = []
        sources: dict[int, Citation] = {}
        for s in self.sentences:
            body = _MARKER.sub("", s.text).strip()
            body = re.sub(r"\s+([.!?])", r"\1", body)
            marks = "".join(f"[{c.marker}]" for c in s.citations)
            parts.append(f"{body} {marks}".strip() if s.supported else f"{body} (unverified)")
            for c in s.citations:
                sources.setdefault(c.marker, c)
        lines = [" ".join(parts), ""]
        for marker in sorted(sources):
            c = sources[marker]
            lines.append(f"[{marker}] {c.doc_id} ({c.start_char}-{c.end_char}): “{c.quote}”")
        return "\n".join(lines).rstrip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "groundedness": round(self.groundedness, 4),
            "sentences": [s.to_dict() for s in self.sentences],
        }


EntailmentFn = Callable[[str, str], float]


def nli_cross_encoder_fn(
    model_name: str = "cross-encoder/nli-deberta-v3-small", model: Any = None
) -> EntailmentFn:
    """Build an ``entailment_fn(premise, hypothesis) -> P(entailment)`` from an NLI cross-encoder.

    Requires ``pip install ragforge[local]`` (sentence-transformers). The default model is a
    small DeBERTa-v3 trained on SNLI/MultiNLI; any 3-way NLI cross-encoder whose labels
    include ``entailment`` works. Probabilities are a softmax over the model's logits.
    """
    if model is not None:
        resolved = model
    else:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "NLI attribution needs sentence-transformers. "
                "Install it with:\n    pip install 'ragforge[local]'"
            ) from exc
        resolved = CrossEncoder(model_name)
    id2label = getattr(getattr(resolved, "config", None), "id2label", None) or {}
    labels = [str(id2label[i]).lower() for i in sorted(id2label)] or [
        "contradiction",
        "entailment",
        "neutral",
    ]
    entail_index = labels.index("entailment")

    def _entailment(premise: str, hypothesis: str) -> float:
        probs = resolved.predict([(premise, hypothesis)], apply_softmax=True)[0]
        return float(probs[entail_index])

    return _entailment


def attribute(
    answer: str,
    contexts: Sequence[ContextLike],
    threshold: float = 0.6,
    max_citations: int = 2,
    embed_fn: EmbedFn | None = None,
    semantic_weight: float = 0.5,
    entailment_fn: EntailmentFn | None = None,
    nli_weight: float = 0.8,
    nli_candidates: int = 2,
) -> CitedAnswer:
    """Attribute each sentence of ``answer`` to the best-supporting context spans.

    Args:
        answer: The generated (or extractive) answer text.
        contexts: Retrieved chunks, in prompt order. Marker ``[n]`` refers to ``contexts[n-1]``.
        threshold: Minimum support score for a sentence to count as supported.
        max_citations: Maximum citations attached per sentence.
        embed_fn: If given, support = ``(1-w) * lexical + w * cosine`` of the best span.
        semantic_weight: Blend weight ``w`` for the embedding similarity.
        entailment_fn: ``(premise, hypothesis) -> P(entailment)``. When given it replaces the
            embedding blend: support = ``(1 - nli_weight) * lexical + nli_weight * entailment``
            for the best of the ``nli_candidates`` lexically closest spans of each context.
        nli_weight: Weight of the entailment probability in the support score.
        nli_candidates: Spans per context that are sent to ``entailment_fn``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if max_citations < 1:
        raise ValueError(f"max_citations must be >= 1, got {max_citations}")
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError(f"semantic_weight must be in [0, 1], got {semantic_weight}")
    if not 0.0 <= nli_weight <= 1.0:
        raise ValueError(f"nli_weight must be in [0, 1], got {nli_weight}")
    if nli_candidates < 1:
        raise ValueError(f"nli_candidates must be >= 1, got {nli_candidates}")

    # Pre-split every context into sentences once.
    ctx_spans = [[(a, b) for a, b in _sentences(c.text)] for c in contexts]
    ctx_sets = [
        [set(_content_tokens(c.text[a:b])) for a, b in spans]
        for c, spans in zip(contexts, ctx_spans, strict=True)
    ]

    out: list[AttributedSentence] = []
    for start, end in _sentences(answer):
        raw = answer[start:end]
        claimed = [int(m) for m in _MARKER.findall(raw)]
        clean = _MARKER.sub(" ", raw)
        tokens = set(_content_tokens(clean))

        scored: list[Citation] = []
        for ci, ctx in enumerate(contexts):
            if not tokens:
                continue
            ranked = sorted(
                (
                    (len(tokens & ctx_tokens) / len(tokens), si)
                    for si, ctx_tokens in enumerate(ctx_sets[ci])
                ),
                key=lambda pair: (-pair[0], pair[1]),
            )
            if entailment_fn is None:
                ranked = [pair for pair in ranked[:1] if pair[0] > 0]
            else:
                ranked = ranked[:nli_candidates]
            best: tuple[float, int] | None = None
            for coverage, si in ranked:
                a, b = ctx_spans[ci][si]
                score = coverage
                if entailment_fn is not None:
                    entailed = min(1.0, max(0.0, entailment_fn(ctx.text[a:b], clean.strip())))
                    score = (1.0 - nli_weight) * coverage + nli_weight * entailed
                elif embed_fn is not None:
                    sim = max(0.0, cosine_similarity(embed_fn(clean), embed_fn(ctx.text[a:b])))
                    score = (1.0 - semantic_weight) * coverage + semantic_weight * sim
                if best is None or score > best[0]:
                    best = (score, si)
            if best is None:
                continue
            a, b = ctx_spans[ci][best[1]]
            scored.append(
                Citation(
                    marker=ci + 1,
                    chunk_id=ctx.chunk_id,
                    doc_id=ctx.doc_id,
                    start_char=ctx.start_char + a,
                    end_char=ctx.start_char + b,
                    quote=ctx.text[a:b],
                    score=best[0],
                )
            )

        scored.sort(key=lambda c: (-c.score, c.marker))
        support = scored[0].score if scored else 0.0
        supported = support >= threshold and bool(tokens)
        citations = [c for c in scored if c.score >= threshold][:max_citations] if supported else []
        citations.sort(key=lambda c: c.marker)

        by_marker = {c.marker: c.score for c in scored}
        invalid = [m for m in claimed if by_marker.get(m, 0.0) < threshold]
        out.append(
            AttributedSentence(
                text=raw,
                start=start,
                end=end,
                support=support,
                supported=supported,
                citations=citations,
                claimed_markers=claimed,
                invalid_markers=invalid,
            )
        )

    groundedness = sum(1 for s in out if s.supported) / len(out) if out else 0.0
    return CitedAnswer(answer=answer, sentences=out, groundedness=groundedness)
