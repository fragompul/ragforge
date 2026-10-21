"""Agentic RAG: a bounded retrieve -> grade -> rewrite loop with abstention.

Single-shot RAG retrieves once and hopes. *Agentic* RAG (Self-RAG, CRAG,
Adaptive-RAG) treats retrieval as a decision process: look at what came back,
decide whether it is enough, and if not, change strategy. :class:`AgenticRag`
implements the smallest loop that captures the idea, with the guard-rails a
production system needs:

1. **Decompose** a multi-part question into sub-questions (optional).
2. **Retrieve** for each sub-question through the normal pipeline (hybrid
   search, reranking, guardrails, tracing all still apply).
3. **Grade** whether the retrieved context actually covers the question.
4. **Rewrite** the query when it does not, and retry -- up to
   ``max_iterations`` times, never repeating a query.
5. **Abstain** instead of hallucinating when the evidence is still insufficient.

Every step is recorded in :class:`AgentStep` objects (and as tracer spans), so
the reasoning is inspectable -- the web demo renders them as a timeline.

All decision points are plain callables. Defaults are deterministic, lexical,
and offline; ``llm_grade_fn`` / ``llm_rewrite_fn`` / ``llm_decompose_fn`` build
model-backed versions from any ``prompt -> text`` function.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ragforge.index import ENGLISH_STOPWORDS, FilterFn
from ragforge.pipeline import RagAnswer, RagPipeline, RetrievedChunk

_WORD = re.compile(r"[a-z0-9]+")
_STOP = ENGLISH_STOPWORDS

ABSTAIN_MESSAGE = "I don't have enough information in the knowledge base to answer that."


def _terms(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


def _stem(term: str) -> str:
    """Crude suffix-insensitive key so 'towers' matches 'tower' and 'builds' matches 'build'."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


@dataclass
class Grade:
    """How well a set of contexts covers a question."""

    score: float  # 0..1
    missing: list[str] = field(default_factory=list)
    reason: str = ""
    contexts: list[str] = field(default_factory=list)  # what was graded; set by the agent loop

    def to_dict(self) -> dict[str, Any]:
        return {"score": round(self.score, 4), "missing": list(self.missing), "reason": self.reason}


GradeFn = Callable[[str, list[str]], Grade]
RewriteFn = Callable[[str, Grade, list[str]], str]
DecomposeFn = Callable[[str], list[str]]
CompleteFn = Callable[[str], str]


def lexical_grade_fn(question: str, contexts: list[str]) -> Grade:
    """Coverage of the question's content terms by the union of the contexts."""
    wanted = {_stem(t): t for t in _terms(question)}
    if not wanted:
        return Grade(1.0, [], "question has no content terms")
    available = {_stem(t) for ctx in contexts for t in _terms(ctx)}
    missing = [original for stem, original in wanted.items() if stem not in available]
    score = 1.0 - len(missing) / len(wanted)
    return Grade(
        score, missing, f"{len(wanted) - len(missing)}/{len(wanted)} question terms covered"
    )


def relax_rewrite_fn(question: str, grade: Grade, previous: list[str]) -> str:
    """Drop terms that no retrieved context contains, keeping the part that can match."""
    drop = {t.lower() for t in grade.missing}
    kept = [w for w in question.split() if re.sub(r"\W", "", w).lower() not in drop]
    relaxed = " ".join(kept).strip()
    return relaxed if _terms(relaxed) else ""  # nothing searchable left: give up


def bridge_rewrite_fn(max_entities: int = 3, top_contexts: int = 2) -> RewriteFn:
    """Multi-hop rewrite: append *bridge entities* found in the best retrieved contexts.

    Questions like "Which team does the player who scored X now play for?" need a second
    retrieval hop whose target is named in the *first* hop's text, not in the question.
    This rewrite reads the entities mentioned by the top retrieved contexts (capitalized
    names, identifiers) that the question does not already contain, and appends the first
    ``max_entities`` of them to the question -- so BM25 and the embedder can reach the
    second document. It is deterministic and model-free; pair it with
    ``AgenticRag(min_iterations=2)``.
    """
    from ragforge.graph import entity_key, extract_entities

    def _rewrite(question: str, grade: Grade, previous: list[str]) -> str:
        seen = {entity_key(e) for e in extract_entities(question)}
        seen |= {entity_key(q) for q in previous}
        bridges: list[str] = []
        for context in grade.contexts[:top_contexts]:
            for entity in extract_entities(context):
                key = entity_key(entity)
                if key and key not in seen and not any(key in q.lower() for q in previous):
                    seen.add(key)
                    bridges.append(entity)
        return f"{question} {' '.join(bridges[:max_entities])}".strip() if bridges else ""

    return _rewrite


def heuristic_decompose_fn(question: str) -> list[str]:
    """Split compound questions on '?' / ' and ' when each part still looks like a question."""
    parts = [
        p.strip(" ?.")
        for p in re.split(
            r"\?\s+|;\s+|\s+and\s+(?=(?:how|what|when|where|who|why|which)\b)", question
        )
    ]
    parts = [p for p in parts if len(_terms(p)) >= 2]
    return parts if len(parts) > 1 else [question]


def llm_grade_fn(complete_fn: CompleteFn) -> GradeFn:
    """Model-backed grader: asks for JSON ``{"score": 0-1, "missing": [...]}``."""

    def _grade(question: str, contexts: list[str]) -> Grade:
        joined = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1)) or "(none)"
        raw = complete_fn(
            "Decide whether the CONTEXT contains enough information to answer the QUESTION. "
            'Reply with ONLY JSON: {"score": <0.0-1.0 sufficiency>, "missing": [<facts still '
            'needed>], "reason": "<short>"}.\n\n'
            f"<question>\n{question}\n</question>\n<context>\n{joined}\n</context>"
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            data = json.loads(match.group()) if match else {}
            score = max(0.0, min(1.0, float(data["score"])))
        except (ValueError, KeyError, TypeError):
            return lexical_grade_fn(question, contexts)  # fail soft, never crash the loop
        return Grade(score, [str(m) for m in data.get("missing", [])], str(data.get("reason", "")))

    return _grade


def llm_rewrite_fn(complete_fn: CompleteFn) -> RewriteFn:
    """Model-backed query rewriter that targets what the grader said was missing."""

    def _rewrite(question: str, grade: Grade, previous: list[str]) -> str:
        tried = "\n".join(f"- {q}" for q in previous)
        out = complete_fn(
            "Rewrite the search query to retrieve the missing information. Use different "
            "wording than earlier attempts. Reply with ONLY the new query.\n\n"
            f"Question: {question}\nStill missing: {', '.join(grade.missing) or 'unknown'}\n"
            f"Earlier queries:\n{tried}"
        )
        return out.strip().splitlines()[0].strip().strip("\"'") if out.strip() else ""

    return _rewrite


def llm_decompose_fn(complete_fn: CompleteFn, max_parts: int = 4) -> DecomposeFn:
    """Model-backed decomposition into independent sub-questions (JSON list)."""

    def _decompose(question: str) -> list[str]:
        raw = complete_fn(
            f"Split the question into at most {max_parts} independent sub-questions needed to "
            "answer it. If it is already simple, return it unchanged. Reply with ONLY a JSON "
            f"list of strings.\n\nQuestion: {question}"
        )
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        try:
            parts = [str(p).strip() for p in json.loads(match.group())] if match else []
        except (ValueError, TypeError):
            parts = []
        return [p for p in parts if p][:max_parts] or [question]

    return _decompose


@dataclass
class AgentStep:
    """One recorded decision or action of the agent."""

    kind: str  # decompose | retrieve | grade | rewrite | stop | answer | abstain
    query: str = ""
    detail: str = ""
    score: float | None = None
    chunk_ids: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "query": self.query,
            "detail": self.detail,
            "score": None if self.score is None else round(self.score, 4),
            "chunk_ids": list(self.chunk_ids),
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass
class AgentResult:
    """Outcome of an agentic run: the answer plus the full decision trail."""

    question: str
    answer: RagAnswer
    steps: list[AgentStep]
    sub_questions: list[str]
    sufficiency: float
    abstained: bool
    retrievals: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "sub_questions": list(self.sub_questions),
            "sufficiency": round(self.sufficiency, 4),
            "abstained": self.abstained,
            "retrievals": self.retrievals,
        }


class AgenticRag:
    """Bounded retrieve/grade/rewrite loop on top of a :class:`RagPipeline`.

    Args:
        pipeline: Any configured pipeline (its guardrails, reranker, tracer apply).
        grade_fn: Sufficiency grader (default :func:`lexical_grade_fn`).
        rewrite_fn: Query rewriter (default :func:`relax_rewrite_fn`).
        decompose_fn: Optional question decomposer; ``None`` keeps one sub-question.
        max_iterations: Max retrieval attempts per sub-question.
        min_iterations: Minimum attempts even when the first retrieval already looks sufficient
            (multi-hop questions: the first hop can cover every question term and still miss
            the second document). Stops early only if the rewrite yields no new query.
        min_sufficiency: Grade at which retrieval is considered good enough.
        abstain_below: If the mean best grade ends below this, abstain instead of answering.
    """

    def __init__(
        self,
        pipeline: RagPipeline,
        grade_fn: GradeFn = lexical_grade_fn,
        rewrite_fn: RewriteFn = relax_rewrite_fn,
        decompose_fn: DecomposeFn | None = None,
        max_iterations: int = 3,
        min_sufficiency: float = 0.7,
        abstain_below: float = 0.3,
        min_iterations: int = 1,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if not 1 <= min_iterations <= max_iterations:
            raise ValueError(
                f"min_iterations must be in [1, {max_iterations}], got {min_iterations}"
            )
        for name, value in (("min_sufficiency", min_sufficiency), ("abstain_below", abstain_below)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        self.pipeline = pipeline
        self.grade_fn = grade_fn
        self.rewrite_fn = rewrite_fn
        self.decompose_fn = decompose_fn
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.min_sufficiency = min_sufficiency
        self.abstain_below = abstain_below

    def run(
        self,
        question: str,
        k: int = 5,
        candidate_pool: int = 20,
        filter_fn: FilterFn | None = None,
    ) -> AgentResult:
        """Answer ``question``, retrying retrieval when the evidence looks insufficient."""
        started = time.perf_counter()
        steps: list[AgentStep] = []
        retrievals = 0

        with self.pipeline.tracer.span("agent", question=question):
            sub_questions = [question]
            if self.decompose_fn is not None:
                t0 = time.perf_counter()
                sub_questions = self.decompose_fn(question) or [question]
                steps.append(
                    AgentStep(
                        "decompose",
                        question,
                        " | ".join(sub_questions),
                        duration_ms=(time.perf_counter() - t0) * 1000,
                    )
                )

            per_sub: list[list[RetrievedChunk]] = []
            best_grades: list[float] = []
            most_attempts = 1
            for sub in sub_questions:
                pool: dict[str, RetrievedChunk] = {}
                tried: list[str] = []
                query = sub
                best = 0.0
                for attempt in range(1, self.max_iterations + 1):
                    tried.append(query)
                    t0 = time.perf_counter()
                    hop_filter = filter_fn
                    if self.min_iterations > 1 and pool:
                        # Later hops must surface *new* evidence, not re-rank the first hop.
                        hop_filter = _exclude_seen(set(pool), filter_fn)
                    with self.pipeline.tracer.span("agent.retrieve", query=query, attempt=attempt):
                        found = self.pipeline.retrieve(
                            query, k=k, candidate_pool=candidate_pool, filter_fn=hop_filter
                        )
                    retrievals += 1
                    for chunk in found:
                        kept = pool.get(chunk.chunk_id)
                        if kept is None or chunk.score > kept.score:
                            pool[chunk.chunk_id] = chunk
                    steps.append(
                        AgentStep(
                            "retrieve",
                            query,
                            f"attempt {attempt}",
                            chunk_ids=[c.chunk_id for c in found],
                            duration_ms=(time.perf_counter() - t0) * 1000,
                        )
                    )

                    ranked = sorted(pool.values(), key=lambda c: c.score, reverse=True)
                    t0 = time.perf_counter()
                    graded_texts = [c.text for c in ranked[: max(k, 1)]]
                    grade = self.grade_fn(sub, graded_texts)
                    grade.contexts = graded_texts
                    best = max(best, grade.score)
                    steps.append(
                        AgentStep(
                            "grade",
                            sub,
                            grade.reason or f"missing: {', '.join(grade.missing) or 'nothing'}",
                            score=grade.score,
                            duration_ms=(time.perf_counter() - t0) * 1000,
                        )
                    )
                    if grade.score >= self.min_sufficiency and attempt >= self.min_iterations:
                        break
                    if attempt == self.max_iterations:
                        steps.append(AgentStep("stop", sub, "iteration budget exhausted"))
                        break

                    new_query = self.rewrite_fn(sub, grade, tried).strip()
                    if not new_query or new_query in tried:
                        steps.append(AgentStep("stop", sub, "no new query to try"))
                        break
                    steps.append(AgentStep("rewrite", new_query, f"was: {query}"))
                    query = new_query

                per_sub.append(sorted(pool.values(), key=lambda c: c.score, reverse=True))
                best_grades.append(best)
                most_attempts = max(most_attempts, len(tried))

            # Multi-hop runs keep each hop's evidence: the context budget grows with the hops taken.
            budget = k * most_attempts if self.min_iterations > 1 else k
            contexts = _interleave(per_sub, limit=max(budget, len(sub_questions)))
            sufficiency = sum(best_grades) / len(best_grades)
            abstained = sufficiency < self.abstain_below or not contexts

            if abstained:
                steps.append(AgentStep("abstain", question, f"sufficiency {sufficiency:.2f}"))
                text = ABSTAIN_MESSAGE
                prompt = ""
                generation_ms = 0.0
            else:
                t0 = time.perf_counter()
                texts = [c.text for c in contexts]
                prompt = self.pipeline.prompt_formatter(question, texts)
                with self.pipeline.tracer.span("agent.generate", contexts=len(texts)):
                    text = self.pipeline.generate_fn(question, texts)
                generation_ms = (time.perf_counter() - t0) * 1000
                steps.append(
                    AgentStep(
                        "answer",
                        question,
                        f"{len(texts)} contexts",
                        chunk_ids=[c.chunk_id for c in contexts],
                        duration_ms=generation_ms,
                    )
                )

        total_ms = (time.perf_counter() - started) * 1000
        answer = RagAnswer(
            query=question,
            answer=text,
            contexts=contexts,
            prompt=prompt,
            retrieval_latency_ms=total_ms - generation_ms,
            generation_latency_ms=generation_ms,
            total_latency_ms=total_ms,
            metadata={"agentic": True, "retrievals": retrievals, "abstained": abstained},
        )
        return AgentResult(
            question=question,
            answer=answer,
            steps=steps,
            sub_questions=sub_questions,
            sufficiency=sufficiency,
            abstained=abstained,
            retrievals=retrievals,
        )


def _exclude_seen(seen: set[str], base: FilterFn | None) -> FilterFn:
    """A chunk filter that rejects already-retrieved chunk ids (and applies ``base``)."""

    def _accept(chunk: Any) -> bool:
        return chunk.id not in seen and (base is None or base(chunk))

    return _accept


def _interleave(groups: list[list[RetrievedChunk]], limit: int) -> list[RetrievedChunk]:
    """Round-robin across sub-question results so every sub-question stays represented."""
    out: list[RetrievedChunk] = []
    seen: set[str] = set()
    depth = max((len(g) for g in groups), default=0)
    for i in range(depth):
        for group in groups:
            if i < len(group) and group[i].chunk_id not in seen:
                seen.add(group[i].chunk_id)
                out.append(group[i])
                if len(out) >= limit:
                    return out
    return out
