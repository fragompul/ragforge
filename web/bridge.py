"""Python side of the RagForge web playground.

This module runs **inside the browser** (Pyodide / WebAssembly) next to the real
``ragforge`` package: every number the UI shows comes from the library's own
code paths, not from a JavaScript re-implementation. All public functions take
and return JSON strings so the Web Worker boundary stays trivial.

Semantic embeddings
-------------------
Pyodide cannot call an asynchronous model synchronously, so the page embeds texts in a
separate worker (transformers.js) and hands the vectors over. The bridge makes that
transparent with a *fixed point*: when semantic mode is on and a function needs the
embedding of a text it has not seen, it records the text, returns ``{"__need_embeddings":
[...]}`` instead of a result, the page embeds those texts, registers them with
``add_embeddings`` and calls the function again. Retrieval is deterministic given the
embeddings, so the second pass is exactly the answer the first pass would have given.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from typing import Any

from ragforge import (
    AgenticRag,
    GraphIndex,
    Guardrails,
    InjectionDetector,
    PIIRedactor,
    QueryRouter,
    RagPipeline,
    attribute,
    evaluate_pipeline,
)
from ragforge.agent import bridge_rewrite_fn, heuristic_decompose_fn
from ragforge.ann import HNSWIndex
from ragforge.chunking import (
    FixedSizeChunker,
    MarkdownChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
)
from ragforge.contextual import title_context_fn
from ragforge.embeddings import hashing_embed
from ragforge.evaluation import EvalCase
from ragforge.extractive import extractive_generate_fn
from ragforge.guardrails import spotlight_prompt_formatter
from ragforge.index import ScoredChunk
from ragforge.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from ragforge.reranking import MaxMarginalRelevanceReranker, NoopReranker
from ragforge.sample import SAMPLE_DOCUMENTS, SAMPLE_EVAL_CASES, SAMPLE_QUERIES
from ragforge.telemetry import InMemoryExporter, Tracer

SEMANTIC_DIMS = 384  # all-MiniLM-L6-v2

_STATE: dict[str, Any] = {"pipeline": None, "exporter": None, "docs": [], "graph": None}
_SEM: dict[str, Any] = {"semantic": False, "table": {}, "missing": []}


# ----------------------------------------------------------------------------- embeddings


def _embed(text: str) -> list[float]:
    """Embedding function handed to every pipeline (hashing, or a registered semantic vector)."""
    if not _SEM["semantic"]:
        return hashing_embed(text)
    vector = _SEM["table"].get(text)
    if vector is None:
        _SEM["missing"].append(text)
        return [0.0] * SEMANTIC_DIMS
    return list(vector)  # type: ignore[no-any-return]


def _fixed_point(fn: Callable[[str], str]) -> Callable[[str], str]:
    """Return ``{"__need_embeddings": [...]}`` until every embedding the call needed is known."""

    def wrapper(config_json: str = "{}") -> str:
        _SEM["missing"] = []
        result = fn(config_json)
        if _SEM["missing"]:
            return json.dumps({"__need_embeddings": list(dict.fromkeys(_SEM["missing"]))})
        return result

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def set_embedding(config_json: str) -> str:
    """Switch between hashing (``semantic: false``) and registered semantic vectors."""
    _SEM["semantic"] = bool(json.loads(config_json).get("semantic", False))
    return json.dumps({"semantic": _SEM["semantic"], "known_vectors": len(_SEM["table"])})


def add_embeddings(config_json: str) -> str:
    """Register ``{"texts": [...], "vectors": [[...], ...]}`` computed by the page."""
    data = json.loads(config_json)
    for text, vector in zip(data["texts"], data["vectors"], strict=True):
        _SEM["table"][text] = vector
    return json.dumps({"known_vectors": len(_SEM["table"])})


# ----------------------------------------------------------------------------- helpers


def _chunker(cfg: dict[str, Any]):
    kind = cfg.get("type", "recursive")
    size = int(cfg.get("size", 500))
    overlap = int(cfg.get("overlap", 50))
    if kind == "sentence":
        return SentenceChunker(max_chars=size, overlap_sentences=max(0, min(overlap, 3)))
    if kind == "markdown":
        return MarkdownChunker(max_chars=size)
    if kind == "fixed":
        words = max(5, size // 6)
        return FixedSizeChunker(chunk_size=words, overlap=min(max(0, overlap // 6), words - 1))
    return RecursiveCharacterChunker(chunk_size=size, chunk_overlap=min(overlap, max(size - 1, 0)))


def _hit(sc: ScoredChunk, rank: int) -> dict[str, Any]:
    c = sc.chunk
    return {
        "rank": rank,
        "chunk_id": c.id,
        "doc_id": c.doc_id,
        "text": c.text,
        "score": sc.score,
        "start": c.start_char,
        "end": c.end_char,
        "ranks": dict(sc.ranks),
        "provenance": sc.provenance,
        "metadata": dict(c.metadata),
        "context": c.context,
    }


def _spans(exporter: InMemoryExporter) -> list[dict[str, Any]]:
    spans = exporter.spans
    if not spans:
        return []
    t0 = min(s.start_time for s in spans)
    return [
        {
            "name": s.name,
            "id": s.span_id,
            "parent": s.parent_id,
            "start_ms": (s.start_time - t0) * 1000,
            "duration_ms": s.duration_ms,
            "attributes": {k: (v if isinstance(v, int | float | bool) else str(v))
                           for k, v in s.attributes.items()},
        }
        for s in sorted(spans, key=lambda s: s.start_time)
    ]  # fmt: skip


def _pipeline() -> RagPipeline:
    pipeline = _STATE["pipeline"]
    if pipeline is None:
        raise RuntimeError("Call init() before using the pipeline")
    return pipeline  # type: ignore[no-any-return]


# ------------------------------------------------------------------------- corpus/index


def sample(_config_json: str = "{}") -> str:
    """The built-in handbook, example queries and evaluation cases."""
    return json.dumps(
        {"documents": SAMPLE_DOCUMENTS, "queries": SAMPLE_QUERIES, "eval_cases": SAMPLE_EVAL_CASES}
    )


@_fixed_point
def init(config_json: str) -> str:
    """(Re)build the index. Config: docs, chunker, analyzer, backend, graph, contextual."""
    cfg = json.loads(config_json)
    docs = cfg.get("docs") or SAMPLE_DOCUMENTS
    exporter = InMemoryExporter()
    graph = GraphIndex() if cfg.get("graph") else None
    pipeline = RagPipeline(
        chunker=_chunker(cfg.get("chunker", {})),
        embed_fn=_embed,
        bm25_analyzer=cfg.get("analyzer", "english"),
        vector_backend=cfg.get("backend", "brute_force"),
        generate_fn=extractive_generate_fn(max_sentences=2),
        tracer=Tracer(exporters=[exporter]),
        extra_indexes={"graph": graph} if graph is not None else None,
        contextualizer=title_context_fn() if cfg.get("contextual") == "title" else None,
    )
    t0 = time.perf_counter()
    chunks = pipeline.ingest_batch(
        [{"id": d["id"], "text": d["text"], "metadata": d.get("metadata")} for d in docs]
    )
    build_ms = (time.perf_counter() - t0) * 1000
    _STATE.update(pipeline=pipeline, exporter=exporter, docs=docs, graph=graph)
    return json.dumps(
        {
            "documents": pipeline.document_count,
            "chunks": len(chunks),
            "build_ms": build_ms,
            "backend": pipeline.vector_backend,
            "embedding": "semantic" if _SEM["semantic"] else "hashing",
            "graph": graph is not None,
            "chunk_sizes": [len(c.text) for c in chunks],
        }
    )


def chunk_document(config_json: str) -> str:
    """Chunk one text with the given chunker config and return offsets (for the visualizer)."""
    cfg = json.loads(config_json)
    chunks = _chunker(cfg.get("chunker", {})).chunk(cfg["text"], doc_id=cfg.get("id", "doc"))
    return json.dumps(
        [{"start": c.start_char, "end": c.end_char, "text": c.text, "id": c.id} for c in chunks]
    )


# ----------------------------------------------------------------------------- search


def _configure(cfg: dict[str, Any]) -> RagPipeline:
    pipeline = _pipeline()
    retriever = pipeline.retriever
    retriever.k_rrf = int(cfg.get("k_rrf", 60))
    mode = cfg.get("mode", "hybrid")
    retriever.weight_bm25 = 0.0 if mode == "vector" else float(cfg.get("w_bm25", 1.0))
    retriever.weight_vector = 0.0 if mode == "bm25" else float(cfg.get("w_vec", 1.0))
    pipeline.weight_bm25, pipeline.weight_vector = retriever.weight_bm25, retriever.weight_vector
    pipeline.router = QueryRouter() if cfg.get("router") else None
    pipeline.reranker = (
        MaxMarginalRelevanceReranker(
            lambda_mult=float(cfg.get("mmr_lambda", 0.7)), relevance="fused"
        )
        if cfg.get("reranker") == "mmr"
        else NoopReranker()
    )
    guard = cfg.get("guardrails", "off")
    pipeline.guardrails = None if guard == "off" else Guardrails(action=guard)
    pipeline.prompt_formatter = (
        spotlight_prompt_formatter if cfg.get("spotlight") else pipeline.prompt_formatter
    )
    return pipeline


@_fixed_point
def search(config_json: str) -> str:
    """Run one query: retrieval views, final answer, citations, and the span trace."""
    cfg = json.loads(config_json)
    pipeline = _configure(cfg)
    query: str = cfg["query"]
    k = int(cfg.get("k", 5))
    pool = int(cfg.get("pool", 20))

    columns = {
        "bm25": [_hit(h, i) for i, h in enumerate(pipeline.bm25_index.search(query, k=k), 1)],
        "vector": [_hit(h, i) for i, h in enumerate(pipeline.vector_index.search(query, k=k), 1)],
    }
    graph = _STATE["graph"]
    if graph is not None:
        columns["graph"] = [_hit(h, i) for i, h in enumerate(graph.search(query, k=k), 1)]
    exporter: InMemoryExporter = _STATE["exporter"]
    exporter.spans.clear()
    response = pipeline.answer(query, k=k, candidate_pool=pool)
    trace = _spans(exporter)

    base = pipeline._base_retriever
    saved = (base.weight_bm25, base.weight_vector)
    base.weight_bm25, base.weight_vector = 1.0, 1.0
    columns["hybrid"] = [
        _hit(h, i) for i, h in enumerate(base.search(query, k=k, candidate_pool=pool), 1)
    ]
    base.weight_bm25, base.weight_vector = saved

    cited = attribute(response.answer, response.contexts)
    return json.dumps(
        {
            "query": query,
            "columns": columns,
            "final": [c.to_dict() for c in response.contexts],
            "answer": response.answer,
            "citations": cited.to_dict(),
            "prompt": response.prompt,
            "trace": trace,
            "latency_ms": response.total_latency_ms,
            "embedding": "semantic" if _SEM["semantic"] else "hashing",
        }
    )


# --------------------------------------------------------------------------- guardrails


def guard_scan(config_json: str) -> str:
    """Screen free text: findings with offsets, risk, sanitized/redacted output."""
    cfg = json.loads(config_json)
    text: str = cfg["text"]
    action = cfg.get("action", "sanitize")
    guard = Guardrails(action=action, risk_threshold=float(cfg.get("threshold", 0.5)))
    result = guard.screen(text)
    injection = InjectionDetector().scan(text)
    pii = PIIRedactor().scan(text)
    return json.dumps(
        {
            "risk": result.risk,
            "blocked": result.blocked,
            "text": result.text,
            "findings": [f.to_dict() for f in [*injection, *pii]],
            "prompt": spotlight_prompt_formatter("(question)", [text]),
        }
    )


# ------------------------------------------------------------------------------ agent


@_fixed_point
def agent_run(config_json: str) -> str:
    """Run the agentic loop and return its full decision trail."""
    cfg = json.loads(config_json)
    pipeline = _configure({**cfg, "mode": "hybrid", "reranker": "none", "guardrails": "off"})
    multihop = cfg.get("multihop", False)
    options: dict[str, Any] = {
        "max_iterations": int(cfg.get("max_iterations", 3)),
        "min_sufficiency": float(cfg.get("min_sufficiency", 0.7)),
        "abstain_below": float(cfg.get("abstain_below", 0.3)),
    }
    if multihop:  # two forced hops; the second reads bridge entities from the first
        options.update(max_iterations=2, min_iterations=2, rewrite_fn=bridge_rewrite_fn())
    agent = AgenticRag(
        pipeline,
        decompose_fn=heuristic_decompose_fn if cfg.get("decompose", True) else None,
        **options,
    )
    result = agent.run(cfg["question"], k=int(cfg.get("k", 3)))
    payload = result.to_dict()
    payload["citations"] = attribute(result.answer.answer, result.answer.contexts).to_dict()
    return json.dumps(payload)


# ------------------------------------------------------------------------------- graph


def graph_view(config_json: str) -> str:
    """Entity graph of the current corpus (requires ``graph: true`` at init) + optional query."""
    cfg = json.loads(config_json)
    graph: GraphIndex | None = _STATE["graph"]
    if graph is None:
        raise RuntimeError("The entity graph is off: rebuild the index with graph enabled")
    data = graph.graph_data(
        max_entities=int(cfg.get("max_entities", 70)), min_df=int(cfg.get("min_df", 1))
    )
    explain = graph.explain(cfg["query"]) if cfg.get("query") else None
    data["explain"] = explain
    return json.dumps(data)


# ------------------------------------------------------------------------------- eval


@_fixed_point
def evaluate(config_json: str) -> str:
    """Score the current pipeline on the built-in cases with RAGAS-style and IR metrics."""
    cfg = json.loads(config_json)
    pipeline = _configure({**cfg, "guardrails": "off"})
    k = int(cfg.get("k", 3))
    cases = [
        EvalCase(
            query=c["query"],
            relevant_doc_ids=c["relevant_doc_ids"],
            ground_truth_answer=c["ground_truth_answer"],
        )
        for c in SAMPLE_EVAL_CASES
    ]
    summary = evaluate_pipeline(pipeline, cases, k=k)
    per_case = []
    ndcgs, recalls, mrrs = [], [], []
    for case, result in zip(SAMPLE_EVAL_CASES, summary.results, strict=True):
        ranked = [c.doc_id for c in pipeline.retrieve(case["query"], k=10)]
        rel = dict.fromkeys(case["relevant_doc_ids"], 1.0)
        ndcgs.append(ndcg_at_k(ranked, rel, 10))
        recalls.append(recall_at_k(ranked, rel, k))
        mrrs.append(mrr_at_k(ranked, rel, 10))
        per_case.append(
            {
                "query": case["query"],
                "relevant": case["relevant_doc_ids"],
                "retrieved": result.retrieved_doc_ids,
                "answer": result.answer,
                "overall": result.overall,
                "context_recall": result.context_recall,
                "context_precision": result.context_precision,
                "ndcg": ndcgs[-1],
                "hit": bool(set(result.retrieved_doc_ids) & set(case["relevant_doc_ids"])),
            }
        )
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    return json.dumps(
        {
            "metrics": {
                "overall": summary.mean_overall,
                "faithfulness": summary.mean_faithfulness,
                "answer_relevancy": summary.mean_answer_relevancy,
                "context_precision": summary.mean_context_precision,
                "context_recall": summary.mean_context_recall,
                "context_f1": summary.mean_context_f1,
                "map": summary.mean_ranked_context_precision,
                "ndcg@10": mean(ndcgs),
                f"recall@{k}": mean(recalls),
                "mrr@10": mean(mrrs),
                "latency_ms": summary.mean_latency_ms,
            },
            "cases": per_case,
        }
    )


# ------------------------------------------------------------------------------ HNSW


def _lift(x: float, y: float) -> list[float]:
    """Lift a point of the unit disk onto the unit hemisphere.

    Cosine similarity on the sphere is monotone in Euclidean distance, so the
    cosine-based HNSW behaves exactly like a 2-D Euclidean index on (x, y).
    """
    return [x, y, math.sqrt(max(0.0, 1.0 - x * x - y * y))]


def hnsw_build(config_json: str) -> str:
    """Build an HNSW graph over clustered 2-D points and export every layer for drawing."""
    cfg = json.loads(config_json)
    n = int(cfg.get("n", 400))
    clusters = int(cfg.get("clusters", 6))
    rng = random.Random(int(cfg.get("seed", 7)))
    centers = [
        (rng.uniform(-0.62, 0.62), rng.uniform(-0.62, 0.62)) for _ in range(max(1, clusters))
    ]
    points: list[list[float]] = []
    while len(points) < n:
        cx, cy = rng.choice(centers)
        x, y = rng.gauss(cx, 0.1), rng.gauss(cy, 0.1)
        if x * x + y * y < 0.92:
            points.append([x, y])

    graph = HNSWIndex(
        m=int(cfg.get("m", 6)),
        ef_construction=int(cfg.get("ef_construction", 40)),
        seed=int(cfg.get("seed", 7)),
        heuristic=bool(cfg.get("heuristic", True)),
    )
    for x, y in points:
        graph.add(_lift(x, y))
    _STATE["hnsw"] = graph
    _STATE["points"] = points

    levels = [graph.level_of(i) for i in range(len(points))]
    edges: dict[str, list[list[int]]] = {}
    for level in range(graph.max_level + 1):
        layer = []
        for i in range(len(points)):
            if levels[i] >= level:
                layer.extend([i, j] for j in graph.neighbors(i, level) if i < j or
                             i not in graph.neighbors(j, level))  # fmt: skip
        edges[str(level)] = layer
    return json.dumps(
        {
            "points": points,
            "levels": levels,
            "edges": edges,
            "entry": graph.entry_point,
            "max_level": graph.max_level,
            "unreachable": len(graph.unreachable()),
        }
    )


def hnsw_search(config_json: str) -> str:
    """Search the explorer graph from a 2-D query point; returns the traversal per layer."""
    cfg = json.loads(config_json)
    graph: HNSWIndex = _STATE["hnsw"]
    points: list[list[float]] = _STATE["points"]
    qx, qy = float(cfg["x"]), float(cfg["y"])
    k = int(cfg.get("k", 5))
    ef = int(cfg.get("ef", 20))
    trace = graph.search_trace(_lift(qx, qy), k=k, ef_search=ef)

    exact = sorted(
        range(len(points)), key=lambda i: (points[i][0] - qx) ** 2 + (points[i][1] - qy) ** 2
    )[:k]
    found = [nid for _, nid in trace["results"]]
    visited_total = sum(len(step["visited"]) for step in trace["path"])
    return json.dumps(
        {
            "results": found,
            "exact": exact,
            "recall": len(set(found) & set(exact)) / max(len(exact), 1),
            "path": trace["path"],
            "visited": visited_total,
            "n": len(points),
        }
    )
