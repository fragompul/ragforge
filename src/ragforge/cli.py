"""Command-line interface for ragforge RAG workflows."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from ragforge._version import __version__
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.evaluation import EvalCase, evaluate_pipeline
from ragforge.guardrails import Guardrails
from ragforge.pipeline import RagPipeline
from ragforge.reranking import (
    HeuristicReranker,
    MaxMarginalRelevanceReranker,
    NoopReranker,
    Reranker,
)
from ragforge.server import serve_forever_blocking
from ragforge.telemetry import Tracer, console_exporter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragforge",
        description="ragforge: Production-hardened RAG pipeline CLI",
    )
    parser.add_argument("-v", "--version", action="version", version=f"ragforge {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest files or directories into an index"
    )
    ingest_parser.add_argument("path", help="Path to text/markdown file or directory")
    ingest_parser.add_argument(
        "--index",
        "-i",
        default="ragforge_index.json",
        help="Path to save the pipeline index (default: ragforge_index.json)",
    )
    ingest_parser.add_argument(
        "--chunk-size", type=int, default=500, help="Chunk size in characters"
    )
    ingest_parser.add_argument(
        "--chunk-overlap", type=int, default=50, help="Chunk overlap in characters"
    )
    ingest_parser.add_argument(
        "--ann",
        action="store_true",
        help="Use the approximate HNSW vector backend instead of brute-force cosine scan "
        "(recommended for large corpora; see docs/benchmarks.md). Shorthand for --backend hnsw",
    )
    ingest_parser.add_argument(
        "--backend",
        choices=["brute_force", "hnsw", "numpy", "int8", "binary"],
        default=None,
        help="Vector backend: exact pure-Python scan, approximate HNSW graph, the exact "
        "NumPy-accelerated scan (pip install 'ragforge[fast]'), or int8 / binary quantized "
        "vectors with rescoring (4x / 32x smaller)",
    )
    ingest_parser.add_argument(
        "--format",
        choices=["json", "dir"],
        default="json",
        help="Index format: a single JSON file, or a memory-mapped directory "
        "(incremental updates, near-instant open; brute_force/numpy/hnsw backends)",
    )
    ingest_parser.add_argument(
        "--contextual",
        choices=["none", "title", "lead"],
        default="none",
        help="Contextual retrieval: prepend the document title or opening sentence to each "
        "chunk before indexing (answers and citations still use the original text)",
    )
    ingest_parser.add_argument(
        "--analyzer",
        choices=["simple", "english"],
        default="simple",
        help="BM25 text analyzer; 'english' drops stopwords and applies light stemming",
    )

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Query an indexed RAG pipeline")
    query_parser.add_argument("query", help="Question or query string")
    query_parser.add_argument(
        "--index",
        "-i",
        default="ragforge_index.json",
        help="Path to the saved pipeline index (default: ragforge_index.json)",
    )
    query_parser.add_argument(
        "-k", type=int, default=3, help="Number of context chunks to retrieve"
    )
    query_parser.add_argument(
        "--reranker",
        choices=["none", "heuristic", "mmr"],
        default="heuristic",
        help="Reranker strategy to apply (default: heuristic)",
    )
    query_parser.add_argument(
        "--trace",
        action="store_true",
        help="Print a per-stage latency trace (fusion search, rerank, generate)",
    )
    query_parser.add_argument(
        "--agentic",
        action="store_true",
        help="Use the agentic loop (grade -> rewrite -> retry, abstain if evidence is weak)",
    )
    query_parser.add_argument(
        "--cite",
        action="store_true",
        help="Attribute each answer sentence to source spans and print groundedness",
    )
    _add_guardrails_option(query_parser)
    _add_route_option(query_parser)

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a pipeline against test cases")
    eval_parser.add_argument("cases", help="Path to JSON file containing evaluation cases")
    eval_parser.add_argument(
        "--index",
        "-i",
        default="ragforge_index.json",
        help="Path to the saved pipeline index (default: ragforge_index.json)",
    )
    eval_parser.add_argument("-k", type=int, default=3, help="Top-k contexts for evaluation")
    eval_parser.add_argument(
        "--output", "-o", help="Optional output JSON path for evaluation report"
    )

    # Benchmark subcommand
    subparsers.add_parser("benchmark", help="Run synthetic performance and latency benchmarks")

    # Serve subcommand
    serve_parser = subparsers.add_parser(
        "serve", help="Serve a pipeline over a minimal dependency-free HTTP JSON API"
    )
    serve_parser.add_argument(
        "--index",
        "-i",
        default="ragforge_index.json",
        help="Path to the saved pipeline index (default: ragforge_index.json)",
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    serve_parser.add_argument(
        "--reranker",
        choices=["none", "heuristic", "mmr"],
        default="heuristic",
        help="Reranker strategy to apply (default: heuristic)",
    )
    _add_guardrails_option(serve_parser)
    _add_route_option(serve_parser)
    serve_parser.add_argument(
        "--asgi",
        action="store_true",
        help="Serve through uvicorn (ASGI, concurrent generation; pip install uvicorn) "
        "instead of the built-in stdlib server",
    )
    serve_parser.add_argument(
        "--api-key",
        default=None,
        help="Require 'Authorization: Bearer <key>' (ASGI mode only)",
    )

    # MCP subcommand
    mcp_parser = subparsers.add_parser(
        "mcp", help="Expose an index to AI assistants as a Model Context Protocol (stdio) server"
    )
    mcp_parser.add_argument(
        "--index",
        "-i",
        default="ragforge_index.json",
        help="Path to the saved pipeline index (default: ragforge_index.json)",
    )
    mcp_parser.add_argument(
        "--allow-ingest",
        action="store_true",
        help="Also expose the ingest_document tool (the index stays read-only by default)",
    )
    mcp_parser.add_argument(
        "--reranker",
        choices=["none", "heuristic", "mmr"],
        default="none",
        help="Reranker strategy to apply (default: none)",
    )
    _add_guardrails_option(mcp_parser)
    _add_route_option(mcp_parser)

    # Chat subcommand
    chat_parser = subparsers.add_parser(
        "chat", help="Interactive terminal chat over an index (streaming, citations, agent mode)"
    )
    chat_parser.add_argument(
        "--index",
        "-i",
        default="ragforge_index.json",
        help="Path to the saved pipeline index (default: ragforge_index.json)",
    )
    chat_parser.add_argument("-k", type=int, default=3, help="Passages per answer")
    chat_parser.add_argument(
        "--reranker", choices=["none", "heuristic", "mmr"], default="none",
        help="Reranker strategy to apply (default: none)",
    )  # fmt: skip
    _add_guardrails_option(chat_parser)
    _add_route_option(chat_parser)

    # Demo subcommand
    demo_parser = subparsers.add_parser(
        "demo", help="Guided, offline tour of hybrid retrieval, guardrails, citations and agents"
    )
    demo_parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")

    return parser


def _add_route_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--route",
        action="store_true",
        help="Per-query fusion weights: boost BM25 for identifiers / rare terms, vectors for "
        "long natural-language questions",
    )


def _get_router(args: argparse.Namespace) -> Any:
    if not getattr(args, "route", False):
        return None
    from ragforge.routing import QueryRouter

    return QueryRouter()


def load_pipeline(path: str | Path, **kwargs: Any) -> RagPipeline:
    """Load a JSON index or a memory-mapped directory index (opened read-only)."""
    if Path(path).is_dir():
        from ragforge.storage import DiskIndex

        disk = DiskIndex.open(path, readonly=True, **kwargs)
        disk.pipeline._disk_index = disk  # type: ignore[attr-defined]  # keep the mapping alive
        return disk.pipeline
    return RagPipeline.load(path, **kwargs)


def _add_guardrails_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--guardrails",
        choices=["off", "flag", "sanitize", "drop"],
        default="off",
        help="Screen retrieved contexts for prompt injection and PII (default: off)",
    )


def _get_guardrails(mode: str) -> Guardrails | None:
    return None if mode == "off" else Guardrails(action=mode)  # type: ignore[arg-type]


def _get_reranker(strategy: str) -> Reranker:
    if strategy == "heuristic":
        return HeuristicReranker()
    if strategy == "mmr":
        return MaxMarginalRelevanceReranker(lambda_mult=0.7)
    return NoopReranker()


def handle_ingest(args: argparse.Namespace) -> int:
    source_path = Path(args.path)
    if not source_path.exists():
        sys.stderr.write(f"Error: Path '{args.path}' does not exist.\n")
        return 1

    chunker = RecursiveCharacterChunker(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    backend = args.backend or ("hnsw" if args.ann else "brute_force")
    contextualizer = None
    if args.contextual != "none":
        from ragforge.contextual import lead_context_fn, title_context_fn

        contextualizer = title_context_fn() if args.contextual == "title" else lead_context_fn()
    if args.format == "dir" and backend in ("int8", "binary"):
        sys.stderr.write("Error: --format dir supports brute_force, numpy and hnsw backends.\n")
        return 1
    disk = None
    try:
        if args.format == "dir":
            from ragforge.storage import DiskIndex

            disk = DiskIndex.create(
                args.index,
                vector_backend=backend,
                chunker=chunker,
                bm25_analyzer=args.analyzer,
                contextualizer=contextualizer,
            )
            pipeline = disk.pipeline
        else:
            pipeline = RagPipeline(
                chunker=chunker,
                vector_backend=backend,
                bm25_analyzer=args.analyzer,
                contextualizer=contextualizer,
            )
    except ImportError as exc:  # NumPy backend without the extra installed
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    except Exception as exc:  # e.g. an existing directory index
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    files_to_ingest: list[Path] = []
    if source_path.is_file():
        files_to_ingest.append(source_path)
    else:
        for ext in ("*.txt", "*.md", "*.rst"):
            files_to_ingest.extend(source_path.rglob(ext))

    if not files_to_ingest:
        sys.stderr.write(f"Warning: No text or markdown files found in '{args.path}'.\n")
        return 0

    total_chunks = 0
    start_time = time.perf_counter()

    for file_path in files_to_ingest:
        try:
            content = file_path.read_text(encoding="utf-8")
            doc_id = str(file_path.stem)
            ingest = disk.ingest if disk is not None else pipeline.ingest
            chunks = ingest(
                doc_id=doc_id,
                text=content,
                metadata={"source": str(file_path), "title": doc_id},
            )
            total_chunks += len(chunks)
        except Exception as e:
            sys.stderr.write(f"Failed to ingest '{file_path}': {e}\n")

    elapsed = time.perf_counter() - start_time
    output_index = Path(args.index)
    if disk is not None:
        disk.close()
    else:
        pipeline.save(output_index)

    backend = {
        "hnsw": "HNSW (approximate)",
        "numpy": "NumPy (exact, vectorized)",
        "brute_force": "brute-force (exact)",
        "int8": "int8-quantized (4x smaller, exact rescoring)",
        "binary": "binary-quantized (32x smaller, exact rescoring)",
    }[backend]
    print(
        f"Successfully ingested {len(files_to_ingest)} documents "
        f"({total_chunks} chunks) in {elapsed:.3f}s using the {backend} vector backend"
    )
    print(f"Saved pipeline index to: {output_index.resolve()}")
    return 0


def handle_query(args: argparse.Namespace) -> int:
    index_path = Path(args.index)
    if not index_path.exists():
        sys.stderr.write(
            f"Error: Index file '{args.index}' does not exist. Run 'ragforge ingest' first.\n"
        )
        return 1

    reranker = _get_reranker(args.reranker)
    tracer = Tracer(exporters=[console_exporter]) if args.trace else None
    pipeline = load_pipeline(
        index_path,
        reranker=reranker,
        tracer=tracer,
        guardrails=_get_guardrails(args.guardrails),
        router=_get_router(args),
    )

    if args.trace:
        print("--- TRACE ---")
    agent_result = None
    if args.agentic:
        from ragforge.agent import AgenticRag

        agent_result = AgenticRag(pipeline).run(args.query, k=args.k)
        answer = agent_result.answer
    else:
        answer = pipeline.answer(args.query, k=args.k)
    if args.trace:
        print("-------------\n")

    if agent_result is not None:
        print("\nAGENT STEPS:")
        for step in agent_result.steps:
            score = "" if step.score is None else f" (score {step.score:.2f})"
            print(f"  - {step.kind:<9} {step.query}{score}")

    print("\n" + "=" * 60)
    print(f"QUERY: {answer.query}")
    print("=" * 60)
    print(f"ANSWER:\n{answer.answer}")
    print("-" * 60)
    print(
        f"RETRIEVED CONTEXTS ({len(answer.contexts)} chunks, {answer.retrieval_latency_ms:.2f}ms):"
    )
    for i, ctx in enumerate(answer.contexts, start=1):
        print(f"\n[{i}] Doc ID: {ctx.doc_id} | Score: {ctx.score:.4f} | Source: {ctx.provenance}")
        print(f"    {ctx.text}")
        for finding in ctx.guardrails.get("findings", []):
            print(f"    ! {finding['kind']}:{finding['rule']} -> {finding['excerpt']}")
    if args.cite:
        from ragforge.citations import attribute

        cited = attribute(answer.answer, answer.contexts)
        print("-" * 60)
        print(f"CITATIONS (groundedness {cited.groundedness:.2f}):")
        for sentence in cited.sentences:
            status = "supported " if sentence.supported else "UNVERIFIED"
            print(f"  [{status}] {sentence.text[:70]}")
            for cite in sentence.citations:
                span = f"chars {cite.start_char}-{cite.end_char}"
                print(f"      -> [{cite.marker}] {cite.doc_id} {span}")
    print("=" * 60 + "\n")
    return 0


def handle_evaluate(args: argparse.Namespace) -> int:
    cases_path = Path(args.cases)
    index_path = Path(args.index)

    if not cases_path.exists():
        sys.stderr.write(f"Error: Eval cases file '{args.cases}' does not exist.\n")
        return 1
    if not index_path.exists():
        sys.stderr.write(f"Error: Index file '{args.index}' does not exist.\n")
        return 1

    with open(cases_path, encoding="utf-8") as f:
        cases_data = json.load(f)

    eval_cases = [
        EvalCase(
            query=c["query"],
            relevant_doc_ids=c.get("relevant_doc_ids", []),
            ground_truth_answer=c.get("ground_truth_answer"),
        )
        for c in cases_data
    ]

    pipeline = load_pipeline(index_path)
    summary = evaluate_pipeline(pipeline, eval_cases, k=args.k)

    print("\n" + summary.to_markdown_table() + "\n")

    if args.output:
        out_path = Path(args.output)
        summary.save(out_path)
        print(f"Report saved to: {out_path.resolve()}")

    return 0


def handle_benchmark() -> int:
    print("Running ragforge performance and latency benchmark...")
    corpus_size = 500
    queries_count = 50

    sample_docs = [
        (
            f"doc_{i}",
            f"Article number {i} covers distributed consensus, vector search algorithms, "
            f"and RAG evaluation metrics. Key index id is REF_{i * 7}.",
        )
        for i in range(corpus_size)
    ]

    chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=30)
    pipeline = RagPipeline(chunker=chunker, reranker=HeuristicReranker())

    # Ingestion benchmark
    t0 = time.perf_counter()
    for doc_id, text in sample_docs:
        pipeline.ingest(doc_id, text)
    t_ingest = time.perf_counter() - t0

    # Retrieval benchmark
    latencies: list[float] = []
    t_query_start = time.perf_counter()
    for q_idx in range(queries_count):
        query = f"consensus vector search REF_{q_idx * 7}"
        res = pipeline.answer(query, k=5)
        latencies.append(res.total_latency_ms)
    t_query_total = time.perf_counter() - t_query_start

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    print("\n--- BENCHMARK RESULTS ---")
    print(
        f"Ingested Docs: {corpus_size} in {t_ingest:.3f}s ({corpus_size / t_ingest:.1f} docs/sec)"
    )
    print(f"Total Chunks:  {pipeline.chunk_count}")
    print(
        f"Queries Run:   {queries_count} in {t_query_total:.3f}s "
        f"({queries_count / t_query_total:.1f} QPS)"
    )
    print(f"Latency P50:   {p50:.2f} ms")
    print(f"Latency P95:   {p95:.2f} ms")
    print(f"Latency P99:   {p99:.2f} ms")
    print("-------------------------\n")
    return 0


def handle_serve(args: argparse.Namespace) -> int:
    index_path = Path(args.index)
    if not index_path.exists():
        sys.stderr.write(
            f"Error: Index file '{args.index}' does not exist. Run 'ragforge ingest' first.\n"
        )
        return 1

    reranker = _get_reranker(args.reranker)
    pipeline = load_pipeline(
        index_path,
        reranker=reranker,
        guardrails=_get_guardrails(args.guardrails),
        router=_get_router(args),
    )

    if args.asgi:
        try:
            import uvicorn
        except ImportError:
            sys.stderr.write("Error: --asgi needs uvicorn. Install it with: pip install uvicorn\n")
            return 1
        from ragforge.asgi import create_app

        print(
            f"Serving {pipeline.document_count} documents on http://{args.host}:{args.port} "
            "(ASGI: concurrent generation)"
        )
        uvicorn.run(create_app(pipeline, api_key=args.api_key), host=args.host, port=args.port)
        return 0

    print(
        f"Serving {pipeline.document_count} documents ({pipeline.chunk_count} chunks) "
        f"on http://{args.host}:{args.port}"
    )
    print(
        'Endpoints: GET /health, POST /query {"query": str, "k": int, "agentic": bool}, '
        "POST /query/stream (SSE)"
    )
    print("Press Ctrl+C to stop.")
    serve_forever_blocking(pipeline, host=args.host, port=args.port)
    return 0


def handle_mcp(args: argparse.Namespace) -> int:
    from ragforge.mcp_server import serve_stdio

    index_path = Path(args.index)
    if not index_path.exists():
        sys.stderr.write(
            f"Error: Index file '{args.index}' does not exist. Run 'ragforge ingest' first.\n"
        )
        return 1

    pipeline = load_pipeline(
        index_path,
        reranker=_get_reranker(args.reranker),
        guardrails=_get_guardrails(args.guardrails),
        router=_get_router(args),
    )
    # stdout is the protocol channel: everything human-readable goes to stderr.
    sys.stderr.write(
        f"ragforge MCP server ready: {pipeline.document_count} documents, "
        f"{pipeline.chunk_count} chunks, ingest {'ON' if args.allow_ingest else 'off'}\n"
    )
    serve_stdio(pipeline, allow_ingest=args.allow_ingest)
    return 0


def handle_chat(args: argparse.Namespace) -> int:
    from ragforge.chat import ChatState, run_chat

    index_path = Path(args.index)
    if not index_path.exists():
        sys.stderr.write(
            f"Error: Index file '{args.index}' does not exist. Run 'ragforge ingest' first.\n"
        )
        return 1
    with contextlib.suppress(ImportError):  # line editing and history for input()
        import readline  # noqa: F401
    pipeline = load_pipeline(
        index_path,
        reranker=_get_reranker(args.reranker),
        guardrails=_get_guardrails(args.guardrails),
        router=_get_router(args),
    )
    return run_chat(pipeline, state=ChatState(k=args.k))


def handle_demo(args: argparse.Namespace) -> int:
    from ragforge.demo import run_demo

    run_demo(color=False if args.no_color else None)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        return handle_ingest(args)
    if args.command == "query":
        return handle_query(args)
    if args.command == "evaluate":
        return handle_evaluate(args)
    if args.command == "benchmark":
        return handle_benchmark()
    if args.command == "serve":
        return handle_serve(args)
    if args.command == "mcp":
        return handle_mcp(args)
    if args.command == "chat":
        return handle_chat(args)
    if args.command == "demo":
        return handle_demo(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
