# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ragforge.ann.HNSWIndex`: a from-scratch, dependency-free Hierarchical
  Navigable Small World graph for approximate nearest-neighbor search, with
  lazy-tombstone deletion and full JSON serialization.
- `ragforge.index.ApproxVectorIndex`: an HNSW-backed drop-in alternative to
  `VectorIndex`, selectable via `RagPipeline(use_ann=True, ann_params=...)`
  and persisted/restored through the pipeline's existing `save`/`load`.
- `docs/math.md`: derivations behind every scoring function in the
  library -- BM25's probabilistic origin, why RRF fuses ranks instead of
  scores, MMR's `(1 - 1/e)` greedy approximation guarantee via
  submodularity, HNSW's expected `O(log n)` query complexity, and the
  birthday-bound collision rate of the CRC32 hashing embedding.
- `ragforge.embeddings_providers`: lazy-import adapters for real embedding
  providers (`openai_embed_fn`, `cohere_embed_fn`,
  `sentence_transformers_embed_fn`, `ollama_embed_fn`) plus a
  `cached_embed_fn` LRU wrapper, with corresponding optional extras
  (`ragforge[openai]`, `ragforge[cohere]`, `ragforge[local]`).
- `ragforge.query_expansion.MultiQueryRetriever`: fans a query out into
  rewritten variants (`llm_query_expansion_fn` or a custom
  `QueryExpansionFn`) and fuses per-variant results via RRF, mitigating
  vocabulary-mismatch retrieval misses. Wired into `RagPipeline` via
  `query_expansion_fn`.
- `ragforge.telemetry`: dependency-free nested tracing (`Tracer`, `Span`)
  with console, in-memory, and OpenTelemetry-bridge exporters, integrated
  into `RagPipeline.answer`/`retrieve` as per-stage spans. Exposed in the
  CLI via `ragforge query --trace`.
- `ragforge.server`: a minimal HTTP JSON API (`GET /health`, `POST /query`)
  built entirely on `http.server`, with no framework dependency. Exposed as
  `ragforge serve`.
- `ragforge ingest --ann`: opt into the HNSW vector backend from the CLI.
- Hypothesis property-based tests (`tests/test_properties.py`) asserting
  invariants across arbitrary inputs: chunk offsets always round-trip to
  exact source text, chunking drops no characters, and BM25/vector/HNSW
  search always respect `k` and return score-sorted, duplicate-free results.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue/PR templates.

## [0.1.0] - Initial release

### Added
- Multi-strategy chunking: `FixedSizeChunker`, `SentenceChunker`,
  `RecursiveCharacterChunker`, `MarkdownChunker`, all with source character
  offsets for citation.
- Dependency-free deterministic embeddings (`hashing_embed`) and vector math
  (`cosine_similarity`, `normalize_vector`).
- Hybrid retrieval: `BM25Index` (Okapi BM25), `VectorIndex` (brute-force
  cosine), and `HybridRetriever` (weighted Reciprocal Rank Fusion), each with
  metadata filtering, document deletion, and JSON persistence.
- Reranking stage: `NoopReranker`, `HeuristicReranker` (Jaccard overlap),
  `CrossEncoderReranker` (adapter for any scoring function), and
  `MaxMarginalRelevanceReranker` (MMR diversity).
- `RagPipeline`: end-to-end orchestration with batch ingestion, latency
  tracking, and configurable prompt formatting.
- RAGAS-style `evaluate_pipeline`: context precision (set and ranked
  MAP@k), context recall, context F1, faithfulness, answer relevancy, and
  answer similarity, with a Markdown summary table and JSON report export.
- `ragforge` CLI: `ingest`, `query`, `evaluate`, `benchmark` subcommands.
- Runnable examples covering the basic pipeline, hybrid vs. single-strategy
  retrieval, evaluation, metadata filtering with MMR, and persistence.
- Strict tooling: `ruff` (lint + format), `mypy --strict`-style
  configuration, `pytest` with a 90% branch-coverage gate, and a GitHub
  Actions CI matrix across Python 3.11-3.13 plus a Docker build smoke test.

[Unreleased]: https://github.com/fragompul/ragforge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/fragompul/ragforge/releases/tag/v0.1.0
