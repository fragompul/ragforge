# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.0]

### Added
- **Guardrails** (`ragforge.guardrails`): prompt-injection detector (EN/ES rules, hidden Unicode,
  exfiltration), PII/secret redactor (Luhn-checked cards), flag/sanitize/drop policies, spotlighting
  prompt formatter; `RagPipeline(guardrails=...)` integration with tracing.
- **Verifiable citations** (`ragforge.citations`): sentence-level attribution with exact source
  character offsets, groundedness score, and verification of LLM-written `[n]` markers.
- **Agentic RAG** (`ragforge.agent`): decompose / retrieve / grade / rewrite loop with a bounded budget
  and abstention; lexical defaults plus `llm_*` factories.
- **MCP server** (`ragforge mcp`): zero-dependency stdio JSON-RPC server exposing search, cited answers and
  stats; ingestion opt-in.
- **LLM-as-judge** (`ragforge.judge`) with rubric prompts, robust verdict parsing and aggregation.
- **LLM provider adapters** (`ragforge.llm_providers`): Anthropic, OpenAI and Ollama, including streaming.
- **Streaming** (`RagPipeline.answer_stream`, `POST /query/stream` as SSE).
- **Semantic cache** (`ragforge.cache`) with TTL, namespaces and corpus-change invalidation.
- **NumPy backend** (`ragforge.fast.NumpyVectorIndex`, extra `ragforge[fast]`) and `vector_backend=` selector.
- **IR metrics** (`ragforge.metrics`): nDCG, Recall, Precision, MRR, MAP.
- **BM25 `english` analyzer** (stopwords + light stemming).
- `ragforge.extractive`: keyless extractive answerer; `ragforge.sample`: built-in demo corpus.
- CLI: `ragforge demo`, `ragforge mcp`; `query --agentic --cite --guardrails`; `ingest --backend --analyzer`.
- **Web playground** (`web/`, GitHub Pages): the real library running in the browser via Pyodide, with an
  HNSW explorer, security lab, agent lab, corpus/chunk visualizer and live evaluation.
- Reproducible benchmarks: `scripts/eval_beir.py` (BEIR SciFact) and a rewritten
  `scripts/generate_benchmarks.py`; light/dark SVG charts.
- **Quantized vector backends** (`ragforge.quantization`): int8 (4x) and binary (32x) codes with
  full-precision rescoring; `vector_backend="int8" | "binary"`.
- **N-way fusion and sparse retrieval** (`ragforge.fusion`, `ragforge.sparse`): weighted RRF over any
  number of rankers, a sparse-vector index and a SPLADE encoder (optional dependency).
- **Query router** (`ragforge.routing`), **contextual retrieval** (`ragforge.contextual`) and
  **HyDE** query expansion (`hyde_expansion_fn`).
- **Entity graph retrieval** (`ragforge.graph`): GraphRAG-lite with activation diffusion and explanations;
  agent gained a bridge-entity rewriter for multi-hop questions.
- **Entailment-checked citations** (`entailment_fn`, `nli_cross_encoder_fn`).
- **Memory-mapped index directories** (`ragforge.storage.DiskIndex`): mmap vectors, CSR graph, append-only
  chunk log, atomic crash-safe commits, single-writer lock, `compact()`.
- **Async + ASGI** (`ragforge.aio`, `ragforge.asgi`) and **multi-tenancy** (`ragforge.tenancy`).
- **Terminal chat** (`ragforge chat`), MCP **resources and prompts**, and a one-click Claude Desktop
  extension builder (`scripts/build_mcpb.py`).
- **Evaluation suite**: multi-dataset BEIR runner with ablations (`scripts/eval_beir.py`), HotpotQA multi-hop
  (`scripts/eval_multihop.py`), guardrail benchmarks against public injection datasets with a
  train/test split (`scripts/eval_guardrails.py`), contextual-retrieval ablation.
- **Testing**: property-based fuzzing of every parser and a stateful model test of `DiskIndex`; mutation
  testing of the metrics module (survivors turned into regression tests).
- **Docs**: MkDocs Material site (`mkdocs.yml`), Colab quickstart notebook, Hugging Face static Space packaging
  (`scripts/build_space.py`), blog post on the HNSW recall bug.

### Changed
- **HNSW rewritten for correctness**: paper's heuristic neighbor selection (Algorithm 4), exact-duplicate
  handling, unit-normalized vectors with dot-product similarity, tombstone-aware beam search, `m_max0`
  defaults to `2*m`, search tracing and graph introspection. Fixes a recall plateau (~0.4–0.6 on clustered
  data regardless of `ef_search`); old 0.1 saved graphs still load.
- **BM25 uses an inverted index**: 12–15x faster queries with identical rankings.
- `MaxMarginalRelevanceReranker(relevance="fused")` option to diversify over the fused score.
- HTTP server hardening: 1 MB body limit, validated `k` / `candidate_pool`, JSON-object bodies only;
  `agentic` queries supported.
- HNSW: **filters are applied inside the traversal** with an exact-scan fallback, a pinned parent-edge
  backbone keeps every node reachable (duplicate floods can no longer orphan documents), and `compact()`
  rebuilds after heavy deletion.
- Version is single-sourced in `ragforge/_version.py`; package ships `py.typed`.

## [0.1.0]

Initial release: multi-strategy chunking, BM25 + vector hybrid retrieval with RRF, from-scratch HNSW,
multi-query expansion, MMR/heuristic/cross-encoder reranking, RAGAS-style evaluation, tracing,
persistence, CLI and a dependency-free HTTP server.
