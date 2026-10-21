# Guides

Task-oriented recipes. Every snippet is runnable; the larger ones live in
[`examples/`](https://github.com/fragompul/ragforge/tree/main/examples) and are executed by the test suite.

- [Pick a vector backend](#pick-a-vector-backend)
- [Hybrid search tuning](#hybrid-search-tuning)
- [Guardrails](#guardrails)
- [Verifiable citations](#verifiable-citations)
- [Agentic RAG](#agentic-rag)
- [Streaming and the HTTP API](#streaming-and-the-http-api)
- [MCP: use an index from Claude](#mcp-use-an-index-from-claude)
- [Semantic cache](#semantic-cache)
- [Evaluation](#evaluation)
- [Real LLM and embedding providers](#real-llm-and-embedding-providers)
- [Observability](#observability)
- [The web playground](#the-web-playground)

## Pick a vector backend

| Backend | Exactness | Needs | Use when |
| :--- | :--- | :--- | :--- |
| `brute_force` (default) | exact | nothing | up to a few thousand chunks |
| `numpy` | exact | `pip install "ragforge[fast]"` | up to hundreds of thousands of chunks; 100 % recall, no tuning |
| `hnsw` | approximate | nothing | beyond that, or when you want the graph (see `docs/benchmarks.md`) |

```python
RagPipeline(vector_backend="numpy")
RagPipeline(vector_backend="hnsw", ann_params={"m": 16, "ef_construction": 100, "ef_search": 50})
```

`ef_search` is the recall/latency dial; `ef_construction` and `m` set the ceiling.
The backend is stored in saved indexes, so `RagPipeline.load(path)` restores it.

## Hybrid search tuning

```python
rag = RagPipeline(bm25_analyzer="english",      # stopwords + light stemming for BM25
                  k_rrf=60, weight_bm25=1.0, weight_vector=1.0)
```

- RRF fuses *ranks*, so BM25's unbounded scores and cosine's `[-1, 1]` never need calibrating.
- Lower `k_rrf` lets top ranks dominate; weights bias toward one retriever (BEIR SciFact: equal weights won).
- `MaxMarginalRelevanceReranker(relevance="fused")` applies MMR as a pure diversity pass on top of the fused
  score instead of replacing it with dense similarity.

## Guardrails

```python
from ragforge import Guardrails, InjectionDetector, PIIRedactor
from ragforge.guardrails import spotlight_prompt_formatter

guard = Guardrails(
    detector=InjectionDetector(extra_rules=[("canary", "high", r"CANARY-\d+")]),
    redactor=PIIRedactor(kinds={"email", "api_key"}),
    action="sanitize",        # "flag" keeps text, "drop" removes the whole context
    risk_threshold=0.5,
)
rag = RagPipeline(guardrails=guard, prompt_formatter=spotlight_prompt_formatter)
```

Screening happens after reranking and before the prompt is built, so `answer`, `answer_stream`, the agent, the
HTTP server and the MCP server are all covered. Findings are attached to each `RetrievedChunk.guardrails` and
emitted as a `guardrails` tracer span. PII is always redacted; the injection policy applies at the threshold.

Limits, stated plainly: rules catch the common attacks and make poisoning auditable; they do not stop a
determined adversary. Combine with least-privilege tools and output filtering.

## Verifiable citations

```python
from ragforge import attribute
cited = attribute(answer.answer, answer.contexts, threshold=0.6)
cited.groundedness            # supported sentences / all sentences
cited.unsupported             # sentences with no evidence
cited.sentences[0].citations  # Citation(doc_id, start_char, end_char, quote, score)
print(cited.to_markdown())
```

If the LLM already wrote `[1]`-style markers, they are *verified*: a marker whose chunk does not support the
sentence is listed in `invalid_markers`. Pass `embed_fn=` to blend embedding similarity into the support score.

## Agentic RAG

```python
from ragforge import AgenticRag
from ragforge.agent import heuristic_decompose_fn
agent = AgenticRag(rag, decompose_fn=heuristic_decompose_fn,
                   max_iterations=3, min_sufficiency=0.7, abstain_below=0.3)
result = agent.run("What does ERR-4012 mean? How does RRF work?")
for step in result.steps: print(step.kind, step.query, step.score)
```

Defaults are deterministic and lexical. `llm_grade_fn`, `llm_rewrite_fn` and `llm_decompose_fn` take any
`prompt -> text` function (see `anthropic_complete_fn`). The loop is bounded, never repeats a query, and
abstains rather than answering when the best sufficiency stays below `abstain_below`.

## Streaming and the HTTP API

```python
for event in rag.answer_stream("question"):
    ...  # {"type": "contexts"} -> {"type": "token", "text": ...}... -> {"type": "done", "answer": RagAnswer}
```

```bash
ragforge serve -i kb.json
curl -N -X POST localhost:8000/query/stream -d '{"query": "How does RRF work?", "k": 3}'
curl    -X POST localhost:8000/query        -d '{"query": "...", "agentic": true}'
```

Provide `stream_generate_fn` (e.g. `anthropic_stream_fn()`) for true token streaming; otherwise the
`generate_fn` output is re-emitted word by word. Request bodies are capped at 1 MB and `k` at 100.

## MCP: use an index from Claude

```bash
ragforge mcp -i kb.json --guardrails sanitize          # stdio JSON-RPC server
claude mcp add ragforge -- ragforge mcp -i kb.json     # Claude Code
```

Claude Desktop (`claude_desktop_config.json`):

```json
{ "mcpServers": { "ragforge": { "command": "ragforge", "args": ["mcp", "--index", "/abs/path/kb.json"] } } }
```

Tools: `search_knowledge_base(query, k, mode)`, `ask_knowledge_base(question, k)` (answer + citations +
groundedness), `knowledge_base_stats`, and `ingest_document` only with `--allow-ingest`.
Logs go to stderr; stdout is the protocol channel.

## Semantic cache

```python
from ragforge import CachedPipeline, SemanticCache
cached = CachedPipeline(rag, SemanticCache(embed_fn=rag.embed_fn, threshold=0.92, ttl_seconds=3600),
                        namespace="tenant-42")
cached.answer("How do I cancel?")      # miss
cached.answer("how to cancel?")        # hit if similar enough (use a real embedding model)
cached.ingest("new", "text")           # clears the cache: stale answers never outlive their sources
```

Entries are partitioned by `(namespace, k, candidate_pool, cache_key)`. Calls with a `filter_fn` bypass the
cache unless you pass a stable `cache_key`. With the bag-of-words `hashing_embed` only near-identical word
sets match; use a semantic embedder in production.

## Evaluation

```python
from ragforge.metrics import ndcg_at_k, recall_at_k, mrr_at_k       # BEIR/TREC definitions
from ragforge.evaluation import EvalCase, evaluate_pipeline          # RAGAS-style, deterministic
from ragforge.judge import evaluate_with_judge, lexical_judge_fn     # LLM-as-judge (lexical = offline stand-in)
```

For a public benchmark run `python scripts/eval_beir.py --download scifact --data-dir data` then
`python scripts/eval_beir.py --data-dir data/scifact --embedder minilm` (needs `sentence-transformers`).

## Quantized vectors

`int8` stores one byte per dimension (4x smaller), `binary` one bit (32x smaller). Candidates are ranked on the
compact codes and then **rescored against the full-precision vectors**, which is what keeps quality:

```python
from ragforge import RagPipeline
rag = RagPipeline(vector_backend="int8")   # or "binary"
```

On SciFact (MiniLM, 384 dims) int8 + rescoring matches the float index and binary + rescoring costs about
0.3 nDCG points -- see [Evaluation](evaluation.md). Without rescoring, binary codes alone lose a lot: measured,
not assumed. `QuantizedVectorIndex(mode=..., rescore_factor=..., keep_full_precision=False)` drops the float
copy when memory matters more than the last points of quality.

## Filtered search and tombstones

`filter_fn` is evaluated *inside* the HNSW traversal, so a selective predicate ("only tenant 7", "only 2024")
returns a full, correct top-k instead of whatever survived an over-fetch. If the walk exceeds its visit budget
(`filter_max_visits`) the index scans the matching rows exactly; the hit provenance says so (`vector:hnsw+scan`).
Deleted chunks are tombstones that go through the same predicate; `rag.vector_index.compact()` rebuilds the graph
once many are dead. The story behind this is in the [blog post](blog/hnsw-recall-bug.md).

## Sparse, SPLADE and N-way fusion

Any number of rankers can be fused with weighted RRF. `SparseVectorIndex` takes an encoder that maps text to
`{term: weight}`; `splade_encode_fn()` uses a SPLADE model for learned term expansion (needs
`sentence-transformers`; the default `lexical_sparse_encode_fn` needs nothing):

```python
from ragforge import RagPipeline, SparseVectorIndex, lexical_sparse_encode_fn

rag = RagPipeline(
    extra_indexes={"sparse": SparseVectorIndex(lexical_sparse_encode_fn)},
    extra_weights={"sparse": 1.0},
)
```

`FusionRetriever([FusionSource("bm25", ...), FusionSource("dense", ...)])` is the same machinery without the pipeline.

## Query router

`QueryRouter` picks the BM25-vs-vector weights *per query* from cheap signals (identifier-like tokens and rare
terms push toward BM25, long natural-language questions toward vectors). It is a heuristic, not a model; the
[evaluation](evaluation.md) reports where it helps and where it is neutral.

```python
rag = RagPipeline(router=QueryRouter())
```

## Contextual retrieval and HyDE

A chunk like "It was completed in 1889." is unfindable on its own. `contextualizer=` prepends context
(`title_context_fn`, `heading_context_fn`, `lead_context_fn`, `llm_context_fn(complete_fn)`, or several via
`combine_context_fns`) to the text that is *indexed*, while answers and citations keep the original span.
`hyde_expansion_fn(complete_fn)` is the query-side analogue: embed a hypothetical answer instead of the question.

## Entity graph (GraphRAG-lite)

`GraphIndex` links chunks that share entities and spreads activation from the query's entities across that graph
(personalised PageRank-style diffusion). That is how a second-hop document that never mentions the query's
entities can still rank:

```python
from ragforge import GraphIndex
graph = GraphIndex(); graph.add(rag.chunks)
graph.search("Where is the manufacturer of the ZX-9 sensor headquartered?", k=4)
graph.explain("...")   # seeds and activations, used by the playground's Knowledge Graph tab
```

The default extractor is a deterministic heuristic (capitalised names, acronyms, identifiers); pass `extract_fn=`
to plug in an LLM or NER model. Results on HotpotQA are in [Multi-hop](multihop.md).

## Multi-hop agent

`AgenticRag(rag, rewrite_fn=bridge_rewrite_fn(), min_iterations=2)` retrieves, extracts the bridge entities from
the best passages, rewrites the question around them and retrieves again while excluding chunks already seen.

## Entailment-checked citations

`attribute(..., entailment_fn=nli_cross_encoder_fn())` re-checks every cited sentence with a natural-language-inference
cross-encoder, so a citation that merely *overlaps* a passage (but contradicts or doesn't support it) is rejected.
The model is optional (`sentence-transformers`); any `(premise, hypothesis) -> float` function works.

## Memory-mapped index directories

A single JSON file is rewritten on every save and fully parsed on every open. `DiskIndex` keeps an index as a
directory instead: vectors in a memory-mapped file, the graph in CSR form, an append-only chunk log and a crash-safe
manifest committed atomically.

```python
from ragforge import DiskIndex
ix = DiskIndex.create("kb.rag", vector_backend="hnsw")
ix.ingest("handbook", text)
ix.commit()           # atomic; a crash before this leaves the previous version intact
ix.compact()          # reclaim space from deleted documents
ix = DiskIndex.open("kb.rag", readonly=True)   # near-instant, safe for many readers
ix.pipeline.retrieve("...")
```

CLI: `ragforge ingest ./docs --format dir -i kb.rag`. A single writer is enforced with an advisory lock.

## Async, ASGI and multi-tenancy

`AsyncRagPipeline(rag, agenerate_fn=...)` runs retrieval in a thread pool and awaits generation with a concurrency
cap. `create_app(pipeline, api_key=...)` is a dependency-free ASGI application you can run under any ASGI server
(`ragforge serve --asgi`). `MultiTenantRag(factory, root=...)` gives every tenant its own isolated pipeline with an
LRU cap, per-tenant chunk quotas and `erase()` for right-to-be-forgotten requests.

## Terminal chat and Claude Desktop

`ragforge chat -i kb.json` is a streaming terminal chat with citations (`/agent on` switches to the multi-step loop, `/cite`, `/trace`; `/help` lists them). For
Claude Desktop, `python scripts/build_mcpb.py` builds a one-click `.mcpb` extension that vendors the package (it has
no dependencies) and asks for an index file in its settings; ingestion is off unless you tick it. The MCP server also
exposes documents as **resources** and ready-made **prompts**.

## Real LLM and embedding providers

```python
from ragforge import anthropic_generate_fn, anthropic_stream_fn, anthropic_complete_fn   # pip install ragforge[anthropic]
from ragforge.embeddings_providers import openai_embed_fn, cached_embed_fn

rag = RagPipeline(embed_fn=cached_embed_fn(openai_embed_fn()),
                  generate_fn=anthropic_generate_fn(model="claude-sonnet-5-5"),
                  stream_generate_fn=anthropic_stream_fn())
```

Every provider import is lazy and raises an error naming the exact `pip install` if missing.

## Observability

```python
from ragforge.telemetry import Tracer, InMemoryExporter, console_exporter, otel_exporter
rag = RagPipeline(tracer=Tracer(exporters=[console_exporter, otel_exporter()]))
```

Spans: `answer › retrieve › fusion_search · rerank · guardrails`, `generate`, and `agent.*`.

## The web playground

`python scripts/build_site.py && python -m http.server -d site 8000` serves the playground locally.
It loads a zip of `src/ragforge` into Pyodide inside a Web Worker and talks to `web/bridge.py` over JSON
strings, so what you see is the library's own output. The GitHub Pages workflow publishes the same build.
