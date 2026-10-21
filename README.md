<div align="center">

<img src="docs/assets/banner.svg" alt="RagForge — hybrid retrieval, forged from scratch" width="100%">

<br>

[![CI](https://github.com/fragompul/ragforge/actions/workflows/ci.yml/badge.svg)](https://github.com/fragompul/ragforge/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/live%20demo-try%20it%20in%20your%20browser-ff6b2c?logo=webassembly&logoColor=white)](https://fragompul.github.io/ragforge/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Dependencies: 0](https://img.shields.io/badge/runtime%20dependencies-0-34d399)](pyproject.toml)
[![Type checked: mypy strict](https://img.shields.io/badge/types-mypy%20strict-2dd4ef)](https://mypy-lang.org/)
[![Coverage](https://img.shields.io/badge/coverage-98%25-34d399)](#engineering-quality)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)
[![MCP server](https://img.shields.io/badge/MCP-server-8b7cf6)](#-use-it-from-claude-mcp)
[![License: MIT](https://img.shields.io/badge/license-MIT-ffb020)](LICENSE)

**[Live demo](https://fragompul.github.io/ragforge/)** · **[Benchmarks](docs/benchmarks.md)** · **[Evaluation](docs/evaluation.md)** · **[Guides](docs/guides.md)** · **[Testing](docs/testing.md)** · **[Math](docs/math.md)** · **[Architecture](docs/architecture.md)** · **[Changelog](CHANGELOG.md)**

</div>

---

**RagForge is the retrieval stack production RAG systems depend on — hybrid search, reranking, guardrails, verifiable citations, agentic retrieval, evaluation and an MCP server — in a codebase small enough to read in an afternoon.** The core has **zero runtime dependencies**; everything heavy (NumPy, Claude, OpenAI, Cohere, sentence-transformers) is optional and lazily imported.

It is built to be *understood*, *measured* and *trusted*: strict typing, property-based tests, reproducible benchmarks on a public dataset, and a [web playground](https://fragompul.github.io/ragforge/) that runs **the real Python library in your browser** (Pyodide / WebAssembly) — no server, no API keys.

<div align="center">
<a href="https://fragompul.github.io/ragforge/"><img src="docs/assets/screenshots/playground.jpg" alt="RagForge playground: BM25, vector and hybrid RRF results side by side" width="92%"></a>
<br><sub>BM25 · vector · hybrid RRF side by side, with answer citations and a per-stage trace — <a href="https://fragompul.github.io/ragforge/">open the live demo</a></sub>
</div>

## Highlights

| | |
| :--- | :--- |
| **Hybrid retrieval** | BM25 (inverted index) + dense vectors fused with weighted **Reciprocal Rank Fusion** — no score normalization. **nDCG@10 0.701 on BEIR SciFact** (+2.3 pts over the best single retriever); on FiQA and SCIDOCS plain vectors win, and the [report](docs/evaluation.md) says so. |
| **HNSW from scratch** | Pure-Python approximate-nearest-neighbor graph with the paper's neighbor-selection heuristic: **recall@10 0.99 at `ef_search=50`** on real embeddings, with a visualizable search trace. |
| **Guardrails** | Indirect **prompt-injection** screening (OWASP LLM01), **PII/secret redaction** and untrusted-data spotlighting on every retrieved context — benchmarked on public injection datasets with a train/test split ([`docs/guardrails.md`](docs/guardrails.md)). |
| **Multi-hop & graph** | Entity-graph retrieval (GraphRAG-lite) and a bridge-entity agent: **both gold paragraphs found for 88.7 %** of HotpotQA questions vs. 79.3 % for plain hybrid ([`docs/multihop.md`](docs/multihop.md)). |
| **Compact, fast indexes** | int8 / binary quantization (4× / 32× smaller, quality kept by rescoring), in-graph filtered HNSW, SPLADE, and a memory-mapped on-disk index. |
| **Verifiable citations** | Each answer sentence is traced to **exact character offsets** in the source, with a groundedness score that doubles as a hallucination signal. |
| **Agentic RAG** | Decompose → retrieve → grade → rewrite → retry — and **abstain** instead of hallucinating. Every decision is inspectable. |
| **MCP server** | Expose any index to Claude and other assistants over the Model Context Protocol. Zero-dependency stdio JSON-RPC; read-only by default. |
| **Evaluation** | nDCG / Recall / MRR (BEIR conventions), RAGAS-style metrics, an LLM-as-judge harness, and a one-command public-benchmark runner. |
| **Production plumbing** | Streaming (SSE), semantic cache, OpenTelemetry-bridged tracing, NumPy backend, JSON persistence, HTTP server, Docker. |

## Measured, not claimed

Retrieval quality on **BEIR SciFact** (5,183 abstracts, 300 test queries) with ragforge's own indexes and `all-MiniLM-L6-v2` vectors — component scores match the published BEIR numbers, which is the sanity check that the implementations are right. Four datasets, including where hybrid fusion does *not* win, are in [`docs/evaluation.md`](docs/evaluation.md):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/eval_scifact_dark.svg">
  <img alt="nDCG@10 on BEIR SciFact: BM25 0.664, BM25 english 0.678, vector 0.648, hybrid RRF 0.701" src="docs/assets/eval_scifact_light.svg" width="680">
</picture>

Search at scale (32-d clustered vectors, top-10, p50): exact pure-Python scan **51 ms** at 20k chunks vs. **HNSW 0.6 ms** (recall@10 ≈ 1.0) vs. NumPy exact **0.18 ms**. The BM25 inverted index is **12–15× faster** than the per-query scan it replaced. Full tables, ablations and the honest story of a recall bug found and fixed along the way are in [`docs/benchmarks.md`](docs/benchmarks.md) and [`docs/evaluation.md`](docs/evaluation.md) — both regenerated by single commands (`scripts/generate_benchmarks.py`, `scripts/eval_beir.py`).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/ann_latency_dark.svg">
  <img alt="Query latency vs corpus size: exact scan vs HNSW" src="docs/assets/ann_latency_light.svg" width="680">
</picture>

## Quickstart

```bash
pip install ragforge                 # zero dependencies
pip install "ragforge[fast]"         # + NumPy exact search     (also: [anthropic] [openai] [cohere] [local])
ragforge demo                        # guided offline tour: hybrid search, guardrails, citations, agents
```

```python
from ragforge import RagPipeline, Guardrails, attribute
from ragforge.extractive import extractive_generate_fn      # keyless baseline answerer

rag = RagPipeline(
    bm25_analyzer="english",
    vector_backend="hnsw",                  # or "brute_force" | "numpy"
    guardrails=Guardrails(action="sanitize"),
    generate_fn=extractive_generate_fn(),   # swap in anthropic_generate_fn() / openai_generate_fn()
)
rag.ingest("eiffel", "The Eiffel Tower is in Paris. It was completed in 1889.")

answer = rag.answer("When was the Eiffel Tower completed?", k=3)
cited = attribute(answer.answer, answer.contexts)
print(cited.to_markdown())          # → "The Eiffel Tower is in Paris. [1] It was completed in 1889. [1]" + source spans
print(cited.groundedness)           # → 1.0
```

<details>
<summary><b>See the full guided tour (<code>ragforge demo</code>)</b> — real program output</summary>
<br>
<img src="docs/assets/demo_terminal.svg" alt="Output of the ragforge demo command" width="100%">
</details>

## What's inside

<table>
<tr>
<td width="50%" valign="top">

**Retrieval**
- Chunkers: recursive, sentence, markdown, fixed-window — all keep source **character offsets**
- BM25 inverted index (`analyzer="english"`: stopwords + light stemming)
- Vector backends: exact scan · **NumPy** exact · **HNSW** (in-graph filters) · **int8 / binary** quantized
- **SPLADE** sparse, N-way fusion, **query router**, contextual retrieval, HyDE, **entity graph**
- **Memory-mapped index directories** (atomic commits, compaction), async + ASGI, multi-tenancy
- Weighted **RRF** hybrid fusion, **multi-query** expansion
- Rerankers: MMR (diversity, optionally over the fused score), heuristic, cross-encoder adapter
- Metadata filtering (`filter_fn`) on every path

**Generation & grounding**
- Pluggable `generate_fn`; **Claude / OpenAI / Ollama** adapters; keyless extractive answerer
- **Streaming** (`answer_stream`, SSE endpoint, provider stream adapters)
- **Verifiable citations** + groundedness; `[n]` markers are *verified*, not trusted

</td>
<td width="50%" valign="top">

**Safety**
- Prompt-injection detector (EN + ES rules, hidden-Unicode, exfiltration patterns)
- PII / secret redaction (e-mail, phone, Luhn-valid cards, API keys, IPs)
- Policies: flag · sanitize · drop; spotlighting prompt formatter
- Hardened HTTP server (body/`k` limits) and **read-only-by-default MCP tools**

**Agents & operations**
- `AgenticRag`: decompose, grade, rewrite, abstain — LLM or lexical decision functions
- **MCP server** (`ragforge mcp`) for Claude Desktop / Claude Code / any MCP client
- **Semantic cache** with TTL, namespaces and corpus-change invalidation
- Tracing → console · memory · **OpenTelemetry**; JSON persistence; Docker

**Evaluation**
- `ragforge.metrics`: nDCG, Recall, Precision, MRR, MAP
- RAGAS-style metrics, **LLM-as-judge** (rubrics, JSON verdicts, robust parsing)
- BEIR runner (4 datasets), HotpotQA multi-hop, guardrail benchmarks, charts and Markdown reports
- Hypothesis fuzzing, a stateful disk-index model test and mutation testing ([`docs/testing.md`](docs/testing.md))

</td>
</tr>
</table>

## 🔌 Use it from Claude (MCP)

```bash
ragforge ingest ./docs -i kb.json --backend hnsw --analyzer english
claude mcp add ragforge -- ragforge mcp -i kb.json --guardrails sanitize
```

Claude can now call `search_knowledge_base` (hybrid / BM25 / vector), `ask_knowledge_base` (answer **with sentence-level citations and a groundedness score**) and `knowledge_base_stats`. Ingestion is off unless you pass `--allow-ingest`: anything an LLM can call, an attacker's prompt can call. [`examples/mcp_client_demo.py`](examples/mcp_client_demo.py) drives the server with raw JSON-RPC so you can see the whole protocol.

## Guides in 60 seconds

<details>
<summary><b>🛡️ Guardrails against indirect prompt injection and PII</b></summary>

```python
from ragforge import Guardrails, RagPipeline
from ragforge.guardrails import spotlight_prompt_formatter

rag = RagPipeline(guardrails=Guardrails(action="sanitize"),       # or "flag" / "drop"
                  prompt_formatter=spotlight_prompt_formatter)    # wraps contexts as untrusted data
ctx = rag.retrieve("What is the shipping time?")[0]
ctx.text         # "... [removed: suspected prompt injection] ... [REDACTED:email]"
ctx.guardrails   # {"risk": 0.9, "findings": [{"kind": "injection", "rule": "instruction_override", ...}]}
```
Pattern rules make poisoning visible and auditable; they are a first layer, not a complete defense — see the [Security Lab](https://fragompul.github.io/ragforge/#security).
</details>

<details>
<summary><b>🧭 Agentic RAG that knows when to abstain</b></summary>

```python
from ragforge import AgenticRag
from ragforge.agent import heuristic_decompose_fn, llm_grade_fn, llm_rewrite_fn
from ragforge import anthropic_complete_fn

complete = anthropic_complete_fn()                       # any prompt -> text function
agent = AgenticRag(rag, decompose_fn=heuristic_decompose_fn,
                   grade_fn=llm_grade_fn(complete), rewrite_fn=llm_rewrite_fn(complete))
result = agent.run("What does ERR-4012 mean? How does RRF work?")
[s.kind for s in result.steps]   # decompose, retrieve, grade, retrieve, grade, answer
result.abstained                 # True when evidence stays insufficient
```
</details>

<details>
<summary><b>⚡ Streaming, semantic cache and the HTTP API</b></summary>

```python
for event in rag.answer_stream("How does hybrid search work?"):   # contexts → token… → done
    if event["type"] == "token": print(event["text"], end="")

from ragforge import CachedPipeline
cached = CachedPipeline(rag)            # rephrased repeats are served without re-running the pipeline
```
```bash
ragforge serve -i kb.json --guardrails sanitize        # POST /query · POST /query/stream (SSE) · GET /health
```
</details>

<details>
<summary><b>📏 Evaluate retrieval and answers</b></summary>

```python
from ragforge.metrics import ndcg_at_k, recall_at_k
from ragforge.judge import evaluate_with_judge
from ragforge import anthropic_complete_fn

ndcg_at_k(["d3", "d1", "d9"], {"d1": 1, "d2": 1}, k=10)
evaluate_with_judge(rag, cases, anthropic_complete_fn(), rubrics=("faithfulness", "correctness"))
```
```bash
python scripts/eval_beir.py --download scifact --data-dir data && python scripts/eval_beir.py --data-dir data/scifact --embedder minilm
```
</details>

More in [`docs/guides.md`](docs/guides.md) and the runnable [`examples/`](examples/).

## Architecture

```mermaid
flowchart LR
    subgraph Ingest["Ingestion"]
        D[Document] --> C["Chunker<br/>offsets + metadata"]
    end
    subgraph Index["Indexes"]
        C --> B["BM25<br/>inverted index"]
        C --> V["Vectors<br/>exact · NumPy · HNSW"]
    end
    subgraph Retrieve["Retrieval"]
        Q[Query] -.->|multi-query| MQ[Expansion]
        MQ --> B & V
        Q --> B & V
        B --> F["Weighted RRF"]
        V --> F
        F --> R["Rerank<br/>MMR · cross-encoder"]
    end
    subgraph Safe["Safety"]
        R --> G["Guardrails<br/>injection · PII"]
    end
    subgraph Gen["Generation"]
        G --> P[Prompt] --> L["LLM<br/>Claude · OpenAI · local"]
        L --> CI["Citations<br/>spans · groundedness"]
    end
    A["Agent loop<br/>grade · rewrite · abstain"] -.-> Retrieve
    CI --> S["Serve<br/>CLI · HTTP/SSE · MCP · Web"]
    S -.-> T["Tracer → OpenTelemetry"]
```

Design rationale for every box lives in [`docs/architecture.md`](docs/architecture.md); derivations (BM25, RRF, MMR's $(1-1/e)$ guarantee, HNSW complexity and the neighbor-selection heuristic, groundedness) are in [`docs/math.md`](docs/math.md).

<div align="center">
<img src="docs/assets/screenshots/hnsw.jpg" alt="HNSW Explorer: a query navigating the real graph" width="48%">
<img src="docs/assets/screenshots/agent.jpg" alt="Agent Lab: decision trail with grading and citations" width="48%">
<br><sub>Left: the <b>HNSW Explorer</b> animates the library's real graph. Right: the <b>Agent Lab</b> shows every decision.</sub>
</div>

## Command line

```bash
ragforge demo                                                    # offline guided tour
ragforge ingest ./docs -i kb.json --backend hnsw --analyzer english
ragforge query "How does hybrid search work?" -i kb.json --reranker mmr --agentic --cite --guardrails sanitize --trace
ragforge evaluate cases.json -i kb.json -o report.json
ragforge serve -i kb.json --port 8000 --guardrails sanitize      # HTTP JSON + SSE
ragforge mcp -i kb.json                                          # Model Context Protocol over stdio
ragforge benchmark                                               # quick throughput/latency check
ragforge chat -i kb.json                                         # streaming terminal chat with citations
python scripts/build_mcpb.py                                     # one-click Claude Desktop extension
```

## Runnable examples

| Example | Shows |
| :--- | :--- |
| [`basic_pipeline.py`](examples/basic_pipeline.py) | ingestion → hybrid retrieval → answer |
| [`hybrid_vs_single.py`](examples/hybrid_vs_single.py) | hybrid RRF beating BM25 and vector alone |
| [`guardrails_demo.py`](examples/guardrails_demo.py) | poisoned document + PII neutralized before the prompt |
| [`citations_and_agent.py`](examples/citations_and_agent.py) | verifiable citations, multi-hop agent, abstention |
| [`mcp_client_demo.py`](examples/mcp_client_demo.py) | raw JSON-RPC conversation with `ragforge mcp` |
| [`cache_and_judge.py`](examples/cache_and_judge.py) | semantic cache and LLM-as-judge evaluation |
| [`evaluate_suite.py`](examples/evaluate_suite.py) · [`metadata_filtering_and_mmr.py`](examples/metadata_filtering_and_mmr.py) · [`persistence_and_serialization.py`](examples/persistence_and_serialization.py) | evaluation, filtering + MMR, persistence |

## Engineering quality

- **Strictly typed** (`mypy --strict`, `py.typed` shipped), formatted and linted with `ruff`, pre-commit hooks included.
- **~98 % test coverage** with branch coverage, **Hypothesis property-based tests**, a real-socket HTTP test suite, a real-subprocess MCP test, and examples executed as integration tests.
- **CI matrix** (Python 3.11–3.13) runs lint, types, tests with and without NumPy, a CLI smoke test, a Docker build, and **builds the Pages site**.
- **Honest benchmarking**: every figure is produced by a script in `scripts/`; a defect found in the original HNSW (recall plateau) was diagnosed, fixed with the paper's heuristic, and documented rather than hidden.
- **Zero runtime dependencies**: the library, the CLI, the HTTP server and the MCP server are standard-library Python.

```bash
pip install -e ".[dev]"
ruff format --check src tests examples scripts web && ruff check src tests examples scripts web
mypy src
pytest --cov=ragforge --cov-fail-under=90
python scripts/build_site.py && python -m http.server -d site 8000     # the web playground
```

<details>
<summary><b>Repository map</b></summary>

```
src/ragforge/   chunking · index (BM25, vectors) · ann (HNSW) · fast (NumPy) · reranking · pipeline
                quantization · fusion · sparse · graph · routing · contextual · storage · aio · asgi · tenancy · chat
                guardrails · citations · agent · judge · metrics · cache · extractive · mcp_server · server
                llm_providers · embeddings_providers · telemetry · evaluation · cli · demo · sample
web/            playground (HTML + ES modules + Pyodide worker + Python bridge)
scripts/        generate_benchmarks · eval_beir · eval_multihop · eval_guardrails · eval_contextual
                build_site · build_mcpb · build_space · make_notebook · render_terminal · svg_charts
docs/           architecture · math · benchmarks · evaluation · multihop · guardrails · testing · guides · blog/
notebooks/      Colab quickstart
tests/          unit, property-based, integration, protocol and example tests
```
</details>

## Roadmap

- [ ] Late-interaction (ColBERT-style) reranker adapter
- [ ] LLM-backed entity extraction and agent grading evaluated on HotpotQA
- [ ] Streaming MCP tool results and resource subscriptions
- [ ] Native (Rust/C) distance kernels for the quantized scans

Issues and ideas are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License & author

[MIT](LICENSE). Built by **Francisco Javier Gómez Pulido** — *AI Lead @ AAPEX · Double Degree in Mathematics & Computer Science · Master's in Artificial Intelligence*.

[LinkedIn](https://www.linkedin.com/in/frangomezpulido) · [GitHub](https://github.com/fragompul) · [Email](mailto:frangomezpulido2002@gmail.com)

If RagForge is useful or interesting to you, a ⭐ helps others find it.
