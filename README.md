# ragforge

[![CI](https://github.com/fragompul/ragforge/actions/workflows/ci.yml/badge.svg)](https://github.com/fragompul/ragforge/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type_checked-mypy_strict-blue)](https://mypy-lang.org/)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/fragompul/ragforge)

A production-hardened, zero-external-dependency RAG (Retrieval-Augmented Generation) engine designed with clean architecture, strict typing, and high reliability:
- **Multi-Strategy Chunking**: Hierarchical recursive, sentence-boundary, markdown header-aware, and sliding window chunkers with source character offsets.
- **Hybrid Retrieval**: Okapi BM25 sparse keyword search combined with dense vector cosine search via **Weighted Reciprocal Rank Fusion (RRF)**.
- **Precision & Diversity Reranking**: Second-stage scoring with **Maximal Marginal Relevance (MMR)** (anti-redundancy), neural cross-encoder adapters, and heuristic token overlap.
- **Metadata Filtering**: First-class attribute filtering (`filter_fn`) for multi-tenant and category isolation.
- **RAGAS-Style Evaluation Engine**: Context Precision (Set & MAP@k), Context Recall, Context F1, Faithfulness (hallucination detection), Answer Relevancy, and Semantic Similarity.
- **Index Serialization & Persistence**: Clean JSON save/load routines for offline batch indexing and fast serving startup.
- **Production CLI**: Command-line interface for ingestion, querying, automated evaluation, and latency benchmarking.

---

## The Engineering Problem

Most RAG failures in production are not model failures—they are retrieval failures wearing an LLM's face:

1. **Diluted vs. Truncated Contexts**: Fixed-size chunking blindly splits sentences across boundaries or produces massive chunks that dilute keyword and vector matches.
2. **Dense-Only Blindspots**: Pure vector search consistently fails on exact identifiers (error codes, order IDs, product SKUs, proper nouns) that sparse keyword search nails instantly.
3. **Context Redundancy**: Top-k retrieval often returns 5 variations of the same paragraph, saturating the LLM context window without adding net new information.
4. **Lack of Regression Testing**: Chunking tweaks, embedding model upgrades, or prompt changes ship without automated signals measuring whether retrieval precision or faithfulness regressed.

`ragforge` addresses all four challenges with modular components that can be inspected, tested, benchmarked, and customized.

---

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        Doc["Document"] --> Chunker["Chunker\n(Recursive / Sentence / Markdown)"]
        Chunker --> Chunks["Chunk[]\n(Offsets + Metadata)"]
    end

    subgraph Indexing
        Chunks --> BM25["BM25 Index\n(Sparse Lexical)"]
        Chunks --> Vector["Vector Index\n(Dense Semantic)"]
    end

    subgraph Retrieval
        Query["Query"] --> BM25
        Query --> Vector
        BM25 -->|Top-N| Hybrid["Hybrid RRF Fusion\nΣ w_i / (k + rank_i)"]
        Vector -->|Top-N| Hybrid
    end

    subgraph Reranking
        Hybrid --> Rerank["Reranker\n(MMR Diversity / Cross-Encoder)"]
        Rerank --> TopK["Top-K Contexts"]
    end

    subgraph Generation
        TopK --> Prompt["Prompt Formatter"]
        Query --> Prompt
        Prompt --> LLM["LLM Generation"]
        LLM --> Answer["RagAnswer\n(Attribution + Latency)"]
    end
```

See [`docs/architecture.md`](docs/architecture.md) for deep architectural diagrams and mathematical formulations.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/fragompul/ragforge.git
cd ragforge

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

`ragforge` has **zero required third-party dependencies** for core operations, keeping installations instant and secure.

---

## Quickstart

### 1. Basic Ingestion, Hybrid Search & Answering

```python
from ragforge import RagPipeline
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.reranking import HeuristicReranker

# Initialize pipeline with recursive chunker and heuristic reranker
pipeline = RagPipeline(
    chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=40),
    reranker=HeuristicReranker(),
)

# Ingest documents with metadata
pipeline.ingest(
    doc_id="eiffel_tower",
    text="The Eiffel Tower is in Paris, France. It was completed in 1889 and stands 330 meters tall.",
    metadata={"category": "monuments", "year": 1889},
)

# Retrieve contexts and generate an answer
response = pipeline.answer("How tall is the Eiffel Tower and when was it built?", k=2)

print(f"Answer: {response.answer}")
print(f"Retrieval Latency: {response.retrieval_latency_ms:.2f} ms")
for ctx in response.contexts:
    print(f"  [{ctx.doc_id}] (Score: {ctx.score:.3f}, Source: {ctx.provenance}) :: {ctx.text}")
```

---

### 2. Production Integration (OpenAI / Anthropic / Local Transformers)

Drop in external embedding and generation functions without changing the pipeline interface:

```python
import openai
from ragforge import RagPipeline
from ragforge.chunking import SentenceChunker
from ragforge.reranking import CrossEncoderReranker

client = openai.OpenAI()

def openai_embed(text: str) -> list[float]:
    res = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return res.data[0].embedding

def openai_generate(query: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(contexts)
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer the question strictly based on the provided context."},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
        ],
        temperature=0.0,
    )
    return res.choices[0].message.content or ""

pipeline = RagPipeline(
    chunker=SentenceChunker(max_chars=500, overlap_sentences=1),
    embed_fn=openai_embed,
    generate_fn=openai_generate,
)
```

---

### 3. Metadata Filtering & MMR Diversity Reranking

```python
from ragforge import RagPipeline
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.reranking import MaxMarginalRelevanceReranker

pipeline = RagPipeline(
    chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=30),
    reranker=MaxMarginalRelevanceReranker(lambda_mult=0.7),  # Balance relevance & novelty
)

pipeline.ingest_batch([
    {"id": "doc1", "text": "Enterprise SLA guarantees 99.99% uptime.", "metadata": {"tier": "enterprise"}},
    {"id": "doc2", "text": "Standard SLA guarantees 99.9% uptime.", "metadata": {"tier": "standard"}},
])

# Filter strictly for enterprise tier during retrieval
res = pipeline.answer(
    "What is the SLA uptime guarantee?",
    k=1,
    filter_fn=lambda chunk: chunk.metadata.get("tier") == "enterprise",
)
```

---

### 4. RAGAS-Style Automated Quality Evaluation

Run automated regressions to evaluate retrieval precision, context recall, faithfulness, and answer relevancy:

```python
from ragforge.evaluation import EvalCase, evaluate_pipeline

eval_cases = [
    EvalCase(
        query="How tall is the Eiffel Tower?",
        relevant_doc_ids=["eiffel_tower"],
        ground_truth_answer="The Eiffel Tower is 330 meters tall.",
    ),
]

summary = evaluate_pipeline(pipeline, eval_cases, k=2)

# Print formatted markdown table
print(summary.to_markdown_table())

# Save report for CI artifact tracking
summary.save("eval_report.json")
```

Output:
| Metric | Mean Score |
| :--- | :--- |
| **Overall Score** | `0.9850` |
| **Faithfulness** | `1.0000` |
| **Answer Relevancy** | `0.9520` |
| **Context Precision** | `1.0000` |
| **Ranked Precision (MAP@k)** | `1.0000` |
| **Context Recall** | `1.0000` |
| **Context F1** | `1.0000` |
| **Answer Similarity** | `0.9880` |
| **Mean Latency (ms)** | `1.42 ms` |

---

### 5. Index Persistence & Serving

```python
# Build and persist index in an offline batch job
pipeline.save("production_index.json")

# Load pre-built index in serving container in sub-milliseconds
serving_pipeline = RagPipeline.load("production_index.json")
answer = serving_pipeline.answer("What is our refund policy?", k=3)
```

---

## Command Line Interface (CLI)

`ragforge` provides a built-in CLI for indexing, querying, evaluating, and benchmarking:

```bash
# Ingest markdown and text documentation into a persistent index
ragforge ingest ./docs --index rag_index.json --chunk-size 400

# Query the index with MMR reranking
ragforge query "How does hybrid search work?" --index rag_index.json --reranker mmr -k 3

# Run automated evaluation test suite
ragforge evaluate eval_cases.json --index rag_index.json -o report.json

# Run synthetic throughput and latency benchmark
ragforge benchmark
```

---

## Runnable Examples

Complete, self-contained examples are available in the [`examples/`](examples/) directory:

```bash
python examples/basic_pipeline.py                  # End-to-end ingestion, retrieval & answers
python examples/hybrid_vs_single.py                # Demonstrates hybrid RRF beating BM25/Vector alone
python examples/evaluate_suite.py                  # Automated RAGAS-style evaluation and reporting
python examples/metadata_filtering_and_mmr.py      # Metadata filtering and MMR context diversity
python examples/persistence_and_serialization.py   # Index serialization and zero-recomputation loading
```

---

## Key Design Decisions & Rationale

- **Rank-Based Fusion (RRF) vs Score Blending**: BM25 scores are unbounded ($0 \to \infty$) whereas cosine similarities range in $[-1.0, 1.0]$. RRF fuses rank positions rather than raw magnitudes, preventing scale distortion.
- **MMR for Diversity**: Mitigates "context redundancy," selecting diverse chunks across relevant sources so LLMs receive rich, non-repetitive contexts.
- **Rank-Aware Context Precision (MAP@k)**: Measures whether relevant documents were placed at top ranks, catching ordering regressions that set-based precision misses.
- **Deterministic Embeddings for CI**: Built-in `hashing_embed` uses deterministic CRC32 hashing to enable zero-network unit tests and reproducible benchmarks in CI environments.

---

## Development & Verification

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
ruff format src tests examples

# Lint code
ruff check src tests examples

# Strict static type checking
mypy src

# Run test suite with branch coverage validation
pytest --cov=ragforge --cov-report=term-missing --cov-fail-under=90
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Francisco Javier Gómez Pulido**

*AI Lead @ AAPEX | Double Degree in Mathematics & Computer Science | Master's in Artificial Intelligence*

📫 **Let's connect:**
- **LinkedIn:** [linkedin.com/in/frangomezpulido](https://www.linkedin.com/in/frangomezpulido)
- **GitHub:** [github.com/fragompul](https://github.com/fragompul)
- **Email:** [frangomezpulido2002@gmail.com](mailto:frangomezpulido2002@gmail.com)
