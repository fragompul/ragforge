# Architecture & System Design

`ragforge` is designed around a modular, two-stage retrieve-and-rerank architecture with end-to-end evaluation and offline indexing capabilities.

---

## 1. End-to-End Pipeline Overview

```mermaid
flowchart TD
    subgraph Ingestion["1. Ingestion & Chunking"]
        RawDocs["Source Documents\n(Text, Markdown, Docs)"]
        Chunker["Chunker Strategy\n• RecursiveCharacterChunker\n• SentenceChunker\n• MarkdownChunker\n• FixedSizeChunker"]
        Chunks["Chunk Stream\n[id, text, doc_id, metadata, offsets]"]
        RawDocs --> Chunker --> Chunks
    end

    subgraph Indexing["2. Dual Dual-Index Ingestion"]
        BM25["BM25Index\n(Sparse Lexical / Okapi BM25+)"]
        Vector["VectorIndex\n(Dense Semantic / Embedding)"]
        Chunks --> BM25
        Chunks --> Vector
    end

    subgraph Retrieval["3. Hybrid Retrieval & Filtering"]
        Query["User Query"]
        Filter["Metadata Filter\n(filter_fn: category, tenant, etc.)"]
        RRF["Weighted Reciprocal Rank Fusion (RRF)\nscore = Σ (w_i / (k_rrf + rank_i))"]
        Query --> BM25
        Query --> Vector
        Filter -.-> BM25
        Filter -.-> Vector
        BM25 -->|Sparse Top-N| RRF
        Vector -->|Dense Top-N| RRF
    end

    subgraph Reranking["4. Second-Stage Reranking & Diversity"]
        Candidates["Fused Candidate Pool\n(N=20..50)"]
        Reranker["Reranking Engine\n• MaxMarginalRelevance (MMR Diversity)\n• CrossEncoderReranker (Neural)\n• HeuristicReranker (Jaccard)"]
        TopK["Top-K Context Chunks\n(Enriched with Provenance & Offsets)"]
        RRF --> Candidates --> Reranker --> TopK
    end

    subgraph Generation["5. Generation & Attribution"]
        PromptFormatter["Prompt Formatter\n(System + Structured Contexts)"]
        LLM["LLM Generator / generate_fn"]
        Answer["RagAnswer\n[answer, contexts, latencies, prompt]"]
        TopK --> PromptFormatter
        Query --> PromptFormatter
        PromptFormatter --> LLM --> Answer
    end
```

---

## 2. Evaluation Flow (RAGAS-Style)

```mermaid
flowchart TB
    subgraph EvaluationSuite["RAGAS-Style Quality Evaluation Engine"]
        Cases["EvalCase[]\n• query\n• relevant_doc_ids\n• ground_truth_answer"]
        Pipeline["RagPipeline Instance"]
        Evaluator["evaluate_pipeline()"]
        
        Cases --> Evaluator
        Pipeline --> Evaluator
        
        subgraph Metrics["Independent Failure-Mode Metrics"]
            CP["Context Precision\n(Set precision of retrieved docs)"]
            RCP["Ranked Context Precision\n(Mean Average Precision MAP@k)"]
            CR["Context Recall\n(Ground-truth document coverage)"]
            F1["Context F1\n(Harmonic mean of CP & CR)"]
            F["Faithfulness\n(Answer grounding in retrieved contexts)"]
            AR["Answer Relevancy\n(Query-to-answer semantic alignment)"]
            AS["Answer Similarity\n(Semantic cosine similarity to ground truth)"]
        end
        
        Evaluator --> Metrics
        
        Summary["EvaluationSummary\n• Sequence[EvalResult]\n• Aggregated Means\n• Markdown Summary Table\n• JSON Report Export"]
        Metrics --> Summary
    end
```

---

## 3. Key Architectural Decisions & Tradeoffs

### A. Reciprocal Rank Fusion (RRF) over Weighted Score Blending
- **The Problem:** BM25 produces unbounded positive scores ($0.0 \to \infty$) while cosine similarity bounded in $[-1.0, 1.0]$. Normalizing disparate score distributions via min-max scaling is brittle and sensitive to outlier chunks.
- **The Solution:** RRF operates purely on ordinal ranks:
  $$\text{RRF}(d) = \sum_{m \in \{\text{bm25}, \text{vector}\}} \frac{w_m}{k_{\text{rrf}} + \text{rank}_m(d)}$$
  This eliminates calibration drift and ensures neither signal dominates due to score magnitude.

### B. Maximal Marginal Relevance (MMR) for Context Window Optimization
- **The Problem:** Dense vector retrieval often returns top-5 chunks that are syntactic variations of the same paragraph, wasting precious LLM context tokens without introducing new information.
- **The Solution:** MMR dynamically balances relevance against intra-context redundancy:
  $$\text{MMR}(d_i) = \lambda \cdot \text{Sim}(d_i, q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}(d_i, d_j)$$
  Setting $\lambda = 0.7$ provides optimal query relevance while ensuring diversity across retrieved chunks.

### C. Multi-Strategy Chunking Hierarchy
- `RecursiveCharacterChunker`: Standard recursive splitting respecting natural document hierarchy (paragraphs $\to$ sentences $\to$ words).
- `SentenceChunker`: Preserves complete semantic sentence boundaries for prose.
- `MarkdownChunker`: Extracts document section headers and attaches them to chunk metadata for structured citation.
- `FixedSizeChunker`: Fast word-count sliding window for homogeneous unstructured text.

### D. Zero-Downtime Index Persistence
Indices serialize cleanly to JSON:
- Offline batch indexing jobs build and save the index (`pipeline.save("index.json")`).
- Online serving instances load pre-built indices (`RagPipeline.load("index.json")`) without re-computing embeddings on startup.

---

## 4. Further Reading

See [`docs/math.md`](math.md) for full derivations behind every scoring
function referenced above: the probabilistic origin of BM25, why RRF fuses
ranks instead of scores, the submodularity argument behind MMR's
approximation guarantee, and the complexity analysis of the HNSW
approximate nearest-neighbor index (`src/ragforge/ann.py`).
