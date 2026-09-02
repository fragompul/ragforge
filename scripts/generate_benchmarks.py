"""Regenerates docs/benchmarks.md, its data file, and its charts from real,
reproducible measurements taken with ragforge's own code -- no external
benchmarking framework, no synthetic numbers.

    python scripts/generate_benchmarks.py

Two things are measured end to end:

1. Brute-force ``VectorIndex`` vs. the from-scratch HNSW ``ApproxVectorIndex``
   (see ``src/ragforge/ann.py``): query latency and recall@10 as corpus size
   grows, to show where the ANN crossover point actually falls -- rather
   than asserting it, per the complexity argument in ``docs/math.md#4``.
2. End-to-end ingestion throughput (chunking + dual BM25/vector indexing)
   at the same corpus sizes.

All measurements use ragforge's built-in deterministic ``hashing_embed`` so
results are reproducible across machines and require no network calls or
model downloads (see ``docs/math.md#5`` for its documented bag-of-words
limitations -- irrelevant here since this benchmark measures *retrieval
mechanics*, not semantic quality; see ``examples/hybrid_vs_single.py`` for a
worked example of the semantic-vs-lexical quality tradeoff instead).
"""

from __future__ import annotations

import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from svg_charts import render_line_chart  # noqa: E402

from ragforge import __version__  # noqa: E402
from ragforge.chunking import Chunk, RecursiveCharacterChunker  # noqa: E402
from ragforge.embeddings import hashing_embed  # noqa: E402
from ragforge.index import ApproxVectorIndex, BM25Index, VectorIndex  # noqa: E402

ASSETS_DIR = ROOT / "docs" / "assets"
DOCS_PATH = ROOT / "docs" / "benchmarks.md"

# A pure-Python, dependency-free cosine scan spends most of its time inside
# per-dimension float loops (see ragforge.embeddings.cosine_similarity), so
# embedding dimensionality dominates wall-clock time for *both* backends
# far more than it would with a vectorized (numpy/BLAS) implementation.
# 32 dims keeps the whole sweep runnable in well under a minute while still
# exercising the same code paths as the library default (128 dims).
_BENCHMARK_DIMS = 32


def _benchmark_embed(text: str) -> list[float]:
    return hashing_embed(text, dims=_BENCHMARK_DIMS)


# The HNSW build in benchmark_ann_vs_brute_force is O(n log n) with a real
# per-node constant (greedy layer search + neighbor pruning, all pure
# Python -- see HNSWIndex.add), so this sweep stays modest to keep the
# script runnable in ~1-2 minutes. Query latency's *shape* is already clear
# at these sizes: see docs/benchmarks.md for the actual numbers.
ANN_CORPUS_SIZES = [300, 1500, 4000]
# Ingestion has no such bottleneck (chunking + appending vectors is O(n)
# with a small constant), so it can sweep further to show throughput trend.
INGEST_CORPUS_SIZES = [500, 2000, 6000, 15000]
QUERIES_PER_SIZE = 20
TOP_K = 10
HNSW_PARAMS = {"m": 8, "ef_construction": 40, "ef_search": 30}
_TOPICS = [
    "consensus", "vector search", "reranking", "chunking", "evaluation",
    "latency", "throughput", "embeddings", "indexing", "caching",
]  # fmt: skip


def _make_corpus(n: int, rng: random.Random) -> list[Chunk]:
    chunks = []
    for i in range(n):
        topic_a, topic_b = rng.sample(_TOPICS, 2)
        text = (
            f"Document {i} discusses {topic_a} and {topic_b} in distributed RAG systems. "
            f"Unique reference code REF-{i:06d} identifies this record."
        )
        chunks.append(Chunk(id=f"c{i}", text=text, doc_id=f"doc{i}", position=0))
    return chunks


def benchmark_ann_vs_brute_force() -> list[dict[str, Any]]:
    rng = random.Random(42)
    results = []

    for n in ANN_CORPUS_SIZES:
        chunks = _make_corpus(n, rng)

        brute = VectorIndex(embed_fn=_benchmark_embed)
        t0 = time.perf_counter()
        brute.add(chunks)
        brute_build_ms = (time.perf_counter() - t0) * 1000

        ann = ApproxVectorIndex(embed_fn=_benchmark_embed, **HNSW_PARAMS)
        t0 = time.perf_counter()
        ann.add(chunks)
        ann_build_ms = (time.perf_counter() - t0) * 1000

        queries = [
            f"REF-{rng.randrange(n):06d} distributed systems" for _ in range(QUERIES_PER_SIZE)
        ]

        brute_latencies_ms = []
        ground_truth = []
        for query in queries:
            t0 = time.perf_counter()
            res = brute.search(query, k=TOP_K)
            brute_latencies_ms.append((time.perf_counter() - t0) * 1000)
            ground_truth.append({r.chunk.id for r in res})

        ann_latencies_ms = []
        recalls = []
        for query, exact in zip(queries, ground_truth, strict=True):
            t0 = time.perf_counter()
            res = ann.search(query, k=TOP_K)
            ann_latencies_ms.append((time.perf_counter() - t0) * 1000)
            approx = {r.chunk.id for r in res}
            recalls.append(len(exact & approx) / max(len(exact), 1))

        row = {
            "n": n,
            "brute_build_ms": round(brute_build_ms, 2),
            "ann_build_ms": round(ann_build_ms, 2),
            "brute_p50_query_ms": round(statistics.median(brute_latencies_ms), 4),
            "ann_p50_query_ms": round(statistics.median(ann_latencies_ms), 4),
            "recall_at_10": round(statistics.mean(recalls), 4),
        }
        results.append(row)
        print(
            f"n={n:>6}  brute_p50={row['brute_p50_query_ms']:>8.4f}ms  "
            f"ann_p50={row['ann_p50_query_ms']:>8.4f}ms  recall@10={row['recall_at_10']:.3f}"
        )

    return results


def benchmark_ingestion_throughput() -> list[dict[str, Any]]:
    rng = random.Random(7)
    results = []

    for n in INGEST_CORPUS_SIZES:
        chunks_raw = [
            (f"doc{i}", f"Article {i} covers {rng.choice(_TOPICS)} in production RAG systems. " * 3)
            for i in range(n)
        ]
        chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=30)
        bm25 = BM25Index()
        vector = VectorIndex()

        t0 = time.perf_counter()
        total_chunks = 0
        for doc_id, text in chunks_raw:
            doc_chunks = chunker.chunk(text, doc_id=doc_id)
            bm25.add(doc_chunks)
            vector.add(doc_chunks)
            total_chunks += len(doc_chunks)
        elapsed_s = time.perf_counter() - t0

        results.append(
            {
                "n": n,
                "total_chunks": total_chunks,
                "elapsed_s": round(elapsed_s, 3),
                "docs_per_sec": round(n / elapsed_s, 1),
            }
        )
        print(
            f"n={n:>6}  chunks={total_chunks:>6}  elapsed={elapsed_s:6.3f}s  "
            f"{n / elapsed_s:8.1f} docs/sec"
        )

    return results


def _render_charts(ann_results: list[dict[str, Any]]) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    latency_svg = render_line_chart(
        title="Query Latency: Brute-Force vs. HNSW",
        x_label="Corpus size (chunks, log scale)",
        y_label="p50 query latency (ms)",
        x_log=True,
        series=[
            {
                "name": "VectorIndex (brute-force)",
                "points": [(r["n"], r["brute_p50_query_ms"]) for r in ann_results],
                "color": "#dc2626",
            },
            {
                "name": "ApproxVectorIndex (HNSW)",
                "points": [(r["n"], r["ann_p50_query_ms"]) for r in ann_results],
                "color": "#2563eb",
            },
        ],
    )
    (ASSETS_DIR / "ann_latency.svg").write_text(latency_svg, encoding="utf-8")

    recall_svg = render_line_chart(
        title="HNSW Recall@10 vs. Brute-Force Ground Truth",
        x_label="Corpus size (chunks, log scale)",
        y_label="Recall@10",
        x_log=True,
        series=[
            {
                "name": "ApproxVectorIndex (HNSW)",
                "points": [(r["n"], r["recall_at_10"]) for r in ann_results],
                "color": "#059669",
            },
        ],
    )
    (ASSETS_DIR / "ann_recall.svg").write_text(recall_svg, encoding="utf-8")


def _render_markdown(
    ann_results: list[dict[str, Any]], ingestion_results: list[dict[str, Any]]
) -> str:
    import platform

    ann_rows = "\n".join(
        f"| {r['n']:,} | {r['brute_p50_query_ms']:.4f} | {r['ann_p50_query_ms']:.4f} | "
        f"{r['brute_build_ms']:.1f} | {r['ann_build_ms']:.1f} | {r['recall_at_10']:.3f} |"
        for r in ann_results
    )
    ingest_rows = "\n".join(
        f"| {r['n']:,} | {r['total_chunks']:,} | {r['elapsed_s']:.3f} | {r['docs_per_sec']:,.1f} |"
        for r in ingestion_results
    )

    crossover = next(
        (r for r in ann_results if r["ann_p50_query_ms"] < r["brute_p50_query_ms"]), None
    )
    crossover_note = (
        f"HNSW overtakes brute force in query latency starting around **{crossover['n']:,} chunks** "
        f"in this run."
        if crossover
        else "Brute force stayed faster than HNSW at every corpus size measured here -- expected "
        "at these sizes, since HNSW's O(log n) traversal has higher constant overhead per "
        "query than a tight O(n) scan until n grows large enough to dominate."
    )

    return f"""# Benchmarks

Real numbers from `ragforge`'s own code, measured with
[`scripts/generate_benchmarks.py`](../scripts/generate_benchmarks.py) -- no
external benchmarking framework, no synthetic or hand-typed figures.
Regenerate with:

```bash
python scripts/generate_benchmarks.py
```

Machine: `{platform.system()} {platform.release()}`, Python `{platform.python_version()}`,
ragforge `{__version__}`. Absolute numbers will vary by machine; the *shapes*
of the curves (crossover point, recall trend) are what matters here and are
tied to the complexity argument in [`docs/math.md`](math.md#4-hnsw-approximate-nearest-neighbor-search-srcragforgeannpy).

---

## 1. Brute-Force vs. HNSW Approximate Search

`VectorIndex` scans every chunk per query (`O(n)`); `ApproxVectorIndex`
(HNSW, see [`src/ragforge/ann.py`](../src/ragforge/ann.py)) trades exactness
for expected `O(log n)` graph traversal. {crossover_note}

Both backends use {_BENCHMARK_DIMS}-dimensional `hashing_embed` vectors
here (the library default is 128) purely so the full sweep finishes in
well under a minute on a laptop -- this repository's pure-Python, zero-
dependency `cosine_similarity` (no numpy/BLAS vectorization, see
`docs/math.md#6`) spends time roughly linear in dimensionality, so lower
dims speed up *both* curves proportionally without changing which one
wins or the recall trend.

![Query latency: brute-force vs HNSW](assets/ann_latency.svg)

![HNSW recall@10 vs brute-force ground truth](assets/ann_recall.svg)

| Corpus size | Brute-force p50 (ms) | HNSW p50 (ms) | Brute-force build (ms) | HNSW build (ms) | Recall@10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
{ann_rows}

Recall@10 is measured against the brute-force result set on the *same*
corpus and query, not a held-out ground truth -- it answers "how often does
the approximate index agree with the exact one," which is exactly the
tradeoff `ApproxVectorIndex`'s docstring warns about. HNSW build time is
higher than brute-force's simple append because it does real work per
insertion (greedy layer search + neighbor pruning, see
`HNSWIndex.add`) instead of just storing the vector.

---

## 2. Ingestion Throughput

End-to-end throughput of `RecursiveCharacterChunker` + dual BM25/vector
indexing, single-threaded, using the default `hashing_embed`.

| Corpus size (docs) | Total chunks | Elapsed (s) | Docs/sec |
| ---: | ---: | ---: | ---: |
{ingest_rows}

A real embedding provider (see
[`src/ragforge/embeddings_providers.py`](../src/ragforge/embeddings_providers.py))
will dominate this cost in practice -- network/model latency per chunk is
orders of magnitude larger than `hashing_embed`'s in-process hashing, so
production ingestion throughput is bounded by the embedding provider's
batch API limits and concurrency, not by ragforge's own chunking or
indexing code measured here.
"""


def main() -> None:
    print("--- ANN vs. brute-force ---")
    ann_results = benchmark_ann_vs_brute_force()

    print("\n--- Ingestion throughput ---")
    ingestion_results = benchmark_ingestion_throughput()

    print("\nRendering charts...")
    _render_charts(ann_results)

    print(f"Writing {DOCS_PATH}...")
    DOCS_PATH.write_text(_render_markdown(ann_results, ingestion_results), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
