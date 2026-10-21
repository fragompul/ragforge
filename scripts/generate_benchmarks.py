"""Regenerates docs/benchmarks.md, docs/assets/benchmarks.json and the charts from
real, reproducible measurements taken with ragforge's own code.

    python scripts/generate_benchmarks.py            # full run (about a minute)
    python scripts/generate_benchmarks.py --quick    # small sizes, for a smoke test

What is measured:

1. **Vector search**: exact pure-Python scan (``VectorIndex``), exact NumPy scan
   (``NumpyVectorIndex``, optional) and the from-scratch HNSW graph
   (``ApproxVectorIndex``) on clustered unit vectors -- latency and recall@10 as the
   corpus grows, plus an ``ef_search`` sweep and an ablation of the HNSW
   neighbor-selection heuristic.
2. **BM25**: the inverted-index implementation vs. the naive "re-tokenize and
   scan every chunk" scan it replaced.
3. **Ingestion**: end-to-end chunking + dual indexing throughput.

Vectors are synthetic (Gaussian clusters on the unit sphere) *on purpose*: it
isolates the index's search behavior from embedding quality. For retrieval
quality on a real dataset with real embeddings see ``scripts/eval_beir.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from svg_charts import PALETTE, render_bar_chart, render_line_chart  # noqa: E402

from ragforge._version import __version__  # noqa: E402
from ragforge.ann import HNSWIndex  # noqa: E402
from ragforge.chunking import Chunk, RecursiveCharacterChunker  # noqa: E402
from ragforge.embeddings import normalize_vector  # noqa: E402
from ragforge.index import ApproxVectorIndex, BM25Index, VectorIndex, _tokenize  # noqa: E402

try:
    from ragforge.fast import NumpyVectorIndex
except ImportError:  # pragma: no cover
    NumpyVectorIndex = None  # type: ignore[assignment,misc]

ASSETS = ROOT / "docs" / "assets"
DOCS = ROOT / "docs" / "benchmarks.md"

DIMS = 32
CLUSTERS = 20
TOP_K = 10
HNSW = {"m": 16, "ef_construction": 100}
# The ablation uses a lean graph (small M / ef_construction): the regime where edge quality matters.
HNSW_ABLATION = {"m": 8, "ef_construction": 40}
EF_DEFAULT = 50
EF_SWEEP = [4, 8, 16, 32, 64, 128]


def _numpy_available() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------- vectors


def make_vectors(n: int, rng: random.Random, centers: list[list[float]]) -> list[list[float]]:
    out = []
    for _ in range(n):
        c = rng.choice(centers)
        out.append(normalize_vector([x + rng.gauss(0, 0.15) for x in c]))
    return out


def _lookup_embed(table: dict[str, list[float]]):
    return lambda text: table[text]


def _run_ann(
    ann: ApproxVectorIndex, truth: list[set[str]], n_queries: int, ef: int
) -> tuple[float, float]:
    """Median latency (ms) and mean recall@10 of ``ann`` at beam width ``ef``."""
    ann.ef_search = ef
    latencies, recalls = [], []
    for i in range(n_queries):
        t0 = time.perf_counter()
        got = {r.chunk.id for r in ann.search(f"q{i}", k=TOP_K)}
        latencies.append((time.perf_counter() - t0) * 1000)
        recalls.append(len(truth[i] & got) / TOP_K)
    return statistics.median(latencies), statistics.mean(recalls)


def benchmark_vectors(sizes: list[int], n_queries: int, ablation_n: int) -> dict[str, Any]:
    rng = random.Random(42)
    centers = [normalize_vector([rng.gauss(0, 1) for _ in range(DIMS)]) for _ in range(CLUSTERS)]
    rows = []
    sweep: dict[int, list[dict[str, float]]] = {}

    for n in sizes:
        vectors = make_vectors(n, rng, centers)
        queries = make_vectors(n_queries, rng, centers)
        table = {f"q{i}": q for i, q in enumerate(queries)}
        chunks = [Chunk(id=f"c{i}", text=f"c{i}", doc_id=f"d{i}", position=0) for i in range(n)]

        brute = VectorIndex(embed_fn=_lookup_embed(table))
        brute.add_vectors(chunks, vectors)
        ann = ApproxVectorIndex(embed_fn=_lookup_embed(table), ef_search=EF_DEFAULT, **HNSW)
        t0 = time.perf_counter()
        ann.add_vectors(chunks, vectors)
        build_s = time.perf_counter() - t0

        brute_ms, truth = [], []
        for i in range(n_queries):
            t0 = time.perf_counter()
            res = brute.search(f"q{i}", k=TOP_K)
            brute_ms.append((time.perf_counter() - t0) * 1000)
            truth.append({r.chunk.id for r in res})

        numpy_ms: list[float] = []
        if NumpyVectorIndex is not None and _numpy_available():
            fast = NumpyVectorIndex(embed_fn=_lookup_embed(table))
            fast.add_vectors(chunks, vectors)
            for i in range(n_queries):
                t0 = time.perf_counter()
                fast.search(f"q{i}", k=TOP_K)
                numpy_ms.append((time.perf_counter() - t0) * 1000)

        p50_default, recall_default = _run_ann(ann, truth, n_queries, EF_DEFAULT)
        _, recall_200 = _run_ann(ann, truth, n_queries, 200)
        row = {
            "n": n,
            "brute_p50_ms": statistics.median(brute_ms),
            "numpy_p50_ms": statistics.median(numpy_ms) if numpy_ms else None,
            "hnsw_p50_ms": p50_default,
            "recall_ef50": recall_default,
            "recall_ef200": recall_200,
            "hnsw_build_s": build_s,
        }
        rows.append(row)
        print(
            f"  n={n:>6}  brute={row['brute_p50_ms']:8.3f}ms  "
            f"numpy={(row['numpy_p50_ms'] or 0):7.3f}ms  hnsw={p50_default:7.3f}ms  "
            f"recall@10 ef50={recall_default:.3f} ef200={recall_200:.3f}  build={build_s:.1f}s"
        )

        sweep[n] = []
        for ef in EF_SWEEP:
            p50, rec = _run_ann(ann, truth, n_queries, ef)
            sweep[n].append({"ef_search": ef, "p50_ms": p50, "recall": rec})

    # Heuristic ablation at a mid-sized corpus.
    n_ab = ablation_n
    vectors = make_vectors(n_ab, rng, centers)
    queries = make_vectors(n_queries, rng, centers)
    truth_ids = [
        set(
            sorted(
                range(n_ab), key=lambda j: -sum(a * b for a, b in zip(q, vectors[j], strict=True))
            )[:TOP_K]
        )
        for q in queries
    ]
    ablation = {}
    for label, heuristic in (("closest-M (no heuristic)", False), ("heuristic selection", True)):
        graph = HNSWIndex(heuristic=heuristic, **HNSW_ABLATION)
        for v in vectors:
            graph.add(v)
        recall = statistics.mean(
            len(truth_ids[i] & {nid for _, nid in graph.search(q, TOP_K, ef_search=EF_DEFAULT)})
            / TOP_K
            for i, q in enumerate(queries)
        )
        ablation[label] = recall
        print(f"  ablation n={n_ab} {label}: recall@10={recall:.3f}")
    return {"rows": rows, "sweep": {str(k): v for k, v in sweep.items()}, "ablation": ablation,
            "ablation_n": n_ab}  # fmt: skip


# --------------------------------------------------------------------------- bm25


def _naive_bm25_search(
    chunks: list[Chunk], tokens: list[list[str]], df: Counter[str], query: str
) -> int:
    """The original algorithm: for every query, re-count terms in every chunk."""
    n = len(chunks)
    avg = sum(len(t) for t in tokens) / n
    q_terms = _tokenize(query)
    hits = 0
    for toks in tokens:
        tf = Counter(toks)
        score = 0.0
        for term in q_terms:
            if term in tf:
                idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
                score += idf * tf[term] * 2.5 / (tf[term] + 1.5 * (0.25 + 0.75 * len(toks) / avg))
        hits += score > 0
    return hits


def benchmark_bm25(sizes: list[int], n_queries: int) -> list[dict[str, Any]]:
    rng = random.Random(7)
    vocab = [f"term{i}" for i in range(3000)]
    weights = [1 / (i + 1) for i in range(len(vocab))]  # Zipf-like
    rows = []
    for n in sizes:
        chunks = [
            Chunk(
                id=f"c{i}",
                text=" ".join(rng.choices(vocab, weights, k=60)),
                doc_id=f"d{i}",
                position=0,
            )
            for i in range(n)
        ]
        index = BM25Index()
        t0 = time.perf_counter()
        index.add(chunks)
        build_ms = (time.perf_counter() - t0) * 1000
        queries = [" ".join(rng.choices(vocab, weights, k=4)) for _ in range(n_queries)]

        inv = []
        for q in queries:
            t0 = time.perf_counter()
            index.search(q, k=10)
            inv.append((time.perf_counter() - t0) * 1000)

        tokens = [_tokenize(c.text) for c in chunks]
        df: Counter[str] = Counter()
        for toks in tokens:
            df.update(set(toks))
        naive = []
        for q in queries[: max(5, n_queries // 4)]:
            t0 = time.perf_counter()
            _naive_bm25_search(chunks, tokens, df, q)
            naive.append((time.perf_counter() - t0) * 1000)

        row = {
            "n": n,
            "inverted_p50_ms": statistics.median(inv),
            "naive_p50_ms": statistics.median(naive),
            "build_ms": build_ms,
        }
        row["speedup"] = row["naive_p50_ms"] / row["inverted_p50_ms"]
        rows.append(row)
        print(
            f"  n={n:>6}  inverted={row['inverted_p50_ms']:7.3f}ms  "
            f"naive={row['naive_p50_ms']:8.3f}ms  x{row['speedup']:.0f}"
        )
    return rows


# ------------------------------------------------------------------------ ingestion

TOPICS = ["consensus", "vector search", "reranking", "chunking", "evaluation", "latency",
          "throughput", "embeddings", "indexing", "caching"]  # fmt: skip


def benchmark_ingestion(sizes: list[int]) -> list[dict[str, Any]]:
    rng = random.Random(11)
    rows = []
    for n in sizes:
        docs = [
            (f"doc{i}", f"Article {i} covers {rng.choice(TOPICS)} in production RAG systems. " * 3)
            for i in range(n)
        ]
        chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=30)
        bm25, vector = BM25Index(), VectorIndex()
        t0 = time.perf_counter()
        total = 0
        for doc_id, text in docs:
            cs = chunker.chunk(text, doc_id=doc_id)
            bm25.add(cs)
            vector.add(cs)
            total += len(cs)
        elapsed = time.perf_counter() - t0
        rows.append({"n": n, "chunks": total, "elapsed_s": elapsed, "docs_per_s": n / elapsed})
        print(f"  n={n:>6}  chunks={total:>6}  {elapsed:6.2f}s  {n / elapsed:9.1f} docs/s")
    return rows


# ------------------------------------------------------------------------- outputs


def benchmark_filtered(n: int, n_queries: int) -> list[dict[str, Any]]:
    """Filtered HNSW search at several selectivities: old over-fetch vs. in-graph filter."""
    rng = random.Random(21)
    centers = [normalize_vector([rng.gauss(0, 1) for _ in range(DIMS)]) for _ in range(CLUSTERS)]
    vectors = make_vectors(n, rng, centers)
    queries = make_vectors(n_queries, rng, centers)
    table = {f"q{i}": q for i, q in enumerate(queries)}
    chunks = [
        Chunk(id=f"c{i}", text=f"c{i}", doc_id=f"d{i}", position=0, metadata={"g": i % 1000})
        for i in range(n)
    ]
    embed = _lookup_embed(table)
    ann = ApproxVectorIndex(embed_fn=embed, ef_search=EF_DEFAULT, **HNSW)
    exact = VectorIndex(embed_fn=embed)
    ann.add_vectors(chunks, vectors)
    exact.add_vectors(chunks, vectors)
    by_node = ann._chunks_by_node

    rows = []
    for selectivity in (0.5, 0.1, 0.02, 0.005):
        limit = int(1000 * selectivity)

        def keep(chunk: Chunk, limit: int = limit) -> bool:
            return bool(chunk.metadata["g"] < limit)

        truth = [
            {r.chunk.id for r in exact.search(f"q{i}", k=TOP_K, filter_fn=keep)}
            for i in range(n_queries)
        ]
        record: dict[str, Any] = {
            "selectivity": selectivity,
            "matching_chunks": sum(1 for c in chunks if keep(c)),
        }
        for label in ("overfetch", "in_graph"):
            recalls, counts, latencies = [], [], []
            for i in range(n_queries):
                t0 = time.perf_counter()
                if label == "overfetch":  # the previous strategy: fetch 10x, then filter
                    raw = ann._graph.search(
                        table[f"q{i}"], k=min(n, TOP_K * 10), ef_search=max(EF_DEFAULT, TOP_K * 10)
                    )
                    got = [by_node[nid].id for _, nid in raw if keep(by_node[nid])][:TOP_K]
                else:
                    got = [r.chunk.id for r in ann.search(f"q{i}", k=TOP_K, filter_fn=keep)]
                latencies.append((time.perf_counter() - t0) * 1000)
                recalls.append(len(truth[i] & set(got)) / max(len(truth[i]), 1))
                counts.append(len(got))
            record[label] = {
                "recall": statistics.mean(recalls),
                "mean_results": statistics.mean(counts),
                "p50_ms": statistics.median(latencies),
            }
        rows.append(record)
        print(
            f"  selectivity={selectivity:>6.3f} ({record['matching_chunks']:>5} chunks)  "
            f"over-fetch recall={record['overfetch']['recall']:.3f} (returns {record['overfetch']['mean_results']:.1f})  "
            f"in-graph recall={record['in_graph']['recall']:.3f} (returns {record['in_graph']['mean_results']:.1f})"
        )
    return rows


_STORAGE_PROBE = """
import json, resource, sys, time
from ragforge import RagPipeline
from ragforge.storage import DiskIndex
fmt, path = sys.argv[1], sys.argv[2]
t0 = time.perf_counter()
if fmt == "json":
    pipeline = RagPipeline.load(path)
    holder = None
else:
    holder = DiskIndex.open(path, readonly=True)
    pipeline = holder.pipeline
open_s = time.perf_counter() - t0
rss_open = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
queries = [" ".join(w) for w in json.loads(sys.argv[3])]
lat = []
for q in queries:
    t0 = time.perf_counter()
    pipeline.vector_index.search(q, k=10)
    lat.append((time.perf_counter() - t0) * 1000)
lat.sort()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
div = 1024 * 1024 if sys.platform == "darwin" else 1024
print(json.dumps({"open_s": open_s, "peak_rss_mb": rss / div, "rss_at_open_mb": rss_open / div, "p50_ms": lat[len(lat) // 2]}))
"""


def benchmark_storage(n: int, workdir: Path) -> dict[str, Any]:
    """JSON index vs. memory-mapped directory index: size, open time, peak memory."""
    import subprocess
    import tempfile

    from ragforge import RagPipeline
    from ragforge.storage import DiskIndex

    rng = random.Random(5)
    vocab = [f"term{i}" for i in range(2500)]
    weights = [1 / (i + 1) for i in range(len(vocab))]
    pipeline = RagPipeline(chunker=RecursiveCharacterChunker(chunk_size=600, chunk_overlap=0), vector_backend="hnsw",
                           ann_params={"m": 16, "ef_construction": 60, "ef_search": 40})  # fmt: skip
    t0 = time.perf_counter()
    for i in range(n):
        pipeline.ingest(f"d{i}", " ".join(rng.choices(vocab, weights, k=14)))
    build_s = time.perf_counter() - t0
    queries = [rng.choices(vocab, weights, k=6) for _ in range(150)]

    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        json_path, dir_path = Path(tmp) / "index.json", Path(tmp) / "index_dir"
        pipeline.save(json_path)
        DiskIndex.from_pipeline(pipeline, dir_path).close()
        sizes = {
            "json": json_path.stat().st_size,
            "dir": sum(f.stat().st_size for f in dir_path.iterdir() if f.is_file()),
        }
        result: dict[str, Any] = {"n": n, "build_s": build_s, "bytes": sizes}
        env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
        for fmt, path in (("json", json_path), ("dir", dir_path)):
            out = subprocess.run(
                [sys.executable, "-c", _STORAGE_PROBE, fmt, str(path), json.dumps(queries)],
                capture_output=True, text=True, env=env, check=True,
            )  # fmt: skip
            result[fmt] = json.loads(out.stdout)
            print(f"  {fmt:<5} size={sizes[fmt] / 1e6:7.1f} MB  open={result[fmt]['open_s']:.2f}s  "
                  f"peak RSS={result[fmt]['peak_rss_mb']:.0f} MB  p50={result[fmt]['p50_ms']:.2f}ms")  # fmt: skip
    return result


def render_charts(data: dict[str, Any]) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = data["vectors"]["rows"]
    sweep = data["vectors"]["sweep"]
    ablation = data["vectors"]["ablation"]
    for theme in ("light", "dark"):
        series = [
            {"name": "Exact · pure Python", "points": [(r["n"], r["brute_p50_ms"]) for r in rows],
             "color": PALETTE[5]},
            {"name": "HNSW (ef=50)", "points": [(r["n"], r["hnsw_p50_ms"]) for r in rows],
             "color": PALETTE[0]},
        ]  # fmt: skip
        if rows[0]["numpy_p50_ms"] is not None:
            series.insert(
                1,
                {"name": "Exact · NumPy", "points": [(r["n"], r["numpy_p50_ms"]) for r in rows],
                 "color": PALETTE[1]},
            )  # fmt: skip
        (ASSETS / f"ann_latency_{theme}.svg").write_text(
            render_line_chart(
                "Query latency vs. corpus size",
                "Corpus size (chunks, log scale)",
                "p50 query latency (ms)",
                series,
                x_log=True,
                y_log=True,
                subtitle=f"{DIMS}-d clustered vectors · top-{TOP_K} · log-log, lower is better",
                theme=theme,
            ),
            encoding="utf-8",
        )
        largest = str(rows[-1]["n"])
        recall_series = [
            {"name": f"n = {int(n):,}", "points": [(p["ef_search"], p["recall"]) for p in pts],
             "color": PALETTE[i % len(PALETTE)]}
            for i, (n, pts) in enumerate(sweep.items())
        ]  # fmt: skip
        (ASSETS / f"ann_recall_{theme}.svg").write_text(
            render_line_chart(
                "HNSW recall@10 vs. ef_search",
                "ef_search (beam width, log scale)",
                "recall@10 vs. exact search",
                recall_series,
                x_log=True,
                y_min=0.6,
                y_max=1.0,
                subtitle=f"M={HNSW['m']}, ef_construction={HNSW['ef_construction']} · "
                f"largest corpus: {int(largest):,}",
                theme=theme,
            ),
            encoding="utf-8",
        )
        filt = data["filtered"]
        (ASSETS / f"filtered_recall_{theme}.svg").write_text(
            render_line_chart(
                "Filtered search: recall vs. filter selectivity",
                "Share of chunks passing the filter (%, log scale)",
                "recall@10 vs. exact filtered search",
                [
                    {
                        "name": "Over-fetch then filter (before)",
                        "points": [
                            (r["selectivity"] * 100, r["overfetch"]["recall"]) for r in filt
                        ],
                        "color": PALETTE[5],
                    },
                    {
                        "name": "In-graph filter + exact fallback (now)",
                        "points": [(r["selectivity"] * 100, r["in_graph"]["recall"]) for r in filt],
                        "color": PALETTE[3],
                    },
                ],  # fmt: skip
                x_log=True,
                y_min=0.0,
                y_max=1.0,
                subtitle=f"HNSW over {data['filtered_n']:,} chunks, top-{TOP_K}",
                theme=theme,
            ),
            encoding="utf-8",
        )
        (ASSETS / f"ann_ablation_{theme}.svg").write_text(
            render_bar_chart(
                "Why neighbor selection matters",
                [
                    {"label": k, "value": v, "color": PALETTE[i * 3]}
                    for i, (k, v) in enumerate(ablation.items())
                ],
                x_label=f"recall@10 at ef_search={EF_DEFAULT}, n={data['vectors']['ablation_n']:,}",
                subtitle=f"M={HNSW_ABLATION['m']}, ef_construction={HNSW_ABLATION['ef_construction']}; only the edge-selection rule differs",
                x_max=1.0,
                theme=theme,
            ),
            encoding="utf-8",
        )


def render_markdown(data: dict[str, Any]) -> str:
    v = data["vectors"]

    def numpy_cell(r: dict[str, Any]) -> str:
        return "-" if r["numpy_p50_ms"] is None else f"{r['numpy_p50_ms']:.3f}"

    ann_rows = "\n".join(
        f"| {r['n']:,} | {r['brute_p50_ms']:.3f} | {numpy_cell(r)} | "
        f"{r['hnsw_p50_ms']:.3f} | {r['recall_ef50']:.3f} | {r['recall_ef200']:.3f} | "
        f"{r['hnsw_build_s']:.1f} |"
        for r in v["rows"]
    )
    last = str(v["rows"][-1]["n"])
    sweep_rows = "\n".join(
        f"| {p['ef_search']} | {p['p50_ms']:.3f} | {p['recall']:.3f} |" for p in v["sweep"][last]
    )
    bm25_rows = "\n".join(
        f"| {r['n']:,} | {r['naive_p50_ms']:.2f} | {r['inverted_p50_ms']:.3f} | **{r['speedup']:.0f}x** |"
        for r in data["bm25"]
    )
    ingest_rows = "\n".join(
        f"| {r['n']:,} | {r['chunks']:,} | {r['elapsed_s']:.2f} | {r['docs_per_s']:,.0f} |"
        for r in data["ingestion"]
    )
    ab = v["ablation"]
    ab_rows = "\n".join(f"| {k} | {val:.3f} |" for k, val in ab.items())
    crossover = next((r["n"] for r in v["rows"] if r["hnsw_p50_ms"] < r["brute_p50_ms"]), None)
    cross_text = (
        f"HNSW overtakes the exact pure-Python scan from about **{crossover:,} chunks**."
        if crossover
        else "At these sizes the exact scan stayed faster than HNSW."
    )
    picture = lambda name, alt: (  # noqa: E731
        f'<picture><source media="(prefers-color-scheme: dark)" srcset="assets/{name}_dark.svg">'
        f'<img alt="{alt}" src="assets/{name}_light.svg" width="680"></picture>'
    )
    filt_rows = "\n".join(
        f"| {r['selectivity'] * 100:g}% ({r['matching_chunks']:,} chunks) | {r['overfetch']['recall']:.3f} | "
        f"{r['overfetch']['mean_results']:.1f} | {r['in_graph']['recall']:.3f} | "
        f"{r['in_graph']['mean_results']:.1f} | {r['overfetch']['p50_ms']:.2f} / {r['in_graph']['p50_ms']:.2f} |"
        for r in data["filtered"]
    )
    st = data["storage"]
    mb = lambda b: f"{b / 1e6:,.1f}"  # noqa: E731
    return f"""# Benchmarks

Measured with ragforge's own code by
[`scripts/generate_benchmarks.py`](https://github.com/fragompul/ragforge/blob/main/scripts/generate_benchmarks.py) -- no external
benchmarking framework and no hand-typed figures. Regenerate with:

```bash
python scripts/generate_benchmarks.py
```

Machine: `{data["machine"]}`, Python `{data["python"]}`, ragforge `{__version__}`.
Absolute numbers vary by machine; the *shapes* (crossover, recall curve, speedups)
are what matter. Quality on real data with real embeddings lives in
[`evaluation.md`](evaluation.md).

---

## 1. Vector search: exact vs. HNSW

Clustered unit vectors ({DIMS}-d, {CLUSTERS} Gaussian clusters), {data["queries"]} queries per
size, top-{TOP_K}. Recall is measured against the exact result set on the same corpus.
{cross_text}

{picture("ann_latency", "Query latency vs corpus size")}

| Corpus | Exact pure-Python p50 (ms) | Exact NumPy p50 (ms) | HNSW p50 (ms), ef=50 | Recall@10, ef=50 | Recall@10, ef=200 | HNSW build (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{ann_rows}

HNSW build is the price of sub-linear queries: each insertion runs a graph search.
The pure-Python implementation is built for clarity and zero dependencies; the NumPy
backend shows how far a vectorized exact scan goes before an ANN index is needed.

### `ef_search` is a real recall dial

{picture("ann_recall", "HNSW recall vs ef_search")}

At the largest corpus ({int(last):,} chunks):

| `ef_search` | p50 query (ms) | Recall@10 |
| ---: | ---: | ---: |
{sweep_rows}

Widening the beam trades latency for recall along a smooth curve, as the HNSW
paper predicts. (An earlier version of this repository showed recall *plateauing*
near 0.55 regardless of `ef_search`; that was a defect of the implementation, not of
HNSW -- see the next section.)

### Ablation: the neighbor-selection heuristic

The first HNSW implementation linked each new node to its `M` *closest* candidates.
On clustered data that wires a node to a clump of mutually redundant neighbors, the
graph fragments into islands, and recall collapses as the corpus grows. The paper's
heuristic (Algorithm 4) keeps a candidate only if it is closer to the new node than
to every neighbor already chosen, which preserves edges that leave the cluster.

{picture("ann_ablation", "Neighbor selection ablation")}

| Edge selection | Recall@10 (ef_search={EF_DEFAULT}, n={v["ablation_n"]:,}) |
| :--- | ---: |
{ab_rows}

---

## 2. BM25: inverted index vs. naive scan

`BM25Index` keeps postings (`term -> {{chunk: tf}}`), so a query only touches chunks
that contain a query term. The previous implementation re-tokenized counts for every
chunk on every query. Zipf-distributed vocabulary, 60-token chunks, 4-term queries:

| Chunks | Naive scan p50 (ms) | Inverted index p50 (ms) | Speedup |
| ---: | ---: | ---: | ---: |
{bm25_rows}

---

## 3. Filtered vector search

Metadata filters (tenant, category, date) are everywhere in production, and graph indexes are
notoriously bad at them: the graph does not know which nodes pass the filter. The earlier
implementation fetched `10 x k` neighbors and filtered afterwards, which **returns too few -- or no --
results when the filter is selective**. Filters now run *inside* the traversal (every node
helps navigation, only passing nodes can be results, the beam widens until it holds `ef_search`
accepted candidates) with a visit cap, and an **exact scan over the passing chunks** completes the
answer when the cap is hit. {data["filtered_n"]:,} chunks, top-{TOP_K}; recall against exact filtered search:

{picture("filtered_recall", "Filtered search recall vs selectivity")}

| Filter passes | Over-fetch recall | Returns (of {TOP_K}) | In-graph recall | Returns (of {TOP_K}) | p50 ms (before / now) |
| :--- | ---: | ---: | ---: | ---: | ---: |
{filt_rows}

---

## 4. Persistence: JSON vs. memory-mapped directory

`RagPipeline.save` writes one JSON file; `DiskIndex` writes a directory with float32
vectors in a memory-mapped file, the HNSW graph in compact binary adjacency arrays, chunks as
JSON lines, and a manifest as the atomic commit record. Each format is opened in a **fresh Python
process** (so peak RSS is honest), then 150 queries are run. {st["n"]:,} chunks, HNSW backend, 128-d vectors.

| | JSON | Directory (mmap) |
| :--- | ---: | ---: |
| Size on disk (MB) | {mb(st["bytes"]["json"])} | {mb(st["bytes"]["dir"])} |
| Open time (s) | {st["json"]["open_s"]:.2f} | {st["dir"]["open_s"]:.2f} |
| Peak memory (MB) | {st["json"]["peak_rss_mb"]:,.0f} | {st["dir"]["peak_rss_mb"]:,.0f} |
| Vector query p50 (ms) | {st["json"]["p50_ms"]:.2f} | {st["dir"]["p50_ms"]:.2f} |

The directory format also supports **incremental ingest, deletion and compaction** without rewriting the
index, and survives a crash mid-append (an interrupted tail is ignored and overwritten).
BM25 postings are rebuilt from `chunks.jsonl` at open, which is a linear pass over the text.

---

## 5. Ingestion throughput

Chunking (`RecursiveCharacterChunker`) plus BM25 and vector indexing, single-threaded,
using the default `hashing_embed`.

| Documents | Chunks | Elapsed (s) | Docs / s |
| ---: | ---: | ---: | ---: |
{ingest_rows}

With a real embedding provider the model call dominates; production ingestion is
bounded by the provider's batch limits, not by ragforge's chunking or indexing.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--quick", action="store_true", help="small sizes for a smoke test")
    args = parser.parse_args()

    vec_sizes = [500, 1500] if args.quick else [1000, 5000, 20000]
    queries = 20 if args.quick else 60
    bm25_sizes = [500, 2000] if args.quick else [1000, 5000, 20000]
    ingest_sizes = [200, 500] if args.quick else [1000, 4000, 12000]

    print("--- vector search ---")
    vectors = benchmark_vectors(vec_sizes, queries, ablation_n=1500 if args.quick else 4000)
    print("--- bm25 ---")
    bm25 = benchmark_bm25(bm25_sizes, 40)
    print("--- ingestion ---")
    ingestion = benchmark_ingestion(ingest_sizes)
    print("--- filtered search ---")
    filtered_n = 1500 if args.quick else 5000
    filtered = benchmark_filtered(filtered_n, 15 if args.quick else 40)
    print("--- storage ---")
    storage = benchmark_storage(2000 if args.quick else 20000, ASSETS)

    data = {
        "ragforge_version": __version__,
        "machine": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "queries": queries,
        "vectors": vectors,
        "bm25": bm25,
        "ingestion": ingestion,
        "filtered": filtered,
        "filtered_n": filtered_n,
        "storage": storage,
    }
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "benchmarks.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    render_charts(data)
    DOCS.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {DOCS} and charts in {ASSETS}")


if __name__ == "__main__":
    main()
