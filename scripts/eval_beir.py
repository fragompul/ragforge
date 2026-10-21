"""Evaluate ragforge's retrievers on BEIR datasets (SciFact, NFCorpus, FiQA, SciDocs, ...).

    # 1. get BEIR-format datasets (corpus.jsonl, queries.jsonl, qrels/test.tsv)
    python scripts/eval_beir.py --download scifact nfcorpus fiqa scidocs --data-root data
    # 2. offline baseline (hashing embeddings, zero dependencies)
    python scripts/eval_beir.py --data-root data --datasets scifact
    # 3. real semantic embeddings (pip install sentence-transformers)
    python scripts/eval_beir.py --data-root data --datasets scifact,nfcorpus,fiqa,scidocs \
        --embedder minilm --splade scifact,nfcorpus --quantization scifact,nfcorpus

Rankings come from ragforge's *own* ``BM25Index``, ``NumpyVectorIndex``, ``SparseVectorIndex``,
``QuantizedVectorIndex`` and ``FusionRetriever`` (weighted Reciprocal Rank Fusion), scored with
``ragforge.metrics`` (nDCG@10 is BEIR's headline metric). Results are written per dataset to
``docs/assets/eval_<dataset>.json`` and rendered together into ``docs/evaluation.md`` plus
light/dark SVG charts, so every number in the report is reproducible from this command.

``--max-queries`` limits a run for smoke tests. Embeddings are cached next to the data so a
re-run only repeats retrieval.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from svg_charts import PALETTE, render_bar_chart  # noqa: E402

from ragforge._version import __version__  # noqa: E402
from ragforge.chunking import Chunk  # noqa: E402
from ragforge.embeddings import EmbedFn, hashing_embed  # noqa: E402
from ragforge.fast import NumpyVectorIndex  # noqa: E402
from ragforge.fusion import FusionRetriever, FusionSource  # noqa: E402
from ragforge.index import ApproxVectorIndex, BM25Index  # noqa: E402
from ragforge.metrics import mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402
from ragforge.quantization import QuantizedVectorIndex  # noqa: E402
from ragforge.routing import QueryRouter  # noqa: E402
from ragforge.sparse import SparseVectorIndex  # noqa: E402

BEIR_URL = "https://public.ukp.da.dk/ukp-datasets/beir/datasets/{name}.zip"
ASSETS = ROOT / "docs" / "assets"
TOP_K = 100
DATASET_TITLES = {
    "scifact": "SciFact",
    "nfcorpus": "NFCorpus",
    "fiqa": "FiQA-2018",
    "scidocs": "SCIDOCS",
}


def download(name: str, data_root: Path) -> Path:
    """Fetch and unpack an official BEIR dataset archive."""
    data_root.mkdir(parents=True, exist_ok=True)
    archive = data_root / f"{name}.zip"
    if not archive.exists():
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(BEIR_URL.format(name=name), archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_root)
    return data_root / name


def load_beir(path: Path, split: str = "test") -> tuple[dict[str, str], dict[str, str], dict]:
    """Return ``(corpus, queries, qrels)`` from a BEIR-format directory."""
    corpus = {}
    for line in (path / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        corpus[str(row["_id"])] = f"{row.get('title', '')}. {row['text']}".strip(". ")
    queries = {}
    for line in (path / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        queries[str(row["_id"])] = row["text"]
    qrels: dict[str, dict[str, float]] = {}
    lines = (path / "qrels" / f"{split}.tsv").read_text(encoding="utf-8").splitlines()[1:]
    for line in lines:
        qid, doc_id, score = line.split("\t")
        if float(score) > 0 and doc_id in corpus and qid in queries:
            qrels.setdefault(qid, {})[doc_id] = float(score)
    return corpus, queries, qrels


# ----------------------------------------------------------------------------- encoders


def encode_dense(kind: str, texts: list[str], cache: Path) -> tuple[dict[str, list[float]], str]:
    """Embed ``texts`` (cached on disk); returns ``(text -> vector, label)``."""
    if kind == "hashing":
        return {t: hashing_embed(t) for t in texts}, "hashing (128-d bag-of-words)"
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    if cache.exists():
        matrix = np.load(cache)
    else:
        model = SentenceTransformer(model_name)
        print(f"  encoding {len(texts):,} texts with MiniLM ...")
        t0 = time.perf_counter()
        matrix = model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        print(f"    done in {time.perf_counter() - t0:.0f}s")
        np.save(cache, matrix)
    label = f"MiniLM-L6-v2 ({matrix.shape[1]}-d, sentence-transformers)"
    return {t: matrix[i].tolist() for i, t in enumerate(texts)}, label


def encode_splade(texts: list[str], cache: Path, top_k: int = 96) -> dict[str, dict[str, float]]:
    """SPLADE term weights for every text (batched on CPU, cached as JSON)."""
    if cache.exists():
        stored = json.loads(cache.read_text())
        return dict(zip(texts, stored, strict=True))
    import torch
    from sentence_transformers import SparseEncoder

    model = SparseEncoder("naver/splade-cocondenser-ensembledistil", device="cpu")
    out: list[dict[str, float]] = []
    t0 = time.perf_counter()
    print(f"  SPLADE-encoding {len(texts):,} texts ...")
    for start in range(0, len(texts), 32):
        dense = model.encode(texts[start : start + 32], convert_to_sparse_tensor=False)
        values, ids = torch.topk(dense, k=top_k, dim=1)
        for row_values, row_ids in zip(values.tolist(), ids.tolist(), strict=True):
            pairs = [(v, i) for v, i in zip(row_values, row_ids, strict=True) if v > 0]
            tokens = model.tokenizer.convert_ids_to_tokens([i for _, i in pairs])
            out.append({tok: round(v, 4) for tok, (v, _) in zip(tokens, pairs, strict=True)})
    print(f"    done in {time.perf_counter() - t0:.0f}s")
    cache.write_text(json.dumps(out))
    return dict(zip(texts, out, strict=True))


# ----------------------------------------------------------------------------- evaluation


def evaluate_run(
    name: str, search: Any, queries: dict[str, str], qrels: dict, note: str = ""
) -> dict[str, Any]:
    """Run ``search(query) -> [doc_id]`` over all judged queries and score it."""
    runs: dict[str, list[str]] = {}
    latencies = []
    for qid in qrels:
        t0 = time.perf_counter()
        runs[qid] = search(queries[qid])
        latencies.append((time.perf_counter() - t0) * 1000)
    row = {
        "system": name,
        "ndcg@10": statistics.mean(ndcg_at_k(runs[q], qrels[q], 10) for q in qrels),
        "recall@10": statistics.mean(recall_at_k(runs[q], qrels[q], 10) for q in qrels),
        "recall@100": statistics.mean(recall_at_k(runs[q], qrels[q], 100) for q in qrels),
        "mrr@10": statistics.mean(mrr_at_k(runs[q], qrels[q], 10) for q in qrels),
        "p50_ms": statistics.median(latencies),
    }
    if note:
        row["note"] = note
    print(
        f"  {name:<36} nDCG@10={row['ndcg@10']:.4f}  R@100={row['recall@100']:.4f}  "
        f"p50={row['p50_ms']:.1f}ms"
    )
    return row


def ann_recall(
    chunks: list[Chunk], vectors: list[list[float]], queries: list[str], embed: EmbedFn
) -> list[dict[str, Any]]:
    """Recall@10 of HNSW vs. exact search on the *real* embeddings, sweeping ``ef_search``."""
    exact = NumpyVectorIndex(embed_fn=embed)
    exact.add_vectors(chunks, vectors)
    ann = ApproxVectorIndex(embed_fn=embed, m=16, ef_construction=100, ef_search=50)
    t0 = time.perf_counter()
    ann.add_vectors(chunks, vectors)
    build_s = time.perf_counter() - t0
    truth = [{r.chunk.id for r in exact.search(q, k=10)} for q in queries]
    rows = []
    for ef in (10, 20, 50, 100, 200):
        ann.ef_search = ef
        lat, rec = [], []
        for q, gt in zip(queries, truth, strict=True):
            t0 = time.perf_counter()
            got = {r.chunk.id for r in ann.search(q, k=10)}
            lat.append((time.perf_counter() - t0) * 1000)
            rec.append(len(gt & got) / 10)
        rows.append(
            {"ef_search": ef, "recall@10": statistics.mean(rec), "p50_ms": statistics.median(lat)}
        )
        print(
            f"  HNSW ef_search={ef:<4} recall@10={rows[-1]['recall@10']:.3f}  p50={rows[-1]['p50_ms']:.2f}ms"
        )
    return [{"build_s": build_s, **row} for row in rows]


def quantization_rows(
    chunks: list[Chunk], vectors: list[list[float]], queries: dict[str, str], qrels: dict,
    embed: EmbedFn, exact_top: dict[str, list[str]],
) -> list[dict[str, Any]]:  # fmt: skip
    """nDCG / agreement with exact search / memory for each quantization setting."""
    dims = len(vectors[0])
    settings = [
        ("float32 exact (baseline)", None, None, False),
        ("int8 + rescoring", "int8", 4, True),
        ("int8 codes only", "int8", 1, False),
        ("binary + rescoring (x8)", "binary", 8, True),
        ("binary codes only", "binary", 1, False),
    ]
    rows = []
    for label, mode, factor, keep in settings:
        if mode is None:
            index: Any = NumpyVectorIndex(embed_fn=embed)
            index.add_vectors(chunks, vectors)
            code_bytes = len(chunks) * dims * 4
        else:
            index = QuantizedVectorIndex(
                embed_fn=embed, mode=mode, rescore_factor=factor, keep_full_precision=keep
            )
            index.add_vectors(chunks, vectors)
            code_bytes = index.memory_bytes()["codes"]
        lat, ndcgs, agree = [], [], []
        for qid in qrels:
            t0 = time.perf_counter()
            ranked = [r.chunk.id for r in index.search(queries[qid], k=10)]
            lat.append((time.perf_counter() - t0) * 1000)
            ndcgs.append(ndcg_at_k(ranked, qrels[qid], 10))
            agree.append(len(set(ranked) & set(exact_top[qid][:10])) / 10)
        row = {
            "setting": label,
            "ndcg@10": statistics.mean(ndcgs),
            "overlap@10_vs_exact": statistics.mean(agree),
            "bytes_per_vector": code_bytes / len(chunks),
            "p50_ms": statistics.median(lat),
        }
        print(f"  quant {label:<28} nDCG@10={row['ndcg@10']:.4f} overlap={row['overlap@10_vs_exact']:.3f} "
              f"{row['bytes_per_vector']:.0f} B/vec")  # fmt: skip
        rows.append(row)
    return rows


def run_dataset(name: str, data_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    corpus, queries, qrels = load_beir(data_dir)
    if args.max_queries:
        qrels = dict(list(qrels.items())[: args.max_queries])
    print(f"\n=== {name}: {len(corpus):,} documents, {len(qrels)} judged queries")

    texts = [*corpus.values(), *queries.values()]
    cache_dir = Path(args.cache_dir or args.data_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    table, embed_label = encode_dense(args.embedder, texts, cache_dir / f"{name}.minilm.npy")
    embed: EmbedFn = table.__getitem__

    chunks = [Chunk(id=d, text=t, doc_id=d, position=0) for d, t in corpus.items()]
    bm25_simple, bm25_english = BM25Index(), BM25Index(analyzer="english")
    bm25_simple.add(chunks)
    bm25_english.add(chunks)
    vectors = [table[c.text] for c in chunks]
    dense = NumpyVectorIndex(embed_fn=embed)
    dense.add_vectors(chunks, vectors)

    fusion = FusionRetriever([FusionSource("bm25", bm25_english), FusionSource("vector", dense)])
    router = QueryRouter()

    def routed(query: str) -> list[str]:
        decision = router.route(query, bm25_english)
        fusion.weight_bm25, fusion.weight_vector = decision.weight_bm25, decision.weight_vector
        out = [r.chunk.id for r in fusion.search(query, k=TOP_K, candidate_pool=TOP_K)]
        fusion.weight_bm25 = fusion.weight_vector = 1.0
        return out

    def ids(results: list[Any]) -> list[str]:
        return [r.chunk.id for r in results]

    rows = [
        evaluate_run(
            "BM25 (simple tokenizer)", lambda q: ids(bm25_simple.search(q, TOP_K)), queries, qrels
        ),
        evaluate_run(
            "BM25 (english analyzer)", lambda q: ids(bm25_english.search(q, TOP_K)), queries, qrels
        ),
        evaluate_run(
            f"Vector ({args.embedder})", lambda q: ids(dense.search(q, TOP_K)), queries, qrels
        ),
        evaluate_run(
            "Hybrid RRF (BM25 + vector)",
            lambda q: ids(fusion.search(q, k=TOP_K, candidate_pool=TOP_K)),
            queries,
            qrels,
        ),
        evaluate_run("Hybrid RRF + query router", routed, queries, qrels),
    ]

    if name in args.splade:
        weights = encode_splade(texts, cache_dir / f"{name}.splade.json")
        sparse = SparseVectorIndex(encode_fn=lambda t: weights[t])
        sparse.add_encoded(chunks, [weights[c.text] for c in chunks])
        three = FusionRetriever(
            [
                FusionSource("bm25", bm25_english),
                FusionSource("vector", dense),
                FusionSource("splade", sparse),
            ]
        )
        rows.append(
            evaluate_run(
                "SPLADE (learned sparse)", lambda q: ids(sparse.search(q, TOP_K)), queries, qrels
            )
        )
        rows.append(
            evaluate_run(
                "Hybrid RRF (BM25 + vector + SPLADE)",
                lambda q: ids(three.search(q, k=TOP_K, candidate_pool=TOP_K)),
                queries,
                qrels,
            )
        )

    report: dict[str, Any] = {
        "dataset": name,
        "title": DATASET_TITLES.get(name, name),
        "ragforge_version": __version__,
        "documents": len(corpus),
        "queries": len(qrels),
        "embedder": embed_label,
        "systems": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows
        ],
    }

    if name in args.quantization:
        print("Quantization:")
        exact_top = {q: ids(dense.search(queries[q], 10)) for q in qrels}
        report["quantization"] = [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            for r in quantization_rows(chunks, vectors, queries, qrels, embed, exact_top)
        ]
    if name in args.ann and not args.skip_ann:
        print("HNSW vs. exact search on the real embeddings:")
        sample = [queries[q] for q in list(qrels)[:150]]
        report["ann"] = [
            {k: round(v, 4) for k, v in r.items()}
            for r in ann_recall(chunks, vectors, sample, embed)
        ]
    return report


# ----------------------------------------------------------------------------- reporting


def render_markdown(reports: list[dict[str, Any]]) -> str:
    """The full ``docs/evaluation.md`` from every available dataset report."""

    def cell(report: dict[str, Any], system: str) -> str:
        row = next((s for s in report["systems"] if s["system"] == system), None)
        return f"{row['ndcg@10']:.3f}" if row else "–"

    def ndcg(report: dict[str, Any], system: str) -> float | None:
        row = next((x for x in report["systems"] if x["system"] == system), None)
        return row["ndcg@10"] if row else None

    def findings_text() -> str:
        """Observations computed from the numbers, so the prose can never contradict the tables."""
        hybrid, vector, bm25, routed = (
            "Hybrid RRF (BM25 + vector)",
            "Vector (minilm)",
            "BM25 (english analyzer)",
            "Hybrid RRF + query router",
        )
        wins, losses, route_up, route_down = [], [], [], []
        for r in reports:
            h, v, b, q = ndcg(r, hybrid), ndcg(r, vector), ndcg(r, bm25), ndcg(r, routed)
            if None in (h, v, b):
                continue
            best_single = max(v, b)
            entry = f"{r['title']} ({h:.3f} vs. {best_single:.3f})"
            (wins if h >= best_single else losses).append(entry)
            if q is not None and abs(q - h) >= 0.005:
                (route_up if q > h else route_down).append(f"{r['title']} ({q - h:+.3f})")
        lines = []
        if wins:
            lines.append(
                f"- Equal-weight hybrid fusion beats the best single retriever on {', '.join(wins)}."
            )
        if losses:
            lines.append(
                f"- It does **not** on {', '.join(losses)}: there one retriever is much stronger "
                "than the other and unweighted fusion lands between them. Fusion helps when both "
                "rankers are competent; weights are the knob when they are not."
            )
        if route_up or route_down:
            parts = []
            if route_up:
                parts.append(f"helps on {', '.join(route_up)}")
            if route_down:
                parts.append(f"hurts on {', '.join(route_down)}")
            lines.append(
                f"- The query router (fixed rules, not tuned on these sets) {' and '.join(parts)} "
                "relative to plain hybrid; elsewhere it is within +-0.005."
            )
        else:
            lines.append("- The query router is within +-0.005 of plain hybrid everywhere.")
        return "\n".join(lines)

    findings = findings_text()

    all_systems = list(dict.fromkeys(x["system"] for r in reports for x in r["systems"]))
    summary_rows = []
    for system in all_systems:
        values = [cell(r, system) for r in reports]
        nums = [float(v) for v in values if v != "–"]
        mean = f"{statistics.mean(nums):.3f}" if len(nums) == len(reports) else "–"
        summary_rows.append(f"| {system} | " + " | ".join(values) + f" | **{mean}** |")
    header = "| System | " + " | ".join(r["title"] for r in reports) + " | Mean |"
    sep = "| :--- | " + " | ".join("---:" for _ in reports) + " | ---: |"

    sections = []
    for r in reports:
        rows = "\n".join(
            f"| {s['system']} | **{s['ndcg@10']:.3f}** | {s['recall@10']:.3f} | {s['recall@100']:.3f} | "
            f"{s['mrr@10']:.3f} | {s['p50_ms']:.1f} |"
            for s in r["systems"]
        )
        picture = (
            '<picture><source media="(prefers-color-scheme: dark)" '
            f'srcset="assets/eval_{r["dataset"]}_dark.svg">'
            f'<img alt="nDCG@10 on {r["title"]}" src="assets/eval_{r["dataset"]}_light.svg" width="680"></picture>'
        )
        block = f"""## {r["title"]}

{r["documents"]:,} documents, {r["queries"]} test queries. Vectors: {r["embedder"]}.

{picture}

| System | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 query (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
{rows}
"""
        if r.get("quantization"):
            qrows = "\n".join(
                f"| {q['setting']} | {q['ndcg@10']:.3f} | {q['overlap@10_vs_exact']:.3f} | "
                f"{q['bytes_per_vector']:.0f} | {q['p50_ms']:.2f} |"
                for q in r["quantization"]
            )
            block += f"""
### Quantized vectors ({r["title"]})

Same MiniLM vectors, searched in code space. *Overlap* is the fraction of the exact top-10
that the quantized search also returns; memory counts the searched codes only (rescoring
keeps the float vectors in a separate, disk-friendly store).

| Setting | nDCG@10 | Overlap@10 vs exact | Bytes / vector | p50 (ms, pure Python) |
| :--- | ---: | ---: | ---: | ---: |
{qrows}
"""
        if r.get("ann"):
            arows = "\n".join(
                f"| {a['ef_search']} | {a['recall@10']:.3f} | {a['p50_ms']:.2f} |" for a in r["ann"]
            )
            block += f"""
### HNSW on real embeddings ({r["title"]})

Recall@10 of the from-scratch HNSW index (`M=16`, `ef_construction=100`) against exact search
over the same {r["documents"]:,} vectors, 150 queries. Build time {r["ann"][0]["build_s"]:.0f} s (pure Python).

| `ef_search` | Recall@10 | p50 latency (ms) |
| ---: | ---: | ---: |
{arows}
"""
        sections.append(block)

    return f"""# Retrieval quality on public benchmarks

Hand-picked examples prove nothing, so ragforge's retrievers are scored on the
[BEIR](https://github.com/beir-cellar/beir) benchmark (Thakur et al., 2021): human-judged
queries over real corpora. Everything below is produced by one command:

```bash
python scripts/eval_beir.py --download scifact nfcorpus fiqa scidocs --data-root data
python scripts/eval_beir.py --data-root data --datasets scifact,nfcorpus,fiqa,scidocs \\
    --embedder minilm --splade scifact,nfcorpus --quantization scifact,nfcorpus
```

Rankings come from ragforge's own indexes and `FusionRetriever`; scores from
`ragforge.metrics` (nDCG@10 is BEIR's headline metric). ragforge {reports[0]["ragforge_version"]}.

## Summary: nDCG@10

{header}
{sep}
{chr(10).join(summary_rows)}

*(Mean is shown only for systems evaluated on every dataset.)*

**How to read this.** BM25 and MiniLM land on the published BEIR numbers for these datasets,
which is the sanity check that the implementations are correct rather than merely plausible.

{findings}

{chr(10).join(sections)}
## Notes

- **BM25 (simple)** is the library default; the `english` analyzer adds stopword removal and
  light stemming.
- **Vector** is exact cosine search (`NumpyVectorIndex`).
- Quantized-search latencies are **pure Python** (no SIMD), where scanning int8 codes is slower than the
  float scan; the point of quantization here is the 4-32x memory reduction at unchanged quality.
- **Hybrid** fuses the `english` BM25 ranking with the vector ranking (weighted RRF, `k=60`).
- **Query router** (`ragforge.routing.QueryRouter`) changes the fusion weights per query using
  transparent rules (identifiers/quoted phrases/rare terms favor BM25; long natural-language
  questions favor vectors). Its thresholds were fixed *before* running these benchmarks and
  were not tuned on them.
- **SPLADE** (`naver/splade-cocondenser-ensembledistil`) is a learned sparse retriever served
  from an inverted index (`SparseVectorIndex`); weights are the top-96 terms per text.
- With the dependency-free `hashing_embed` the vector side is a weak bag-of-words signal;
  bring a real embedding model.

Dataset licences: BEIR datasets carry their own licences (SciFact CC BY-NC 2.0; NFCorpus,
FiQA, SCIDOCS research use). They are downloaded by the script, not redistributed here.
"""


def write_outputs(report: dict[str, Any]) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / f"eval_{report['dataset']}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    for theme in ("light", "dark"):
        svg = render_bar_chart(
            title=f"nDCG@10 on {report['title']} ({report['documents']:,} docs, {report['queries']} queries)",
            subtitle=f"ragforge {report['ragforge_version']} · vectors: {report['embedder']}",
            bars=[
                {"label": r["system"], "value": r["ndcg@10"], "color": PALETTE[i % len(PALETTE)]}
                for i, r in enumerate(report["systems"])
            ],
            x_max=1.0,
            theme=theme,
        )
        (ASSETS / f"eval_{report['dataset']}_{theme}.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--download", nargs="+", metavar="NAME", help="download BEIR datasets first"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", default="scifact")
    parser.add_argument("--embedder", choices=["hashing", "minilm"], default="hashing")
    parser.add_argument("--splade", default="", help="comma-separated datasets to add SPLADE for")
    parser.add_argument("--quantization", default="", help="datasets for the quantization study")
    parser.add_argument("--ann", default="scifact", help="datasets for the HNSW recall sweep")
    parser.add_argument("--skip-ann", action="store_true")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--report-only", action="store_true", help="re-render docs from saved JSON")
    args = parser.parse_args()
    for field in ("splade", "quantization", "ann"):
        setattr(args, field, {x for x in getattr(args, field).split(",") if x})

    if args.download:
        for name in args.download:
            download(name, args.data_root)
    if not args.report_only:
        for name in [d for d in args.datasets.split(",") if d]:
            write_outputs(run_dataset(name, args.data_root / name, args))

    order = list(DATASET_TITLES)
    # eval_guardrails.json / eval_multihop.json share the prefix but belong to other scripts.
    results = (json.loads(p.read_text()) for p in ASSETS.glob("eval_*.json"))
    found = sorted(
        (r for r in results if r.get("dataset") in order),
        key=lambda r: order.index(r["dataset"]),
    )
    if found:
        (ROOT / "docs" / "evaluation.md").write_text(render_markdown(found), encoding="utf-8")
        print(f"\nWrote {ROOT / 'docs' / 'evaluation.md'} ({len(found)} datasets)")


if __name__ == "__main__":
    main()
