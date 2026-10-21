"""Does contextual retrieval help? Chunk-level retrieval with and without document context.

    python scripts/eval_contextual.py --data-root data --datasets scifact,nfcorpus --embedder minilm

Long documents are retrieved as *chunks*. A chunk often lacks the words that identify its
document ("the compound", "this protein", "the trial"), so neither BM25 nor an embedder can
tell what it is about. *Contextual retrieval* prepends a short situating context -- here the
document **title**, the cheapest deterministic context -- to the text that is indexed
(``Chunk.index_text``) while answers still use the original text.

Each BEIR document body is split into ~300-character chunks; chunk rankings are collapsed to
document rankings (first appearance), scored with nDCG@10 / Recall@100 against the human
judgments. Compared on identical chunks: **no context** vs **title context** for BM25,
vector, and hybrid RRF. Results: ``docs/contextual.md``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from svg_charts import render_bar_chart  # noqa: E402

from ragforge._version import __version__  # noqa: E402
from ragforge.chunking import RecursiveCharacterChunker  # noqa: E402
from ragforge.contextual import title_context_fn  # noqa: E402
from ragforge.fast import NumpyVectorIndex  # noqa: E402
from ragforge.fusion import FusionRetriever, FusionSource  # noqa: E402
from ragforge.index import BM25Index  # noqa: E402
from ragforge.metrics import ndcg_at_k, recall_at_k  # noqa: E402

ASSETS = ROOT / "docs" / "assets"
TITLES = {"scifact": "SciFact", "nfcorpus": "NFCorpus"}
CHUNK = 300


def load(path: Path) -> tuple[list[dict[str, str]], dict[str, str], dict[str, dict[str, float]]]:
    docs = [
        json.loads(line)
        for line in (path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    queries = {
        j["_id"]: j["text"]
        for j in map(json.loads, (path / "queries.jsonl").read_text(encoding="utf-8").splitlines())
    }
    qrels: dict[str, dict[str, float]] = {}
    for line in (path / "qrels" / "test.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        qid, did, score = line.split("\t")
        if float(score) > 0 and qid in queries:
            qrels.setdefault(qid, {})[did] = float(score)
    return docs, queries, qrels


def encode(texts: list[str], cache: Path, kind: str):
    if kind == "hashing":
        from ragforge.embeddings import hashing_embed

        return {t: hashing_embed(t) for t in texts}, "hashing"
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    if cache.exists():
        matrix = np.load(cache)
    else:
        matrix = model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        np.save(cache, matrix)
    return {t: matrix[i].tolist() for i, t in enumerate(texts)}, "MiniLM-L6-v2"


def doc_ranking(chunk_ids: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for cid in chunk_ids:
        seen.setdefault(cid.split("::")[0], None)
    return list(seen)


def run_variant(
    docs, queries, qrels, contextual: bool, embed_kind: str, cache_dir: Path, name: str
):
    chunker = RecursiveCharacterChunker(chunk_size=CHUNK, chunk_overlap=0)
    context_fn = title_context_fn()
    chunks = []
    for d in docs:
        title = d.get("title", "") or d["_id"]
        body = d["text"]
        for chunk in chunker.chunk(body, doc_id=d["_id"], metadata={"title": title}):
            if contextual:
                chunk.context = context_fn(body, chunk)
            chunks.append(chunk)
    texts = list(dict.fromkeys([c.index_text for c in chunks] + [queries[q] for q in qrels]))
    table, label = encode(
        texts, cache_dir / f"{name}.{'ctx' if contextual else 'plain'}.npy", embed_kind
    )
    bm25 = BM25Index(analyzer="english")
    bm25.add(chunks)
    dense = NumpyVectorIndex(embed_fn=table.__getitem__)
    dense.add_vectors(chunks, [table[c.index_text] for c in chunks])
    fusion = FusionRetriever([FusionSource("bm25", bm25), FusionSource("vector", dense)])
    systems = {
        "BM25": lambda q: [r.chunk.id for r in bm25.search(q, 300)],
        "Vector": lambda q: [r.chunk.id for r in dense.search(q, 300)],
        "Hybrid RRF": lambda q: [r.chunk.id for r in fusion.search(q, k=300, candidate_pool=300)],
    }
    out = {}
    for system, search in systems.items():
        runs = {q: doc_ranking(search(queries[q])) for q in qrels}
        out[system] = {
            "ndcg@10": statistics.mean(ndcg_at_k(runs[q], qrels[q], 10) for q in qrels),
            "recall@100": statistics.mean(recall_at_k(runs[q], qrels[q], 100) for q in qrels),
        }
    return out, len(chunks), label


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", default="scifact,nfcorpus")
    parser.add_argument("--embedder", choices=["hashing", "minilm"], default="hashing")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir or args.data_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"ragforge_version": __version__, "chunk_chars": CHUNK, "datasets": []}
    for name in args.datasets.split(","):
        docs, queries, qrels = load(args.data_root / name)
        print(f"\n=== {name}: {len(docs):,} docs, {len(qrels)} queries (chunks of ~{CHUNK} chars)")
        plain, n_chunks, label = run_variant(
            docs, queries, qrels, False, args.embedder, cache_dir, name
        )
        ctx, _, _ = run_variant(docs, queries, qrels, True, args.embedder, cache_dir, name)
        for system in plain:
            print(
                f"  {system:<12} plain {plain[system]['ndcg@10']:.4f} -> title context {ctx[system]['ndcg@10']:.4f}"
            )
        report["datasets"].append(
            {
                "dataset": name,
                "title": TITLES.get(name, name),
                "documents": len(docs),
                "chunks": n_chunks,
                "queries": len(qrels),
                "embedder": label,
                "plain": plain,
                "title_context": ctx,
            }  # fmt: skip
        )
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "eval_contextual.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for theme in ("light", "dark"):
        bars = []
        for d in report["datasets"]:
            for system in ("BM25", "Vector", "Hybrid RRF"):
                gain = d["title_context"][system]["ndcg@10"] - d["plain"][system]["ndcg@10"]
                bars.append(
                    {
                        "label": f"{d['title']} · {system}",
                        "value": max(gain, 0.0),
                        "color": "var(--ember)" if False else "#ff6b2c",
                        "suffix": f"  ({gain:+.3f})",
                    }
                )
        (ASSETS / f"eval_contextual_{theme}.svg").write_text(
            render_bar_chart(
                "nDCG@10 gain from title context",
                bars,
                subtitle="chunk-level retrieval, same chunks, context added at indexing time",
                x_max=max(0.1, max(b["value"] for b in bars) * 1.3),
                theme=theme,
            ),
            encoding="utf-8",
        )
    (ROOT / "docs" / "contextual.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {ROOT / 'docs' / 'contextual.md'}")


def render_markdown(report: dict[str, Any]) -> str:
    sections = []
    for d in report["datasets"]:
        rows = "\n".join(
            f"| {s} | {d['plain'][s]['ndcg@10']:.3f} | {d['title_context'][s]['ndcg@10']:.3f} | "
            f"**{d['title_context'][s]['ndcg@10'] - d['plain'][s]['ndcg@10']:+.3f}** | "
            f"{d['plain'][s]['recall@100']:.3f} | {d['title_context'][s]['recall@100']:.3f} |"
            for s in d["plain"]
        )
        sections.append(
            f"""## {d["title"]}

{d["documents"]:,} documents split into {d["chunks"]:,} chunks; {d["queries"]} test queries; vectors: {d["embedder"]}.

| System | nDCG@10 plain | nDCG@10 + title context | Δ | Recall@100 plain | Recall@100 + context |
| :--- | ---: | ---: | ---: | ---: | ---: |
{rows}
"""
        )
    picture = (
        '<picture><source media="(prefers-color-scheme: dark)" srcset="assets/eval_contextual_dark.svg">'
        '<img alt="Gain from contextual retrieval" src="assets/eval_contextual_light.svg" width="680"></picture>'
    )
    return f"""# Contextual retrieval: measured

A chunk like *"The compound reduced tumour growth by 40 %"* cannot be matched to a query about a specific
drug unless the chunk says which drug. **Contextual retrieval** prepends a short situating
context to each chunk **at indexing time** (`RagPipeline(contextualizer=...)`, `Chunk.index_text`);
answers and citations still use the original chunk text. Here the context is the cheapest
deterministic one -- the document **title** (`title_context_fn`) -- and the experiment holds
everything else fixed (same chunks, same ~{report["chunk_chars"]}-char splitting, same metrics).
Chunk rankings are collapsed to document rankings and scored against BEIR's human judgments.
ragforge {report["ragforge_version"]}.

```bash
python scripts/eval_contextual.py --data-root data --datasets scifact,nfcorpus --embedder minilm
```

{picture}

{chr(10).join(sections)}
## Reading it

- A positive Δ means the title made chunks easier to find. The effect is largest for **BM25**, which has no
  other way to learn what a chunk is about, and for corpora whose titles are informative.
- The deterministic title context is free. `llm_context_fn` (one model call per chunk, as in Anthropic's
  *contextual retrieval* write-up) produces richer context at a real cost; it is not measured here
  because it needs an LLM key -- the plumbing is identical.
- Context lengthens every indexed chunk, which can dilute BM25's length normalization and the embedding;
  check the **Recall@100** columns and the rows where Δ is small or negative before enabling it everywhere.
"""


if __name__ == "__main__":
    main()
