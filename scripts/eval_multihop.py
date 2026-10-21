"""Multi-hop retrieval and abstention on HotpotQA (distractor setting).

    python scripts/eval_multihop.py --download --data-root data         # needs: pip install pyarrow
    python scripts/eval_multihop.py --data-root data --embedder minilm

A multi-hop question needs *two* paragraphs ("Which magazine was started first, Arthur's
Magazine or First for Women?"; "the director of film X also directed which other film?").
Single-pass retrieval often finds one and misses the other. This script measures how well
ragforge's retrieval strategies recover **all** gold paragraphs, using a shared corpus built
from the paragraphs of ``--questions`` validation questions (so distractors are real, and
retrieval is genuinely hard):

* BM25 / vector / hybrid RRF at a fixed context budget of ``k`` passages,
* hybrid + **entity-graph** retrieval (``GraphIndex``, a third RRF source),
* hybrid + the **agentic bridge loop** (``AgenticRag`` with ``bridge_rewrite_fn``: retrieve,
  read the entities in the top passages, retrieve again for the *new* evidence; each of the two
  hops retrieves ``k/2`` so the total budget is the same).

It also measures **abstention**: the same lexical sufficiency grader that drives the agent is
run on answerable questions and on *unanswerable* ones (questions whose gold paragraphs are
not in the corpus) at several thresholds -- how often it wrongly refuses vs. correctly refuses.

No thresholds or rules were tuned on this data. Results: ``docs/multihop.md``.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from svg_charts import PALETTE, render_bar_chart  # noqa: E402

from ragforge import RagPipeline  # noqa: E402
from ragforge._version import __version__  # noqa: E402
from ragforge.agent import AgenticRag, bridge_rewrite_fn, lexical_grade_fn  # noqa: E402
from ragforge.chunking import RecursiveCharacterChunker  # noqa: E402
from ragforge.embeddings import hashing_embed  # noqa: E402
from ragforge.graph import GraphIndex  # noqa: E402

HOTPOT_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)
ASSETS = ROOT / "docs" / "assets"


def load_hotpot(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def paragraphs(row: dict[str, Any]) -> dict[str, str]:
    ctx = row["context"]
    return {
        t: f"{t}. {' '.join(s).strip()}".strip()
        for t, s in zip(ctx["title"], ctx["sentences"], strict=True)
    }


def build_embedder(kind: str, texts: list[str], cache: Path):
    """``(embed_fn, label)``; texts seen at build time come from a cache, new ones (agent rewrites) are embedded live."""
    if kind == "hashing":
        return hashing_embed, "hashing (128-d bag-of-words)"
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
    table = {t: matrix[i].tolist() for i, t in enumerate(texts)}

    def embed(text: str) -> list[float]:
        found = table.get(text)
        if found is None:
            found = table[text] = model.encode(text, normalize_embeddings=True).tolist()
        return found

    return embed, f"MiniLM-L6-v2 ({matrix.shape[1]}-d)"


def make_pipeline(embed: Any, docs: dict[str, str], with_graph: bool) -> RagPipeline:
    pipeline = RagPipeline(
        chunker=RecursiveCharacterChunker(chunk_size=8000, chunk_overlap=0),
        embed_fn=embed,
        bm25_analyzer="english",
        vector_backend="numpy",
        extra_indexes={"graph": GraphIndex()} if with_graph else None,
    )
    for title, text in docs.items():
        pipeline.ingest(title, text)
    return pipeline


def score(found_titles: list[str], gold: set[str]) -> tuple[float, float]:
    got = set(found_titles)
    return len(gold & got) / len(gold), float(gold <= got)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--embedder", choices=["hashing", "minilm"], default="hashing")
    parser.add_argument("--questions", type=int, default=300)
    parser.add_argument("--unanswerable", type=int, default=150)
    parser.add_argument("--k", type=int, default=10, help="context budget (passages)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    parquet = args.data_root / "hotpot_val.parquet"
    if args.download and not parquet.exists():
        args.data_root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HOTPOT_URL, parquet)
    rows = load_hotpot(parquet)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    answerable = rows[: args.questions]
    corpus: dict[str, str] = {}
    for row in answerable:
        corpus.update(paragraphs(row))
    gold_of = {r["id"]: set(r["supporting_facts"]["title"]) for r in rows}
    unanswerable = [r for r in rows[args.questions :] if not (gold_of[r["id"]] & set(corpus))][
        : args.unanswerable
    ]
    print(
        f"corpus: {len(corpus):,} paragraphs; {len(answerable)} answerable, {len(unanswerable)} unanswerable questions"
    )

    texts = [*corpus.values(), *(r["question"] for r in answerable + unanswerable)]
    cache = Path(args.cache_dir or args.data_root) / "hotpot.minilm.npy"
    embed, label = build_embedder(args.embedder, list(dict.fromkeys(texts)), cache)
    # keep the cache aligned with the de-duplicated text order
    plain = make_pipeline(embed, corpus, with_graph=False)
    graphy = make_pipeline(embed, corpus, with_graph=True)
    k, half = args.k, max(1, args.k // 2)

    def top_titles(pipeline: RagPipeline, q: str, kk: int) -> list[str]:
        return [c.doc_id for c in pipeline.retrieve(q, k=kk)]

    systems: dict[str, Any] = {
        "BM25 (english)": lambda q: [r.chunk.doc_id for r in plain.bm25_index.search(q, k)],
        "Vector (MiniLM)" if args.embedder == "minilm" else "Vector": lambda q: [
            r.chunk.doc_id for r in plain.vector_index.search(q, k)
        ],
        "Hybrid RRF": lambda q: top_titles(plain, q, k),
        "Graph only": lambda q: [
            r.chunk.doc_id for r in graphy.extra_indexes["graph"].search(q, k)
        ],
        "Hybrid RRF + entity graph": lambda q: top_titles(graphy, q, k),
    }
    agent = AgenticRag(
        plain,
        rewrite_fn=bridge_rewrite_fn(),
        min_iterations=2,
        max_iterations=2,
        min_sufficiency=0.0,
        abstain_below=0.0,
    )
    systems["Hybrid RRF + agentic bridge hop"] = lambda q: [
        c.doc_id for c in agent.run(q, k=half).answer.contexts
    ]

    results = []
    for name, search in systems.items():
        recalls, both = [], []
        by_type: dict[str, list[float]] = {"bridge": [], "comparison": []}
        for row in answerable:
            r, b = score(search(row["question"]), gold_of[row["id"]])
            recalls.append(r)
            both.append(b)
            by_type[row["type"]].append(b)
        entry = {
            "system": name,
            "recall": statistics.mean(recalls),
            "all_gold": statistics.mean(both),
            "all_gold_bridge": statistics.mean(by_type["bridge"]),
            "all_gold_comparison": statistics.mean(by_type["comparison"]),
        }
        results.append(entry)
        print(f"  {name:<34} recall={entry['recall']:.3f}  both-gold={entry['all_gold']:.3f} "
              f"(bridge {entry['all_gold_bridge']:.3f}, comparison {entry['all_gold_comparison']:.3f})")  # fmt: skip

    abstention = []
    for threshold in (0.3, 0.5, 0.7):

        def refuses(q: str, threshold: float = threshold) -> bool:
            contexts = [c.text for c in plain.retrieve(q, k=5)]
            return lexical_grade_fn(q, contexts).score < threshold

        false_refusal = statistics.mean(refuses(r["question"]) for r in answerable)
        correct_refusal = statistics.mean(refuses(r["question"]) for r in unanswerable)
        abstention.append(
            {"abstain_below": threshold, "wrongly_refused_answerable": false_refusal,
             "correctly_refused_unanswerable": correct_refusal}
        )  # fmt: skip
        print(
            f"  abstain<{threshold}: refuses {false_refusal:.1%} of answerable, {correct_refusal:.1%} of unanswerable"
        )

    report = {
        "ragforge_version": __version__,
        "dataset": "HotpotQA distractor (validation sample)",
        "paragraphs": len(corpus),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "k": k,
        "embedder": label,
        "systems": [
            {k_: round(v, 4) if isinstance(v, float) else v for k_, v in r.items()} for r in results
        ],
        "abstention": [{k_: round(v, 4) for k_, v in a.items()} for a in abstention],
    }
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "eval_multihop.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for theme in ("light", "dark"):
        svg = render_bar_chart(
            title=f"Both gold paragraphs within {k} passages",
            subtitle=f"HotpotQA distractor · {len(answerable)} questions · {len(corpus):,} paragraphs",
            bars=[
                {"label": r["system"], "value": r["all_gold"], "color": PALETTE[i % len(PALETTE)]}
                for i, r in enumerate(report["systems"])
            ],
            x_max=1.0,
            theme=theme,
        )
        (ASSETS / f"eval_multihop_{theme}.svg").write_text(svg, encoding="utf-8")
    (ROOT / "docs" / "multihop.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {ROOT / 'docs' / 'multihop.md'}")


def render_markdown(r: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {s['system']} | {s['recall']:.3f} | **{s['all_gold']:.3f}** | {s['all_gold_bridge']:.3f} | "
        f"{s['all_gold_comparison']:.3f} |"
        for s in r["systems"]
    )
    abst = "\n".join(
        f"| {a['abstain_below']} | {a['wrongly_refused_answerable']:.1%} | {a['correctly_refused_unanswerable']:.1%} |"
        for a in r["abstention"]
    )
    picture = (
        '<picture><source media="(prefers-color-scheme: dark)" srcset="assets/eval_multihop_dark.svg">'
        '<img alt="Both gold paragraphs retrieved" src="assets/eval_multihop_light.svg" width="680"></picture>'
    )
    best = max(r["systems"], key=lambda s: s["all_gold"])
    base = next(s for s in r["systems"] if s["system"] == "Hybrid RRF")
    return f"""# Multi-hop retrieval and abstention

Single-pass retrieval often finds one of the two paragraphs a multi-hop question needs and
misses the other. This report measures how well ragforge's strategies recover **all** gold
paragraphs on **HotpotQA** (distractor setting, validation split; Yang et al., 2018; CC BY-SA 4.0),
using a shared corpus of {r["paragraphs"]:,} paragraphs built from {r["answerable"]} questions, so every
distractor is a real Wikipedia paragraph. Vectors: {r["embedder"]}. ragforge {r["ragforge_version"]}.

```bash
python scripts/eval_multihop.py --download --data-root data     # needs: pip install pyarrow
python scripts/eval_multihop.py --data-root data --embedder minilm
```

## Retrieval: are both supporting paragraphs in the context?

Context budget: **{r["k"]} passages** for every system (the agent spends {r["k"] // 2} per hop).
*Recall* is the mean fraction of gold paragraphs found; *both gold* is the share of questions where
**all** of them are in the context. *Bridge* questions need a hop through an entity named only in the
first paragraph; *comparison* questions name both entities up front.

{picture}

| System | Recall | Both gold | Bridge | Comparison |
| :--- | ---: | ---: | ---: | ---: |
{rows}

**Reading it.** The best system here is *{best["system"]}* ({best["all_gold"]:.1%} vs. {base["all_gold"]:.1%}
for plain hybrid). Whatever the ranking, check the bridge and comparison columns separately: bridge questions are
where a second hop *should* matter, comparison questions are mostly solved by the first retrieval.
No rule or threshold was tuned on this data; the bridge rewrite is the deterministic
`ragforge.agent.bridge_rewrite_fn` and the graph is the heuristic `GraphIndex` (no model, no LLM).
An LLM-driven decompose/rewrite loop (`llm_rewrite_fn`) is the natural next step and is not measured here.

## Abstention: does the agent know when it cannot answer?

The agent's default lexical sufficiency grader (share of question terms covered by the retrieved
passages) run on the {r["answerable"]} answerable questions and on {r["unanswerable"]} **unanswerable** ones --
HotpotQA questions whose gold paragraphs are *not* in the corpus.

| Abstain below | Wrongly refused (answerable) | Correctly refused (unanswerable) |
| ---: | ---: | ---: |
{abst}

A purely lexical grader is a blunt instrument: unanswerable multi-hop questions still share many common
words with unrelated paragraphs, and answerable ones paraphrase their evidence. Treat this as the
baseline that an LLM- or NLI-based grader (`llm_grade_fn`) should beat, and pick the threshold
for your cost of a wrong refusal vs. a wrong answer.
"""


if __name__ == "__main__":
    main()
