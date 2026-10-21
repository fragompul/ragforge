"""Measure ragforge's prompt-injection detector on public datasets.

    python scripts/eval_guardrails.py --download --data-root data     # needs: pip install pyarrow
    python scripts/eval_guardrails.py --data-root data [--beir data/scifact]

Rules like these are easy to *claim* works; this script measures it, on three public labelled
collections of injection / jailbreak prompts and one large benign corpus:

* ``deepset/prompt-injections`` (Apache-2.0; English + German; 546 train / 116 test),
* ``xTRam1/safe-guard-prompt-injection`` (8,236 train / 2,060 test),
* ``jackhhao/jailbreak-classification`` (Apache-2.0; jailbreak vs benign prompts),
* benign *documents*: BEIR SciFact abstracts (false positives on real retrieved text, the
  thing the detector actually screens) and the bundled sample corpus.

Metrics use the library default (``risk >= 0.5``). It also measures the **indirect** setting:
an attack sentence is injected into benign abstracts, and we check that it is detected and
that the sanitized text keeps the benign content.

**Protocol.** The rules were written from the published attack taxonomy *before* these
datasets were run. They were then refined once by reading false negatives on the **train**
splits only; the **test** splits were never inspected. ``--baseline`` loads the numbers
recorded before that refinement so the report shows both honestly.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from svg_charts import PALETTE, render_bar_chart  # noqa: E402

from ragforge._version import __version__  # noqa: E402
from ragforge.guardrails import Guardrails, InjectionDetector  # noqa: E402
from ragforge.sample import SAMPLE_DOCUMENTS  # noqa: E402

ASSETS = ROOT / "docs" / "assets"
HF = "https://huggingface.co/datasets"
SOURCES = {
    "deepset_train": f"{HF}/deepset/prompt-injections/resolve/main/data/train-00000-of-00001-9564e8b05b4757ab.parquet",
    "deepset_test": f"{HF}/deepset/prompt-injections/resolve/main/data/test-00000-of-00001-701d16158af87368.parquet",
    "safeguard_train": f"{HF}/xTRam1/safe-guard-prompt-injection/resolve/main/data/train-00000-of-00001.parquet",
    "safeguard_test": f"{HF}/xTRam1/safe-guard-prompt-injection/resolve/main/data/test-00000-of-00001.parquet",
    "jailbreak_train": f"{HF}/jackhhao/jailbreak-classification/resolve/main/default/jailbreak_dataset_train.csv",
    "jailbreak_test": f"{HF}/jackhhao/jailbreak-classification/resolve/main/default/jailbreak_dataset_test.csv",
}
TITLES = {
    "deepset": "deepset/prompt-injections",
    "safeguard": "safe-guard-prompt-injection",
    "jailbreak": "jailbreak-classification",
}


def download(root: Path) -> None:
    raw = root / "guardrails"
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = raw / (name + (".csv" if url.endswith(".csv") else ".parquet"))
        if not target.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, target)


def load_labeled(root: Path, name: str) -> list[tuple[str, int]]:
    """``[(text, is_attack)]`` for one split."""
    raw = root / "guardrails"
    if name.startswith("jailbreak"):
        with open(raw / f"{name}.csv", encoding="utf-8", newline="") as f:
            return [(row["prompt"], int(row["type"] == "jailbreak")) for row in csv.DictReader(f)]
    import pyarrow.parquet as pq

    table = pq.read_table(raw / f"{name}.parquet").to_pylist()
    return [(row["text"], int(row["label"])) for row in table]


def predict(guard: Guardrails, texts: list[str]) -> tuple[list[bool], list[float], list[list[str]]]:
    detector = guard.detector
    assert detector is not None
    flags, risks, rules = [], [], []
    for text in texts:
        findings = detector.scan(text)
        risk = detector.risk(findings)
        flags.append(risk >= guard.risk_threshold)
        risks.append(risk)
        rules.append(sorted({f.rule for f in findings}))
    return flags, risks, rules


def confusion(labels: list[int], flags: list[bool]) -> dict[str, float]:
    tp = sum(1 for y, p in zip(labels, flags, strict=True) if y and p)
    fp = sum(1 for y, p in zip(labels, flags, strict=True) if not y and p)
    fn = sum(1 for y, p in zip(labels, flags, strict=True) if y and not p)
    tn = sum(1 for y, p in zip(labels, flags, strict=True) if not y and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(labels), "attacks": tp + fn, "benign": fp + tn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }  # fmt: skip


def evaluate_split(guard: Guardrails, root: Path, name: str) -> dict[str, Any]:
    rows = load_labeled(root, name)
    texts, labels = [t for t, _ in rows], [y for _, y in rows]
    t0 = time.perf_counter()
    flags, _, rules = predict(guard, texts)
    seconds = time.perf_counter() - t0
    result = confusion(labels, flags)
    result["kb_per_second"] = sum(len(t) for t in texts) / 1024 / max(seconds, 1e-9)
    result["rule_hits_on_attacks"] = dict(
        Counter(
            r for y, rs, p in zip(labels, rules, flags, strict=True) if y and p for r in rs
        ).most_common(8)
    )
    result["missed_examples"] = [
        t[:140] for t, y, p in zip(texts, labels, flags, strict=True) if y and not p
    ][:5]
    return result


def false_positive_rate_on_documents(
    guard: Guardrails, beir: Path | None, limit: int
) -> dict[str, Any]:
    docs = [d["text"] for d in SAMPLE_DOCUMENTS if d["id"] != "poisoned-faq"]
    out: dict[str, Any] = {
        "sample_corpus": {"n": len(docs), "flagged": predict(guard, docs)[0].count(True)}
    }
    if beir is not None and (beir / "corpus.jsonl").exists():
        corpus = [
            json.loads(line)
            for line in (beir / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        texts = [f"{c.get('title', '')}. {c['text']}" for c in corpus][:limit]
        flags, _, rules = predict(guard, texts)
        out["scifact_abstracts"] = {
            "n": len(texts), "flagged": flags.count(True),
            "rules": dict(Counter(r for rs, f in zip(rules, flags, strict=True) if f for r in rs)),
            "examples": [t[:160] for t, f in zip(texts, flags, strict=True) if f][:3],
        }  # fmt: skip
    return out


def indirect_injection(
    guard: Guardrails, beir: Path, attacks: list[str], n: int, seed: int
) -> dict[str, Any]:
    """Inject an attack into benign abstracts; check detection and benign-content retention."""
    rng = random.Random(seed)
    corpus = [
        json.loads(line)
        for line in (beir / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    abstracts = [c["text"] for c in rng.sample(corpus, n)]
    detected = neutral = 0
    retained: list[float] = []
    for abstract in abstracts:
        sentences = re.split(r"(?<=[.!?])\s+", abstract)
        attack = rng.choice(attacks)[:300]
        pos = rng.randrange(len(sentences) + 1)
        poisoned = " ".join([*sentences[:pos], attack, *sentences[pos:]])
        screened = Guardrails(action="sanitize", redactor=None).screen(poisoned)
        detected += bool(screened.risk >= 0.5)
        residual = (
            InjectionDetector.risk(guard.detector.scan(screened.text)) if guard.detector else 0.0
        )
        neutral += bool(screened.risk >= 0.5 and residual < 0.5)
        original = set(re.findall(r"\w+", abstract.lower()))
        kept = set(re.findall(r"\w+", screened.text.lower()))
        retained.append(len(original & kept) / max(len(original), 1))
    return {"n": n, "detected": detected / n, "neutralized": neutral / n,
            "benign_words_retained": statistics.mean(retained)}  # fmt: skip


def run(args: argparse.Namespace) -> dict[str, Any]:
    guard = Guardrails(action="sanitize")
    report: dict[str, Any] = {
        "ragforge_version": __version__,
        "threshold": guard.risk_threshold,
        "splits": {},
    }
    for name in SOURCES:
        result = evaluate_split(guard, args.data_root, name)
        report["splits"][name] = result
        print(f"  {name:<18} recall={result['recall']:.3f} precision={result['precision']:.3f} "
              f"FPR={result['fpr']:.3f} (n={result['n']})")  # fmt: skip
    beir = Path(args.beir) if args.beir else None
    report["documents"] = false_positive_rate_on_documents(guard, beir, args.max_docs)
    for k, v in report["documents"].items():
        print(f"  benign documents [{k}]: {v['flagged']}/{v['n']} flagged")
    if beir is not None and (beir / "corpus.jsonl").exists():
        attacks = [
            t
            for t, y in load_labeled(args.data_root, "deepset_test")
            + load_labeled(args.data_root, "safeguard_test")
            if y and t.isascii()
        ]
        report["indirect"] = indirect_injection(guard, beir, attacks, 300, seed=5)
        print(f"  indirect injection: {report['indirect']}")
    return report


def render_markdown(report: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    def row(name: str, r: dict[str, Any], base: dict[str, Any] | None) -> str:
        delta = f" ({r['recall'] - base['recall']:+.1%} vs. initial)" if base else ""
        return (f"| {TITLES[name.rsplit('_', 1)[0]]} · {name.rsplit('_', 1)[1]} | {r['n']:,} | {r['recall']:.1%}{delta} | "
                f"{r['precision']:.1%} | {r['f1']:.3f} | {r['fpr']:.1%} |")  # fmt: skip

    rows = "\n".join(
        row(n, r, baseline["splits"].get(n) if baseline else None)
        for n, r in report["splits"].items()
    )
    docs = report["documents"]
    sci = docs.get("scifact_abstracts")
    doc_lines = [
        f"- bundled sample corpus (excluding the deliberately poisoned FAQ): **{docs['sample_corpus']['flagged']} / {docs['sample_corpus']['n']}** flagged"
    ]
    if sci:
        doc_lines.append(
            f"- BEIR SciFact abstracts: **{sci['flagged']} / {sci['n']:,}** flagged ({sci['flagged'] / sci['n']:.2%})"
        )
    ind = report.get("indirect")
    indirect = ""
    if ind:
        indirect = f"""
## Indirect injection inside retrieved documents

The setting the detector exists for: an attack sentence taken from the test splits is inserted at
a random position in a benign SciFact abstract ({ind["n"]} trials), then screened with
`Guardrails(action="sanitize")`.

| Metric | Result |
| :--- | ---: |
| Attack detected (risk ≥ 0.5) | **{ind["detected"]:.1%}** |
| Neutralized (re-scan of the sanitized text is below the threshold) | **{ind["neutralized"]:.1%}** |
| Benign abstract words preserved | **{ind["benign_words_retained"]:.1%}** |
"""
    missed = "\n".join(
        f"- *{TITLES[n.rsplit('_', 1)[0]]} {n.rsplit('_', 1)[1]}*: “{ex}…”"
        for n, r in report["splits"].items()
        if n.endswith("_test")
        for ex in r["missed_examples"][:2]
    )
    picture = (
        '<picture><source media="(prefers-color-scheme: dark)" srcset="assets/eval_guardrails_dark.svg">'
        '<img alt="Detector recall per dataset" src="assets/eval_guardrails_light.svg" width="680"></picture>'
    )
    return f"""# Guardrails: measured detection and false positives

`ragforge.guardrails` screens retrieved text for **indirect prompt injection** and PII. Pattern
rules are only worth shipping if their recall and false-positive rate are *measured*, so this report
scores the default detector (`risk >= {report["threshold"]}`) on public labelled data. Reproduce:

```bash
python scripts/eval_guardrails.py --download --data-root data       # needs: pip install pyarrow
python scripts/eval_guardrails.py --data-root data --beir data/scifact
```

**Protocol.** The rules were written from the published attack taxonomy before these datasets were
run, then refined once by reading false negatives on the **train** splits only. The **test** splits were
never inspected. {('The "vs. initial" column compares with the rules as first written.' if baseline else "")}

{picture}

## Detection on labelled prompts

| Dataset · split | Prompts | Recall | Precision | F1 | False-positive rate |
| :--- | ---: | ---: | ---: | ---: | ---: |
{rows}

## False positives on real documents

{chr(10).join(doc_lines)}
{indirect}
## What the rules miss

Pattern matching catches the *lazy* attacks: override phrases, role hijacks, prompt exfiltration,
chat-template tokens, markdown exfiltration, hidden Unicode (English and Spanish). It does not
understand intent, so semantically disguised or non-English attacks (the deepset set is partly German)
slip through. Examples from the test splits that were **not** flagged:

{missed}

Treat the detector as a *defense-in-depth layer that makes common poisoning visible and auditable*,
not as a guarantee: combine it with least-privilege tools, spotlighting
(`spotlight_prompt_formatter`), and output filtering. A learned classifier can be plugged in through
`InjectionDetector(extra_rules=...)` or by wrapping `Guardrails`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--beir", default=None, help="BEIR dataset dir (e.g. data/scifact) for document tests"
    )
    parser.add_argument("--max-docs", type=int, default=5183)
    parser.add_argument("--baseline", type=Path, default=ASSETS / "eval_guardrails_baseline.json")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="record this run as the 'initial rules' baseline",
    )
    args = parser.parse_args()
    if args.download:
        download(args.data_root)
    report = run(args)
    ASSETS.mkdir(parents=True, exist_ok=True)
    if args.save_baseline:
        args.baseline.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved baseline to {args.baseline}")
        return
    baseline = json.loads(args.baseline.read_text()) if args.baseline.exists() else None
    (ASSETS / "eval_guardrails.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for theme in ("light", "dark"):
        bars = [
            {
                "label": f"{TITLES[n.rsplit('_', 1)[0]]} (test)",
                "value": r["recall"],
                "color": PALETTE[i % len(PALETTE)],
            }
            for i, (n, r) in enumerate(
                x for x in report["splits"].items() if x[0].endswith("_test")
            )
        ]
        (ASSETS / f"eval_guardrails_{theme}.svg").write_text(
            render_bar_chart(
                "Attack recall on held-out test splits",
                bars,
                subtitle="default rules, risk ≥ 0.5",
                x_max=1.0,
                theme=theme,
            ),
            encoding="utf-8",
        )
    (ROOT / "docs" / "guardrails.md").write_text(
        render_markdown(report, baseline), encoding="utf-8"
    )
    print(f"Wrote {ROOT / 'docs' / 'guardrails.md'}")


if __name__ == "__main__":
    main()
