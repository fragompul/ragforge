"""Package the playground as a Hugging Face *static* Space.

    python scripts/build_space.py [--out dist/space]

The playground is a static site (Pyodide in the browser), so a Space needs no server: the output
directory is the site plus a ``README.md`` with the Space front matter. Push it with::

    git clone https://huggingface.co/spaces/<user>/ragforge && cp -r dist/space/* ragforge/

Nothing here talks to Hugging Face; publishing is a manual step.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FRONT_MATTER = """---
title: RagForge Playground
emoji: 🔎
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
license: mit
short_description: Hybrid RAG, guardrails and citations running in your browser
---

# RagForge playground

The full [RagForge](https://github.com/fragompul/ragforge) library, running client-side with
Pyodide: hybrid retrieval, an HNSW explorer, a prompt-injection lab, a knowledge graph and a
multi-hop agent. Documents you add never leave your browser.
"""


def build(out: Path) -> Path:
    site = ROOT / "site"
    if not (site / "index.html").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_site.py")], check=True)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(site, out)
    (out / "README.md").write_text(FRONT_MATTER, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "space")
    args = parser.parse_args()
    print(f"Space ready in {build(args.out)}")


if __name__ == "__main__":
    main()
