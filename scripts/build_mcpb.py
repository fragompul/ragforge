"""Build a Claude Desktop extension bundle (``ragforge.mcpb``) for the ragforge MCP server.

    python scripts/build_mcpb.py            # writes dist/ragforge-<version>.mcpb
    python scripts/build_mcpb.py --out x.mcpb

An ``.mcpb`` (MCP Bundle) is a zip with a ``manifest.json`` that Claude Desktop installs with a
double-click and configures through a settings form -- no ``pip``, no JSON editing. Because
ragforge's core has **zero dependencies**, the bundle simply vendors the package under
``server/lib``; it only needs a Python 3.11+ interpreter on the machine.

The manifest asks the user for the index file to serve (built with ``ragforge ingest``), whether
to allow the ingest tool (off by default: anything an LLM can call, an attacker's prompt can
call) and the guardrails policy for retrieved text.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "ragforge"
sys.path.insert(0, str(ROOT / "src"))
from ragforge._version import __version__  # noqa: E402

LAUNCHER = '''"""Launcher for the ragforge MCP server inside a Claude Desktop extension bundle."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

from ragforge.cli import main  # noqa: E402

index = os.environ.get("RAGFORGE_INDEX", "").strip()
if not index:
    sys.stderr.write("ragforge: choose an index file in the extension settings.\\n")
    raise SystemExit(2)

args = ["mcp", "--index", index]
if os.environ.get("RAGFORGE_ALLOW_INGEST", "false").lower() == "true":
    args.append("--allow-ingest")
guardrails = os.environ.get("RAGFORGE_GUARDRAILS", "sanitize").strip().lower()
if guardrails in ("flag", "sanitize", "drop"):
    args += ["--guardrails", guardrails]
if os.environ.get("RAGFORGE_ROUTE", "false").lower() == "true":
    args.append("--route")

raise SystemExit(main(args))
'''


def manifest() -> dict:
    return {
        "manifest_version": "0.3",
        "name": "ragforge",
        "display_name": "RagForge knowledge base",
        "version": __version__,
        "description": "Search and ask questions over your own documents with hybrid retrieval, "
        "guardrails and verifiable citations.",
        "long_description": "Serves a RagForge index to Claude: hybrid BM25 + vector search, "
        "answers with sentence-level citations and a groundedness score, and whole documents as "
        "resources. Build an index with `ragforge ingest ./docs -i kb.json`, then point this "
        "extension at it. Read-only by default.",
        "author": {"name": "Francisco Javier Gómez Pulido", "url": "https://github.com/fragompul"},
        "homepage": "https://github.com/fragompul/ragforge",
        "repository": {"type": "git", "url": "https://github.com/fragompul/ragforge.git"},
        "license": "MIT",
        "keywords": ["rag", "retrieval", "search", "citations", "knowledge-base"],
        "server": {
            "type": "python",
            "entry_point": "server/main.py",
            "mcp_config": {
                "command": "python3",
                "args": ["${__dirname}/server/main.py"],
                "env": {
                    "RAGFORGE_INDEX": "${user_config.index_path}",
                    "RAGFORGE_ALLOW_INGEST": "${user_config.allow_ingest}",
                    "RAGFORGE_GUARDRAILS": "${user_config.guardrails}",
                    "RAGFORGE_ROUTE": "${user_config.route}",
                },
            },
        },
        "tools": [
            {
                "name": "search_knowledge_base",
                "description": "Hybrid search returning ranked passages with offsets.",
            },
            {
                "name": "ask_knowledge_base",
                "description": "Answer with sentence-level citations and a groundedness score.",
            },
            {"name": "knowledge_base_stats", "description": "Document and chunk counts."},
        ],
        "tools_generated": False,
        "user_config": {
            "index_path": {
                "type": "file",
                "title": "Index file",
                "description": "A ragforge index (JSON) built with `ragforge ingest`.",
                "required": True,
            },
            "guardrails": {
                "type": "string",
                "title": "Guardrails policy",
                "description": "Screen retrieved text for prompt injection and PII: "
                "flag, sanitize, drop, or off.",
                "default": "sanitize",
            },
            "route": {
                "type": "boolean",
                "title": "Query router",
                "description": "Choose BM25 vs. vector weights per query.",
                "default": False,
            },
            "allow_ingest": {
                "type": "boolean",
                "title": "Allow adding documents",
                "description": "Expose the ingest_document tool. Off by default.",
                "default": False,
            },
        },
        "compatibility": {"platforms": ["darwin", "win32", "linux"]},
    }


def build(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest(), indent=2))
        zf.writestr("server/main.py", LAUNCHER)
        for path in sorted(SRC.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                zf.write(path, Path("server/lib/ragforge") / path.relative_to(SRC))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / f"ragforge-{__version__}.mcpb")
    args = parser.parse_args()
    out = build(args.out)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
