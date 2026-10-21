"""Assemble the static GitHub Pages site into ./site.

    python scripts/build_site.py            # writes site/
    python -m http.server -d site 8000      # then open http://localhost:8000

The site is the RagForge playground: plain HTML/CSS/ES-modules (no bundler, no
Node toolchain) plus a zipped copy of the *actual* ``ragforge`` package, which the
browser loads into Pyodide (CPython compiled to WebAssembly). There is no server:
the library runs entirely in the visitor's browser.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "ragforge"
WEB = ROOT / "web"
OUT = ROOT / "site"

sys.path.insert(0, str(ROOT / "src"))
from ragforge._version import __version__  # noqa: E402

DATA_PATTERNS = ["benchmarks.json", "eval_*.json"]


def build_package_zip(dest: Path) -> None:
    """Zip the real package sources (no bytecode) for Pyodide's ``unpackArchive``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SRC.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                zf.write(path, Path("ragforge") / path.relative_to(SRC))


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(WEB, OUT, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "bridge.py"))
    build_package_zip(OUT / "py" / "ragforge.zip")
    shutil.copy(WEB / "bridge.py", OUT / "py" / "bridge.py")

    (OUT / "data").mkdir(exist_ok=True)
    shipped = sorted(
        {p.name for pattern in DATA_PATTERNS for p in (ROOT / "docs" / "assets").glob(pattern)}
    )
    for name in shipped:
        shutil.copy(ROOT / "docs" / "assets" / name, OUT / "data" / name)
    (OUT / "data" / "manifest.json").write_text(json.dumps({"files": shipped}), encoding="utf-8")
    if not shipped:
        print(
            "warning: no benchmark JSON found in docs/assets; the evaluation panels will be empty"
        )

    # Cache-busting token derived from everything that ships.
    digest = hashlib.sha256()
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            digest.update(path.read_bytes())
    build = digest.hexdigest()[:10]
    index = OUT / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8")
        .replace("{{VERSION}}", __version__)
        .replace("{{BUILD}}", build),
        encoding="utf-8",
    )
    (OUT / "version.json").write_text(json.dumps({"version": __version__, "build": build}))
    (OUT / ".nojekyll").touch()
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"Built site v{__version__} ({build}) -> {OUT}  [{size / 1024:.0f} KiB]")


if __name__ == "__main__":
    main()
