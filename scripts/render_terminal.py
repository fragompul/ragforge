"""Render real CLI output as a crisp, theme-aware SVG "terminal screenshot".

    python scripts/render_terminal.py            # writes docs/assets/demo_terminal.svg

It runs ``ragforge demo`` (forcing ANSI color), parses the escape codes and draws
each line as SVG text, so the README image is always genuine program output --
never a mock-up. Regenerate it whenever the demo changes.
"""

from __future__ import annotations

import html
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ragforge.demo import _ANSI, run_demo  # noqa: E402

PALETTE = {
    "ember": "#ff8a4c", "gold": "#ffc247", "cyan": "#4fd8f0", "green": "#4ade9a",
    "red": "#ff6b81", "grey": "#7d86a3", "dim": "#98a1bd", "reset": "#e6e9f4",
}  # fmt: skip
CODE = {v: k for k, v in _ANSI.items()}
TOKEN = re.compile(r"(\x1b\[[0-9;]*m)")
CELL_W, LINE_H, PAD = 8.1, 19.0, 22


def parse(line: str) -> list[tuple[str, str, bool]]:
    """Split a line into ``(text, color, bold)`` runs."""
    runs: list[tuple[str, str, bool]] = []
    color, bold = "reset", False
    for part in TOKEN.split(line):
        if TOKEN.fullmatch(part):
            name = CODE.get(part, "reset")
            if name == "reset":
                color, bold = "reset", False
            elif name == "bold":
                bold = True
            else:
                color = name
        elif part:
            runs.append((part, color, bold))
    return runs


def render(lines: list[str]) -> str:
    width = max(len(re.sub(r"\x1b\[[0-9;]*m", "", ln)) for ln in lines) * CELL_W + 2 * PAD
    height = len(lines) * LINE_H + 2 * PAD + 34
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        'font-family="JetBrains Mono, SF Mono, Menlo, Consolas, monospace" font-size="13.5">',
        f'<rect width="{width:.0f}" height="{height:.0f}" rx="14" fill="#0b0f1a"/>',
        f'<rect width="{width:.0f}" height="34" rx="14" fill="#141a2c"/>'
        f'<rect y="20" width="{width:.0f}" height="14" fill="#141a2c"/>',
        '<circle cx="22" cy="17" r="6" fill="#ff5f57"/>'
        '<circle cx="42" cy="17" r="6" fill="#febc2e"/>'
        '<circle cx="62" cy="17" r="6" fill="#28c840"/>',
        f'<text x="{width / 2:.0f}" y="22" text-anchor="middle" fill="#7d86a3" font-size="12">'
        "ragforge demo</text>",
    ]
    for i, line in enumerate(lines):
        y = 34 + PAD + i * LINE_H
        spans, x = [], PAD
        for text, color, bold in parse(line):
            weight = ' font-weight="700"' if bold else ""
            spans.append(
                f'<tspan x="{x:.1f}" fill="{PALETTE.get(color, PALETTE["reset"])}"{weight}>'
                f"{html.escape(text)}</tspan>"
            )
            x += len(text) * CELL_W
        out.append(f'<text y="{y:.1f}" xml:space="preserve">{"".join(spans)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    buffer = io.StringIO()
    run_demo(buffer, color=True)
    lines = buffer.getvalue().rstrip("\n").split("\n")
    target = ROOT / "docs" / "assets" / "demo_terminal.svg"
    target.write_text(render(lines), encoding="utf-8")
    print(f"Wrote {target} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
