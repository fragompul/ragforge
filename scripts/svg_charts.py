"""A tiny, dependency-free SVG chart renderer for the docs and README.

Kept deliberately out of the installable ``ragforge`` package: it exists so the
repository's charts don't require matplotlib (or any plotting library) to
regenerate, in keeping with the project's zero-dependency ethos -- but it is a
docs tool, not a library feature.

Every chart renders in a ``light`` and a ``dark`` theme with the RagForge
palette, so the README can swap them with a ``<picture>`` element and look right
on both GitHub themes.
"""

from __future__ import annotations

import html
import math
from typing import Any

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "bg": "#ffffff",
        "panel": "#f6f7fb",
        "text": "#10162b",
        "muted": "#5b6585",
        "grid": "#e4e7f1",
        "axis": "#9aa3bf",
    },
    "dark": {
        "bg": "#0b0f1a",
        "panel": "#121829",
        "text": "#e7eaf3",
        "muted": "#8b93ab",
        "grid": "#232b42",
        "axis": "#4a5575",
    },
}

# Brand palette, tuned to hold contrast on both themes.
PALETTE = ["#ff6b2c", "#2dd4ef", "#8b7cf6", "#34d399", "#ffb020", "#f43f5e"]
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif"


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _scale(value: float, lo: float, hi: float, log: bool) -> float:
    if log:
        value, lo, hi = (math.log10(max(v, 1e-9)) for v in (value, lo, hi))
    if hi == lo:
        return 0.5
    return (value - lo) / (hi - lo)


def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    for step in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if value <= step * magnitude:
            return step * magnitude
    return 10 * magnitude


def _fmt(value: float) -> str:
    if value >= 100 or float(value).is_integer():
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".") if value >= 0.01 else f"{value:.4f}".rstrip("0")


def _frame(width: int, height: int, title: str, subtitle: str, t: dict[str, str]) -> list[str]:
    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="{FONT}" role="img" aria-label="{_esc(title)}">',
        f'<rect width="{width}" height="{height}" rx="14" fill="{t["bg"]}"/>',
        f'<text x="24" y="34" font-size="17" font-weight="700" fill="{t["text"]}">'
        f"{_esc(title)}</text>",
    ]
    if subtitle:
        parts.append(
            f'<text x="24" y="54" font-size="12" fill="{t["muted"]}">{_esc(subtitle)}</text>'
        )
    return parts


def render_line_chart(
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, Any]],
    x_log: bool = False,
    y_max: float | None = None,
    y_min: float = 0.0,
    y_log: bool = False,
    subtitle: str = "",
    theme: str = "light",
    width: int = 680,
    height: int = 400,
) -> str:
    """Multi-series line chart. ``series`` items: ``{"name", "points": [(x, y)], "color"?}``."""
    t = THEMES[theme]
    m = {"top": 84, "right": 28, "bottom": 58, "left": 66}
    pw, ph = width - m["left"] - m["right"], height - m["top"] - m["bottom"]

    xs = [x for s in series for x, _ in s["points"]]
    ys = [y for s in series for _, y in s["points"]]
    x_min, x_max = min(xs), max(xs)
    if y_log:
        y_lo = 10 ** math.floor(math.log10(min(ys) * 0.8))
        y_top = 10 ** math.ceil(math.log10(max(ys) * 1.2))
    else:
        y_lo = y_min
        y_top = y_max if y_max is not None else _nice_ceiling(max(ys) * 1.08)

    def px(x: float) -> float:
        return m["left"] + _scale(x, x_min, x_max, x_log) * pw

    def py(y: float) -> float:
        return m["top"] + ph - _scale(y, y_lo, y_top, y_log) * ph

    parts = _frame(width, height, title, subtitle, t)

    if y_log:
        ticks = [10**e for e in range(round(math.log10(y_lo)), round(math.log10(y_top)) + 1)]
    else:
        ticks = [y_lo + (y_top - y_lo) * i / 4 for i in range(5)]
    for value in ticks:
        y = py(value)
        parts.append(
            f'<line x1="{m["left"]}" y1="{y:.1f}" x2="{width - m["right"]}" y2="{y:.1f}" '
            f'stroke="{t["grid"]}"/>'
            f'<text x="{m["left"] - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="{t["muted"]}">{_fmt(value)}</text>'
        )
    for x in sorted(set(xs)):
        parts.append(
            f'<text x="{px(x):.1f}" y="{height - m["bottom"] + 20}" text-anchor="middle" '
            f'font-size="11" fill="{t["muted"]}">{_fmt(x)}</text>'
        )
    parts.append(
        f'<text x="{m["left"] + pw / 2}" y="{height - 14}" text-anchor="middle" font-size="12" '
        f'fill="{t["muted"]}">{_esc(x_label)}</text>'
        f'<text x="18" y="{m["top"] + ph / 2}" text-anchor="middle" font-size="12" '
        f'fill="{t["muted"]}" transform="rotate(-90 18 {m["top"] + ph / 2})">{_esc(y_label)}</text>'
    )

    lx = 24
    ly = 72
    for idx, s in enumerate(series):
        color = s.get("color") or PALETTE[idx % len(PALETTE)]
        pts = sorted(s["points"])
        path = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        parts.append(
            f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.6" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y in pts:
            parts.append(
                f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="{t["bg"]}" '
                f'stroke="{color}" stroke-width="2.4"/>'
            )
        parts.append(
            f'<rect x="{lx}" y="{ly - 8}" width="14" height="4" rx="2" fill="{color}"/>'
            f'<text x="{lx + 20}" y="{ly - 2}" font-size="12" fill="{t["text"]}">'
            f"{_esc(s['name'])}</text>"
        )
        lx += 30 + 7 * len(str(s["name"]))
    parts.append("</svg>")
    return "\n".join(parts)


def render_bar_chart(
    title: str,
    bars: list[dict[str, Any]],
    x_label: str = "",
    subtitle: str = "",
    theme: str = "light",
    x_max: float | None = None,
    width: int = 680,
    row_height: int = 38,
) -> str:
    """Horizontal bar chart. ``bars`` items: ``{"label", "value", "color"?, "note"?}``."""
    t = THEMES[theme]
    label_w = 190
    m = {"top": 78, "right": 64, "bottom": 46, "left": 24 + label_w}
    height = m["top"] + m["bottom"] + row_height * len(bars)
    pw = width - m["left"] - m["right"]
    top = x_max if x_max is not None else _nice_ceiling(max(b["value"] for b in bars) * 1.05)

    parts = _frame(width, height, title, subtitle, t)
    for i in range(5):
        value = top * i / 4
        x = m["left"] + pw * i / 4
        parts.append(
            f'<line x1="{x:.1f}" y1="{m["top"] - 6}" x2="{x:.1f}" y2="{height - m["bottom"]}" '
            f'stroke="{t["grid"]}"/>'
            f'<text x="{x:.1f}" y="{height - m["bottom"] + 18}" text-anchor="middle" '
            f'font-size="11" fill="{t["muted"]}">{_fmt(value)}</text>'
        )
    for idx, bar in enumerate(bars):
        y = m["top"] + idx * row_height
        color = bar.get("color") or PALETTE[idx % len(PALETTE)]
        w = max(2.0, bar["value"] / top * pw)
        parts.append(
            f'<text x="{m["left"] - 12}" y="{y + row_height / 2 + 1:.1f}" text-anchor="end" '
            f'font-size="12.5" fill="{t["text"]}">{_esc(bar["label"])}</text>'
            f'<rect x="{m["left"]}" y="{y + 6}" width="{w:.1f}" height="{row_height - 14}" rx="5" '
            f'fill="{color}"/>'
            f'<text x="{m["left"] + w + 8:.1f}" y="{y + row_height / 2 + 1:.1f}" font-size="12.5" '
            f'font-weight="700" fill="{t["text"]}">{_fmt(bar["value"])}'
            f"{_esc(bar.get('suffix', ''))}</text>"
        )
    if x_label:
        parts.append(
            f'<text x="{m["left"] + pw / 2}" y="{height - 10}" text-anchor="middle" font-size="12" '
            f'fill="{t["muted"]}">{_esc(x_label)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)
