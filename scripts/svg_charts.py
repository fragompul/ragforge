"""A tiny, dependency-free SVG chart renderer used only to generate the
static images embedded in docs/benchmarks.md.

Kept deliberately out of the installable ``ragforge`` package: it exists so
this repository's own benchmark charts don't require matplotlib (or any
other plotting library) to regenerate, in keeping with the project's
zero-dependency ethos -- but it is a docs tool, not a library feature.
"""

from __future__ import annotations

import math
from typing import Any

_PALETTE = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]


def _scale(value: float, domain_min: float, domain_max: float, log: bool) -> float:
    if log:
        value = math.log10(max(value, 1e-9))
        domain_min = math.log10(max(domain_min, 1e-9))
        domain_max = math.log10(max(domain_max, 1e-9))
    if domain_max == domain_min:
        return 0.5
    return (value - domain_min) / (domain_max - domain_min)


def render_line_chart(
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, Any]],
    x_log: bool = False,
    width: int = 640,
    height: int = 380,
) -> str:
    """Render a multi-series line chart as a standalone SVG string.

    Each entry in ``series`` is ``{"name": str, "points": [(x, y), ...],
    "color": optional str}``.
    """
    margin = {"top": 56, "right": 30, "bottom": 56, "left": 74}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    all_x = [x for s in series for x, _ in s["points"]]
    all_y = [y for s in series for _, y in s["points"]]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = 0.0, max(all_y) * 1.15 if all_y else 1.0

    def px(x: float) -> float:
        return margin["left"] + _scale(x, x_min, x_max, x_log) * plot_w

    def py(y: float) -> float:
        return margin["top"] + plot_h - _scale(y, y_min, y_max, False) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="20" text-anchor="middle" font-size="16" '
        f'font-weight="600" fill="#111827">{title}</text>',
    ]

    n_ticks = 5
    for i in range(n_ticks + 1):
        y_val = y_min + (y_max - y_min) * i / n_ticks
        y_px = py(y_val)
        parts.append(
            f'<line x1="{margin["left"]}" y1="{y_px:.1f}" x2="{width - margin["right"]}" '
            f'y2="{y_px:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin["left"] - 10}" y="{y_px + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6b7280">{y_val:,.2f}</text>'
        )

    for x_val in sorted(set(all_x)):
        x_px = px(x_val)
        parts.append(
            f'<line x1="{x_px:.1f}" y1="{margin["top"]}" x2="{x_px:.1f}" '
            f'y2="{height - margin["bottom"]}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x_px:.1f}" y="{height - margin["bottom"] + 22:.1f}" '
            f'text-anchor="middle" font-size="11" fill="#6b7280">{x_val:,.0f}</text>'
        )

    parts.append(
        f'<text x="{width / 2}" y="{height - 8}" text-anchor="middle" '
        f'font-size="12" fill="#374151">{x_label}</text>'
    )
    parts.append(
        f'<text x="18" y="{height / 2}" text-anchor="middle" font-size="12" '
        f'fill="#374151" transform="rotate(-90 18 {height / 2})">{y_label}</text>'
    )

    parts.append(
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" '
        f'y2="{height - margin["bottom"]}" stroke="#9ca3af" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{margin["left"]}" y1="{height - margin["bottom"]}" '
        f'x2="{width - margin["right"]}" y2="{height - margin["bottom"]}" '
        f'stroke="#9ca3af" stroke-width="1.5"/>'
    )

    legend_x = margin["left"] + 10
    legend_y = margin["top"] - 12
    for idx, s in enumerate(series):
        color = s.get("color") or _PALETTE[idx % len(_PALETTE)]
        points = sorted(s["points"])
        path_pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
        parts.append(
            f'<polyline points="{path_pts}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for x, y in points:
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{color}"/>')

        lx = legend_x + idx * 190
        parts.append(
            f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 20}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{lx + 26}" y="{legend_y + 4}" font-size="12" '
            f'fill="#111827">{s["name"]}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)
