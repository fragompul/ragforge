/* Theme-aware SVG charts built from the benchmark JSON (colors are CSS variables). */
import { esc } from "./ui.js";

const NS = "http://www.w3.org/2000/svg";
const PALETTE = ["var(--ember)", "var(--cyan)", "var(--violet)", "var(--mint)", "var(--amber)", "var(--rose)"];

function niceCeil(v) {
  if (v <= 0) return 1;
  const mag = 10 ** Math.floor(Math.log10(v));
  return ([1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find((s) => v <= s * mag) || 10) * mag;
}
const fmt = (v) => (v >= 100 || Number.isInteger(v) ? Math.round(v).toLocaleString() : v >= 10 ? v.toFixed(1) : v === 0 ? "0" : String(+v.toFixed(v < 0.01 ? 4 : 3)));

function svgRoot(w, h, label) {
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", label);
  svg.style.width = "100%";
  svg.style.height = "auto";
  svg.style.fontFamily = "var(--sans)";
  return svg;
}

export function lineChart({ title, subtitle, series, xLog = false, yLog = false, yMin = 0, yMax, xLabel, yLabel, width = 640, height = 340 }) {
  const m = { t: 18, r: 22, b: 50, l: 58 };
  const pw = width - m.l - m.r, ph = height - m.t - m.b;
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => s.points.map((p) => p[1]));
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const lo = yLog ? 10 ** Math.floor(Math.log10(Math.min(...ys) * 0.8)) : yMin;
  const top = yLog ? 10 ** Math.ceil(Math.log10(Math.max(...ys) * 1.2)) : (yMax ?? niceCeil(Math.max(...ys) * 1.08));
  const sx = (x) => m.l + ((xLog ? Math.log10(x) - Math.log10(x0) : x - x0) / ((xLog ? Math.log10(x1) - Math.log10(x0) : x1 - x0) || 1)) * pw;
  const sy = (y) => m.t + ph - (yLog ? (Math.log10(y) - Math.log10(lo)) / (Math.log10(top) - Math.log10(lo)) : (y - lo) / (top - lo)) * ph;
  let g = "";
  const ticks = yLog ? Array.from({ length: Math.round(Math.log10(top) - Math.log10(lo)) + 1 }, (_, i) => lo * 10 ** i) : Array.from({ length: 5 }, (_, i) => lo + ((top - lo) * i) / 4);
  for (const v of ticks) {
    const y = sy(v);
    g += `<line x1="${m.l}" x2="${width - m.r}" y1="${y}" y2="${y}" stroke="var(--line)"/><text x="${m.l - 9}" y="${y + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${fmt(v)}</text>`;
  }
  [...new Set(xs)].sort((a, b) => a - b).forEach((x) => {
    g += `<text x="${sx(x)}" y="${height - m.b + 18}" text-anchor="middle" font-size="11" fill="var(--muted)">${fmt(x)}</text>`;
  });
  series.forEach((s, i) => {
    const color = s.color || PALETTE[i % PALETTE.length];
    const pts = [...s.points].sort((a, b) => a[0] - b[0]);
    g += `<polyline fill="none" stroke="${color}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round" points="${pts.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")}"/>`;
    pts.forEach((p) => (g += `<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="4" fill="var(--panel)" stroke="${color}" stroke-width="2.4"><title>${esc(s.name)}: ${fmt(p[1])} @ ${fmt(p[0])}</title></circle>`));
  });
  g += `<text x="${m.l + pw / 2}" y="${height - 8}" text-anchor="middle" font-size="12" fill="var(--muted)">${esc(xLabel || "")}</text>`;
  g += `<text transform="rotate(-90 15 ${m.t + ph / 2})" x="15" y="${m.t + ph / 2}" text-anchor="middle" font-size="12" fill="var(--muted)">${esc(yLabel || "")}</text>`;
  const svg = svgRoot(width, height, title);
  svg.innerHTML = g;
  return { svg, legend: series.map((s, i) => [s.name, s.color || PALETTE[i % PALETTE.length]]) };
}

export function barChart({ title, bars, max = 1, width = 640, rowHeight = 38, label = 190 }) {
  const m = { t: 8, r: 70, b: 30, l: label };
  const height = m.t + m.b + bars.length * rowHeight;
  const pw = width - m.l - m.r;
  let g = "";
  for (let i = 0; i <= 4; i++) {
    const x = m.l + (pw * i) / 4;
    g += `<line x1="${x}" x2="${x}" y1="${m.t}" y2="${height - m.b}" stroke="var(--line)"/><text x="${x}" y="${height - m.b + 17}" text-anchor="middle" font-size="11" fill="var(--muted)">${fmt((max * i) / 4)}</text>`;
  }
  bars.forEach((b, i) => {
    const y = m.t + i * rowHeight, w = Math.max(2, (b.value / max) * pw);
    const color = b.color || PALETTE[i % PALETTE.length];
    g += `<text x="${m.l - 10}" y="${y + rowHeight / 2 + 4}" text-anchor="end" font-size="12.5" fill="var(--text)" ${b.bold ? 'font-weight="700"' : ""}>${esc(b.label)}</text>`;
    g += `<rect x="${m.l}" y="${y + 6}" width="${w}" height="${rowHeight - 14}" rx="5" fill="${color}" ${b.dim ? 'opacity=".55"' : ""}/>`;
    g += `<text x="${m.l + w + 8}" y="${y + rowHeight / 2 + 4}" font-size="12.5" font-weight="700" fill="var(--text)">${fmt(b.value)}${esc(b.suffix || "")}</text>`;
  });
  const svg = svgRoot(width, height, title);
  svg.innerHTML = g;
  return { svg };
}
