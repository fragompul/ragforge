/* Knowledge Graph: entities, co-mentions, and how a query's activation diffuses through them. */
import { h, slider, ms } from "../ui.js";

const W = 880, H = 600;
const PRESETS = [
  "Where is the manufacturer of the ZX-9 sensor headquartered?",
  "Who founded Helio Dynamics?",
  "What does error ERR-4012 mean?",
  "How does HNSW relate to MMR?",
];

export async function mount(root, ctx) {
  let data = null, nodes = [], byKey = new Map(), hover = null, raf = 0, theme = {};
  const cfg = { max_entities: 70, min_df: 1 };
  const input = h("input", { type: "text", value: PRESETS[0], "aria-label": "Query", autocomplete: "off", spellcheck: "false" });
  input.addEventListener("keydown", (e) => e.key === "Enter" && refresh());
  const canvas = h("canvas", { width: W * 2, height: H * 2, "aria-label": "Entity co-mention graph" });
  const c = canvas.getContext("2d");
  const side = h("div", { class: "stack" });
  const notice = h("div");

  function readTheme() {
    const s = getComputedStyle(document.documentElement);
    const v = (n) => s.getPropertyValue(n).trim();
    theme = { line: v("--line-2"), text: v("--text"), muted: v("--muted"), ember: v("--ember"), cyan: v("--cyan"), mint: v("--mint"), bg: v("--panel"), amber: v("--amber") };
  }

  function layout() {
    const n = nodes.length;
    nodes.forEach((node, i) => {
      const a = (i / Math.max(n, 1)) * Math.PI * 2;
      node.x = W / 2 + Math.cos(a) * 220 * (0.6 + 0.4 * ((i * 7919) % 100) / 100);
      node.y = H / 2 + Math.sin(a) * 200 * (0.6 + 0.4 * ((i * 104729) % 100) / 100);
      node.vx = node.vy = 0;
    });
    const edges = data.edges.map((e) => [byKey.get(e.source), byKey.get(e.target), e.weight]);
    const k = Math.sqrt((W * H) / Math.max(n, 1)) * 0.62;
    for (let it = 0; it < 320; it++) {
      const cool = 1 - it / 320;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const a = nodes[i], b = nodes[j];
          let dx = a.x - b.x, dy = a.y - b.y;
          const d2 = dx * dx + dy * dy + 0.01, d = Math.sqrt(d2);
          const f = (k * k) / d2;
          dx /= d; dy /= d;
          a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
        }
      }
      for (const [a, b, w] of edges) {
        if (!a || !b) continue;
        const dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 0.01;
        const f = (d * d) / k / 60 * (1 + Math.log(w));
        a.vx += (dx / d) * f; a.vy += (dy / d) * f; b.vx -= (dx / d) * f; b.vy -= (dy / d) * f;
      }
      for (const node of nodes) {
        node.vx += (W / 2 - node.x) * 0.02; node.vy += (H / 2 - node.y) * 0.02;
        const speed = Math.hypot(node.vx, node.vy) + 0.01, cap = 22 * cool + 1.5;
        node.x += (node.vx / speed) * Math.min(speed, cap); node.y += (node.vy / speed) * Math.min(speed, cap);
        node.vx = node.vy = 0;
        node.x = Math.max(40, Math.min(W - 40, node.x)); node.y = Math.max(30, Math.min(H - 30, node.y));
      }
    }
  }

  function draw() {
    readTheme();
    c.setTransform(2, 0, 0, 2, 0, 0);
    c.clearRect(0, 0, W, H);
    const act = data?.explain?.activation || {};
    const maxAct = Math.max(1e-9, ...Object.values(act));
    const seeds = new Set((data?.explain?.seeds || []).map((s) => s.key));
    c.lineCap = "round";
    for (const e of data?.edges || []) {
      const a = byKey.get(e.source), b = byKey.get(e.target);
      if (!a || !b) continue;
      const lit = (act[e.source] || 0) > 0 && (act[e.target] || 0) > 0;
      c.strokeStyle = lit ? theme.ember : theme.line;
      c.globalAlpha = lit ? 0.5 + 0.4 * Math.min(1, (act[e.source] + act[e.target]) / maxAct / 2) : 0.35;
      c.lineWidth = lit ? 1.8 : 0.8 + Math.log(e.weight);
      c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
    }
    c.globalAlpha = 1;
    for (const node of nodes) {
      const a = (act[node.id] || 0) / maxAct;
      const r = 5 + Math.sqrt(node.degree) * 2.6 + a * 7;
      c.fillStyle = a > 0 ? mix(theme.cyan, theme.ember, a) : theme.muted;
      c.globalAlpha = a > 0 ? 1 : 0.55;
      c.beginPath(); c.arc(node.x, node.y, r, 0, Math.PI * 2); c.fill();
      c.globalAlpha = 1;
      if (seeds.has(node.id)) { c.strokeStyle = theme.ember; c.lineWidth = 2.5; c.beginPath(); c.arc(node.x, node.y, r + 5, 0, Math.PI * 2); c.stroke(); }
      if (hover === node || a > 0 || seeds.has(node.id) || node.degree >= 3 || nodes.length < 26) {
        c.font = '600 12px Inter, system-ui, sans-serif'; c.textAlign = "center";
        c.fillStyle = theme.text; c.globalAlpha = hover === node || a > 0 || seeds.has(node.id) ? 1 : 0.75;
        c.fillText(node.label, node.x, node.y - r - 6);
        c.globalAlpha = 1;
      }
    }
  }

  function mix(a, b, t) {
    const pa = parse(a), pb = parse(b);
    return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * Math.min(1, t))).join(",")})`;
  }
  function parse(color) {
    const el = parse.el || (parse.el = document.createElement("canvas").getContext("2d"));
    el.fillStyle = color; const hex = el.fillStyle;
    return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  }

  async function refresh() {
    if (!ctx.index.graph) return;
    const query = input.value.trim();
    data = await ctx.call("graph_view", { ...cfg, query });
    nodes = data.nodes.map((n) => ({ ...n }));
    byKey = new Map(nodes.map((n) => [n.id, n]));
    layout();
    draw();
    renderSide();
  }

  async function ensureGraph() {
    if (ctx.index.graph) return refresh();
    notice.replaceChildren(h("div", { class: "callout" }, h("span", {}, "Building the entity graph for the corpus…")));
    ctx.index.graph = true;
    ctx.store.set("index", ctx.index);
    await ctx.rebuild();
    notice.replaceChildren();
    await refresh();
  }

  function renderSide() {
    const ex = data.explain;
    const seeds = ex ? ex.seeds : [];
    const docs = ex ? ex.chunks.slice(0, 6) : [];
    const maxScore = Math.max(1e-9, ...docs.map((d) => d.score));
    side.replaceChildren(
      h("div", { class: "card" }, h("div", { class: "card-h" }, h("h3", {}, "Seeds & activation"), h("span", { class: "badge neutral" }, `${data.nodes.length} entities · ${data.edges.length} links`)),
        h("div", { class: "card-b" },
          seeds.length ? h("div", { class: "tags" }, seeds.map((s) => h("span", { class: "badge warn" }, `◎ ${s.entity}`))) : h("div", { class: "hint" }, ex ? "No entity in the query matches the graph — graph retrieval returns nothing (hybrid fusion simply ignores it)." : "Type a question to see which entities it activates."),
          h("div", { style: { height: "10px" } }),
          docs.length ? docs.map((d) => h("div", { class: "hit" }, h("div", { class: "hit-h" }, h("span", { class: "docid" }, d.doc_id), h("span", { class: "score-bar" }, h("i", { style: { width: `${Math.max(4, (d.score / maxScore) * 100)}%` } })), h("span", { class: "score-num" }, d.score.toFixed(3))))) : null,
          h("div", { class: "hint", style: { marginTop: "10px" } }, "Mass flows entity → document → entity for a few steps. A document that never mentions the query's entities can still rank if it shares one with a document that does — that is how the second hop of a multi-hop question is found."))));
  }

  canvas.addEventListener("pointermove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W, y = ((e.clientY - rect.top) / rect.height) * H;
    const near = nodes.find((n) => Math.hypot(n.x - x, n.y - y) < 12 + Math.sqrt(n.degree) * 3);
    if (near !== hover) { hover = near || null; canvas.style.cursor = near ? "pointer" : "default"; draw(); }
  });
  canvas.addEventListener("click", () => { if (hover) { input.value = hover.label; refresh(); } });
  window.addEventListener("themechange", () => data && draw());

  root.append(
    h("div", { class: "hero" }, h("h1", {}, "Retrieval that follows ", h("span", { class: "grad-text" }, "the links")),
      h("p", {}, "The library's ", h("code", {}, "GraphIndex"), " links every chunk to the entities it mentions. A query activates its entities and the activation diffuses through the graph — so the document about a company is found even though the question only names its product. Click a node to ask about it.")),
    h("div", { class: "searchbar" }, input, h("button", { class: "btn primary", onclick: refresh }, "Activate")),
    h("div", { class: "chips" }, PRESETS.map((q) => h("button", { class: "chip", onclick: () => { input.value = q; refresh(); } }, q))),
    h("div", { style: { height: "16px" } }), notice,
    h("div", { class: "split" },
      h("div", { class: "card" }, h("div", { class: "card-h" }, h("h3", {}, "Graph")), h("div", { class: "card-b" },
        slider({ label: "Entities shown", min: 15, max: 120, value: cfg.max_entities, onInput: (v) => (cfg.max_entities = v) }).el,
        slider({ label: "Min. mentions", min: 1, max: 4, value: cfg.min_df, onInput: (v) => (cfg.min_df = v) }).el,
        h("button", { class: "btn small", style: { width: "100%" }, onclick: refresh }, "Re-layout"),
        h("div", { class: "hint", style: { marginTop: "10px" } }, "Extraction is a deterministic heuristic (capitalized names, acronyms, identifiers). Plug an LLM extractor into GraphIndex(extract_fn=...) for better recall."))),
      h("div", { class: "stack" }, h("div", { class: "canvas-wrap" }, canvas,
        h("div", { class: "canvas-legend" }, h("span", {}, h("i", { style: { background: "var(--muted)" } }), "entity"), h("span", {}, h("i", { style: { background: "var(--cyan)" } }), "activated"), h("span", {}, h("i", { style: { background: "var(--ember)" } }), "strongly activated / query entity"))), side)));

  let first = true;
  return { onShow() { if (first) { first = false; ensureGraph(); } else if (ctx.index.graph) refresh(); } };
}
