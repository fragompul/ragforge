/* HNSW Explorer: build the real graph on 2-D points, then watch a query navigate it. */
import { h, slider, toggle, segmented, toast } from "../ui.js";

const W = 800, H = 680;

export async function mount(root, ctx) {
  const cfg = { n: 500, clusters: 7, m: 6, ef_construction: 40, heuristic: true, seed: 7 };
  const q = { k: 5, ef: 24, speed: 3 };
  let graph = null;
  let adjacency = [];          // per level: Map(node -> Set(neighbors))
  let layerView = "all";
  let query = null;            // {x, y}
  let result = null;           // last hnsw_search payload
  let reveal = 0;              // number of visited events revealed
  let events = [];             // flattened visited events [{level, node}]
  let raf = 0;
  let theme = {};

  const canvas = h("canvas", { width: W * 2, height: H * 2, "aria-label": "HNSW graph visualization. Click to place a query." });
  const ctx2d = canvas.getContext("2d");
  const stats = h("div", { class: "stack" });
  const layerCtl = h("div");
  const bench = h("div", { class: "hint" });

  function readTheme() {
    const s = getComputedStyle(document.documentElement);
    const v = (n) => s.getPropertyValue(n).trim();
    theme = { bg: v("--bg-2"), line: v("--line-2"), text: v("--text"), muted: v("--muted"), ember: v("--ember"), amber: v("--amber"), cyan: v("--cyan"), violet: v("--violet"), mint: v("--mint"), rose: v("--rose") };
  }

  const toPx = (p) => [W / 2 + p[0] * (W / 2) * 0.93, H / 2 - p[1] * (H / 2) * 0.93];
  const fromPx = (x, y) => [((x - W / 2) / (W / 2)) / 0.93, -((y - H / 2) / (H / 2)) / 0.93];

  async function build() {
    cancelAnimationFrame(raf);
    result = null; events = [];
    graph = await ctx.call("hnsw_build", cfg);
    adjacency = Object.entries(graph.edges).map(([lvl, list]) => {
      const m = new Map();
      list.forEach(([a, b]) => { (m.get(a) ?? m.set(a, new Set()).get(a)).add(b); (m.get(b) ?? m.set(b, new Set()).get(b)).add(a); });
      return [Number(lvl), m];
    }).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
    buildLayerControl();
    if (query) await search(query); else { draw(); renderStats(); }
  }

  function buildLayerControl() {
    const options = [["all", "All"], ...Array.from({ length: graph.max_level + 1 }, (_, i) => [String(i), `L${i}`])];
    if (layerView !== "all" && Number(layerView) > graph.max_level) layerView = "all";
    const ctl = segmented({ label: "Layer", options, value: layerView, onChange: (v) => { layerView = v; draw(); } });
    layerCtl.replaceChildren(ctl.el);
  }

  async function search(point) {
    query = point;
    cancelAnimationFrame(raf);
    result = await ctx.call("hnsw_search", { x: point.x, y: point.y, k: q.k, ef: q.ef });
    events = result.path.flatMap((step) => step.visited.map((node) => ({ level: step.level, node })));
    reveal = 0;
    renderStats();
    animate();
  }

  function animate() {
    reveal = Math.min(events.length, reveal + Math.max(1, q.speed));
    draw();
    if (reveal < events.length) raf = requestAnimationFrame(animate);
  }

  function levelColor(l) { return [theme.muted, theme.violet, theme.amber, theme.ember, theme.rose][Math.min(l, 4)]; }

  function draw() {
    if (!graph) return;
    readTheme();
    const c = ctx2d;
    c.setTransform(2, 0, 0, 2, 0, 0);
    c.clearRect(0, 0, W, H);
    // domain ring
    c.strokeStyle = theme.line; c.globalAlpha = 0.5; c.setLineDash([4, 6]);
    c.beginPath(); c.arc(W / 2, H / 2, (W / 2) * 0.93 * 0.96, 0, Math.PI * 2); c.stroke(); c.setLineDash([]); c.globalAlpha = 1;

    const showLevel = layerView === "all" ? null : Number(layerView);
    const edgeLevels = showLevel === null ? [...Array(graph.max_level + 1).keys()] : [showLevel];
    for (const lvl of edgeLevels) {
      const color = levelColor(lvl);
      c.strokeStyle = color;
      c.globalAlpha = showLevel === null ? (lvl === 0 ? 0.13 : 0.45) : 0.7;
      c.lineWidth = showLevel === null ? (lvl === 0 ? 0.7 : 1.4) : 1.3;
      c.beginPath();
      for (const [a, b] of graph.edges[String(lvl)] || []) {
        const [x1, y1] = toPx(graph.points[a]), [x2, y2] = toPx(graph.points[b]);
        c.moveTo(x1, y1); c.lineTo(x2, y2);
      }
      c.stroke();
    }
    c.globalAlpha = 1;

    // traversal tree (revealed part)
    const seen = new Set();
    const shown = events.slice(0, reveal);
    c.lineWidth = 1.8;
    shown.forEach((e) => {
      const adj = adjacency[e.level];
      c.strokeStyle = theme.cyan; c.globalAlpha = 0.85;
      if (adj && adj.has(e.node)) {
        for (const nb of adj.get(e.node)) {
          if (seen.has(`${e.level}:${nb}`)) {
            const [x1, y1] = toPx(graph.points[e.node]), [x2, y2] = toPx(graph.points[nb]);
            c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
            break;
          }
        }
      }
      seen.add(`${e.level}:${e.node}`);
    });
    c.globalAlpha = 1;

    // nodes
    const visitedNodes = new Set(shown.map((e) => e.node));
    graph.points.forEach((p, i) => {
      const lvl = graph.levels[i];
      const dim = showLevel !== null && lvl < showLevel;
      const [x, y] = toPx(p);
      const r = 1.9 + lvl * 1.35;
      c.globalAlpha = dim ? 0.18 : 1;
      c.fillStyle = visitedNodes.has(i) ? theme.cyan : levelColor(lvl);
      c.beginPath(); c.arc(x, y, visitedNodes.has(i) ? r + 1.1 : r, 0, Math.PI * 2); c.fill();
    });
    c.globalAlpha = 1;

    // entry point
    if (graph.entry !== null) {
      const [x, y] = toPx(graph.points[graph.entry]);
      c.strokeStyle = theme.rose; c.lineWidth = 2; c.beginPath(); c.arc(x, y, 9, 0, Math.PI * 2); c.stroke();
    }

    // results once fully revealed
    if (result && reveal >= events.length) {
      const found = new Set(result.results), exact = new Set(result.exact);
      exact.forEach((i) => {
        const [x, y] = toPx(graph.points[i]);
        c.strokeStyle = found.has(i) ? theme.mint : theme.rose; c.lineWidth = 2.2;
        c.setLineDash(found.has(i) ? [] : [3, 3]);
        c.beginPath(); c.arc(x, y, 8.5, 0, Math.PI * 2); c.stroke(); c.setLineDash([]);
      });
      found.forEach((i) => {
        const [x, y] = toPx(graph.points[i]);
        c.fillStyle = exact.has(i) ? theme.mint : theme.amber;
        c.beginPath(); c.arc(x, y, 5, 0, Math.PI * 2); c.fill();
      });
    }

    // query marker
    if (query) {
      const [x, y] = toPx([query.x, query.y]);
      c.strokeStyle = theme.text; c.lineWidth = 2;
      c.beginPath(); c.moveTo(x - 9, y); c.lineTo(x + 9, y); c.moveTo(x, y - 9); c.lineTo(x, y + 9); c.stroke();
      c.fillStyle = theme.ember; c.beginPath(); c.arc(x, y, 3.2, 0, Math.PI * 2); c.fill();
    }
  }

  function renderStats() {
    if (!result) {
      stats.replaceChildren(h("div", { class: "callout" }, h("span", {}, h("b", {}, "Click anywhere on the canvas "), "to drop a query point. HNSW enters at the top layer, hops greedily toward the query, and descends layer by layer.")));
      return;
    }
    const pct = (result.visited / result.n) * 100;
    const perLayer = result.path.map((s) => h("tr", {}, h("td", {}, `L${s.level}`), h("td", { class: "num" }, s.visited.length), h("td", {}, h("div", { class: "meter-track" }, h("div", { class: "meter-fill", style: { width: `${Math.min(100, (s.visited.length / result.n) * 100 * 4)}%`, background: levelColor(s.level) } })))));
    stats.replaceChildren(
      h("div", { class: "kpis" },
        h("div", { class: "kpi" }, h("small", {}, "Nodes visited"), h("b", { class: "grad-text" }, result.visited), h("span", { class: "sub" }, `of ${result.n} (${pct.toFixed(1)}%)`)),
        h("div", { class: "kpi" }, h("small", {}, `Recall@${q.k}`), h("b", { style: { color: result.recall >= 1 ? "var(--mint)" : "var(--amber)" } }, result.recall.toFixed(2)), h("span", { class: "sub" }, "vs. exact scan")),
        h("div", { class: "kpi" }, h("small", {}, "Saved work"), h("b", {}, `${(result.n / Math.max(result.visited, 1)).toFixed(1)}×`), h("span", { class: "sub" }, "fewer distance evals"))),
      h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Layer"), h("th", { class: "num" }, "visited"), h("th", {}, ""))), h("tbody", {}, perLayer)));
  }

  async function benchmark() {
    bench.textContent = "Running 60 random queries…";
    let recall = 0, visited = 0;
    for (let i = 0; i < 60; i++) {
      const a = Math.random() * Math.PI * 2, r = Math.sqrt(Math.random()) * 0.85;
      const res = await ctx.call("hnsw_search", { x: r * Math.cos(a), y: r * Math.sin(a), k: q.k, ef: q.ef });
      recall += res.recall; visited += res.visited;
    }
    bench.textContent = `Mean recall@${q.k}: ${(recall / 60).toFixed(3)} · mean nodes visited: ${(visited / 60).toFixed(0)} of ${cfg.n} (${((visited / 60 / cfg.n) * 100).toFixed(1)}%)`;
  }

  const rebuild = async () => { try { await build(); } catch (e) { toast(e.message); } };
  const controls = h("div", { class: "card" },
    h("div", { class: "card-h" }, h("h3", {}, "Graph")),
    h("div", { class: "card-b" },
      slider({ label: "Points", min: 100, max: 1500, step: 50, value: cfg.n, onInput: (v) => (cfg.n = v) }).el,
      slider({ label: "Clusters", min: 1, max: 14, value: cfg.clusters, onInput: (v) => (cfg.clusters = v) }).el,
      slider({ label: "M (links per node)", min: 2, max: 16, value: cfg.m, onInput: (v) => (cfg.m = v), hint: "Layer 0 keeps up to 2·M." }).el,
      slider({ label: "ef_construction", min: 4, max: 200, value: cfg.ef_construction, onInput: (v) => (cfg.ef_construction = v) }).el,
      toggle({ label: "Diversity heuristic (paper's Algorithm 4)", checked: cfg.heuristic, onChange: (v) => (cfg.heuristic = v) }).el,
      h("button", { class: "btn primary", style: { width: "100%" }, onclick: rebuild }, "Rebuild graph")));

  const searchCtl = h("div", { class: "card" },
    h("div", { class: "card-h" }, h("h3", {}, "Query")),
    h("div", { class: "card-b" },
      slider({ label: "k (neighbors wanted)", min: 1, max: 12, value: q.k, onInput: (v) => { q.k = v; query && search(query); } }).el,
      slider({ label: "ef_search (beam width)", min: 1, max: 100, value: q.ef, onInput: (v) => { q.ef = v; query && search(query); }, hint: "The recall ↔ latency dial. Try 1–5, then 50." }).el,
      slider({ label: "Animation speed", min: 1, max: 12, value: q.speed, onInput: (v) => (q.speed = v) }).el,
      layerCtl,
      h("div", { style: { display: "flex", gap: "8px", flexWrap: "wrap" } },
        h("button", { class: "btn small", onclick: () => { const a = Math.random() * 6.283, r = Math.sqrt(Math.random()) * 0.85; search({ x: r * Math.cos(a), y: r * Math.sin(a) }); } }, "Random query"),
        h("button", { class: "btn small", onclick: benchmark }, "Measure 60 queries")),
      bench));

  canvas.addEventListener("pointerdown", (e) => {
    const rect = canvas.getBoundingClientRect();
    const [x, y] = fromPx(((e.clientX - rect.left) / rect.width) * W, ((e.clientY - rect.top) / rect.height) * H);
    if (x * x + y * y < 1) search({ x, y });
  });
  window.addEventListener("themechange", draw);

  root.append(
    h("div", { class: "hero" }, h("h1", {}, "Watch ", h("span", { class: "grad-text" }, "HNSW"), " navigate a graph"),
      h("p", {}, "This is the library's from-scratch HNSW index running on 2-D points. Upper layers are express lanes; the query drops through them and refines on the dense bottom layer, touching a small fraction of the nodes. Turn the diversity heuristic off and shrink M to see recall fall apart.")),
    h("div", { class: "split wide-left" },
      h("div", { class: "stack" }, controls, searchCtl),
      h("div", { class: "stack" },
        h("div", { class: "canvas-wrap" }, canvas,
          h("div", { class: "canvas-legend" },
            h("span", {}, h("i", { style: { background: "var(--muted)" } }), "layer 0"), h("span", {}, h("i", { style: { background: "var(--violet)" } }), "layer 1"),
            h("span", {}, h("i", { style: { background: "var(--amber)" } }), "layer 2+"), h("span", {}, h("i", { style: { background: "var(--cyan)" } }), "visited"),
            h("span", {}, h("i", { style: { background: "var(--mint)" } }), "found & exact"), h("span", {}, h("i", { style: { background: "var(--rose)" } }), "missed neighbor / entry point"))),
        stats)));

  await build();
  query = { x: 0.1, y: 0.05 };
  await search(query);
  return { onShow() { draw(); } };
}
