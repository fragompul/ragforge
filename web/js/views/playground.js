/* Playground: query the index and compare BM25 / vector / hybrid side by side. */
import { h, $, highlight, queryTerms, fmt, ms, debounce, slider, segmented, select, meter, toggle, toast, COLORS } from "../ui.js";

const EXTRA_QUERIES = [
  "How do I protect a RAG system from prompt injection?",
  "What is the standard shipping time?",
  "How does HNSW trade recall for speed?",
];

export async function mount(root, ctx) {
  const relevant = new Map(ctx.sample.eval_cases.map((c) => [c.query.toLowerCase(), new Set(c.relevant_doc_ids)]));
  const examples = [...new Set([...ctx.sample.eval_cases.map((c) => c.query).slice(0, 5), ...EXTRA_QUERIES])];

  const cfg = { mode: "hybrid", k: 3, k_rrf: 60, w_bm25: 1, w_vec: 1, reranker: "none", mmr_lambda: 0.7, guardrails: "off", router: false };
  let lastQuery = "";

  const input = h("input", { type: "text", placeholder: "Ask the handbook anything…", "aria-label": "Search query", autocomplete: "off", spellcheck: "false" });
  const runBtn = h("button", { class: "btn primary", type: "button", onclick: () => run() }, "Search");
  input.addEventListener("keydown", (e) => e.key === "Enter" && run());

  const rerun = debounce(() => lastQuery && run(), 120);
  const modeCtl = segmented({ label: "Final pipeline retrieval", options: [["hybrid", "Hybrid"], ["bm25", "BM25"], ["vector", "Vector"]], value: cfg.mode, onChange: (v) => { cfg.mode = v; rerun(); } });
  const kCtl = slider({ label: "Top-k", min: 1, max: 8, value: cfg.k, onInput: (v) => { cfg.k = v; rerun(); } });
  const rrfCtl = slider({ label: "RRF constant k", min: 1, max: 120, value: cfg.k_rrf, onInput: (v) => { cfg.k_rrf = v; rerun(); }, hint: "Smaller k lets top ranks dominate the fusion." });
  const wbCtl = slider({ label: "BM25 weight", min: 0, max: 3, step: 0.1, value: 1, format: (v) => v.toFixed(1), onInput: (v) => { cfg.w_bm25 = v; rerun(); } });
  const wvCtl = slider({ label: "Vector weight", min: 0, max: 3, step: 0.1, value: 1, format: (v) => v.toFixed(1), onInput: (v) => { cfg.w_vec = v; rerun(); } });
  const rrCtl = segmented({ label: "Reranker", options: [["none", "None"], ["mmr", "MMR diversity"]], value: cfg.reranker, onChange: (v) => { cfg.reranker = v; lamWrap.hidden = v !== "mmr"; rerun(); } });
  const lamCtl = slider({ label: "MMR λ (relevance ↔ diversity)", min: 0, max: 1, step: 0.05, value: cfg.mmr_lambda, format: (v) => v.toFixed(2), onInput: (v) => { cfg.mmr_lambda = v; rerun(); } });
  const lamWrap = h("div", { hidden: true }, lamCtl.el);
  const gdCtl = select({ label: "Guardrails on retrieved context", options: [["off", "Off"], ["flag", "Flag only"], ["sanitize", "Sanitize"], ["drop", "Drop poisoned"]], value: cfg.guardrails, onChange: (v) => { cfg.guardrails = v; rerun(); } });

  const info = h("div", { class: "hint" });
  const refreshInfo = () => (info.textContent = `${ctx.info.documents} documents · ${ctx.info.chunks} chunks · ${ctx.info.backend} vectors · ${ctx.info.embedding} embeddings`);
  const modelStatus = h("div", { class: "hint" });
  const embCtl = segmented({ label: "Vector embeddings", options: [["hashing", "Hashing (instant)"], ["semantic", "MiniLM (semantic)"]], value: ctx.embedding, onChange: async (v) => {
    const label = v === "semantic" ? "Loading MiniLM into your browser (~23 MB, cached after the first time)…" : "Switching embeddings…";
    modelStatus.textContent = label;
    try {
      await ctx.setEmbedding(v, ({ loaded, total }) => { if (total) modelStatus.textContent = `Downloading MiniLM… ${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`; });
      modelStatus.textContent = v === "semantic" ? "all-MiniLM-L6-v2 (q8) is running in your browser via transformers.js — real semantic vectors." : "Hashing embeddings: deterministic bag-of-words, no model.";
      lastQuery && run();
    } catch (err) {
      embCtl.set("hashing");
      ctx.embedding = "hashing";
      modelStatus.textContent = `Could not load the model (${err.message}). Staying on hashing embeddings.`;
      toast("Semantic mode unavailable — are you offline?");
      try { await ctx.call("set_embedding", {}); } catch { /* ignore */ }
    }
  } });
  modelStatus.textContent = "Hashing embeddings: deterministic bag-of-words, no model. Try MiniLM to see real semantic matching.";
  const routerCtl = toggle({ label: "Query router (per-query fusion weights)", checked: false, onChange: (v) => { cfg.router = v; rerun(); } });
  const graphCtl = toggle({ label: "Entity-graph source (3-way fusion)", checked: !!ctx.index.graph, onChange: async (v) => { ctx.index.graph = v; ctx.store.set("index", ctx.index); await ctx.rebuild(); lastQuery && run(); } });
  const contextCtl = toggle({ label: "Contextual retrieval (title context)", checked: ctx.index.contextual === "title", onChange: async (v) => { ctx.index.contextual = v ? "title" : "none"; ctx.store.set("index", ctx.index); await ctx.rebuild(); lastQuery && run(); } });
  refreshInfo();
  ctx.onIndexChange(() => { refreshInfo(); lastQuery && run(); });

  const controls = h("aside", { class: "card" },
    h("div", { class: "card-h" }, h("h3", {}, "Pipeline")),
    h("div", { class: "card-b" }, modeCtl.el, kCtl.el, rrfCtl.el, wbCtl.el, wvCtl.el, rrCtl.el, lamWrap, gdCtl.el, h("hr", { class: "sep" }), embCtl.el, modelStatus, h("div", { style: { height: "10px" } }), graphCtl.el, contextCtl.el, routerCtl.el, info));

  const compare = h("div", { class: "grid cols-3" });
  const output = h("div", { class: "grid cols-2" });
  const results = h("div", { class: "stack" }, compare, output);

  // Hero with live headline numbers from the evaluation artefacts.
  const stats = h("div", { class: "hero-stats" },
    h("div", { class: "stat" }, h("b", {}, "0"), h("span", {}, "required dependencies")),
    h("div", { class: "stat" }, h("b", {}, "100%"), h("span", {}, "runs in your browser")));
  fetch("data/eval_scifact.json").then((r) => r.json()).then((d) => {
    const hybrid = d.systems.find((s) => s.system.startsWith("Hybrid RRF (BM25"));
    const best = Math.max(...d.systems.filter((s) => !s.system.startsWith("Hybrid")).map((s) => s.ndcg10 ?? s["ndcg@10"]));
    stats.append(h("div", { class: "stat" }, h("b", { class: "grad-text" }, fmt(hybrid["ndcg@10"], 3)), h("span", {}, `nDCG@10 on SciFact (best single retriever ${fmt(best, 3)})`)));
    const ann = d.ann?.find((a) => a.ef_search === 50);
    if (ann) stats.append(h("div", { class: "stat" }, h("b", { class: "grad-text" }, fmt(ann["recall@10"], 3)), h("span", {}, "HNSW recall@10 on real embeddings")));
  }).catch(() => {});

  root.append(
    h("div", { class: "hero" },
      h("h1", {}, "Hybrid retrieval, ", h("span", { class: "grad-text" }, "forged from scratch.")),
      h("p", {}, "You are running the real ", h("code", {}, "ragforge"), " Python library — BM25, vectors, Reciprocal Rank Fusion, reranking, guardrails and citations — compiled to WebAssembly in this tab. No server, no API keys."),
      stats),
    h("div", { class: "searchbar" }, input, runBtn),
    h("div", { class: "chips" }, examples.map((q) => h("button", { class: "chip", type: "button", onclick: () => { input.value = q; run(); } }, q))),
    h("div", { style: { height: "20px" } }),
    h("div", { class: "split" }, controls, results));

  compare.append(h("div", { class: "card empty", style: { gridColumn: "1 / -1" } }, "Pick an example above or type a question to see how each retriever ranks the handbook."));

  function hitRow(hit, col, terms, known, maxScore) {
    const isRel = known && known.has(hit.doc_id);
    const tags = h("div", { class: "tags" });
    if (col === "hybrid") {
      for (const [src, rank] of Object.entries(hit.ranks || {})) tags.append(h("span", { class: `badge ${src === "graph" ? "good" : src}` }, `${src} #${rank}`));
    }
    return h("div", { class: `hit${isRel ? " relevant" : ""}`, onclick: (e) => e.currentTarget.classList.toggle("expanded") },
      h("div", { class: "hit-h" },
        h("span", { class: "rank" }, hit.rank), h("span", { class: "docid" }, hit.doc_id),
        isRel ? h("span", { class: "badge good", title: "Known relevant document for this query" }, "★ relevant") : null,
        h("span", { class: "score-bar" }, h("i", { style: { width: `${Math.max(4, (hit.score / (maxScore || 1)) * 100)}%` } })),
        h("span", { class: "score-num" }, col === "hybrid" ? hit.score.toFixed(4) : hit.score.toFixed(3))),
      tags,
      h("div", { class: "snippet", html: highlight(hit.text, terms) }));
  }

  function column(name, label, color, hits, terms, known) {
    const max = Math.max(...hits.map((x) => x.score), 1e-9);
    return h("div", { class: "card" },
      h("div", { class: "card-b" },
        h("div", { class: "col-title" }, h("span", { class: "swatch", style: { background: color } }), label, h("small", {}, `${hits.length} results`)),
        hits.length ? hits.map((x) => hitRow(x, name, terms, known, max)) : h("div", { class: "empty" }, "No matches")));
  }

  function renderAnswer(data, terms) {
    const cit = data.citations;
    const body = h("div", { class: "answer" });
    if (!cit.sentences.length) body.append(data.answer);
    cit.sentences.forEach((s) => {
      const span = h("span", { class: `sent ${s.supported ? "ok" : "no"}`, title: s.supported ? `support ${s.support.toFixed(2)}` : "No supporting span found" }, s.text);
      s.citations.forEach((c) => span.append(h("sup", { class: "cite" }, c.marker)));
      body.append(span, " ");
    });
    const g = cit.groundedness;
    const sources = h("div", { class: "sources" });
    const seen = new Set();
    cit.sentences.flatMap((s) => s.citations).forEach((c) => {
      if (seen.has(c.chunk_id + c.start_char)) return;
      seen.add(c.chunk_id + c.start_char);
      sources.append(h("div", { class: "source" }, h("b", {}, `[${c.marker}]`), h("div", {}, h("div", {}, h("span", { class: "docid" }, c.doc_id), h("span", { class: "hint" }, `  chars ${c.start_char}–${c.end_char}`)), h("q", {}, c.quote))));
    });
    return h("div", { class: "card" },
      h("div", { class: "card-h" }, h("h3", {}, "Answer · extractive, with verifiable citations"), h("span", { class: `badge ${g >= 0.99 ? "good" : g > 0 ? "warn" : "bad"}` }, `groundedness ${(g * 100).toFixed(0)}%`)),
      h("div", { class: "card-b" }, body, h("div", { style: { height: "12px" } }), meter(g, g >= 0.99 ? "var(--mint)" : g > 0 ? "var(--amber)" : "var(--rose)"), sources,
        h("div", { class: "hint", style: { marginTop: "10px" } }, "Swap in any LLM via generate_fn; the same attribution verifies its sentences against the retrieved spans.")));
  }

  function renderFinal(data, terms, known) {
    const wrap = h("div", { class: "card" }, h("div", { class: "card-h" }, h("h3", {}, "Context sent to the LLM"), h("span", { class: "badge neutral" }, `${cfg.mode} · ${cfg.reranker === "mmr" ? "MMR" : "no rerank"}`)));
    const body = h("div", { class: "card-b" });
    if (!data.final.length) body.append(h("div", { class: "empty" }, "No context retrieved."));
    data.final.forEach((c, i) => {
      const flagged = c.guardrails && c.guardrails.findings;
      const row = h("div", { class: "hit" + (known && known.has(c.doc_id) ? " relevant" : ""), onclick: (e) => e.currentTarget.classList.toggle("expanded") },
        h("div", { class: "hit-h" }, h("span", { class: "rank" }, i + 1), h("span", { class: "docid" }, c.doc_id), c.provenance ? h("span", { class: "badge neutral" }, c.provenance.replace("rerank:", "")) : null,
          flagged ? c.guardrails.findings.slice(0, 2).map((f) => h("span", { class: `badge ${f.kind === "injection" ? "bad" : "warn"}` }, `${f.kind}: ${f.rule}`)) : null),
        h("div", { class: "snippet", html: highlight(c.text, terms) }));
      body.append(row);
    });
    body.append(h("details", { style: { marginTop: "12px" } }, h("summary", {}, "Prompt preview"), h("pre", {}, data.prompt || "(no context)")));
    wrap.append(body);
    return wrap;
  }

  function renderTrace(trace, total) {
    const end = Math.max(...trace.map((s) => s.start_ms + s.duration_ms), 0.001);
    const depth = new Map();
    const byId = new Map(trace.map((s) => [s.id, s]));
    const dOf = (s) => (s.parent && byId.has(s.parent) ? 1 + (depth.get(s.parent) ?? dOf(byId.get(s.parent))) : 0);
    trace.forEach((s) => depth.set(s.id, dOf(s)));
    const rows = trace.map((s) => h("div", { class: "wf-row" },
      h("div", { class: "wf-name", style: { paddingLeft: `${depth.get(s.id) * 12}px` } }, s.name),
      h("div", { class: "wf-track" }, h("div", { class: "wf-bar", style: { left: `${(s.start_ms / end) * 100}%`, width: `${(s.duration_ms / end) * 100}%` } })),
      h("div", { class: "wf-ms" }, ms(s.duration_ms))));
    return h("div", { class: "card" }, h("div", { class: "card-h" }, h("h3", {}, "Pipeline trace"), h("span", { class: "badge neutral" }, `total ${ms(total)}`)),
      h("div", { class: "card-b" }, h("div", { class: "waterfall" }, rows), h("div", { class: "hint", style: { marginTop: "10px" } }, "Nested spans exported by ragforge.telemetry — the same data feeds OpenTelemetry.")));
  }

  let seq = 0;
  async function run() {
    const query = input.value.trim();
    if (!query) return;
    lastQuery = query;
    const mine = ++seq;
    runBtn.disabled = true;
    try {
      const data = await ctx.call("search", { query, ...cfg, pool: 20 });
      if (mine !== seq) return;
      const terms = queryTerms(query);
      const known = relevant.get(query.toLowerCase());
      const cols = [
        column("bm25", "BM25 · keywords", COLORS.bm25, data.columns.bm25, terms, known),
        column("vector", data.embedding === "semantic" ? "Vector · MiniLM (semantic)" : "Vector · hashing", COLORS.vector, data.columns.vector, terms, known),
      ];
      if (data.columns.graph) cols.push(column("graph", "Entity graph", "var(--mint)", data.columns.graph, terms, known));
      cols.push(column("hybrid", `Hybrid · RRF${data.columns.graph ? " (3-way)" : ""}`, COLORS.hybrid, data.columns.hybrid, terms, known));
      compare.className = cols.length === 4 ? "grid cols-2" : "grid cols-3";
      compare.replaceChildren(...cols);
      const left = h("div", { class: "stack" }, renderAnswer(data, terms), renderTrace(data.trace, data.latency_ms));
      output.replaceChildren(left, renderFinal(data, terms, known));
    } catch (err) {
      compare.replaceChildren(h("div", { class: "card callout", style: { gridColumn: "1 / -1" } }, h("span", {}, h("b", {}, "Search failed. "), err.message)));
    } finally {
      runBtn.disabled = false;
    }
  }

  return {
    onShow() { input.focus({ preventScroll: true }); },
    setQuery(q) { input.value = q; run(); },
  };
}
