/* App shell: boot sequence, shared state, tab routing, theming. */
import { startPython, onPy, call as rawCall } from "./py.js";
import { embedTexts, isLoaded, loadModel } from "./embed.js";
import { h, $, ICONS, toast } from "./ui.js";
import * as playground from "./views/playground.js";
import * as corpus from "./views/corpus.js";
import * as hnsw from "./views/hnsw.js";
import * as security from "./views/security.js";
import * as agent from "./views/agent.js";
import * as graph from "./views/graph.js";
import * as evaluation from "./views/evaluation.js";
import * as about from "./views/about.js";

const BUILD = document.querySelector('meta[name="ragforge-build"]')?.content || "dev";

const VIEWS = [
  { id: "playground", label: "Playground", icon: "search", mod: playground },
  { id: "corpus", label: "Corpus", icon: "docs", mod: corpus },
  { id: "hnsw", label: "HNSW Explorer", icon: "graph", mod: hnsw },
  { id: "security", label: "Security Lab", icon: "shield", mod: security },
  { id: "graph", label: "Knowledge Graph", icon: "network", mod: graph },
  { id: "agent", label: "Agent Lab", icon: "agent", mod: agent },
  { id: "evaluation", label: "Evaluation", icon: "chart", mod: evaluation },
  { id: "about", label: "About", icon: "info", mod: about },
];

const store = {
  get(key, fallback) { try { const v = localStorage.getItem(`ragforge.${key}`); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(`ragforge.${key}`, JSON.stringify(value)); } catch { /* storage unavailable */ } },
};

/** Shared context handed to every view. */
export const ctx = {
  /** Call a bridge function; transparently supplies semantic embeddings when the bridge asks for them. */
  async call(fn, arg = {}) {
    for (let attempt = 0; attempt < 6; attempt++) {
      const result = await rawCall(fn, arg);
      if (!result || !result.__need_embeddings) return result;
      const texts = result.__need_embeddings;
      const vectors = await embedTexts(texts);
      await rawCall("add_embeddings", { texts, vectors });
    }
    throw new Error("Could not resolve the embeddings this request needs");
  },
  embedding: "hashing",
  /** Switch the vector side between the instant hashing embedder and MiniLM running in the browser. */
  async setEmbedding(mode, onProgress) {
    if (mode === "semantic" && !isLoaded()) await loadModel(onProgress);
    await rawCall("set_embedding", { semantic: mode === "semantic" });
    ctx.embedding = mode;
    await ctx.rebuild();
  },
  store,
  toast,
  sample: null,
  docs: [],
  index: { chunker: { type: "recursive", size: 500, overlap: 50 }, analyzer: "english", backend: "brute_force", graph: false, contextual: "none" },
  info: null,
  listeners: new Set(),
  async rebuild() {
    ctx.info = await ctx.call("init", { docs: ctx.docs, ...ctx.index });
    ctx.listeners.forEach((fn) => fn());
    return ctx.info;
  },
  onIndexChange(fn) { ctx.listeners.add(fn); },
  /** Replace the corpus. Only *customized* corpora are remembered, so sample-corpus updates reach returning visitors. */
  setDocs(docs, persist = true) {
    ctx.docs = docs;
    if (!persist) return;
    const isSample = JSON.stringify(docs) === JSON.stringify(ctx.sample?.documents);
    try { isSample ? localStorage.removeItem("ragforge.customDocs") : store.set("customDocs", docs); } catch { /* storage unavailable */ }
  },
};

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = $("#theme-btn");
  if (btn) btn.innerHTML = theme === "dark" ? ICONS.sun : ICONS.moon;
}

function initTheme() {
  const saved = store.get("theme", null);
  const prefersLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
  applyTheme(saved || (prefersLight ? "light" : "dark"));
}

function buildShell() {
  const nav = $("#tabs");
  const main = $("#views");
  VIEWS.forEach((v) => {
    nav.append(h("button", { class: "tab", role: "tab", id: `tab-${v.id}`, "aria-selected": "false", onclick: () => (location.hash = v.id), html: `${ICONS[v.icon]}<span>${v.label}</span>` }));
    v.root = h("section", { class: "view", id: `view-${v.id}`, role: "tabpanel", "aria-labelledby": `tab-${v.id}` });
    main.append(v.root);
  });
  $("#theme-btn").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    store.set("theme", next);
    applyTheme(next);
    window.dispatchEvent(new Event("themechange"));
  });
}

function route() {
  const id = VIEWS.some((v) => v.id === location.hash.slice(1)) ? location.hash.slice(1) : "playground";
  VIEWS.forEach((v) => {
    const active = v.id === id;
    v.root.classList.toggle("active", active);
    $(`#tab-${v.id}`).setAttribute("aria-selected", String(active));
    if (active && v.ready && v.api.onShow) v.api.onShow();
  });
  document.title = `${VIEWS.find((v) => v.id === id).label} · RagForge`;
}

function setProgress(pct, msg) {
  $("#boot-bar").style.width = `${pct}%`;
  if (msg) $("#boot-msg").textContent = msg;
}

async function start() {
  initTheme();
  buildShell();
  $("#build-label").textContent = `build ${BUILD}`;

  onPy("progress", ({ pct, msg }) => setProgress(pct, msg));
  onPy("fatal", ({ message }) => {
    $("#status-dot").className = "dot error";
    $("#status-text").textContent = "Python failed to load";
    $("#boot-err").textContent = `Could not start the Python runtime: ${message}. Check your connection and reload.`;
    $("#boot-retry").hidden = false;
  });
  $("#boot-retry").addEventListener("click", () => location.reload());

  onPy("ready", async ({ version, python }) => {
    try {
      ctx.sample = await rawCall("sample");
      ctx.docs = store.get("customDocs", null) || ctx.sample.documents;
      ctx.index = { ...ctx.index, ...store.get("index", {}) };
      await ctx.rebuild();
      $("#status-dot").className = "dot ready";
      $("#status-text").textContent = `ragforge ${version} · Python ${python}`;
      for (const v of VIEWS) {
        v.api = await v.mod.mount(v.root, ctx);
        v.ready = true;
      }
      route();
      $("#boot").classList.add("done");
    } catch (err) {
      console.error(err);
      $("#boot-err").textContent = `Startup failed: ${err.message}`;
      $("#boot-retry").hidden = false;
    }
  });

  window.addEventListener("hashchange", route);
  startPython(BUILD);
}

start();
