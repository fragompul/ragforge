/* Corpus: edit documents, tune the chunker, and SEE where chunk boundaries land. */
import { h, esc, slider, select, toast, debounce } from "../ui.js";

const CHUNK_COLORS = ["c0", "c1", "c2"];

export async function mount(root, ctx) {
  let selected = ctx.docs[0]?.id;
  const chunkerCfg = { ...ctx.index.chunker };
  let analyzer = ctx.index.analyzer;
  let backend = ctx.index.backend;

  const list = h("div", { class: "doclist" });
  const filter = h("input", { type: "text", placeholder: "Filter documents…", "aria-label": "Filter documents" });
  filter.addEventListener("input", () => renderList());

  const idInput = h("input", { type: "text", "aria-label": "Document id" });
  const textArea = h("textarea", { rows: 8, "aria-label": "Document text", spellcheck: "false" });
  const metaArea = h("input", { type: "text", "aria-label": "Metadata JSON", placeholder: '{"topic": "retrieval"}' });
  const view = h("div", { class: "chunk-view" });
  const strip = h("div", { class: "chunk-strip" });
  const table = h("div");
  const summary = h("div", { class: "hint" });
  const fileInput = h("input", { type: "file", accept: ".txt,.md,.rst", multiple: true, hidden: true });

  const current = () => ctx.docs.find((d) => d.id === selected);

  function renderList() {
    const q = filter.value.toLowerCase();
    list.replaceChildren(...ctx.docs.filter((d) => !q || d.id.includes(q) || d.text.toLowerCase().includes(q)).map((d) =>
      h("div", { class: `doc-item${d.id === selected ? " active" : ""}`, role: "button", tabindex: 0, onclick: () => select_(d.id), onkeydown: (e) => e.key === "Enter" && select_(d.id) },
        h("span", { class: "docid" }, d.id), h("span", { class: "badge neutral" }, `${d.text.length}`))));
  }

  function select_(id) {
    selected = id;
    const d = current();
    idInput.value = d?.id ?? "";
    textArea.value = d?.text ?? "";
    metaArea.value = d?.metadata ? JSON.stringify(d.metadata) : "";
    renderList();
    renderChunks();
  }

  const renderChunks = debounce(async () => {
    const text = textArea.value;
    if (!text.trim()) { view.textContent = "(empty document)"; strip.replaceChildren(); table.replaceChildren(); summary.textContent = ""; return; }
    let chunks;
    try { chunks = await ctx.call("chunk_document", { text, id: idInput.value || "doc", chunker: chunkerCfg }); } catch (e) { view.textContent = e.message; return; }
    // Build the highlighted text: each char gets the set of chunks that cover it.
    const cover = new Array(text.length).fill(0);
    const owner = new Array(text.length).fill(-1);
    chunks.forEach((c, i) => { for (let p = c.start; p < Math.min(c.end, text.length); p++) { cover[p]++; owner[p] = i; } });
    let html = "", p = 0;
    while (p < text.length) {
      let q = p;
      while (q < text.length && cover[q] === cover[p] && owner[q] === owner[p]) q++;
      const piece = esc(text.slice(p, q));
      html += cover[p] === 0 ? piece : cover[p] > 1 ? `<span class="ov" title="overlap (${cover[p]} chunks)">${piece}</span>` : `<span class="${CHUNK_COLORS[owner[p] % 3]}">${piece}</span>`;
      p = q;
    }
    view.innerHTML = html;
    strip.replaceChildren(...chunks.map((c, i) => h("i", { title: `chunk ${i + 1}: ${c.start}–${c.end}`, style: { left: `${(c.start / text.length) * 100}%`, width: `${Math.max(((c.end - c.start) / text.length) * 100, 0.6)}%`, background: ["var(--ember)", "var(--cyan)", "var(--violet)"][i % 3] } })));
    const overlapChars = cover.filter((x) => x > 1).length;
    summary.textContent = `${chunks.length} chunk${chunks.length === 1 ? "" : "s"} · ${text.length} chars · ${overlapChars} chars (${((overlapChars / text.length) * 100).toFixed(0)}%) duplicated by overlap`;
    table.replaceChildren(h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "#"), h("th", { class: "num" }, "start"), h("th", { class: "num" }, "end"), h("th", { class: "num" }, "chars"), h("th", {}, "preview"))),
      h("tbody", {}, chunks.map((c, i) => h("tr", {}, h("td", {}, i + 1), h("td", { class: "num" }, c.start), h("td", { class: "num" }, c.end), h("td", { class: "num" }, c.end - c.start), h("td", { class: "hint" }, c.text.slice(0, 70) + (c.text.length > 70 ? "…" : "")))))));
  }, 120);

  [textArea, idInput].forEach((el) => el.addEventListener("input", renderChunks));

  async function apply() {
    let meta;
    try { meta = metaArea.value.trim() ? JSON.parse(metaArea.value) : undefined; } catch { return toast("Metadata must be valid JSON"); }
    const id = idInput.value.trim();
    if (!id) return toast("Document id is required");
    const docs = [...ctx.docs];
    const at = docs.findIndex((d) => d.id === selected);
    const doc = { id, text: textArea.value, ...(meta ? { metadata: meta } : {}) };
    if (docs.some((d, i) => d.id === id && i !== at)) return toast(`Id “${id}” already exists`);
    at >= 0 ? (docs[at] = doc) : docs.push(doc);
    selected = id;
    ctx.setDocs(docs);
    await reindex("Document saved & index rebuilt");
  }

  async function reindex(message) {
    ctx.index = { chunker: { ...chunkerCfg }, analyzer, backend };
    ctx.store.set("index", ctx.index);
    try {
      const info = await ctx.rebuild();
      toast(`${message} — ${info.documents} docs, ${info.chunks} chunks`);
    } catch (e) { toast(`Index error: ${e.message}`); }
    renderList();
  }

  const sizeCtl = slider({ label: "Chunk size (chars)", min: 80, max: 1500, step: 10, value: chunkerCfg.size, onInput: (v) => { chunkerCfg.size = v; renderChunks(); } });
  const overlapCtl = slider({ label: "Overlap", min: 0, max: 300, step: 5, value: chunkerCfg.overlap, onInput: (v) => { chunkerCfg.overlap = v; renderChunks(); } });
  const typeCtl = select({ label: "Chunker", options: [["recursive", "Recursive (paragraph → sentence → word)"], ["sentence", "Sentence-aware"], ["markdown", "Markdown headers"], ["fixed", "Fixed word windows"]], value: chunkerCfg.type, onChange: (v) => { chunkerCfg.type = v; renderChunks(); } });
  const anaCtl = select({ label: "BM25 analyzer", options: [["english", "English (stopwords + stemming)"], ["simple", "Simple (lowercase tokens)"]], value: analyzer, onChange: (v) => (analyzer = v) });
  const backCtl = select({ label: "Vector backend", options: [["brute_force", "Exact scan"], ["hnsw", "HNSW (approximate)"]], value: backend, onChange: (v) => (backend = v) });

  fileInput.addEventListener("change", async () => {
    const docs = [...ctx.docs];
    for (const f of fileInput.files) {
      const id = f.name.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_-]+/g, "-").toLowerCase();
      const text = await f.text();
      const at = docs.findIndex((d) => d.id === id);
      const doc = { id, text: text.slice(0, 200000), metadata: { source: f.name } };
      at >= 0 ? (docs[at] = doc) : docs.push(doc);
      selected = id;
    }
    fileInput.value = "";
    ctx.setDocs(docs);
    await reindex("Files imported");
    select_(selected);
  });

  root.append(
    h("div", { class: "hero" }, h("h1", {}, "Your corpus, ", h("span", { class: "grad-text" }, "under the microscope")),
      h("p", {}, "Edit the knowledge base, drop in your own .txt / .md files (they never leave your browser) and watch how the chunker slices them. Overlap is hatched in amber.")),
    h("div", { class: "split wide-left" },
      h("div", { class: "stack" },
        h("div", { class: "card" }, h("div", { class: "card-h" }, h("h3", {}, "Documents"), h("span", { class: "badge neutral" }, `${ctx.docs.length}`)),
          h("div", { class: "card-b" }, filter, list,
            h("div", { style: { display: "flex", gap: "8px", flexWrap: "wrap", marginTop: "12px" } },
              h("button", { class: "btn small", onclick: () => { selected = undefined; idInput.value = `doc-${ctx.docs.length + 1}`; textArea.value = ""; metaArea.value = ""; textArea.focus(); renderList(); renderChunks(); } }, "+ New"),
              h("button", { class: "btn small", onclick: () => fileInput.click() }, "Upload files"),
              fileInput,
              h("button", { class: "btn small danger", onclick: async () => { ctx.setDocs(ctx.sample.documents); selected = ctx.docs[0].id; await reindex("Sample handbook restored"); select_(selected); } }, "Reset sample")))),
        h("div", { class: "card" }, h("div", { class: "card-h" }, h("h3", {}, "Indexing")),
          h("div", { class: "card-b" }, typeCtl.el, sizeCtl.el, overlapCtl.el, anaCtl.el, backCtl.el,
            h("button", { class: "btn primary", style: { width: "100%" }, onclick: () => reindex("Index rebuilt") }, "Rebuild index")))),
      h("div", { class: "stack" },
        h("div", { class: "card" }, h("div", { class: "card-h" }, h("h3", {}, "Editor")),
          h("div", { class: "card-b" }, h("div", { class: "grid cols-2" }, h("div", { class: "field" }, h("label", {}, "Id"), idInput), h("div", { class: "field" }, h("label", {}, "Metadata (JSON)"), metaArea)),
            h("div", { class: "field" }, h("label", {}, "Text"), textArea),
            h("div", { style: { display: "flex", gap: "8px", flexWrap: "wrap" } },
              h("button", { class: "btn primary", onclick: apply }, "Save & re-index"),
              h("button", { class: "btn danger", onclick: async () => { ctx.setDocs(ctx.docs.filter((d) => d.id !== selected)); selected = ctx.docs[0]?.id; await reindex("Document deleted"); select_(selected); } }, "Delete")))),
        h("div", { class: "card" }, h("div", { class: "card-h" }, h("h3", {}, "Chunk map"), summary),
          h("div", { class: "card-b" }, view, strip, h("div", { style: { height: "14px" } }), table)))));

  select_(selected);
  return { onShow() { renderList(); } };
}
