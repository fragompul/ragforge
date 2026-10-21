/* About: what RagForge is, how it fits together, how to use it. */
import { h, ICONS, toast } from "../ui.js";

const FEATURES = [
  ["search", "Hybrid retrieval", "BM25 inverted index + dense vectors + optional entity graph and learned sparse (SPLADE), fused with weighted Reciprocal Rank Fusion and a per-query router."],
  ["graph", "HNSW from scratch", "Pure-Python approximate-nearest-neighbor graph: diversity heuristic, guaranteed reachability, in-graph metadata filtering, compaction, a visualizable search trace."],
  ["shield", "Guardrails", "Prompt-injection screening and PII redaction with measured detection rates on public datasets — and honest limits."],
  ["agent", "Agentic & multi-hop", "Decompose, retrieve, grade, rewrite, hop through bridge entities — then abstain instead of hallucinating."],
  ["docs", "Verifiable citations", "Sentence-to-span attribution with exact offsets, groundedness, and optional NLI to catch contradictions."],
  ["chart", "Evaluation", "nDCG / Recall / MRR on four BEIR datasets, HotpotQA multi-hop, LLM-as-judge, contextual-retrieval and quantization studies."],
  ["info", "MCP server", "Tools, resources and prompts over the Model Context Protocol; read-only by default; packageable for Claude Desktop."],
  ["network", "Entity-graph retrieval", "GraphRAG-lite: personalized-PageRank diffusion over a chunk/entity graph finds the second document of a two-hop question."],
  ["graph", "Quantization", "int8 (4×) and binary (32×) vectors with exact rescoring, measured on real embeddings."],
  ["docs", "Memory-mapped index", "A directory format with mmap vectors, incremental ingest, deletes, compaction and crash-safe commits."],
  ["shield", "Multi-tenancy", "Index-per-tenant or shared-index-with-enforced-filter isolation across retrieval, agents and MCP."],
  ["graph", "Zero dependencies", "The core is standard-library Python; async/ASGI, providers and NumPy are optional and lazy."],
];

const code = (text) => {
  const block = h("div", { class: "codeblock" }, h("button", { class: "icon-btn copy", title: "Copy", html: ICONS.copy, onclick: async () => { try { await navigator.clipboard.writeText(text); toast("Copied"); } catch { toast("Copy failed"); } } }), h("pre", {}, text));
  return block;
};

export async function mount(root) {
  const diagram = `<svg viewBox="0 0 900 250" role="img" aria-label="RagForge pipeline: chunk, index, retrieve, rerank, guard, generate, cite" style="width:100%;height:auto;font-family:var(--sans)">
    <defs><linearGradient id="gg" x1="0" x2="1"><stop offset="0" stop-color="#ffb020"/><stop offset="1" stop-color="#ff6b2c"/></linearGradient></defs>
    ${[["Chunk", "recursive · sentence\nmarkdown · offsets", 20, "var(--violet)"], ["Index", "BM25 inverted\nvector · HNSW", 150, "var(--cyan)"], ["Fuse", "weighted RRF\nmulti-query", 280, "var(--ember)"], ["Rerank", "MMR · cross-enc.\nheuristic", 410, "var(--amber)"], ["Guard", "injection · PII\nspotlighting", 540, "var(--rose)"], ["Generate", "any LLM · stream\nextractive", 670, "var(--mint)"]].map(([t, s, x, c], i) => `
      <g transform="translate(${x},70)"><rect width="112" height="96" rx="14" fill="var(--panel-2)" stroke="${c}" stroke-opacity=".7"/>
      <text x="56" y="30" text-anchor="middle" font-weight="800" font-size="15" fill="${c}">${t}</text>
      ${s.split("\n").map((l, j) => `<text x="56" y="${54 + j * 17}" text-anchor="middle" font-size="11.5" fill="var(--muted)">${l}</text>`).join("")}</g>
      ${i < 5 ? `<path d="M${x + 114} 118h16" stroke="var(--faint)" stroke-width="2" marker-end="url(#ar)"/>` : ""}`).join("")}
    <g transform="translate(790,70)"><rect width="92" height="96" rx="14" fill="url(#gg)"/><text x="46" y="42" text-anchor="middle" font-weight="800" font-size="15" fill="#1a0d05">Cite</text><text x="46" y="64" text-anchor="middle" font-size="11.5" fill="#1a0d05">spans ·</text><text x="46" y="81" text-anchor="middle" font-size="11.5" fill="#1a0d05">groundedness</text></g>
    <path d="M766 118h22" stroke="var(--faint)" stroke-width="2"/>
    <path d="M76 190v26h760v-26" fill="none" stroke="var(--line-2)" stroke-dasharray="4 5"/>
    <text x="456" y="236" text-anchor="middle" font-size="12" fill="var(--muted)">spans → tracer → console · memory · OpenTelemetry   ·   evaluation: nDCG · recall · faithfulness · LLM judge</text>
    <defs><marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 10 5 0 10z" fill="var(--faint)"/></marker></defs></svg>`;

  root.append(
    h("div", { class: "hero" }, h("h1", {}, "A RAG toolkit you can ", h("span", { class: "grad-text" }, "read in an afternoon")),
      h("p", {}, "RagForge implements the retrieval stack production systems depend on — and keeps every piece small enough to understand, test and trust. Strict typing, property-based tests and reproducible benchmarks included.")),
    h("div", { class: "card" }, h("div", { class: "card-b", html: diagram })),
    h("div", { style: { height: "18px" } }),
    h("div", { class: "grid", style: { gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))" } }, FEATURES.map(([icon, title, text]) => h("div", { class: "feature" }, h("div", { class: "ico", style: { color: "var(--ember)" }, html: ICONS[icon] }), h("h4", {}, title), h("p", {}, text)))),
    h("div", { style: { height: "18px" } }),
    h("div", { class: "grid cols-2" },
      h("div", { class: "card" }, h("div", { class: "card-h" }, h("h3", {}, "Install & first query")), h("div", { class: "card-b" },
        code(`pip install ragforge            # zero dependencies
pip install "ragforge[fast]"     # + NumPy exact search

from ragforge import RagPipeline
from ragforge.extractive import extractive_generate_fn

rag = RagPipeline(bm25_analyzer="english",
                  generate_fn=extractive_generate_fn())
rag.ingest("eiffel", "The Eiffel Tower is in Paris. "
                     "It was completed in 1889.")
print(rag.answer("When was it completed?").answer)`))),
      h("div", { class: "card" }, h("div", { class: "card-h" }, h("h3", {}, "Use it from Claude (MCP)")), h("div", { class: "card-b" },
        code(`ragforge ingest ./docs -i kb.json --backend hnsw
claude mcp add ragforge -- ragforge mcp -i kb.json

# or in any MCP client config:
{ "mcpServers": { "ragforge": {
    "command": "ragforge",
    "args": ["mcp", "--index", "kb.json"] } } }`),
        h("p", { class: "hint" }, "Tools: search_knowledge_base · ask_knowledge_base (with citations) · knowledge_base_stats. Ingestion stays off unless you pass --allow-ingest.")))),
    h("div", { style: { height: "18px" } }),
    h("div", { class: "card" }, h("div", { class: "card-b", style: { display: "flex", gap: "18px", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between" } },
      h("div", {}, h("h3", { style: { fontSize: "17px" } }, "Built by Francisco Javier Gómez Pulido"), h("div", { class: "hint" }, "AI Lead · Mathematics & Computer Science · MSc Artificial Intelligence")),
      h("div", { style: { display: "flex", gap: "10px", flexWrap: "wrap" } },
        h("a", { class: "btn primary", href: "https://github.com/fragompul/ragforge", target: "_blank", rel: "noopener" }, "Star on GitHub"),
        h("a", { class: "btn", href: "https://www.linkedin.com/in/frangomezpulido", target: "_blank", rel: "noopener" }, "LinkedIn"),
        h("a", { class: "btn", href: "https://github.com/fragompul/ragforge/tree/main/docs", target: "_blank", rel: "noopener" }, "Read the docs")))));
  return {};
}
