/* Agent Lab: retrieve → grade → rewrite → abstain, with the whole decision trail. */
import { h, slider, toggle, meter, ms } from "../ui.js";

const KINDS = {
  decompose: ["var(--violet)", "Split the question into independent sub-questions"],
  retrieve: ["var(--cyan)", "Hybrid retrieval through the normal pipeline"],
  grade: ["var(--amber)", "Does the evidence cover the question?"],
  rewrite: ["var(--ember)", "Retry with a different query"],
  stop: ["var(--faint)", "Stopped"],
  answer: ["var(--mint)", "Generate from the merged evidence"],
  abstain: ["var(--rose)", "Refuse rather than guess"],
};

const PRESETS = [
  ["Two facts, two documents", "What does error ERR-4012 mean? How does reciprocal rank fusion work?"],
  ["Needs a retry", "How do I terminate a subscription and what is the zeppelin refund policy?"],
  ["Unanswerable", "What is the capital of Atlantis and its quantum zeppelin fleet?"],
  ["Single hop", "Why does BM25 fail on synonyms?"],
  ["Two hops (enable multi-hop)", "Where is the manufacturer of the ZX-9 sensor headquartered?"],
];

export async function mount(root, ctx) {
  const cfg = { max_iterations: 3, min_sufficiency: 0.7, abstain_below: 0.3, decompose: true, multihop: false, k: 3 };
  const input = h("input", { type: "text", value: PRESETS[0][1], "aria-label": "Question for the agent", autocomplete: "off", spellcheck: "false" });
  input.addEventListener("keydown", (e) => e.key === "Enter" && run());
  const out = h("div", { class: "stack" });

  async function run() {
    const question = input.value.trim();
    if (!question) return;
    out.replaceChildren(h("div", { class: "card empty" }, "Thinking…"));
    const res = await ctx.call("agent_run", { question, ...cfg });
    const steps = res.steps.map((s, i) => {
      const [color, blurb] = KINDS[s.kind] || ["var(--cyan)", ""];
      return h("div", { class: "step", style: { "--c": color, animationDelay: `${i * 70}ms` } },
        h("div", { class: "step-h" }, h("span", { class: "step-kind" }, s.kind.toUpperCase()), h("span", { class: "step-q" }, s.query),
          s.score !== null ? h("span", { class: `badge ${s.score >= cfg.min_sufficiency ? "good" : s.score >= cfg.abstain_below ? "warn" : "bad"}` }, `sufficiency ${s.score.toFixed(2)}`) : null,
          h("span", { class: "hint", style: { marginLeft: "auto" } }, ms(s.duration_ms))),
        h("div", { class: "step-d" }, s.detail || blurb),
        s.score !== null ? meter(s.score, s.score >= cfg.min_sufficiency ? "var(--mint)" : s.score >= cfg.abstain_below ? "var(--amber)" : "var(--rose)") : null,
        s.chunk_ids.length ? h("div", { class: "tags", style: { marginTop: "8px" } }, s.chunk_ids.map((id) => h("span", { class: "badge neutral" }, id.split("::")[0]))) : null);
    });
    const verdict = res.abstained
      ? h("div", { class: "card", style: { borderColor: "color-mix(in srgb, var(--rose) 50%, var(--line))" } }, h("div", { class: "card-b" }, h("span", { class: "badge bad" }, "ABSTAINED"), h("p", { class: "answer", style: { margin: "10px 0 0" } }, res.answer.answer), h("div", { class: "hint" }, `Best sufficiency ${res.sufficiency.toFixed(2)} < ${cfg.abstain_below} after ${res.retrievals} retrieval${res.retrievals === 1 ? "" : "s"}. A confident-sounding guess would have been a hallucination.`)))
      : h("div", { class: "card" }, h("div", { class: "card-b" }, h("div", { style: { display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" } }, h("span", { class: "badge good" }, "ANSWERED"), h("span", { class: "badge neutral" }, `${res.retrievals} retrievals`), h("span", { class: "badge neutral" }, `mean sufficiency ${res.sufficiency.toFixed(2)}`), h("span", { class: `badge ${res.citations.groundedness >= 0.99 ? "good" : "warn"}` }, `groundedness ${(res.citations.groundedness * 100).toFixed(0)}%`)),
        h("p", { class: "answer", style: { margin: "12px 0 0" } }, res.answer.answer),
        h("div", { class: "sources" }, res.citations.sentences.flatMap((s) => s.citations).map((c) => h("div", { class: "source" }, h("b", {}, `[${c.marker}]`), h("div", {}, h("span", { class: "docid" }, c.doc_id), h("span", { class: "hint" }, `  chars ${c.start_char}–${c.end_char}`), h("div", {}, h("q", {}, c.quote))))))));
    out.replaceChildren(h("div", { class: "grid cols-2" }, h("div", { class: "card" }, h("div", { class: "card-h" }, h("h3", {}, "Decision trail"), res.sub_questions.length > 1 ? h("span", { class: "badge neutral" }, `${res.sub_questions.length} sub-questions`) : null), h("div", { class: "card-b" }, h("div", { class: "timeline" }, steps))), verdict));
  }

  const controls = h("div", { class: "card" }, h("div", { class: "card-h" }, h("h3", {}, "Policy")),
    h("div", { class: "card-b" },
      toggle({ label: "Decompose multi-part questions", checked: cfg.decompose, onChange: (v) => (cfg.decompose = v) }).el,
      toggle({ label: "Multi-hop bridge mode (2 forced hops)", checked: cfg.multihop, onChange: (v) => (cfg.multihop = v) }).el,
      slider({ label: "Max retrieval attempts", min: 1, max: 5, value: cfg.max_iterations, onInput: (v) => (cfg.max_iterations = v) }).el,
      slider({ label: "Sufficient at ≥", min: 0.3, max: 1, step: 0.05, value: cfg.min_sufficiency, format: (v) => v.toFixed(2), onInput: (v) => (cfg.min_sufficiency = v) }).el,
      slider({ label: "Abstain below", min: 0, max: 0.8, step: 0.05, value: cfg.abstain_below, format: (v) => v.toFixed(2), onInput: (v) => (cfg.abstain_below = v) }).el,
      slider({ label: "Top-k per retrieval", min: 1, max: 6, value: cfg.k, onInput: (v) => (cfg.k = v) }).el,
      h("div", { class: "hint" }, "Multi-hop mode forces a second retrieval whose query adds the entities found in the first hop's passages, and excludes passages already seen. "),
      h("div", { class: "hint" }, "Grading, rewriting and decomposition are lexical here so the demo is deterministic and keyless; ", h("code", {}, "llm_grade_fn"), " / ", h("code", {}, "llm_rewrite_fn"), " plug in any model.")));

  root.append(
    h("div", { class: "hero" }, h("h1", {}, "Retrieval that ", h("span", { class: "grad-text" }, "knows when it's wrong")),
      h("p", {}, "Agentic RAG treats retrieval as a decision loop: grade the evidence, rewrite the query when it falls short, retry within a budget — and abstain instead of hallucinating when nothing supports an answer.")),
    h("div", { class: "searchbar" }, input, h("button", { class: "btn primary", onclick: run }, "Run agent")),
    h("div", { class: "chips" }, PRESETS.map(([label, q]) => h("button", { class: "chip", onclick: () => { input.value = q; run(); } }, label))),
    h("div", { style: { height: "18px" } }),
    h("div", { class: "split" }, controls, out));
  run();
  return {};
}
