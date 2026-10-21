/* Tiny DOM helpers shared by all views. No framework. */
export function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === "html") node.innerHTML = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function esc(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const STOP = new Set("a an and are as at be by can do does for from has have how i in is it of on or that the their this to was were what when where which who why will with you your".split(" "));

export function queryTerms(query) {
  return [...new Set((query.toLowerCase().match(/[a-z0-9]+/g) || []).filter((t) => !STOP.has(t) && t.length > 1))];
}

/** HTML-escape `text` and wrap occurrences of the query terms (prefix-stemmed) in <mark>. */
export function highlight(text, terms) {
  const safe = esc(text).replace(/\[REDACTED:[a-z_]+\]|\[removed: suspected prompt injection\]/g, (m) => `<span class="redacted">${m}</span>`);
  if (!terms.length) return safe;
  const stems = terms.map((t) => (t.length > 5 ? t.slice(0, t.length - 1) : t)).map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(?<![A-Za-z0-9])(${stems.join("|")})[A-Za-z0-9]*`, "gi");
  return safe.replace(re, (m, _s, offset, whole) => {
    // never break HTML tags inserted above
    const before = whole.lastIndexOf("<", offset);
    const closed = whole.lastIndexOf(">", offset);
    return before > closed ? m : `<mark>${m}</mark>`;
  });
}

export function fmt(n, digits = 3) {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  return Number(n).toFixed(digits);
}
export function ms(n) {
  if (n === null || n === undefined) return "–";
  return n < 1 ? `${n.toFixed(2)} ms` : n < 100 ? `${n.toFixed(1)} ms` : `${Math.round(n)} ms`;
}

export function debounce(fn, wait = 200) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

export function toast(message) {
  const t = h("div", { class: "toast" }, message);
  Object.assign(t.style, {
    position: "fixed", bottom: "22px", left: "50%", transform: "translateX(-50%)", zIndex: 200,
    background: "var(--panel-2)", border: "1px solid var(--line-2)", padding: "10px 16px", borderRadius: "12px",
    boxShadow: "var(--shadow)", fontWeight: 600,
  });
  document.body.append(t);
  setTimeout(() => t.remove(), 2200);
}

export function fillRange(input) {
  const min = Number(input.min || 0), max = Number(input.max || 100);
  input.style.setProperty("--fill", `${((input.value - min) / (max - min)) * 100}%`);
}

/** Labeled slider returning {el, get}. */
export function slider({ label, min, max, step = 1, value, format = (v) => v, onInput, hint }) {
  const out = h("output", {}, format(value));
  const input = h("input", { type: "range", min, max, step, value });
  fillRange(input);
  input.addEventListener("input", () => {
    out.textContent = format(Number(input.value));
    fillRange(input);
    onInput && onInput(Number(input.value));
  });
  const el = h("div", { class: "field" }, h("label", {}, label, out), input, hint ? h("div", { class: "hint" }, hint) : null);
  return { el, get: () => Number(input.value), set: (v) => { input.value = v; out.textContent = format(v); fillRange(input); } };
}

/** Segmented control returning {el, get, set}. */
export function segmented({ label, options, value, onChange }) {
  let current = value;
  const buttons = options.map(([val, text]) =>
    h("button", { type: "button", "aria-pressed": String(val === current), onclick: () => set(val, true) }, text));
  const el = h("div", { class: "field" }, label ? h("label", {}, label) : null, h("div", { class: "seg" }, buttons));
  function set(val, fire) {
    current = val;
    options.forEach(([v], i) => buttons[i].setAttribute("aria-pressed", String(v === val)));
    if (fire && onChange) onChange(val);
  }
  return { el, get: () => current, set: (v) => set(v, false) };
}

export function toggle({ label, checked = false, onChange }) {
  const input = h("input", { type: "checkbox" });
  input.checked = checked;
  input.addEventListener("change", () => onChange && onChange(input.checked));
  return { el: h("label", { class: "field switch" }, h("span", { class: "lbl" }, label), input, h("i")), get: () => input.checked, set: (v) => (input.checked = v) };
}

export function select({ label, options, value, onChange }) {
  const el = h("select", {}, options.map(([v, t]) => h("option", { value: v }, t)));
  el.value = value;
  el.addEventListener("change", () => onChange && onChange(el.value));
  return { el: h("div", { class: "field" }, h("label", {}, label), el), get: () => el.value, set: (v) => (el.value = v) };
}

export function meter(value, color) {
  const fill = h("div", { class: "meter-fill", style: { background: color } });
  requestAnimationFrame(() => (fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`));
  return h("div", { class: "meter-track" }, fill);
}

export const COLORS = { bm25: "var(--cyan)", vector: "var(--violet)", hybrid: "var(--ember)" };

export const ICONS = {
  search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
  docs: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></svg>',
  graph: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="8" r="2.5"/><circle cx="12" cy="18" r="2.5"/><path d="m8 7 8 1M7 8l4 8M17 10l-4 6"/></svg>',
  shield: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 4 6v6c0 5 3.500 8 8 9 4.500-1 8-4 8-9V6z"/><path d="m9 12 2 2 4-4"/></svg>',
  agent: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="8" width="16" height="11" rx="3"/><path d="M12 4v4M9 13.500h.01M15 13.500h.01M9 17h6"/></svg>',
  chart: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20V4M4 20h16M8 16v-5M13 16V7M18 16v-8"/></svg>',
  network: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="5" r="2.500"/><circle cx="5" cy="18" r="2.500"/><circle cx="19" cy="18" r="2.500"/><circle cx="12" cy="13" r="1.500"/><path d="M12 7.500v4M10.800 14l-4.300 2.700M13.200 14l4.300 2.700M7 17l10 0"/></svg>',
  info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></svg>',
  sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.900 4.900l1.400 1.400M17.700 17.700l1.400 1.400M2 12h2M20 12h2M4.900 19.100l1.400-1.400M17.700 6.300l1.400-1.400"/></svg>',
  moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.800A9 9 0 1 1 11.200 3a7 7 0 0 0 9.800 9.800z"/></svg>',
  github: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 .5a11.500 11.500 0 0 0-3.640 22.410c.580.110.790-.250.790-.560v-2c-3.200.700-3.870-1.350-3.870-1.350-.520-1.330-1.280-1.690-1.280-1.690-1.040-.710.080-.700.080-.700 1.150.080 1.760 1.180 1.760 1.180 1.030 1.760 2.700 1.250 3.360.960.100-.750.400-1.250.730-1.540-2.550-.290-5.240-1.280-5.240-5.690 0-1.260.450-2.280 1.180-3.090-.120-.290-.510-1.460.110-3.040 0 0 .970-.310 3.170 1.180a10.900 10.900 0 0 1 5.780 0c2.200-1.490 3.170-1.180 3.170-1.180.630 1.580.230 2.750.110 3.040.740.810 1.180 1.830 1.180 3.090 0 4.420-2.690 5.390-5.260 5.670.410.360.780 1.060.780 2.140v3.170c0 .310.210.680.800.560A11.500 11.500 0 0 0 12 .5z"/></svg>',
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>',
};
