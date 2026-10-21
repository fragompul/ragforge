/* Main-thread client for the embedding worker (loaded lazily, only if the user asks for semantic mode). */
let worker = null;
let nextId = 1;
const pending = new Map();
let progressFn = null;
let loaded = false;

function ensureWorker() {
  if (worker) return worker;
  worker = new Worker("js/embed-worker.js", { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "progress") return progressFn && progressFn(data.p);
    const slot = pending.get(data.id);
    if (!slot) return;
    pending.delete(data.id);
    data.type === "error" ? slot.reject(new Error(data.message)) : slot.resolve(data);
  };
  worker.onerror = (e) => {
    const err = new Error(e.message || "The embedding worker failed to start");
    for (const slot of pending.values()) slot.reject(err);
    pending.clear();
    worker = null;
  };
  return worker;
}

const send = (message) =>
  new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    ensureWorker().postMessage({ id, ...message });
  });

export const isLoaded = () => loaded;

/** Download (first time) and initialize the model. `onProgress({loaded, total, file})` receives byte counts. */
export async function loadModel(onProgress) {
  progressFn = (p) => {
    if (p.status === "progress" && onProgress) onProgress({ loaded: p.loaded, total: p.total, file: p.file });
  };
  await send({ type: "load" });
  loaded = true;
}

/** Embed texts (batched); returns an array of 384-d unit vectors. */
export async function embedTexts(texts) {
  const out = [];
  for (let i = 0; i < texts.length; i += 32) {
    const { vectors } = await send({ type: "embed", texts: texts.slice(i, i + 32) });
    out.push(...vectors);
  }
  return out;
}
