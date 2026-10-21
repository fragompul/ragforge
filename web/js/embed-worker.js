/* Module worker: sentence embeddings in the browser with transformers.js (ONNX Runtime Web).
 * Model: Xenova/all-MiniLM-L6-v2 (the ONNX export of sentence-transformers/all-MiniLM-L6-v2),
 * 8-bit quantized (~23 MB, downloaded once and cached by the browser). */
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2";

env.allowLocalModels = false;
let extractor = null;

self.onmessage = async ({ data }) => {
  const { id, type, texts } = data;
  try {
    if (type === "load") {
      if (!extractor) {
        extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
          dtype: "q8",
          progress_callback: (p) => postMessage({ type: "progress", p }),
        });
      }
      postMessage({ type: "done", id });
    } else if (type === "embed") {
      const out = await extractor(texts, { pooling: "mean", normalize: true });
      postMessage({ type: "done", id, vectors: out.tolist() });
    }
  } catch (err) {
    postMessage({ type: "error", id, message: String(err && err.message ? err.message : err) });
  }
};
