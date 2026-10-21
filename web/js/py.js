/* Promise-based RPC client for the Pyodide worker. */
let worker = null;
let nextId = 1;
const pending = new Map();
const listeners = { progress: [], ready: [], fatal: [] };
let readyInfo = null;
let readyResolve;
const readyPromise = new Promise((resolve) => (readyResolve = resolve));

export function onPy(event, fn) {
  listeners[event].push(fn);
  if (event === "ready" && readyInfo) fn(readyInfo);
}

export function startPython(build) {
  worker = new Worker(`js/worker.js?v=${build}`);
  worker.onmessage = ({ data }) => {
    if (data.type === "progress") listeners.progress.forEach((fn) => fn(data));
    else if (data.type === "ready") {
      readyInfo = data;
      readyResolve(data);
      listeners.ready.forEach((fn) => fn(data));
    } else if (data.type === "fatal") listeners.fatal.forEach((fn) => fn(data));
    else if (data.type === "result") {
      const slot = pending.get(data.id);
      if (!slot) return;
      pending.delete(data.id);
      data.error ? slot.reject(new Error(data.error)) : slot.resolve(data.data);
    }
  };
  worker.onerror = (e) => listeners.fatal.forEach((fn) => fn({ message: e.message || "Worker failed to start" }));
}

/** Call `bridge.<fn>(JSON)` and resolve with the parsed JSON result. */
export async function call(fn, arg = {}) {
  await readyPromise;
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, {
      resolve: (text) => {
        try { resolve(JSON.parse(text)); } catch (e) { reject(e); }
      },
      reject,
    });
    worker.postMessage({ id, fn, arg: JSON.stringify(arg) });
  });
}
