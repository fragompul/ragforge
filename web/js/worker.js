/* Web Worker hosting Pyodide (CPython on WebAssembly) and the real ragforge package. */
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.0/full/";
const build = new URL(self.location.href).searchParams.get("v") || "dev";
importScripts(PYODIDE + "pyodide.js");

let bridge = null;
const progress = (pct, msg) => postMessage({ type: "progress", pct, msg });

async function boot() {
  progress(8, "Starting the Python runtime (WebAssembly)…");
  const py = await loadPyodide({ indexURL: PYODIDE });
  progress(62, "Unpacking the ragforge package…");
  const zip = await (await fetch(new URL(`../py/ragforge.zip?v=${build}`, self.location.href))).arrayBuffer();
  py.unpackArchive(zip, "zip");
  const src = await (await fetch(new URL(`../py/bridge.py?v=${build}`, self.location.href))).text();
  py.FS.writeFile("bridge.py", src);
  progress(85, "Importing ragforge…");
  py.runPython("import sys; sys.path.insert(0, '.')");
  bridge = py.pyimport("bridge");
  const version = py.runPython("import ragforge; ragforge.__version__");
  progress(100, "Ready");
  postMessage({ type: "ready", version, python: py.runPython("import sys; sys.version.split()[0]") });
}

boot().catch((err) => postMessage({ type: "fatal", message: String(err && err.message ? err.message : err) }));

onmessage = (event) => {
  const { id, fn, arg } = event.data;
  if (!bridge) return postMessage({ type: "result", id, error: "Python is not ready yet" });
  try {
    const out = bridge[fn](arg ?? "{}");
    postMessage({ type: "result", id, data: out });
  } catch (err) {
    postMessage({ type: "result", id, error: String(err && err.message ? err.message : err) });
  }
};
