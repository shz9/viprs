/* One eagerly initialized runtime per worker. A fit() call stays entirely in Python. */
const PYODIDE_VERSION = "0.28.3";
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
let currentRun = null, runsStarted = 0;
const emit = (type, fields = {}) => self.postMessage({type, ...fields});

async function localText(path) {
  const response = await fetch(new URL(path, self.location.href));
  if (!response.ok) throw new Error(`Cannot load ${path} (HTTP ${response.status}).`);
  return response.text();
}

async function initialize() {
  const start = performance.now();
  const manifest = JSON.parse(await localText("runtime.json"));
  if (manifest.pyodide_version !== PYODIDE_VERSION || !manifest.wheels?.length) {
    throw new Error("The VIPRS WASM bundle is missing or incompatible. Rebuild it with tools/wasm/build.py.");
  }
  emit("runtime-status", {message: "Loading runtime"});
  importScripts(INDEX_URL + "pyodide.js");
  const pyodide = await loadPyodide({indexURL: INDEX_URL});
  pyodide.setStdout({batched: (message) => emit("log", {id: currentRun, message})});
  pyodide.setStderr({batched: (message) => emit("log", {id: currentRun, message})});
  emit("runtime-status", {message: "Loading packages"});
  await pyodide.loadPackage(manifest.builtin_packages);
  const wheels = manifest.wheels.map((path) => new URL(path, self.location.href).href);
  pyodide.globals.set("packages_json", JSON.stringify(wheels));
  await pyodide.runPythonAsync(`
import json, micropip, os, gc, shutil
await micropip.install(json.loads(packages_json), deps=False)
os.environ["TQDM_DISABLE"] = "1"
from tqdm import tqdm
tqdm.monitor_interval = 0
import viprs, magenpy
`);
  await pyodide.runPythonAsync(await localText("inference.py"));
  pyodide.globals.delete("packages_json");
  const initializationSeconds = (performance.now() - start) / 1000;
  const sumstatsFormats = JSON.parse(pyodide.runPython(`
from magenpy.parsers.sumstats_parsers import get_supported_sumstats_formats
json.dumps(sorted(get_supported_sumstats_formats(include_aliases=True)))
`));
  emit("ready", {seconds: initializationSeconds, sumstats_formats: sumstatsFormats});
  return {pyodide, manifest, initializationSeconds};
}

// Catch immediately, even if the user never submits a job.
const ready = initialize().catch((error) => {
  emit("init-error", {message: error.message || String(error)});
  return null;
});

async function stageInput(source, FS, path, inferCompression = false) {
  let stream;
  if (source.file) {
    if (!source.file.size) throw new Error("An input is empty.");
    stream = source.file.stream();
  } else {
    const url = new URL(source.url);
    if (!["https:", "http:"].includes(url.protocol)) throw new Error("Only HTTP(S) download URLs are supported.");
    let response;
    try { response = await fetch(url, {credentials: "omit"}); }
    catch { throw new Error("Could not fetch an input. Check its URL, connection, and the host's CORS settings."); }
    if (!response.ok) throw new Error(`Input download failed (HTTP ${response.status}).`);
    stream = response.body;
  }
  if (!stream) throw new Error("An input is empty.");
  const reader = stream.getReader();
  let handle;
  let size = 0;
  const header = new Uint8Array(2);
  let headerSize = 0;
  try {
    handle = FS.open(path, "w");
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      if (inferCompression && headerSize < header.length) {
        const prefix = value.subarray(0, header.length - headerSize);
        header.set(prefix, headerSize);
        headerSize += prefix.length;
      }
      const written = FS.write(handle, value, 0, value.length, size);
      if (written !== value.length) throw new Error("Could not write the complete input to browser memory.");
      size += written;
    }
    if (!size) throw new Error("An input is empty.");
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
    if (handle !== undefined) FS.close(handle);
  }
  if (inferCompression) {
    // pandas infers compression from the path. Sniff actual streamed bytes,
    // including a gzip header split across chunks, then rename without copying.
    const suffix = headerSize === 2 && header[0] === 0x1f && header[1] === 0x8b ? ".tsv.gz" : ".tsv";
    const renamed = path + suffix;
    FS.rename(path, renamed);
    return renamed;
  }
  return path;
}

async function stageRangedInput(source, FS, path, inferCompression = false) {
  // Large Catalog files are served from a geographically distant endpoint for
  // some users. Use independent ranges when EBI exposes range support, while
  // retaining the ordinary streaming path for every other server.
  let metadata;
  try { metadata = await fetch(source.url, {method: "HEAD", credentials: "omit"}); }
  catch { return null; }
  const size = Number(metadata.headers.get("content-length"));
  if (!metadata.ok || !Number.isSafeInteger(size) || size < 8 * 1024 ** 2) return null;
  // Accept-Ranges is not a CORS-safelisted response header and EBI does not
  // expose it. A one-byte request provides an observable, reliable check.
  let probe;
  try {
    probe = await fetch(source.url, {
      credentials: "omit", headers: {Range: "bytes=0-0"},
    });
  } catch { return null; }
  if (probe.status !== 206) {
    await probe.body?.cancel().catch(() => {});
    return null;
  }
  await probe.body?.cancel().catch(() => {});
  const parts = Math.min(4, Math.ceil(size / (8 * 1024 ** 2)));
  const handle = FS.open(path, "w");
  const header = new Uint8Array(2);
  try {
    await Promise.all(Array.from({length: parts}, async (_, index) => {
      const start = Math.floor(size * index / parts);
      const end = Math.floor(size * (index + 1) / parts) - 1;
      const response = await fetch(source.url, {
        credentials: "omit", headers: {Range: `bytes=${start}-${end}`},
      });
      if (response.status !== 206 || !response.body) throw new Error("The GWAS Catalog server stopped accepting parallel byte ranges. Retry the download.");
      const reader = response.body.getReader();
      let position = start;
      try {
        while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          if (start === 0 && position < 2) header.set(value.subarray(0, 2 - position), position);
          const written = FS.write(handle, value, 0, value.length, position);
          if (written !== value.length) throw new Error("Could not write the complete input to browser memory.");
          position += written;
        }
      } catch (error) {
        await reader.cancel().catch(() => {});
        throw error;
      } finally { reader.releaseLock(); }
      if (position !== end + 1) throw new Error("A parallel GWAS Catalog download ended before its byte range was complete.");
    }));
  } finally { FS.close(handle); }
  if (inferCompression) {
    const suffix = header[0] === 0x1f && header[1] === 0x8b ? ".tsv.gz" : ".tsv";
    const renamed = path + suffix;
    FS.rename(path, renamed);
    return renamed;
  }
  return path;
}

async function fitLDArchives(sources, FS, workPath, fitArchive) {
  let downloadSeconds = 0;
  for (let index = 0; index < sources.length; index++) {
    const source = sources[index];
    const path = `${workPath}/ld-current`;
    let stagedPath;
    emit("phase", {
      id: currentRun,
      message: source.stream ? "Streaming LD reference" : `${source.file ? "Reading" : "Downloading"} LD reference`,
      detail: `Chromosome archive ${index + 1} of ${sources.length}.`,
    });
    const downloadStart = performance.now();
    try {
      if (source.stream) {
        await fitArchive(source.url, sources.length - index - 1);
      } else {
        stagedPath = await stageInput(source, FS, path);
        downloadSeconds += (performance.now() - downloadStart) / 1000;
        await fitArchive(stagedPath, sources.length - index - 1);
      }
    } finally {
      // Python has already released the loader, model arrays and extracted
      // store. Remove the compressed archive before staging the next one.
      for (const candidate of [stagedPath, path]) {
        if (!candidate) continue;
        try { FS.unlink(candidate); } catch {}
      }
    }
  }
  return downloadSeconds;
}

function sortLDInputs(sources) {
  const key = (source) => source.file
    ? (source.file.webkitRelativePath || source.file.name)
    : decodeURIComponent(new URL(source.url).pathname).split("/").filter(Boolean).at(-1);
  return [...sources].sort((left, right) =>
    key(left).localeCompare(key(right), undefined, {numeric: true})
  );
}

self.onmessage = async ({data}) => {
  if (data.type !== "run" || currentRun !== null) return;
  currentRun = data.id;
  let runtime, workPath, output, failure, requestStarted = false, requestFinished = false;
  try {
    runtime = await ready;
    if (!runtime) return; // init-error has already been delivered.
    const {pyodide, manifest, initializationSeconds} = runtime;
    for (const kind of ["sumstats", "ld"]) {
      if (!Array.isArray(data[kind]) || !data[kind].length || data[kind].length > 64) {
        throw new Error("Select 1–64 files or URLs for each input type.");
      }
    }
    runsStarted += 1;
    const runStart = performance.now();
    emit("phase", {id: data.id, message: "Preparing inputs", started: true});
    const {expandDataURL} = await import("./sources.mjs");
    for (const kind of ["sumstats", "ld"]) {
      const expanded = new Map();
      for (const source of data[kind]) {
        if (source.file) expanded.set(source.file.webkitRelativePath || source.file.name, source);
        else if (source.catalog) {
          emit("phase", {id: data.id, message: "Resolving GWAS Catalog study", detail: `Finding GRCh37/hg19 summary statistics for ${source.catalog}.`});
          pyodide.globals.set("catalog_accession", source.catalog);
          try {
            const url = await pyodide.runPythonAsync("await resolve_browser_gwas_catalog(catalog_accession)");
            expanded.set(url, {url, parallel: true});
          } finally { pyodide.globals.delete("catalog_accession"); }
        }
        else if (kind === "ld" && source.stream && source.url.startsWith("hf://")) {
          // Keep the native URI and glob: GWADataLoader handles both through
          // magenpy's Hugging Face filesystem integration.
          expanded.set(source.url, source);
        }
        else {
          emit("phase", {id: data.id, message: "Resolving input paths", detail: "Finding matching files before download."});
          for (const url of await expandDataURL(source.url, self.location.href)) expanded.set(url, {url, stream: Boolean(source.stream)});
        }
        if (expanded.size > 64) throw new Error("Patterns match more than 64 " + kind + " files. Narrow the selection.");
      }
      data[kind] = [...expanded.values()];
      if (kind === "ld") data[kind] = sortLDInputs(data[kind]);
    }
    workPath = `/runs/run-${runsStarted}`;
    pyodide.FS.mkdirTree(workPath);
    const request = {options: data.options, sumstats_paths: []};
    // Stream into the virtual filesystem without assembling another full JS buffer.
    // Summary statistics stay available across chromosomes; LD archives do not.
    let downloadSeconds = 0;
    for (let index = 0; index < data.sumstats.length; index++) {
      const source = data.sumstats[index];
      emit("phase", {
        id: data.id,
        message: `${source.file ? "Reading" : "Downloading"} summary statistics`,
        detail: `File ${index + 1} of ${data.sumstats.length}. Model fitting has not started yet.`,
      });
      const downloadStart = performance.now();
      const inputPath = `${workPath}/sumstats-${index}`;
      request.sumstats_paths.push(
        source.parallel
          ? (await stageRangedInput(source, pyodide.FS, inputPath, true) || await stageInput(source, pyodide.FS, inputPath, true))
          : await stageInput(source, pyodide.FS, inputPath, true)
      );
      downloadSeconds += (performance.now() - downloadStart) / 1000;
    }
    pyodide.globals.set("request_json", JSON.stringify(request));
    pyodide.globals.set("progress_callback", (chromosome, completed, total) => {
      emit("chromosome-complete", {
        id: data.id, chromosome, completed, total,
        message: `Finished chromosome ${chromosome} (${completed} of ${total}).`,
      });
    });
    await pyodide.runPythonAsync("start_request(request_json)");
    requestStarted = true;
    downloadSeconds += await fitLDArchives(
      data.ld, pyodide.FS, workPath, async (path, remaining) => {
        emit("phase", {
          id: data.id,
          message: `Fitting ${data.options.model || "VIPRS"}`,
          detail: "Harmonizing and fitting the current chromosome; its LD data will then be released.",
        });
        pyodide.globals.set("ld_path", path);
        pyodide.globals.set("remaining_ld_archives", remaining);
        try {
          await pyodide.runPythonAsync(
            "process_ld_archive(ld_path, remaining_ld_archives, progress_callback)"
          );
        } finally {
          pyodide.globals.delete("ld_path");
          pyodide.globals.delete("remaining_ld_archives");
        }
      }
    );
    output = JSON.parse(await pyodide.runPythonAsync("finish_request()"));
    requestFinished = true;
    emit("log", {id: data.id, message: `Inputs streamed in ${downloadSeconds.toFixed(1)}s. Model: ${data.options.model}; components: ${data.options.components}; maximum iterations: ${data.options.max_iter}; tolerance: ${data.options.tolerance}; seed: ${data.options.seed}.`});
    output.report.pyodide_version = PYODIDE_VERSION;
    output.report.wheels = manifest.wheels;
    output.report.runtime_reused = runsStarted > 1;
    Object.assign(output.report.timings, {
      initialization_seconds: initializationSeconds, download_seconds: downloadSeconds,
      run_seconds: (performance.now() - runStart) / 1000,
    });
    output.report.inputs = Object.fromEntries(["sumstats", "ld"].map((kind) =>
      [kind, data[kind].map((source) => source.file ? source.file.name : source.url)]));
  } catch (error) { failure = error.message || String(error); }
  finally {
    if (runtime && workPath) {
      const {pyodide} = runtime;
      try {
        if (requestStarted && !requestFinished) pyodide.runPython("abort_request()");
        pyodide.globals.delete("request_json");
        pyodide.globals.delete("progress_callback");
        pyodide.globals.set("work_path", workPath);
        pyodide.runPython("shutil.rmtree(work_path, ignore_errors=True); gc.collect()");
      } catch (error) { failure = failure || `Could not clean up run files: ${error.message || error}`; }
      finally { pyodide.globals.delete("work_path"); }
    }
    currentRun = null;
  }
  if (failure) emit("error", {id: data.id, message: failure});
  else if (output) emit("result", {id: data.id, result: output});
};
