importScripts("./errors.js");
/* Created on the first valid Run request; one runtime per worker. A fit() call stays entirely in Python. */
const PYODIDE_VERSION = "0.28.3";
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
let currentRun = null, runsStarted = 0, errorStage = "runtime", errorSource = null;
function setStage(stage, source = null) { errorStage = stage; errorSource = source; }
function deletePythonGlobals(pyodide, ...names) {
  // Pyodide raises KeyError for missing globals, unlike JavaScript Map.delete.
  for (const name of names) {
    if (pyodide.globals.has(name)) pyodide.globals.delete(name);
  }
}
function diagnostic(error) { return VIPRSErrors.describe(error, errorStage, errorSource); }
const emit = (type, fields = {}) => self.postMessage({type, ...fields});

async function localText(path) {
  // These files change independently of immutable, hash-pinned wheels.
  // Never let a cached manifest select wheels from an earlier local build.
  const response = await fetch(new URL(path, self.location.href), {cache: "no-store"});
  if (!response.ok) throw new Error(`Cannot load ${path} (HTTP ${response.status}).`);
  return response.text();
}

function validateManifest(manifest) {
  if (manifest.pyodide_version !== PYODIDE_VERSION || !manifest.wheels?.length) {
    throw new Error("The VIPRS WASM bundle is missing or incompatible. Rebuild it with tools/wasm/build.py.");
  }
  const magenpyWheels = manifest.wheels.filter(path => /\/magenpy-[^/]+\.whl(?:#|$)/.test(path));
  if (magenpyWheels.length !== 1 || !/\/magenpy-0\.2\.5-/.test(magenpyWheels[0])) {
    const found = magenpyWheels.map(path => path.split("/").at(-1).split("#")[0]).join(", ") || "no magenpy wheel";
    throw new Error(`This web app requires magenpy 0.2.5; the manifest lists ${found}. Rebuild the WASM bundle with tools/wasm/build.py and reload the page.`);
  }
}

function runtimeProgress(message, source = null) {
  setStage("runtime", source);
  emit("runtime-status", {message, source});
}

async function installRuntimePackages(pyodide, manifest) {
  runtimeProgress("Downloading Python dependencies", INDEX_URL);
  const errors = [];
  await pyodide.loadPackage(manifest.builtin_packages, {
    messageCallback: message => runtimeProgress(message, INDEX_URL),
    errorCallback: message => { errors.push(message); emit("log", {id: currentRun, message}); },
  });
  if (errors.length) throw new Error("Python dependency loading failed: " + errors.join("; "));
  await pyodide.runPythonAsync("import json, micropip, os, gc, shutil");
  // Report each wheel separately: completion of CDN downloads is not the end
  // of initialization. Preserve micropip's compatibility and hash checks.
  for (const [index, path] of manifest.wheels.entries()) {
    const url = new URL(path, self.location.href);
    const name = url.pathname.split("/").at(-1);
    runtimeProgress(`Installing model package ${index + 1}/${manifest.wheels.length}: ${name}`, url.href);
    pyodide.globals.set("package_url", url.href);
    try { await pyodide.runPythonAsync("await micropip.install(package_url, deps=False)"); }
    finally { pyodide.globals.delete("package_url"); }
    emit("log", {id: currentRun, message: `Installed ${name}.`});
  }
  runtimeProgress("Importing VIPRS and magenpy");
  await pyodide.runPythonAsync(`
os.environ["TQDM_DISABLE"] = "1"
from tqdm import tqdm
tqdm.monitor_interval = 0
import viprs, magenpy
`);
}

async function initialize() {
  const start = performance.now();
  runtimeProgress("Reading runtime manifest", "runtime.json");
  const manifest = JSON.parse(await localText("runtime.json"));
  validateManifest(manifest);
  runtimeProgress("Downloading and starting Python runtime", INDEX_URL);
  importScripts(INDEX_URL + "pyodide.js");
  const pyodide = await loadPyodide({indexURL: INDEX_URL});
  pyodide.setStdout({batched: (message) => emit("log", {id: currentRun, message})});
  pyodide.setStderr({batched: (message) => emit("log", {id: currentRun, message})});
  await installRuntimePackages(pyodide, manifest);
  const versions = JSON.parse(await pyodide.runPythonAsync('json.dumps({"magenpy_version": magenpy.__version__, "viprs_version": viprs.__version__})'));
  runtimeProgress("Loading browser inference adapter", "inference.py");
  await pyodide.runPythonAsync(await localText("inference.py"));
  const initializationSeconds = (performance.now() - start) / 1000;
  emit("ready", {seconds: initializationSeconds, runtime: {pyodide_version: PYODIDE_VERSION, ...versions, wheels: manifest.wheels}});
  return {pyodide, manifest, initializationSeconds};
}

// Catch immediately, even if the user never submits a job.
const ready = initialize().catch((error) => {
  emit("init-error", {error: diagnostic(error)});
  return null;
});

function stageProgress(stage, fields) {
  emit("stage-progress", {id: currentRun, stage, ...fields});
}

function modelProgress(event, archiveIndex, archiveCount) {
  const {stage, chromosome, model, completed = 0, total, iteration, max_iter} = event;
  const exact = Number.isFinite(total) && total > 0;
  const done = event.state === "done" && (exact ? completed >= total : archiveIndex >= archiveCount && event.archive_fraction === 1);
  const state = event.state === "done" ? done ? "done" : "waiting" : "active";
  let detail = event.detail || "";
  let heading, message;
  if (stage === "fitting") {
    const ordinal = completed + (event.state === "done" ? 0 : 1);
    const position = exact ? ` (${ordinal}/${total})` : "";
    heading = `Fitting ${model} · chromosome ${chromosome}${position}`;
    const count = exact ? `${completed}/${total} chromosomes complete` : `${completed} chromosome(s) complete · file ${archiveIndex}/${archiveCount}`;
    const iterations = Number.isFinite(iteration) ? iteration === 0 ? " · initializing fit" : ` · iteration ${iteration}${event.state === "done" ? "" : ` (limit ${max_iter})`}` : "";
    detail = count;
    message = `${Number(event.variants).toLocaleString()} variants · ${count}${event.state === "done" ? "" : iterations}.`;
  } else if (stage === "ld" && chromosome) {
    heading = `Loading LD · chromosome ${chromosome}`;
    message = detail;
  } else {
    heading = stage === "ld" ? "Loading LD reference" : "Harmonizing variants";
    message = detail;
  }
  return {stage, state, completed: exact ? completed : archiveIndex - 1 + (event.archive_fraction || 0),
    total: exact ? total : event.archive_fraction === undefined ? null : archiveCount,
    detail, heading, message, quiet: Boolean(event.quiet)};
}

// Frequent byte/wait updates stay in the status detail without flooding the log.
function inputProgress(message, detail, stage = "sumstats", fileIndex = 0, fileCount = 1) {
  let lastUpdate = -Infinity;
  return ({bytes = 0, total = null, stage: transferStage = "transfer"}) => {
    const now = performance.now();
    if (transferStage === "transfer" && bytes > 0 && bytes !== total && now - lastUpdate < 500) return;
    lastUpdate = now;
    const mib = value => (value / 1024 ** 2).toFixed(1) + " MiB";
    const amount = total > 0
      ? `${mib(bytes)} of ${mib(total)} (${Math.min(100, Math.floor(100 * bytes / total))}%)`
      : `${mib(bytes)} received`;
    const description = `${detail} ${transferStage === "connecting" ? "Waiting for the server response." : transferStage === "checking" ? "Checking download size and parallel-transfer support." : amount + "."}`;
    emit("phase", {id: currentRun, message, detail: description, quiet: true});
    stageProgress(stage, {state: "active", completed: total > 0 ? fileIndex + Math.min(1, bytes / total) : null,
      total: total > 0 ? fileCount : null, detail: transferStage === "connecting" ? "Connecting…" : transferStage === "checking" ? "Checking download…" : amount});
  };
}

function catalogProgress(accession, build) {
  let timer = null;
  const stop = () => { if (timer !== null) clearInterval(timer); timer = null; };
  const report = (event, url) => {
    stop();
    if (event !== "request") return;
    const metadata = url.endsWith("-meta.yaml");
    const name = metadata ? decodeURIComponent(new URL(url).pathname.split("/").at(-1)) : url.endsWith("/harmonised/") ? "harmonised file list" : "study file list";
    const started = performance.now();
    const update = (quiet = false) => emit("phase", {
      id: currentRun,
      message: metadata ? "Checking GWAS Catalog genome build" : "Reading GWAS Catalog file list",
      detail: `${accession} · ${name}. Waiting for the ${metadata ? "metadata" : "directory"} response for ${build} (${Math.floor((performance.now() - started) / 1000)}s).`,
      quiet,
    });
    update();
    timer = setInterval(() => update(true), 1000);
  };
  return {report, stop};
}

async function stageInput(source, FS, path, inferCompression = false, onProgress = null) {
  let stream, total = source.file?.size || null;
  if (source.file) {
    if (!source.file.size) throw new Error("An input is empty.");
    stream = source.file.stream();
  } else {
    const url = new URL(source.url);
    if (!["https:", "http:"].includes(url.protocol)) throw new Error("Only HTTP(S) download URLs are supported.");
    let response;
    onProgress?.({stage: "connecting"});
    try { response = await fetch(url, {credentials: "omit"}); }
    catch { throw new Error("Could not fetch an input. Check its URL, connection, and the host's CORS settings."); }
    if (!response.ok) throw new Error(`Input download failed (HTTP ${response.status}).`);
    total = Number(response.headers?.get("content-length")) || null;
    stream = response.body;
  }
  onProgress?.({bytes: 0, total});
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
      onProgress?.({bytes: size, total});
    }
    if (!size) throw new Error("An input is empty.");
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
    if (handle !== undefined) FS.close(handle);
  }
  onProgress?.({bytes: size, total: size});
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

async function stageRangedInput(source, FS, path, inferCompression = false, onProgress = null) {
  // Large Catalog files are served from a geographically distant endpoint for
  // some users. Use independent ranges when EBI exposes range support, while
  // retaining the ordinary streaming path for every other server.
  let metadata;
  onProgress?.({stage: "checking"});
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
  let received = 0;
  onProgress?.({bytes: 0, total: size});
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
          received += written;
          onProgress?.({bytes: received, total: size});
        }
      } catch (error) {
        await reader.cancel().catch(() => {});
        throw error;
      } finally { reader.releaseLock(); }
      if (position !== end + 1) throw new Error("A parallel GWAS Catalog download ended before its byte range was complete.");
    }));
  } finally { FS.close(handle); }
  onProgress?.({bytes: received, total: size});
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
    setStage("ld", source.file?.name || source.url);
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
        stagedPath = source.staged_path || await stageInput(source, FS, path, false, inputProgress(`${source.file ? "Reading" : "Downloading"} LD reference`, `File ${index + 1} of ${sources.length}.`, "ld", index, sources.length));
        if (!source.staged_path) downloadSeconds += (performance.now() - downloadStart) / 1000;
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

async function prepareCatalogInputs(data, pyodide, workPath) {
  if (!data.sumstats.some(source => source.catalog)) return 0;
  const start = performance.now(), firstLD = data.ld[0];
  setStage("ld", firstLD.file?.name || firstLD.url);
  emit("phase", {id: currentRun, message: "Checking LD genome build", detail: "Reading LD metadata before selecting GWAS Catalog files."});
  stageProgress("ld", {state: "active", detail: "Checking the reference genome build."});
  const metadataPath = firstLD.stream ? firstLD.url : await stageInput(firstLD, pyodide.FS, `${workPath}/ld-first`, false,
    inputProgress("Reading LD metadata", "First reference file.", "ld", 0, data.ld.length));
  // Keep a staged first archive for fitting, avoiding a second download or copy.
  if (!firstLD.stream) firstLD.staged_path = metadataPath;
  pyodide.globals.set("ld_metadata_path", metadataPath);
  pyodide.globals.set("ld_expected_build", data.options.ld_genome_build || null);
  let build;
  try { build = pyodide.runPython("inspect_browser_ld_build(ld_metadata_path, ld_expected_build)"); }
  finally { pyodide.globals.delete("ld_metadata_path"); pyodide.globals.delete("ld_expected_build"); }
  data.options.ld_genome_build = build;
  data.options.sumstats_genome_build = build;
  emit("ld-metadata", {id: currentRun, genome_build: build});
  stageProgress("ld", {state: "waiting", detail: `${build} reference identified; matrices load before fitting.`});
  for (let index = 0; index < data.sumstats.length; index++) {
    const source = data.sumstats[index];
    if (!source.catalog) continue;
    setStage("sumstats", source.catalog);
    if (source.catalog_files?.[build]) {
      data.sumstats[index] = {url: source.catalog_files[build], parallel: true};
      emit("phase", {id: currentRun, message: "GWAS Catalog validation reused", detail: `${source.catalog} · ${build}. Downloading the previously verified file.`});
      continue;
    }
    emit("phase", {id: currentRun, message: "Resolving GWAS Catalog study", detail: `Finding ${build} summary statistics for ${source.catalog}.`});
    pyodide.globals.set("catalog_accession", source.catalog);
    pyodide.globals.set("catalog_genome_build", build);
    const progress = catalogProgress(source.catalog, build);
    pyodide.globals.set("catalog_progress_callback", progress.report);
    try {
      const url = await pyodide.runPythonAsync("await resolve_browser_gwas_catalog(catalog_accession, genome_build=catalog_genome_build, progress_callback=catalog_progress_callback)");
      data.sumstats[index] = {url, parallel: true};
      emit("phase", {id: currentRun, message: "GWAS Catalog file selected", detail: `${source.catalog} · ${build}. Preparing to download ${decodeURIComponent(new URL(url).pathname.split("/").at(-1))}.`});
    } finally {
      progress.stop();
      pyodide.globals.delete("catalog_progress_callback");
      pyodide.globals.delete("catalog_accession");
      pyodide.globals.delete("catalog_genome_build");
    }
  }
  return (performance.now() - start) / 1000;
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
  setStage("runtime");
  let runtime, workPath, output, failure, requestStarted = false, requestFinished = false;
  try {
    runtime = await ready;
    if (!runtime) return; // init-error has already been delivered.
    const {pyodide, manifest, initializationSeconds} = runtime;
    for (const kind of ["sumstats", "ld"]) {
      setStage(kind);
      if (!Array.isArray(data[kind]) || !data[kind].length || data[kind].length > 64) {
        throw new Error("Select 1–64 files or URLs for each input type.");
      }
    }
    runsStarted += 1;
    const runStart = performance.now();
    emit("phase", {id: data.id, message: "Preparing inputs", major: "inputs", started: true});
    const {expandDataURL} = await import("./sources.mjs");
    for (const kind of ["sumstats", "ld"]) {
      setStage(kind);
      const expanded = new Map();
      for (const source of data[kind]) {
        setStage(kind, source.file?.name || source.url || source.catalog);
        if (source.file) expanded.set(source.file.webkitRelativePath || source.file.name, source);
        else if (source.catalog) expanded.set(source.catalog, source);
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
    stageProgress("sumstats", {state: "active", detail: "Preparing summary-statistics inputs."});
    let downloadSeconds = await prepareCatalogInputs(data, pyodide, workPath);
    for (let index = 0; index < data.sumstats.length; index++) {
      const source = data.sumstats[index];
      setStage("sumstats", source.file?.name || source.url);
      emit("phase", {
        id: data.id,
        message: `${source.file ? "Reading" : "Downloading"} summary statistics`,
        detail: `File ${index + 1} of ${data.sumstats.length}. Model fitting has not started yet.`,
      });
      const downloadStart = performance.now();
      const inputPath = `${workPath}/sumstats-${index}`;
      const progress = inputProgress(`${source.file ? "Reading" : "Downloading"} summary statistics`, `File ${index + 1} of ${data.sumstats.length}.`, "sumstats", index, data.sumstats.length);
      request.sumstats_paths.push(
        source.parallel
          ? (await stageRangedInput(source, pyodide.FS, inputPath, true, progress) || await stageInput(source, pyodide.FS, inputPath, true, progress))
          : await stageInput(source, pyodide.FS, inputPath, true, progress)
      );
      downloadSeconds += (performance.now() - downloadStart) / 1000;
    }
    stageProgress("sumstats", {state: "active", detail: "Reading and organizing summary statistics by chromosome."});
    let archiveIndex = 1;
    pyodide.globals.set("fit_progress_callback", payload => {
      const progress = modelProgress(JSON.parse(payload), archiveIndex, data.ld.length);
      stageProgress(progress.stage, progress);
    });
    pyodide.globals.set("memory_callback", () => pyodide._module.HEAPU8.buffer.byteLength);
    pyodide.globals.set("request_json", JSON.stringify(request));
    pyodide.globals.set("progress_callback", (chromosome, completed, total) => {
      emit("chromosome-complete", {
        id: data.id, chromosome, completed, total,
        message: `Finished chromosome ${chromosome}. ${completed} chromosome(s) complete.`,
      });
    });
    pyodide.globals.set("stage_callback", (stage) => {
      setStage(stage, errorSource);
      if (["inference", "results"].includes(stage)) return;
      emit("phase", {id: data.id, message: {sumstats: "Reading summary statistics", ld: "Loading LD reference", harmonization: "Matching summary statistics to LD", inference: "Fitting model", results: "Preparing results"}[stage] || stage});
    });
    setStage("sumstats", data.sumstats.map(s => s.file?.name || s.url).join(", "));
    emit("log", {id: data.id, message: `Reading summary statistics with the ${data.options.sumstats_format} format.`});
    await pyodide.runPythonAsync("start_request(request_json)");
    requestStarted = true;
    stageProgress("sumstats", {state: "done", completed: data.sumstats.length, total: data.sumstats.length, detail: "Summary statistics ready."});
    // The parsed chromosome tables no longer depend on their compressed input.
    // Unlink it before any LD metadata or model arrays enter the wasm heap.
    for (const path of request.sumstats_paths) {
      try { pyodide.FS.unlink(path); } catch {}
    }
    downloadSeconds += await fitLDArchives(
      data.ld, pyodide.FS, workPath, async (path, remaining) => {
        archiveIndex = data.ld.length - remaining;
        emit("phase", {
          id: data.id,
          major: "fitting",
          message: `Fitting ${data.options.model || "VIPRS"}`,
          detail: `Reference file ${archiveIndex} of ${data.ld.length}.`,
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
    for (const stage of ["ld", "harmonization", "fitting"]) stageProgress(stage, {state: "done", completed: 1, total: 1, detail: "Complete"});
    setStage("results");
    emit("phase", {id: data.id, major: "results", message: "Combining chromosome results and preparing downloads"});
    output = JSON.parse(await pyodide.runPythonAsync("finish_request()"));
    requestFinished = true;
    emit("log", {id: data.id, message: `Inputs streamed in ${downloadSeconds.toFixed(1)}s. Summary statistics format: ${data.options.sumstats_format}; model: ${data.options.model}; components: ${data.options.components}; maximum iterations: ${data.options.max_iter}; tolerance: ${data.options.tolerance}; seed: ${data.options.seed}.`});
    output.report.pyodide_version = PYODIDE_VERSION;
    output.report.wheels = manifest.wheels;
    output.report.runtime_reused = runsStarted > 1;
    Object.assign(output.report.timings, {
      initialization_seconds: initializationSeconds, download_seconds: downloadSeconds,
      run_seconds: (performance.now() - runStart) / 1000,
    });
    output.report.inputs = Object.fromEntries(["sumstats", "ld"].map((kind) =>
      [kind, data[kind].map((source) => source.file ? source.file.name : source.url)]));
  } catch (error) { failure = diagnostic(error); }
  finally {
    if (runtime && workPath) {
      const {pyodide} = runtime;
      try {
        if (requestStarted && !requestFinished) pyodide.runPython("abort_request()");
        deletePythonGlobals(pyodide, "fit_progress_callback");
        deletePythonGlobals(pyodide, "memory_callback");
        deletePythonGlobals(pyodide, "request_json");
        deletePythonGlobals(pyodide, "progress_callback");
        deletePythonGlobals(pyodide, "stage_callback");
        pyodide.globals.set("work_path", workPath);
        pyodide.runPython("shutil.rmtree(work_path, ignore_errors=True); gc.collect()");
      } catch (error) { failure = failure || VIPRSErrors.describe(error, "cleanup"); }
      finally { deletePythonGlobals(pyodide, "work_path"); }
    }
    currentRun = null;
  }
  if (failure) emit("error", {id: data.id, error: failure});
  else if (output) emit("result", {id: data.id, result: output});
};
