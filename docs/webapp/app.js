import {EXAMPLE, resolveDataURL, selectLocalFiles} from "./sources.mjs";
const $ = (id) => document.getElementById(id);
let worker = null, runtimeState = "loading", activeRun = null, nextRun = 0;
let timer = null, startedAt = 0, downloadURLs = [], resultRows = {};

function syncInputs() {
  for (const source of ["example", "file", "url", "folder"]) {
    $(source + "-inputs").hidden = $("source").value !== ({file: "files", url: "urls"}[source] || source);
  }
  $("sumstats-format").disabled = $("source").value === "example";
}
function syncModel() {
  const mixture = $("model").value === "VIPRSMix";
  $("mixture-settings").hidden = !mixture;
  $("components").disabled = !mixture;
  $("model-help").textContent = mixture
    ? "A point mass at zero with multiple Gaussian effect-size components."
    : "A point mass at zero and one Gaussian component for nonzero effects.";
}
$("source").addEventListener("change", syncInputs);
$("model").addEventListener("change", syncModel);
$("demo").addEventListener("click", () => { $("source").value = "example"; $("sumstats-format").value = "fastGWA"; syncInputs(); });
for (const id of ["sumstats-file", "ld-file"]) $(id).addEventListener("change", () => {
  $("file-summary").textContent = `${$("sumstats-file").files.length} summary-statistics file(s) · ${$("ld-file").files.length} LD archive(s)`;
});
function updateFolderSummary() {
  if (!$("data-folder").files.length) return;
  try {
    const counts = ["sumstats", "ld"].map(kind => selectLocalFiles($("data-folder").files, $(kind + "-pattern").value).length);
    $("folder-summary").textContent = `${counts[0]} summary-statistics file(s) · ${counts[1]} LD archive(s) matched`;
  } catch (error) { $("folder-summary").textContent = error.message; }
}
$("data-folder").addEventListener("change", updateFolderSummary);
for (const kind of ["sumstats", "ld"]) $(kind + "-pattern").addEventListener("input", updateFolderSummary);

function log(message) {
  const lines = $("log").textContent.split("\n");
  lines.push(`[${new Date().toLocaleTimeString()}] ${message}`);
  $("log").textContent = lines.slice(-100).join("\n");
  $("log").scrollTop = $("log").scrollHeight;
}
function status(label, title, message, error = false) {
  $("status-badge").textContent = label;
  $("status-badge").dataset.state = error ? "error" : "normal";
  $("activity").dataset.state = label.toLowerCase().replaceAll(" ", "-");
  $("activity-title").textContent = title;
  $("status").textContent = message;
}
function runtimeStatus(state, text) {
  runtimeState = state;
  $("runtime-state").dataset.state = state;
  $("runtime-text").textContent = text;
  $("retry-runtime").hidden = state !== "error";
  $("run").disabled = activeRun !== null || state === "error";
}
function releaseResults() {
  downloadURLs.forEach((url) => URL.revokeObjectURL(url)); downloadURLs = [];
  $("results").hidden = $("metrics").hidden = $("timing-summary").hidden = $("elapsed-line").hidden = true;
  $("result-rows").replaceChildren(); resultRows = {};
  $("download").removeAttribute("href"); $("download-report").removeAttribute("href");
}
function finish() {
  activeRun = null; clearInterval(timer); timer = null;
  $("activity").classList.remove("running");
  $("controls").disabled = false;
  $("run").disabled = runtimeState === "error"; $("cancel").disabled = true;
}
function fail(message) {
  log(message); status("Error", "The analysis could not complete", message, true); finish();
}
function beginTimer() {
  startedAt = performance.now(); $("elapsed").textContent = "0s";
  $("elapsed-line").hidden = false;
  timer = setInterval(() => { $("elapsed").textContent = `${Math.floor((performance.now() - startedAt) / 1000)}s`; }, 1000);
}

function startWorker() {
  worker?.terminate();
  runtimeStatus("loading", "Preparing runtime");
  const current = new Worker(new URL("worker.js", import.meta.url));
  worker = current;
  current.onmessage = ({data}) => {
    if (worker !== current) return;
    if (data.type === "runtime-status") runtimeStatus("loading", data.message);
    if (data.type === "ready") {
      runtimeStatus("ready", "Runtime ready");
      if (activeRun === null && $("status-badge").textContent === "Ready") {
        $("status").textContent = "Select your inputs and model.";
      }
      log(`Runtime initialized in ${data.seconds.toFixed(1)}s; ready for reuse.`);
    }
    if (data.type === "init-error") {
      runtimeStatus("error", "Initialization failed"); fail(data.message);
    }
    if (data.type === "log" && (data.id == null || data.id === activeRun)) log(data.message);
    if (data.id !== activeRun || activeRun === null) return;
    if (data.type === "phase") {
      if (data.started) beginTimer();
      status("Running", data.message, data.detail || "Preparing inputs for inference.");
      log(data.message + (data.detail ? ": " + data.detail : ""));
    }
    if (data.type === "result") {
      try { showResult(data.result); }
      catch (error) {
        releaseResults();
        fail(`Fitting completed, but the results could not be displayed: ${error.message || error}`);
      }
    }
    if (data.type === "error") fail(data.message);
  };
  current.onerror = (event) => {
    if (worker !== current) return;
    event.preventDefault(); current.terminate(); worker = null;
    runtimeStatus("error", "Runtime unavailable");
    fail(event.message || "The worker could not start. Check the connection and serve this page over HTTP.");
  };
}
$("retry-runtime").addEventListener("click", () => { try { startWorker(); } catch (error) { runtimeStatus("error", "Runtime unavailable"); fail(error.message); } });

function sourceInputs(kind) {
  if ($("source").value === "example") return [{url: resolveDataURL(EXAMPLE[kind], location.href)}];
  if ($("source").value === "folder") {
    if (!$("data-folder").files.length) throw new Error("Choose a local data folder first.");
    const files = selectLocalFiles($("data-folder").files, $(kind + "-pattern").value);
    if (files.some(file => !file.size)) throw new Error("Each input file must be nonempty.");
    return files.map(file => ({file}));
  }
  if ($("source").value === "files") {
    const files = Array.from($(kind + "-file").files);
    if (!files.length || files.length > 64) throw new Error(`Select 1–64 ${kind === "ld" ? "LD archives" : "summary-statistics files"}.`);
    if (files.some((file) => !file.size)) throw new Error("Each input file must be nonempty.");
    return files.map((file) => ({file}));
  }
  const lines = $(kind + "-url").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (!lines.length || lines.length > 64) throw new Error(`Enter 1–64 ${kind} URLs, one per line.`);
  const urls = lines.map((line) => resolveDataURL(line, location.href));
  if (new Set(urls).size !== urls.length) throw new Error(`Remove duplicate ${kind} URLs.`);
  return urls.map((url) => ({url}));
}
function renderRows() {
  const chromosome = $("chromosome-filter").value;
  const rows = resultRows[chromosome] || [];
  $("result-rows").replaceChildren();
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const key of ["SNP", "CHR", "A1", "A2", "PIP", "BETA", "VAR_BETA"]) {
      const td = document.createElement("td"), value = row[key];
      td.textContent = typeof value === "number" && ["PIP", "BETA", "VAR_BETA"].includes(key) ? value.toPrecision(5) : String(value ?? "—");
      tr.append(td);
    }
    $("result-rows").append(tr);
  }
  $("table-caption").textContent = chromosome ? `Top variants · chromosome ${chromosome}` : "Top variants · all chromosomes";
}
$("chromosome-filter").addEventListener("change", renderRows);
function showResult(result) {
  for (const [id, data, type] of [["download", result.tsv, "text/tab-separated-values"], ["download-report", JSON.stringify(result.report, null, 2), "application/json"]]) {
    const url = URL.createObjectURL(new Blob([data], {type})); downloadURLs.push(url); $(id).href = url;
  }
  resultRows = {"": result.rows, ...result.chromosome_rows};
  $("chromosome-filter").replaceChildren(new Option("All chromosomes", ""));
  for (const c of result.report.chromosomes) $("chromosome-filter").add(new Option(`Chromosome ${c} · ${result.report.variants_per_chromosome[c].toLocaleString()} variants`, c));
  renderRows();
  const report = result.report;
  $("variants").textContent = report.variants.toLocaleString();
  $("chromosomes").textContent = report.chromosomes.length;
  $("iterations").textContent = report.iterations;
  $("timing-summary").textContent = `Input preparation ${report.timings.preparation_seconds.toFixed(1)}s · Model fitting ${report.timings.fit_seconds.toFixed(1)}s. Runtime initialization is excluded.`;
  $("result-summary").textContent = `${report.model}${report.model === "VIPRSMix" ? ` · ${report.components} non-null components` : ""} · ${report.variants.toLocaleString()} variants across ${report.chromosomes.length} chromosome(s). ${report.message}`;
  $("results").hidden = $("metrics").hidden = $("timing-summary").hidden = false;
  $("elapsed").textContent = `${report.timings.run_seconds.toFixed(1)}s`;
  status(report.converged ? "Converged" : "Iteration limit", "Analysis complete", report.message);
  log(`Completed ${report.model}: ${report.variants} variants across ${report.chromosomes.length} chromosome(s).`);
  finish();
}

$("inference-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (activeRun !== null) return;
  releaseResults();
  try {
    if (!worker || runtimeState === "error") throw new Error("Initialize the runtime before starting inference.");
    const request = {
      type: "run", id: ++nextRun,
      sumstats: sourceInputs("sumstats"), ld: sourceInputs("ld"),
      options: {model: $("model").value, components: Number($("components").value), seed: Number($("seed").value), max_iter: Number($("max-iter").value), tolerance: Number($("tolerance").value), sumstats_format: $("source").value === "example" ? "fastGWA" : $("sumstats-format").value},
    };
    activeRun = request.id;
    $("controls").disabled = true; $("run").disabled = true; $("cancel").disabled = false;
    $("elapsed-line").hidden = true;
    $("activity").classList.add("running");
    status(runtimeState === "ready" ? "Running" : "Queued", runtimeState === "ready" ? "Preparing inputs" : "Waiting for runtime", "You can cancel this run at any time.");
    worker.postMessage(request);
  } catch (error) { fail(error.message); }
});
$("cancel").addEventListener("click", () => {
  worker?.terminate(); worker = null; finish(); releaseResults();
  status("Cancelled", "Run cancelled", "The runtime is restarting."); log("Run cancelled.");
  try { startWorker(); } catch (error) { runtimeStatus("error", "Runtime unavailable"); fail(error.message); }
});
$("protocol-warning").hidden = location.protocol !== "file:";
syncInputs(); syncModel();
if (location.protocol !== "file:") {
  try { startWorker(); } catch (error) { runtimeStatus("error", "Runtime unavailable"); fail(error.message); }
} else { runtimeStatus("error", "HTTP server required"); $("retry-runtime").hidden = true; }
