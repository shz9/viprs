import {EXAMPLE, resolveDataURL, selectLocalFiles} from "./sources.mjs";
const $ = (id) => document.getElementById(id);
let worker = null, runtimeState = "loading", activeRun = null, nextRun = 0;
let timer = null, startedAt = 0, downloadURLs = [], resultRows = {};

function syncInputs() {
  for (const kind of ["ld", "sumstats"]) {
    for (const source of ["hosted", "catalog", "example", "files", "urls", "folder"]) {
      const panel = $(kind + "-" + source + "-inputs");
      if (panel) panel.hidden = $(kind + "-source").value !== source;
    }
  }
  if ($("sumstats-source").value === "catalog" && Array.from($("sumstats-format").options).some(option => option.value === "gwascatalog")) {
    $("sumstats-format").value = "gwascatalog";
  }
  $("sumstats-format").disabled = runtimeState !== "ready" || $("sumstats-source").value === "catalog";
}
function setSumstatsFormats(formats) {
  if (!Array.isArray(formats) || !formats.length) throw new Error("The runtime did not report any summary-statistics formats.");
  const select = $("sumstats-format"), current = select.value;
  select.replaceChildren(...formats.map((format) => new Option(format, format)));
  select.value = formats.includes(current) ? current : formats.includes("fastgwa") ? "fastgwa" : formats[0];
  syncInputs();
}
function syncHostedLD() {
  $("ld-hosted-path").value = `hf://datasets/shz9/ukb-ld/${$("ld-ancestry").value}/chr_*.zip`;
}
$("ld-ancestry").addEventListener("change", syncHostedLD);
function syncModel() {
  const mixture = $("model").value === "VIPRSMix";
  const componentInput = $("components");
  const components = mixture ? Math.min(9, Math.max(1, Number.parseInt(componentInput.value, 10) || 1)) : 1;
  $("mixture-settings").hidden = !mixture;
  componentInput.disabled = !mixture;
  $("model-help").textContent = mixture
    ? `A point mass at zero with ${components} Gaussian effect-size ${components === 1 ? "component" : "components"}.`
    : "A point mass at zero and one Gaussian component for nonzero effects.";
  renderPrior(components, mixture);
}
$("model").addEventListener("change", syncModel);
$("components").addEventListener("input", syncModel);
$("components").addEventListener("change", () => {
  const input = $("components");
  input.value = Math.min(9, Math.max(1, Number.parseInt(input.value, 10) || 1));
  syncModel();
});

const SVG_NS = "http://www.w3.org/2000/svg";
const SLAB_COLORS = ["#1f6f9c", "#6b55a1", "#178276", "#b36f12", "#9b3f78", "#4f772d", "#a34f34", "#39758a", "#796328"];
function slabPath(index) {
  const baseline = 140, center = 284;
  const sigma = [25, 38, 54, 72, 94, 120, 150, 184, 222][index];
  const height = [92, 82, 72, 63, 55, 49, 44, 40, 36][index];
  const points = [];
  for (let x = 28; x <= 540; x += 4) {
    const y = baseline - height * Math.exp(-0.5 * ((x - center) / sigma) ** 2);
    points.push(`${x},${y.toFixed(2)}`);
  }
  return `M28,${baseline} L${points.join(" L")} L540,${baseline} Z`;
}
function renderPrior(components, mixture) {
  const curves = [];
  for (let index = components - 1; index >= 0; index -= 1) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", slabPath(index));
    path.setAttribute("fill", SLAB_COLORS[index]);
    path.setAttribute("stroke", SLAB_COLORS[index]);
    path.setAttribute("class", "slab-curve");
    curves.push(path);
  }
  $("slab-curves").replaceChildren(...curves);
  const slabWord = components === 1 ? "slab" : "slabs";
  $("prior-summary").value = `1 spike + ${components} ${slabWord}`;
  $("slab-key").textContent = components === 1 ? "Slab · nonzero effects" : `${components} slabs · different effect scales`;
  $("prior-plot-description").textContent = `A point mass at zero and ${components === 1 ? "one centered Gaussian slab" : `${components} centered Gaussian slabs with different variances`} for nonzero genetic effects.`;
}
$("demo").addEventListener("click", () => {
  $("ld-source").value = "example";
  $("sumstats-source").value = "urls";
  $("sumstats-url").value = resolveDataURL(EXAMPLE.sumstats, location.href);
  $("sumstats-format").value = "fastgwa";
  syncInputs();
});
for (const kind of ["ld", "sumstats"]) {
  $(kind + "-source").addEventListener("change", syncInputs);
  $(kind + "-file").addEventListener("change", () => {
    $(kind + "-file-summary").textContent = `${$(kind + "-file").files.length} file(s) selected`;
  });
  const updateFolderSummary = () => {
    const files = $(kind + "-folder").files;
    if (!files.length) {
      $(kind + "-folder-summary").textContent = "Choose a folder to see matching file counts.";
      return;
    }
    try {
      const count = selectLocalFiles(files, $(kind + "-pattern").value).length;
      $(kind + "-folder-summary").textContent = `${count} file(s) matched`;
    } catch (error) { $(kind + "-folder-summary").textContent = error.message; }
  };
  $(kind + "-folder").addEventListener("change", updateFolderSummary);
  $(kind + "-pattern").addEventListener("input", updateFolderSummary);
}

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
  $("controls").disabled = false; $("demo").disabled = false;
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
      try { setSumstatsFormats(data.sumstats_formats); }
      catch (error) { runtimeStatus("error", "Initialization failed"); fail(error.message); return; }
      runtimeStatus("ready", "Runtime ready");
      syncInputs();
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
    if (data.type === "chromosome-complete") {
      status("Running", "Fit progress", data.message);
      log(data.message);
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
  const source = $(kind + "-source").value;
  if (kind === "ld" && source === "hosted") return [{url: $("ld-hosted-path").value, stream: true}];
  if (kind === "sumstats" && source === "catalog") {
    const accession = $("sumstats-catalog-id").value.trim().toUpperCase();
    if (!/^GCST\d+$/.test(accession)) throw new Error("Enter a GWAS Catalog accession such as GCST005195.");
    return [{catalog: accession}];
  }
  if (source === "example") return [{url: EXAMPLE[kind], stream: kind === "ld"}];
  if (source === "folder") {
    if (!$(kind + "-folder").files.length) throw new Error(`Choose a local ${kind} folder first.`);
    const files = selectLocalFiles($(kind + "-folder").files, $(kind + "-pattern").value);
    if (files.some(file => !file.size)) throw new Error("Each input file must be nonempty.");
    return files.map(file => ({file}));
  }
  if (source === "files") {
    const files = Array.from($(kind + "-file").files);
    if (!files.length || files.length > 64) throw new Error(`Select 1–64 ${kind === "ld" ? "LD archives" : "summary-statistics files"}.`);
    if (files.some((file) => !file.size)) throw new Error("Each input file must be nonempty.");
    return files.map((file) => ({file}));
  }
  const lines = $(kind + "-url").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (!lines.length || lines.length > 64) throw new Error(`Enter 1–64 ${kind} URLs, one per line.`);
  const urls = lines.map((line) => ({
    raw: line,
    url: resolveDataURL(line, location.href),
  }));
  if (new Set(urls.map(({url}) => url)).size !== urls.length) throw new Error(`Remove duplicate ${kind} URLs.`);
  return urls.map(({raw, url}) => ({
    url: kind === "ld" && /^hf:\/\//i.test(raw) ? raw : url,
    stream: kind === "ld" && /^hf:\/\//i.test(raw),
  }));
}
function renderRows() {
  const chromosome = $("chromosome-filter").value;
  const rows = resultRows[chromosome] || [];
  $("result-rows").replaceChildren();
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const key of ["CHR", "SNP", "A1", "A2", "PIP", "BETA", "VAR_BETA"]) {
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
  $("result-summary").textContent = `${report.model}${report.model === "VIPRSMix" ? ` · ${report.components} non-null components` : ""} · ${report.variants.toLocaleString()} variants across ${report.chromosomes.length} chromosome(s).`;
  $("results").hidden = $("metrics").hidden = $("timing-summary").hidden = false;
  $("elapsed").textContent = `${report.timings.run_seconds.toFixed(1)}s`;
  status(report.converged ? "Converged" : "Iteration limit", "Analysis complete", "");
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
      options: {model: $("model").value, components: Number($("components").value), seed: Number($("seed").value), max_iter: Number($("max-iter").value), tolerance: Number($("tolerance").value), sumstats_format: $("sumstats-source").value === "catalog" ? "gwascatalog" : $("sumstats-format").value},
    };
    activeRun = request.id;
    $("controls").disabled = true; $("demo").disabled = true; $("run").disabled = true; $("cancel").disabled = false;
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
syncInputs(); syncModel(); syncHostedLD();
if (location.protocol !== "file:") {
  try { startWorker(); } catch (error) { runtimeStatus("error", "Runtime unavailable"); fail(error.message); }
} else { runtimeStatus("error", "HTTP server required"); $("retry-runtime").hidden = true; }
