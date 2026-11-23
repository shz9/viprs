import {validateCatalog} from "./catalog.mjs";
import {formatDuration} from "./time.js";
import "./errors.js";
import {resultsPanel} from "./results.js";
import {SUMSTATS_FORMAT_HELP} from "./formats.js";
import {EXAMPLE, HOSTED_LD_PANELS, LD_ANCESTRIES, resolveDataURL, selectLocalFiles} from "./sources.mjs";
const $ = (id) => document.getElementById(id);
let worker = null, runtimeState = "idle", activeRun = null, nextRun = 0;
let session = null, runtimeInfo = null, errorURL = null;
let runtimeTimer = null;
const RUNTIME_INACTIVITY_MS = 180000;
let timer = null, startedAt = 0, downloadURLs = [], resultRows = {};
let resultTSV = "", tableSearchVersion = 0, tableSearchTimer = null;
const sumstatsFormatHelp = SUMSTATS_FORMAT_HELP;
const sumstatsFormatAliases = {
  plink: "plink2", ssf: "gwascatalog", "gwas-ssf": "gwascatalog",
  sumher: "ldak", megaprs: "ldak",
};
const sumstatsFormatLabels = {
  magenpy: "Magenpy", plink2: "PLINK 2", "plink1.9": "PLINK 1.9",
  cojo: "GCTA-COJO", fastgwa: "GCTA fastGWA", gwascatalog: "GWAS Catalog",
  saige: "SAIGE", regenie: "REGENIE", ldak: "LDAK / SumHer / MegaPRS",
};
const formatLabel = format => sumstatsFormatLabels[format] || sumstatsFormatHelp[format]?.name || format;

let catalogValidation = null, catalogController = null, catalogVersion = 0;
function currentCatalogValidation() {
  return catalogValidation?.accession === $("sumstats-catalog-id").value.trim().toUpperCase() ? catalogValidation : null;
}
function catalogMessage(message, state = "") {
  $("catalog-validation").textContent = message;
  $("catalog-validation").dataset.state = state;
}
function syncCatalogOptions() {
  const result = $("sumstats-source").value === "catalog" ? currentCatalogValidation() : null;
  const select = $("ld-cohort"), previous = select.value;
  const panels = Object.entries(HOSTED_LD_PANELS).filter(([, panel]) => !result?.complete || result.files[panel.genome_build]);
  const ids = panels.map(([id]) => id);
  if (Array.from(select.options).map(option => option.value).join() !== ids.join()) {
    select.replaceChildren(...panels.map(([id, panel]) => new Option(`${panel.name} · ${panel.build_label}`, id)));
    select.value = ids.includes(previous) ? previous : ids[0] || "";
    if (select.value && select.value !== previous) syncHostedLD();
  }
  for (const option of $("ld-custom-build").options) option.disabled = Boolean(option.value && result?.complete && !result.files[option.value]);
  const build = selectedLDBuild();
  const sizes = result?.file_sizes || {};
  const largeFiles = Object.entries(sizes).filter(([fileBuild, bytes]) => (!build || fileBuild === build) && bytes > 3e9);
  $("catalog-memory-warning").hidden = !largeFiles.length;
  $("catalog-memory-warning-message").textContent = largeFiles.length
    ? `Large download (${largeFiles.map(([, bytes]) => `${(bytes / 1e9).toFixed(1)} GB`).join(", ")}). Parsing may exceed browser memory;` : "";
  if (!result) return;
  if (Object.keys(result.files).length) catalogMessage(`Valid study ID. Select compatible LD reference panel (${Object.keys(result.files).sort().join(" or ")}).`, "valid");
  else catalogMessage("Unable to verify genome build.", "invalid");
}
$("sumstats-catalog-id").addEventListener("input", () => {
  catalogVersion++; catalogController?.abort(); catalogValidation = null;
  $("catalog-validate").disabled = false;
  catalogMessage("ID changed. Please validate again.");
  syncCatalogOptions();
});
$("catalog-validate").addEventListener("click", async () => {
  const version = ++catalogVersion;
  catalogController?.abort(); catalogController = new AbortController(); catalogValidation = null;
  syncCatalogOptions();
  $("catalog-validate").disabled = true;
  catalogMessage("Checking study files and builds.");
  try {
    const result = await validateCatalog($("sumstats-catalog-id").value, {signal: catalogController.signal});
    if (version !== catalogVersion) return;
    catalogValidation = result; syncCatalogOptions();
  } catch (error) {
    if (version === catalogVersion) {
      const message = /accession|out of range/.test(error.message) ? "Invalid ID. Check the accession."
        : /No summary-statistics/.test(error.message) ? "Study files were not found."
        : /no supported TSV/.test(error.message) ? "No supported study files found."
        : "Validation failed. Please try again.";
      catalogMessage(message, "invalid");
    }
  } finally {
    if (version === catalogVersion) $("catalog-validate").disabled = false;
  }
});
function syncInputs() {
  for (const kind of ["ld", "sumstats"]) {
    for (const source of ["hosted", "catalog", "files", "urls", "folder"]) {
      const panel = $(kind + "-" + source + "-inputs");
      if (panel) panel.hidden = $(kind + "-source").value !== source;
    }
  }
  if ($("sumstats-source").value === "catalog" && Array.from($("sumstats-format").options).some(option => option.value === "gwascatalog")) {
    $("sumstats-format").value = "gwascatalog";
  }
  $("sumstats-format").disabled = $("sumstats-source").value === "catalog";
  $("sumstats-format-info").disabled = !sumstatsFormatHelp[$("sumstats-format").value];
  syncGenomeBuild();
}
function setSumstatsFormats(formats) {
  if (!Array.isArray(formats) || !formats.length) throw new Error("The runtime did not report any summary-statistics formats.");
  formats = [...new Set(formats.map(format => sumstatsFormatAliases[format] || format))].sort();
  const select = $("sumstats-format"), current = sumstatsFormatAliases[select.value] || select.value;
  select.replaceChildren(...formats.map((format) => new Option(formatLabel(format), format)));
  select.value = formats.includes(current) ? current : formats.includes("fastgwa") ? "fastgwa" : formats[0];
  syncInputs();
}
function renderFormatHelp() {
  const guide = sumstatsFormatHelp[$("sumstats-format").value];
  if (!guide) return;
  $("sumstats-format-title").textContent = formatLabel($("sumstats-format").value);
  $("sumstats-format-separator-name").textContent = guide.separator;
  $("sumstats-format-separator-symbol").textContent = guide.separator_symbol;
  $("sumstats-format-notes").textContent = guide.notes;
  $("sumstats-format-example").textContent = guide.example;
  $("sumstats-format-docs").href = guide.url;
  $("sumstats-format-columns").replaceChildren(...guide.columns.map(([column, meaning]) => {
    const row = document.createElement("tr");
    for (const value of [column, meaning]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    return row;
  }));
}
$("sumstats-format-info").addEventListener("click", () => {
  renderFormatHelp();
  $("sumstats-format-dialog").showModal();
});
$("ld-dataset-info").addEventListener("click", () => $("ld-dataset-dialog").showModal());
for (const name of ["sumstats-format", "ld-dataset"]) {
  const dialog = $(name + "-dialog");
  $(name + "-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
}
$("sumstats-format").addEventListener("change", () => {
  syncInputs();
  if ($("sumstats-format-dialog").open) renderFormatHelp();
});
function selectedLDBuild() {
  const source = $("ld-source").value;
  if (source === "hosted") return HOSTED_LD_PANELS[$("ld-cohort").value || "ukb"].genome_build;
  return $("ld-custom-build").value || null;
}
function buildLabel(build) {
  return build === "GRCh37" || build === "GRCh38" ? build : "an unknown genome build";
}
function syncGenomeBuild() {
  syncCatalogOptions();
  const build = selectedLDBuild();
  const source = $("ld-source").value;
  $("ld-custom-build-inputs").hidden = source === "hosted";
  $("ld-build-guidance").textContent = source === "hosted"
    ? "Choose the LD reference dataset that matches your GWAS."
    : build ? `Match summary statistics to ${buildLabel(build)}.`
    : "Genome build is read from LD metadata. Select it above only if missing.";
}
function syncHostedLD() {
  const panel = HOSTED_LD_PANELS[$("ld-cohort").value || "ukb"];
  const previous = $("ld-ancestry").value;
  $("ld-ancestry").replaceChildren(...panel.ancestries.map(code => new Option(`${code} · ${LD_ANCESTRIES[code]}`, code)));
  $("ld-ancestry").value = panel.ancestries.includes(previous) ? previous : panel.ancestries[0];
  $("ld-hosted-path").value = `hf://datasets/${panel.repository}/${$("ld-ancestry").value}/chr_*.zip`;
  $("ld-panel-name").textContent = panel.name;
  $("ld-snp-set").textContent = panel.snp_set;
  $("ld-genome-build").textContent = panel.build_label;
  $("ld-panel-description").textContent = panel.description;
  $("ld-panel-link").href = `https://huggingface.co/datasets/${panel.repository}`;
  syncGenomeBuild();
}
$("ld-cohort").replaceChildren(...Object.entries(HOSTED_LD_PANELS).map(([id, panel]) => new Option(`${panel.name} · ${panel.build_label}`, id)));
$("ld-cohort").value = "ukb";
$("ld-cohort").addEventListener("change", syncHostedLD);
$("ld-ancestry").addEventListener("change", syncHostedLD);
$("ld-custom-build").addEventListener("change", syncGenomeBuild);
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
function useExampleLD() {
  $("ld-source").value = "urls";
  $("ld-url").value = EXAMPLE.ld;
  $("ld-custom-build").value = "GRCh37";
}
$("demo").addEventListener("click", () => {
  useExampleLD();
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
function runtimeStatus(state, text, source = null) {
  clearTimeout(runtimeTimer); runtimeTimer = null;
  if (state === "loading") {
    const current = worker;
    // This timer runs on the page, so it can recover even if a synchronous
    // Python import has blocked the worker's own event loop.
    runtimeTimer = setTimeout(() => {
      if (worker !== current || runtimeState !== "loading") return;
      current?.terminate(); worker = null;
      runtimeStatus("error", "Initialization stalled — retry available");
      fail(VIPRSErrors.describe(new Error(`No initialization progress for three minutes. Last step: ${text}. Check the connection to the package host, then retry the runtime.`), "runtime", source));
    }, RUNTIME_INACTIVITY_MS);
  }
  runtimeState = state;
  $("runtime-state").dataset.state = state;
  $("runtime-text").textContent = text;
  $("retry-runtime").hidden = state !== "error";
  $("run").disabled = activeRun !== null || state === "error";
}
function releaseResults() {
  downloadURLs.forEach((url) => URL.revokeObjectURL(url)); downloadURLs = [];
  $("results").hidden = $("metrics").hidden = $("timing-summary").hidden = $("elapsed-line").hidden = true;
  $("result-rows").replaceChildren(); resultRows = {}; resultTSV = "";
  tableSearchVersion++; clearTimeout(tableSearchTimer);
  $("table-rsid").value = ""; $("table-empty").hidden = true;
  resultsPanel.clear();
  $("download").removeAttribute("href");
}
const progressStages = ["sumstats", "harmonization", "fitting"];
function resetProgress() {
  $("run-progress").hidden = false;
  for (const stage of progressStages) updateProgress({stage, state: "waiting", completed: 0, total: 1, detail: "Waiting"});
}
function updateProgress(data) {
  if (![...progressStages, "ld"].includes(data.stage)) return;
  if (progressStages.includes(data.stage)) {
    const bar = $("progress-" + data.stage), row = $("progress-" + data.stage + "-row");
    row.dataset.state = data.state;
    $("progress-" + data.stage + "-detail").textContent = data.detail || (data.state === "done" ? "Complete" : "In progress");
    bar.setAttribute("aria-valuetext", $("progress-" + data.stage + "-detail").textContent);
    if (Number.isFinite(data.total) && data.total > 0 && Number.isFinite(data.completed)) {
      bar.max = data.total; bar.value = Math.max(0, Math.min(data.total, data.completed));
    } else if (data.state === "active") bar.removeAttribute("value");
    else { bar.max = 1; bar.value = data.state === "done" ? 1 : 0; }
  }
  const description = [data.heading, data.message].filter(Boolean).join(" · ");
  if (description) status("Running", data.stage === "fitting" ? "Model fitting" : $("activity-title").textContent, description);
  if (description && !data.quiet) log(description);
}
function stopProgress(state) {
  for (const stage of progressStages) {
    const row = $("progress-" + stage + "-row");
    if (row.dataset.state !== "active") continue;
    const bar = $("progress-" + stage);
    updateProgress({stage, state, completed: bar.value, total: bar.max, detail: state === "error" ? "Stopped after an error" : "Cancelled"});
  }
}
function finish() {
  activeRun = null; clearInterval(timer); timer = null;
  $("activity").classList.remove("running");
  $("controls").disabled = false; $("demo").disabled = false;
  $("run").disabled = runtimeState === "error"; $("cancel").disabled = true;
}
function clearError() {
  if (errorURL) URL.revokeObjectURL(errorURL);
  errorURL = null;
  $("error-support").hidden = true;
  for (const kind of ["sumstats", "ld"]) {
    $(kind + "-error").hidden = true;
    document.querySelector("." + kind + "-card").classList.remove("has-error");
  }
}
function captureSession() {
  return {
    options: {model: $("model").value, components: $("model").value === "VIPRSMix" ? Number($("components").value) : 1, seed: Number($("seed").value), max_iter: Number($("max-iter").value), tolerance: Number($("tolerance").value), sumstats_format: $("sumstats-source").value === "catalog" ? "gwascatalog" : $("sumstats-format").value, ld_genome_build: selectedLDBuild()},
    inputs: Object.fromEntries(["sumstats", "ld"].map(kind => {
      const source = $(kind + "-source").value;
      let inputs;
      if (["files", "folder"].includes(source)) inputs = Array.from($(kind + (source === "files" ? "-file" : "-folder")).files).map(file => ({name: file.name, bytes: file.size, last_modified: file.lastModified}));
      else inputs = source === "catalog" ? $("sumstats-catalog-id").value : source === "hosted" ? $("ld-hosted-path").value : $(kind + "-url").value;
      return [kind, {source, inputs, ...(source === "folder" ? {pattern: $(kind + "-pattern").value} : {})}];
    })),
  };
}
function fail(error, stage = "runtime") {
  const detail = error?.category ? error : VIPRSErrors.describe(error, stage);
  clearError();
  log(detail.title + ": " + detail.message);
  status("Error", detail.title, detail.message, true);
  stopProgress("error");
  finish();
  $("error-advice").textContent = detail.advice;
  const report = VIPRSErrors.safeReport({schema_version: 1, timestamp: new Date().toISOString(),
    error: detail, session: session || captureSession(), runtime: runtimeInfo,
    environment: {page: location.origin + location.pathname, browser: navigator.userAgent,
      language: navigator.language, device_memory_gb: navigator.deviceMemory ?? null, hardware_concurrency: navigator.hardwareConcurrency},
    log: $("log").textContent,
  });
  errorURL = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: "application/json"}));
  $("download-error").href = errorURL;
  $("report-error").href = VIPRSErrors.issueURL(report);
  $("error-support").hidden = false;
  const kind = ["sumstats", "ld"].includes(detail.stage) ? detail.stage : null;
  if (kind) {
    $(kind + "-error").textContent = detail.title + ": " + detail.message;
    $(kind + "-error").hidden = false;
    document.querySelector("." + kind + "-card").classList.add("has-error");
    $(kind + "-error").focus();
  } else $("error-support").focus();
}
function beginTimer() {
  startedAt = performance.now(); $("elapsed").textContent = formatDuration(0);
  $("elapsed-line").hidden = false;
  timer = setInterval(() => { $("elapsed").textContent = formatDuration((performance.now() - startedAt) / 1000); }, 1000);
}

function startWorker() {
  worker?.terminate();
  const current = new Worker(new URL("worker.js?v=runtime-0.2.5-catalog-4", import.meta.url));
  worker = current;
  runtimeStatus("loading", "Preparing runtime");
  log("Starting runtime initialization. Packages will load before inference begins.");
  current.onmessage = ({data}) => {
    if (worker !== current || !data || typeof data !== "object") return;
    if (data.type === "ld-metadata") {
      $("ld-build-guidance").textContent = `LD genome build: ${buildLabel(data.genome_build)}. Match your summary statistics to this build.`;
    }
    if (data.type === "runtime-status") {
      runtimeStatus("loading", data.message, data.source);
      log(data.message);
      status("Loading", "Loading inference environment", data.message);
    }
    if (data.type === "ready") {
      runtimeInfo = data.runtime || runtimeInfo;
      runtimeStatus("ready", "Runtime ready");
      syncInputs();
      if (activeRun === null) {
        status("Ready", "Ready for inference", "Select your inputs and model.");
      }
      log(`Runtime initialized in ${data.seconds.toFixed(1)}s; ready for reuse.`);
    }
    if (data.type === "init-error") {
      runtimeStatus("error", "Initialization failed"); fail(data.error || data.message);
    }
    if (data.type === "log" && (data.id == null || data.id === activeRun)) log(data.message);
    if (data.id !== activeRun || activeRun === null) return;
    if (data.type === "stage-progress") updateProgress(data);
    if (data.type === "phase") {
      if (data.started) beginTimer();
      const majorTitles = {inputs: "Preparing inputs", fitting: "Model fitting", results: "Preparing results"};
      status("Running", majorTitles[data.major] || $("activity-title").textContent,
        data.message + (data.detail ? ": " + data.detail : ""));
      if (!data.quiet) log(data.message + (data.detail ? ": " + data.detail : ""));
    }
    if (data.type === "chromosome-complete") {
      status("Running", $("activity-title").textContent, data.message);
      log(data.message);
    }
    if (data.type === "result") {
      try { showResult(data.result); }
      catch (error) {
        releaseResults();
        fail(error, "results");
      }
    }
    if (data.type === "error") {
      if (["memory", "cleanup"].includes(data.error?.category)) {
        current.terminate(); worker = null; runtimeStatus("error", "Restart runtime before retrying");
      }
      fail(data.error || data.message);
    }
  };
  current.onerror = (event) => {
    if (worker !== current) return;
    event.preventDefault(); current.terminate(); worker = null;
    runtimeStatus("error", "Runtime unavailable");
    fail(event.message || "The worker stopped without providing an error message.", "worker");
  };
}
$("retry-runtime").addEventListener("click", () => { clearError(); session = null; status("Ready", "Preparing runtime", "You can run inference when the runtime is ready."); try { startWorker(); } catch (error) { runtimeStatus("error", "Runtime unavailable"); fail(error.message); } });

function sourceInputs(kind) {
  const source = $(kind + "-source").value;
  if (kind === "ld" && source === "hosted") return [{url: $("ld-hosted-path").value, stream: true}];
  if (kind === "sumstats" && source === "catalog") {
    const accession = $("sumstats-catalog-id").value.trim().toUpperCase();
    if (!/^GCST\d+$/.test(accession)) throw new Error("Enter a GWAS Catalog accession such as GCST005195.");
    const validation = currentCatalogValidation(), build = selectedLDBuild();
    if (validation?.complete && build && !validation.files[build]) throw new Error(`This study has no ${build} file. Select a compatible LD panel shown by validation, or choose another study.`);
    return [{catalog: accession, ...(validation ? {catalog_files: {...validation.files}} : {})}];
  }
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
function drawTableRows(rows, caption) {
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
  $("table-caption").textContent = caption;
  $("table-empty").hidden = rows.length > 0;
}
async function renderRows() {
  const version = ++tableSearchVersion, chromosome = $("chromosome-filter").value;
  const query = $("table-rsid").value.trim().toLowerCase();
  if (!query) {
    drawTableRows(resultRows[chromosome] || [], chromosome ? `Top variants · chromosome ${chromosome}` : "Top variants · all chromosomes");
    return;
  }
  const ids = new Set(query.split(/[\s,;]+/).filter(Boolean));
  const lines = resultTSV.matchAll(/[^\r\n]+/g), header = lines.next().value?.[0].split("\t") || [];
  const snpIndex = header.indexOf("SNP"), chromosomeIndex = header.indexOf("CHR");
  drawTableRows([], "Searching the complete fitted result…"); $("table-empty").hidden = true;
  const matches = []; let scanned = 0, count = 0;
  for (const [line] of lines) {
    const cells = line.split("\t");
    if (ids.has((cells[snpIndex] || "").toLowerCase()) && (!chromosome || cells[chromosomeIndex] === chromosome)) {
      count++;
      if (matches.length < 200) {
        const row = Object.fromEntries(header.map((key, index) => [key, cells[index]]));
        for (const key of ["PIP", "BETA", "VAR_BETA"]) row[key] = row[key] ? Number(row[key]) : null;
        matches.push(row);
      }
    }
    if (++scanned % 10000 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
      if (version !== tableSearchVersion) return;
    }
  }
  if (version !== tableSearchVersion) return;
  matches.sort((a, b) => (b.PIP || 0) - (a.PIP || 0));
  drawTableRows(matches, `${count.toLocaleString()} matching fitted variant(s)${chromosome ? ` · chromosome ${chromosome}` : " · all chromosomes"}${count > 200 ? " · showing the first 200 matches" : ""}`);
}
$("chromosome-filter").addEventListener("change", () => { clearTimeout(tableSearchTimer); renderRows(); });
$("table-rsid").addEventListener("input", () => {
  tableSearchVersion++; clearTimeout(tableSearchTimer);
  tableSearchTimer = setTimeout(renderRows, 180);
});
function showResult(result) {
  resultTSV = result.tsv;
  $("table-rsid").value = "";
  resultRows = {"": result.rows, ...result.chromosome_rows};
  $("chromosome-filter").replaceChildren(new Option("All chromosomes", ""));
  for (const c of result.report.chromosomes) $("chromosome-filter").add(new Option(`Chromosome ${c} · ${result.report.variants_per_chromosome[c].toLocaleString()} variants`, c));
  renderRows();
  const report = result.report;
  resultsPanel.show(result, (id, data, type) => {
    const url = URL.createObjectURL(new Blob([data], {type}));
    downloadURLs.push(url); return url;
  });
  $("variants").textContent = report.variants.toLocaleString();
  $("chromosomes").textContent = report.chromosomes.length;
  $("iterations").textContent = report.iterations;
  $("timing-summary").textContent = `Input preparation ${formatDuration(report.timings.preparation_seconds)} · Model fitting ${formatDuration(report.timings.fit_seconds)}. Runtime initialization is excluded.`;
  $("result-summary").textContent = `${report.model}${report.model === "VIPRSMix" ? ` · ${report.components} non-null components` : ""} · ${report.variants.toLocaleString()} variants across ${report.chromosomes.length} chromosome(s).`;
  $("results").hidden = $("metrics").hidden = $("timing-summary").hidden = false;
  $("elapsed").textContent = formatDuration(report.timings.run_seconds);
  status(report.converged ? "Converged" : "Iteration limit", "Analysis complete", "");
  finish();
}

$("inference-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (activeRun !== null) return;
  releaseResults(); clearError(); session = captureSession();
  let stage = "runtime";
  try {
    if (runtimeState === "error") throw new Error("Restart the runtime before starting inference.");
    stage = "sumstats";
    const sumstats = sourceInputs("sumstats");
    stage = "ld";
    const ld = sourceInputs("ld");
    stage = "validation";
    const request = {
      type: "run", id: ++nextRun, sumstats, ld,
      options: {...session.options},
    };
    if (!worker) startWorker();
    activeRun = request.id;
    resetProgress();
    $("controls").disabled = true; $("demo").disabled = true; $("run").disabled = true; $("cancel").disabled = false;
    $("elapsed-line").hidden = true;
    $("activity").classList.add("running");
    status(runtimeState === "ready" ? "Running" : "Queued", runtimeState === "ready" ? "Preparing inputs" : "Loading inference environment", "You can cancel this run at any time.");
    worker.postMessage(request);
  } catch (error) { fail(error, stage); }
});
$("cancel").addEventListener("click", () => {
  worker?.terminate(); worker = null; stopProgress("cancelled"); finish(); releaseResults(); clearError();
  runtimeStatus("idle", "Starts when you run inference");
  status("Cancelled", "Run cancelled", "Choose your inputs and run again when ready."); log("Run cancelled; runtime released.");
});
$("protocol-warning").hidden = location.protocol !== "file:";
setSumstatsFormats(Object.keys(sumstatsFormatHelp).sort());
syncInputs(); syncModel(); syncHostedLD();
if (location.protocol !== "file:") {
  runtimeStatus("idle", "Starts when you run inference");
} else { runtimeStatus("error", "HTTP server required"); $("retry-runtime").hidden = true; }
