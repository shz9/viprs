import {formatDuration} from "../../docs/webapp/time.js";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";
import {EXAMPLE, HOSTED_LD_PANELS, LD_ANCESTRIES, resolveDataURL, selectLocalFiles} from "../../docs/webapp/sources.mjs";
import "../../docs/webapp/errors.js";
import {SUMSTATS_FORMAT_HELP} from "../../docs/webapp/formats.js";

// Exercise actual page handlers with a minimal DOM; no inference downloads needed.
function page(overrides = {}) {
  const nodes = new Map(), workers = [], blobs = new Map(), timers = new Map();
  let nextTimer = 0;
  const element = () => ({value: "", textContent: "", hidden: false, disabled: false, files: [], options: [], dataset: {}, handlers: {},
    classList: {add() {}, remove() {}}, setAttribute() {}, removeAttribute() {}, append() {}, add() {},
    replaceChildren(...items) { this.options = items; }, focus() { this.focused = true; },
    addEventListener(type, handler) { this.handlers[type] = handler; }});
  const get = id => { if (!nodes.has(id)) nodes.set(id, element()); return nodes.get(id); };
  for (const [id, value] of Object.entries({"sumstats-source": "catalog", "ld-source": "hosted", model: "VIPRS", components: "3", seed: "0", "max-iter": "1000", tolerance: "0.000001"})) get(id).value = value;
  get("status-badge").textContent = "Ready";
  class Worker { constructor() { workers.push(this); } terminate() { this.terminated = true; } postMessage(data) { this.request = data; } }
  class URLs extends URL { static createObjectURL(blob) { const url = `blob:${blobs.size}`; blobs.set(url, blob); return url; } static revokeObjectURL(url) { blobs.delete(url); } }
  const context = vm.createContext({document: {getElementById: get, querySelector: get, createElementNS: element, createElement: element},
    location: new URL("https://example.org/webapp/"), navigator: {userAgent: "Test browser", language: "en"},
    setTimeout: (callback, delay) => { const id = ++nextTimer; timers.set(id, {callback, delay}); return id; },
    clearTimeout: id => timers.delete(id),
    Worker, URL: URLs, Blob, performance, setInterval, clearInterval, EXAMPLE, HOSTED_LD_PANELS, LD_ANCESTRIES, resolveDataURL, selectLocalFiles,
    resultsPanel: {clear() {}, show() {}}, formatDuration,
    VIPRSErrors: globalThis.VIPRSErrors, SUMSTATS_FORMAT_HELP, Option: function(text, value) { return {text, value}; }, ...overrides});
  const source = readFileSync(new URL("../../docs/webapp/app.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/gm, "").replaceAll("import.meta.url", '"https://example.org/webapp/app.js"');
  vm.runInContext(source, context);
  return {get, workers, blobs, timers, submit: () => get("inference-form").handlers.submit({preventDefault() {}})};
}
test("input validation is placed beside summary controls and exports a snapshot", async () => {
  const {get, submit, blobs} = page();
  submit();
  assert.equal(get("status-badge").textContent, "Error");
  assert.match(get("status").textContent, /GWAS Catalog accession/);
  assert.equal(get("sumstats-error").hidden, false);
  assert.equal(get("sumstats-error").focused, true);
  assert.equal(get("error-support").hidden, false);
  const report = JSON.parse(await blobs.get(get("download-error").href).text());
  assert.equal(report.error.stage, "sumstats");
  assert.equal(report.session.options.max_iter, 1000);
  assert.match(get("report-error").href, /github.com/);
});
test("memory failure releases controls and requires restart; stale messages are ignored", async () => {
  const {get, workers, submit, blobs} = page();
  get("sumstats-catalog-id").value = "GCST005195";
  submit();
  const worker = workers[0];
  const error = globalThis.VIPRSErrors.describe({message: "Traceback\nMemoryError: Cannot allocate memory"}, "inference");
  worker.onmessage({data: {type: "error", id: worker.request.id, error}});
  assert.equal(get("activity-title").textContent, "Browser memory limit reached");
  assert.equal(get("status").textContent, "Cannot allocate memory");
  assert.equal(get("controls").disabled, false);
  assert.equal(get("run").disabled, true);
  assert.equal(get("retry-runtime").hidden, false);
  assert(worker.terminated);
  const report = JSON.parse(await blobs.get(get("download-error").href).text());
  assert.match(report.error.details, /Traceback/);
  get("retry-runtime").handlers.click();
  assert.equal(get("error-support").hidden, true);
  assert.equal(blobs.size, 0);
  worker.onmessage({data: {type: "error", id: worker.request.id, error}});
  assert.equal(get("error-support").hidden, true);
});


test("formats and help are available without starting an inference worker", () => {
  const {get, workers} = page();
  assert.equal(workers.length, 0);
  assert(get("sumstats-format").options.some(option => option.value === "gwascatalog" && option.text === "GWAS Catalog"));
  assert(get("sumstats-format").options.some(option => option.value === "fastgwa" && option.text === "GCTA fastGWA"));
  const options = get("sumstats-format").options;
  assert.equal(options.length, 9);
  assert.equal(new Set(options.map(option => option.text)).size, 9);
  assert(!options.some(option => ["plink", "ssf", "gwas-ssf", "sumher", "megaprs"].includes(option.value)));
  assert(options.some(option => option.value === "plink2" && option.text === "PLINK 2"));
  assert.equal(get("sumstats-format-info").disabled, false);
});


test("requests and error reports use the selected model's effective components", async () => {
  for (const [model, stored, expected] of [["VIPRS", "3", 1], ["VIPRSMix", "3", 3], ["VIPRSMix", "9", 9], ["VIPRS", "9", 1]]) {
    const {get, workers, blobs, submit} = page();
    get("model").value = model;
    get("components").value = stored;
    get("model").handlers.change();
    get("sumstats-catalog-id").value = "GCST005195";
    submit();
    const worker = workers[0];
    assert.equal(worker.request.options.components, expected);
    const error = globalThis.VIPRSErrors.describe(new Error("Unreadable summary statistics"), "sumstats");
    worker.onmessage({data: {type: "error", id: worker.request.id, error}});
    const report = JSON.parse(await blobs.get(get("download-error").href).text());
    assert.equal(report.session.options.components, expected);
    const body = new URL(get("report-error").href).searchParams.get("body");
    const summary = JSON.parse(body.split("```json\n")[1].split("\n```")[0]);
    assert.equal(summary.options.components, expected);
    assert.equal(summary.options.model, model);
  }
});


test("changing the example format propagates the selection to the worker", () => {
  for (const format of ["fastgwa", "ldak", "regenie"]) {
    const {get, workers, submit} = page();
    get("demo").handlers.click();
    get("sumstats-format").value = format;
    get("sumstats-format").handlers.change();
    assert.equal(get("sumstats-format").disabled, false);
    submit();
    assert.equal(workers[0].request.options.sumstats_format, format);
  }
});


test("initialization shows its current step and a stall offers a clean retry", async () => {
  const {get, workers, blobs, timers, submit} = page();
  get("sumstats-catalog-id").value = "GCST005195";
  submit();
  const old = workers[0];
  old.onmessage({data: null});
  const status = {type: "runtime-status", message: "Installing model package 6/6: zarr.whl", source: "https://example.org/wheels/zarr.whl"};
  old.onmessage({data: status});
  assert.match(get("log").textContent, /Installing model package 6/);
  assert.equal(get("activity-title").textContent, "Loading inference environment");
  assert.equal(get("status").textContent, status.message);
  assert.equal(timers.size, 1);
  const timeout = [...timers.values()][0];
  assert.equal(timeout.delay, 180000);
  timeout.callback();
  assert(old.terminated);
  assert.equal(get("runtime-state").dataset.state, "error");
  assert.equal(get("retry-runtime").hidden, false);
  assert.equal(get("controls").disabled, false);
  assert.match(get("status").textContent, /Last step: Installing model package 6/);
  const report = JSON.parse(await blobs.get(get("download-error").href).text());
  assert.equal(report.error.source, status.source);
  old.onmessage({data: {type: "ready", seconds: 1}});
  assert.equal(get("runtime-state").dataset.state, "error");
  get("retry-runtime").handlers.click();
  assert.equal(workers.length, 2);
  assert.equal(timers.size, 1);
  workers[1].onmessage({data: {type: "ready", seconds: 1}});
  assert.equal(timers.size, 0);
  assert.equal(get("runtime-state").dataset.state, "ready");
});

test("cancelling initialization clears its timeout", () => {
  const {get, timers, workers, submit} = page();
  get("sumstats-catalog-id").value = "GCST005195";
  submit();
  assert.equal(timers.size, 1);
  get("cancel").handlers.click();
  assert.equal(timers.size, 0);
  assert(workers[0].terminated);
  assert.equal(get("runtime-state").dataset.state, "idle");
});


test("major status headings stay stable across frequent substeps", () => {
  let now = 0, tick;
  const {get, workers, submit} = page({performance: {now: () => now}, setInterval: callback => { tick = callback; return 1; }, clearInterval() {}});
  get("sumstats-catalog-id").value = "GCST005195";
  submit();
  const worker = workers[0], id = worker.request.id;
  worker.onmessage({data: {type: "phase", id, major: "inputs", started: true, message: "Preparing inputs"}});
  assert.equal(get("elapsed").textContent, "00:00");
  for (const [milliseconds, expected] of [[59999, "00:59"], [60000, "01:00"], [125900, "02:05"], [3661000, "61:01"]]) {
    now = milliseconds; tick(); assert.equal(get("elapsed").textContent, expected);
  }
  worker.onmessage({data: {type: "phase", id, message: "Downloading summary statistics", detail: "File 2 of 3"}});
  assert.equal(get("activity-title").textContent, "Preparing inputs");
  assert.equal(get("status").textContent, "Downloading summary statistics: File 2 of 3");
  worker.onmessage({data: {type: "phase", id, major: "fitting", message: "Fitting VIPRS"}});
  for (const message of ["Loading LD reference", "Matching summary statistics to LD", "Fitting model", "Preparing results"]) {
    worker.onmessage({data: {type: "phase", id, message}});
    assert.equal(get("activity-title").textContent, "Model fitting");
    assert.equal(get("status").textContent, message);
  }
  worker.onmessage({data: {type: "phase", id, major: "results", message: "Combining chromosome results"}});
  assert.equal(get("activity-title").textContent, "Preparing results");
});


test("frequent network progress updates status without replacing the major heading or flooding logs", () => {
  const {get, workers, submit} = page();
  get("sumstats-catalog-id").value = "GCST90079133";
  submit();
  const worker = workers[0], id = worker.request.id;
  worker.onmessage({data: {type: "phase", id, major: "inputs", message: "Preparing inputs"}});
  const log = get("log").textContent;
  for (const detail of ["Waiting for metadata (240s).", "12.0 MiB of 100.0 MiB (12%)."]) {
    worker.onmessage({data: {type: "phase", id, message: "Downloading summary statistics", detail, quiet: true}});
    assert.equal(get("activity-title").textContent, "Preparing inputs");
    assert(get("status").textContent.includes(detail));
    assert.equal(get("log").textContent, log);
  }
});


test("stage bars show measured progress, chromosome headings, and reset for a new run", () => {
  const {get, workers, submit} = page();
  get("sumstats-catalog-id").value = "GCST90079133";
  submit();
  const worker = workers[0], id = worker.request.id;
  const send = event => worker.onmessage({data: {type: "stage-progress", id, ...event}});
  assert.equal(get("run-progress").hidden, false);
  assert.equal(get("progress-fitting-row").dataset.state, "waiting");
  send({stage: "sumstats", state: "active", completed: 0.5, total: 1, detail: "50 MiB of 100 MiB (50%)."});
  assert.equal(get("progress-sumstats").value, 0.5);
  assert.match(get("progress-sumstats-detail").textContent, /50%/);
  send({stage: "fitting", state: "active", completed: 1, total: 22, heading: "Fitting VIPRS · chromosome 2 (2/22)", message: "1/22 chromosomes complete · iteration 8 (limit 1000).", detail: "1/22 chromosomes complete"});
  assert.equal(get("activity-title").textContent, "Model fitting");
  assert.match(get("status").textContent, /Fitting VIPRS · chromosome 2 \(2\/22\)/);
  send({stage: "ld", state: "active", heading: "Loading LD · chromosome 4", message: "Loading chromosome 4 LD matrix."});
  assert.equal(get("activity-title").textContent, "Model fitting");
  assert.match(get("status").textContent, /Loading LD · chromosome 4/);
  assert.equal(get("progress-fitting").max, 22);
  assert.equal(get("progress-fitting").value, 1);
  worker.onmessage({data: {type: "chromosome-complete", id, message: "Finished chromosome 2 (2 of 22)."}});
  assert.equal(get("activity-title").textContent, "Model fitting");
  get("cancel").handlers.click();
  assert.equal(get("progress-fitting-row").dataset.state, "cancelled");
  assert.equal(get("progress-fitting").value, 1);
  submit();
  assert.equal(get("progress-fitting").value, 0);
  assert.equal(get("progress-fitting-detail").textContent, "Waiting");
});

test("progress ignores messages from previous runs and freezes on errors", () => {
  const {get, workers, submit} = page();
  get("sumstats-catalog-id").value = "GCST90079133";
  submit();
  const worker = workers[0], id = worker.request.id;
  worker.onmessage({data: {type: "stage-progress", id: id+1, stage: "fitting", state: "done", completed: 22, total: 22, detail: "Stale"}});
  assert.equal(get("progress-fitting-detail").textContent, "Waiting");
  worker.onmessage({data: {type: "stage-progress", id, stage: "fitting", state: "active", completed: 2, total: 22, detail: "2/22 complete"}});
  worker.onmessage({data: {type: "error", id, error: globalThis.VIPRSErrors.describe(new Error("fit failed"), "inference")}});
  assert.equal(get("progress-fitting-row").dataset.state, "error");
  assert.equal(get("progress-fitting").value, 2);
});
