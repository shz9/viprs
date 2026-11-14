import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";
import {EXAMPLE, HOSTED_LD_PANELS, LD_ANCESTRIES, resolveDataURL, selectLocalFiles} from "../../docs/webapp/sources.mjs";
import "../../docs/webapp/errors.js";
import {SUMSTATS_FORMAT_HELP} from "../../docs/webapp/formats.js";

// Exercise actual page handlers with a minimal DOM; no inference downloads needed.
function page() {
  const nodes = new Map(), workers = [], blobs = new Map();
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
    Worker, URL: URLs, Blob, performance, setInterval, clearInterval, EXAMPLE, HOSTED_LD_PANELS, LD_ANCESTRIES, resolveDataURL, selectLocalFiles,
    VIPRSErrors: globalThis.VIPRSErrors, SUMSTATS_FORMAT_HELP, Option: function(text, value) { return {text, value}; }});
  const source = readFileSync(new URL("../../docs/webapp/app.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/gm, "").replaceAll("import.meta.url", '"https://example.org/webapp/app.js"');
  vm.runInContext(source, context);
  return {get, workers, blobs, submit: () => get("inference-form").handlers.submit({preventDefault() {}})};
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
