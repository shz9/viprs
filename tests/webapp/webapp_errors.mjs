import assert from "node:assert/strict";
import test from "node:test";
import "../../docs/webapp/errors.js";
const {describe, safeReport, issueURL} = globalThis.VIPRSErrors;

test("Python traceback is retained only in diagnostics", () => {
  const traceback = 'Traceback (most recent call last):\n  File "inference.py", line 42\nValueError: Missing required column BETA';
  const error = describe({message: traceback}, "sumstats", "study.tsv");
  assert.equal(error.message, "Missing required column BETA");
  assert.equal(error.stage, "sumstats");
  assert.equal(error.source, "study.tsv");
  assert.equal(error.details, traceback);
  assert.match(error.advice, /format/);
});
test("memory failures take priority without losing the failing stage", () => {
  for (const message of ["MemoryError", "RuntimeError: memory access out of bounds", "Cannot enlarge memory arrays", "Out of memory", "Array buffer allocation failed"]) {
    const error = describe({message}, "inference");
    assert.equal(error.category, "memory");
    assert.equal(error.stage, "inference");
    assert.match(error.advice, /4 GiB/);
  }
  assert.equal(describe(new Error("Bad numeric value"), "sumstats").category, "sumstats");
});
test("data matching, inference and results have distinct guidance", () => {
  for (const stage of ["ld", "harmonization", "inference", "results", "runtime", "worker", "cleanup"]) {
    const error = describe(new Error("Failure"), stage);
    assert.equal(error.category, stage);
    assert.equal(error.message, "Failure");
    assert(error.advice.length > 20);
  }
});
test("reports redact URL secrets recursively while retaining multiline errors", () => {
  const report = safeReport({sources: ["https://user:password@example.org/data?token=secret#private"],
    details: "Download https://example.org/file?signature=secret\nValueError: Failed", options: {seed: 1}});
  assert.deepEqual(report.sources, ["https://example.org/data"]);
  assert.equal(report.details, "Download https://example.org/file\nValueError: Failed");
  assert.equal(report.options.seed, 1);
});
test("GitHub draft stays bounded and omits full traceback and logs", () => {
  const report = {error: describe({message: "Traceback\nValueError: " + "x".repeat(10000)}, "sumstats"), session: {options: {seed: 2}}, log: "private log"};
  const url = new URL(issueURL(report));
  assert.equal(url.origin + url.pathname, "https://github.com/shz9/viprs/issues/new");
  assert(url.href.length < 8000);
  assert(!url.searchParams.get("body").includes("Traceback"));
  assert(!url.searchParams.get("body").includes("private log"));
  assert.match(url.searchParams.get("body"), /Attach the downloaded/);
});
