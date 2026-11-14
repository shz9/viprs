/* Shared by the page and classic web worker. Keep diagnostics out of status text. */
(() => {
  const stages = {
    worker: ["Inference worker stopped unexpectedly", "Restart the runtime and try again. If this repeats with large inputs, try fewer chromosomes or the command-line version."],
    runtime: ["Runtime initialization failed", "Check your connection and retry the runtime."],
    sumstats: ["Summary statistics could not be read", "Check the selected format, required columns, numeric values, and file compression."],
    ld: ["LD reference could not be loaded", "Check the LD source and archive format, and try again."],
    harmonization: ["Summary statistics and LD reference could not be matched", "Check chromosome coverage, variant identifiers, alleles, and matching genome builds."],
    inference: ["Inference failed", "Check the input data and model settings. If this persists, export a diagnostic report."],
    results: ["Results could not be prepared", "Export a diagnostic report to help investigate this failure."],
    validation: ["Check your inputs", "Correct the input and run inference again."],
    cleanup: ["Session cleanup failed", "Restart the runtime before running again."],
  };
  function describe(error, stage = "runtime", source = null) {
    const details = String(error?.details || error?.stack || error?.message || error || "Unknown error");
    const raw = String(error?.message || error || "Unknown error");
    const lines = raw.trim().split(/\r?\n/).map(line => line.trim()).filter(Boolean);
    // Pyodide embeds a Python traceback in Error.message. Its final line is the exception.
    const last = lines.at(-1) || "Unknown error";
    const message = last.replace(/^(?:PythonError:\s*)?(?:[\w.]+(?:Error|Exception)):\s*/, "");
    const memory = /MemoryError|out of memory|memory access out of bounds|cannot enlarge memory|allocation failed|could not allocate|cannot allocate|invalid array buffer length|Array buffer allocation failed/i.test(raw);
    const [title, advice] = memory
      ? ["Browser memory limit reached", "Try fewer chromosomes or smaller inputs, close other tabs, or use the command-line version. The wasm32 runtime has a 4 GiB ceiling; the browser may allow less."]
      : stages[stage] || stages.inference;
    return {category: memory ? "memory" : stage, stage, source, title,
      message: message || (memory ? "The browser could not allocate enough memory." : "An unexpected error occurred."), advice, details};
  }
  function redact(value) {
    // Strip credentials, signed query strings and fragments from URLs in all diagnostic text.
    return String(value).replace(/(?:https?|hf):\/\/[^\s<>"']+/gi, raw => {
      try { const url = new URL(raw); url.username = url.password = url.search = url.hash = ""; return url.href; }
      catch { return "[URL omitted]"; }
    });
  }
  function safeReport(value) {
    if (typeof value === "string") return redact(value);
    if (Array.isArray(value)) return value.map(safeReport);
    if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, safeReport(item)]));
    return value;
  }
  function issueURL(report) {
    const summary = {error: {category: report.error.category, stage: report.error.stage,
      message: report.error.message.slice(0, 250), source: report.error.source?.slice(0, 150)},
      options: report.session?.options, environment: {...report.environment, browser: report.environment?.browser?.slice(0, 200)},
      runtime: {pyodide_version: report.runtime?.pyodide_version}};
    const attachment = "\n\nAttach the downloaded viprs-error.json after reviewing it for private information. Input file contents are not included.";
    let body = "### What happened\n" + report.error.title +
      "\n\n### Steps to reproduce\n1. [Describe your inputs and steps]\n2. [Describe expected behavior]\n\n### Session summary\n```json\n" +
      JSON.stringify(summary, null, 2) + "\n```";
    const url = () => "https://github.com/shz9/viprs/issues/new?" + new URLSearchParams({title: "[Web app] " + report.error.title, body: body + attachment});
    if (url().length > 7500) body = "### Error\n" + report.error.title + "\n\n### Steps to reproduce\n[Describe your inputs and steps]";
    return url();
  }
  globalThis.VIPRSErrors = {describe, redact, safeReport, issueURL};
})();
