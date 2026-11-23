import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";

const workerSource = readFileSync(new URL("../../docs/webapp/worker.js", import.meta.url), "utf8");

function worker(fetchInput = () => { throw new Error("Unexpected fetch"); }, overrides = {}) {
  const context = vm.createContext({
    URL, performance, setInterval, clearInterval,
    importScripts: () => vm.runInContext(readFileSync(new URL("../../docs/webapp/errors.js", import.meta.url), "utf8"), context),
    self: {location: {href: "https://example.org/webapp/worker.js"}, postMessage() {}},
    fetch: (url, options) => String(url).endsWith("/runtime.json")
      ? new Promise(() => {}) : fetchInput(url, options),
    ...overrides,
  });
  vm.runInContext(workerSource, context);
  return context;
}

function filesystem(write = (bytes) => bytes.length) {
  const state = {opened: 0, closed: 0, bytes: 0};
  return {
    state,
    open() { state.opened++; return {}; },
    write(handle, bytes, offset, length, position) {
      assert.equal(position, state.bytes);
      const written = write(bytes.subarray(offset, offset + length));
      state.bytes += written;
      return written;
    },
    close() { state.closed++; },
  };
}

test("local and hosted streams preserve the exact input bytes", async () => {
  const expected = new Uint8Array([0, 1, 2, 255, 4]);
  for (const local of [true, false]) {
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(expected.slice(0, 2));
        controller.enqueue(expected.slice(2));
        controller.close();
      },
    });
    const received = [];
    const FS = filesystem((bytes) => { received.push(...bytes); return bytes.length; });
    const context = worker((url, options) => {
      assert.equal(String(url), "https://data.example/input");
      assert.equal(options.credentials, "omit");
      return {ok: true, body: stream};
    });
    await context.stageInput(local ? {file: {size: 5, stream: () => stream}} : {url: "https://data.example/input"}, FS, "/input");
    assert.deepEqual(received, [...expected]);
    assert.equal(FS.state.closed, 1);
    assert.equal(stream.locked, false);
  }
});

test("streams over the old per-file and combined caps without a full-file buffer", async () => {
  const context = worker();
  const chunk = new Uint8Array(1024 ** 2);
  let total = 0;
  for (let file = 0; file < 2; file++) {
    let remaining = 300;
    const stream = new ReadableStream({
      pull(controller) {
        if (remaining-- > 0) controller.enqueue(chunk);
        else controller.close();
      },
    });
    const FS = filesystem();
    await context.stageInput({file: {size: 300 * chunk.length, stream: () => stream}}, FS, "/input");
    total += FS.state.bytes;
    assert.equal(FS.state.closed, 1);
  }
  assert.equal(total, 600 * 1024 ** 2);
});

test("LD archives are fitted and deleted one at a time in source order", async () => {
  const context = worker();
  const resident = new Set(), events = [];
  let peakResident = 0;
  const FS = {
    open(path) {
      resident.add(path);
      peakResident = Math.max(peakResident, resident.size);
      events.push(`open:${path}`);
      return {path};
    },
    write(_handle, _bytes, _offset, length) { return length; },
    close() {},
    unlink(path) {
      if (!resident.delete(path)) throw new Error("missing");
      events.push(`unlink:${path}`);
    },
  };
  const sources = [21, 22].map(chromosome => ({file: {
    name: `chr${chromosome}.zip`, size: 1,
    stream: () => new ReadableStream({start(controller) {
      controller.enqueue(new Uint8Array([chromosome]));
      controller.close();
    }}),
  }}));
  const seen = [];
  await context.fitLDArchives(sources, FS, "/run", async (path, remaining) => {
    assert.deepEqual([...resident], [path]);
    seen.push([path, remaining]);
    events.push(`fit:${path}`);
  });
  assert.equal(peakResident, 1);
  assert.deepEqual(seen, [["/run/ld-current", 1], ["/run/ld-current", 0]]);
  assert.deepEqual(events, [
    "open:/run/ld-current", "fit:/run/ld-current", "unlink:/run/ld-current",
    "open:/run/ld-current", "fit:/run/ld-current", "unlink:/run/ld-current",
  ]);
});

test("hosted LD is fitted from its remote URL without filesystem staging", async () => {
  const context = worker();
  const url = "hf://datasets/shz9/ukb-ld/EUR/chr_22.zip";
  const seen = [];
  const seconds = await context.fitLDArchives(
    [{url, stream: true}],
    {open() { assert.fail("streamed LD must not be staged"); }, unlink() {}},
    "/run",
    async (path, remaining) => seen.push([path, remaining]),
  );
  assert.deepEqual(seen, [[url, 0]]);
  assert.equal(seconds, 0);
});

test("large range-capable inputs are downloaded in parallel into fixed offsets", async () => {
  const size = 32 * 1024 ** 2;
  const ranges = [];
  const context = worker(async (_url, options) => {
    if (options.method === "HEAD") return {
      ok: true,
      headers: {get: name => ({"content-length": String(size), "accept-ranges": "bytes"})[name]},
    };
    const [start, end] = options.headers.Range.match(/\d+/g).map(Number);
    if (start === 0 && end === 0) return {status: 206, body: new ReadableStream({start(controller) {
      controller.enqueue(new Uint8Array([0x1f]));
      controller.close();
    }})};
    ranges.push([start, end]);
    let position = start;
    return {status: 206, body: new ReadableStream({pull(controller) {
      if (position > end) return controller.close();
      const length = Math.min(1024 ** 2, end - position + 1);
      const chunk = new Uint8Array(length);
      if (position === 0) chunk.set([0x1f, 0x8b]);
      position += length;
      controller.enqueue(chunk);
    }})};
  });
  const writes = [], renames = [], progress = [];
  const FS = {
    open: path => ({path}), close() {},
    write(_handle, _bytes, _offset, length, position) { writes.push([position, length]); return length; },
    rename: (from, to) => renames.push([from, to]),
  };
  assert.equal(await context.stageRangedInput({url: "https://ftp.ebi.ac.uk/large"}, FS, "/sumstats", true, value => progress.push(value)), "/sumstats.tsv.gz");
  assert.deepEqual(ranges, [
    [0, 8 * 1024 ** 2 - 1], [8 * 1024 ** 2, 16 * 1024 ** 2 - 1],
    [16 * 1024 ** 2, 24 * 1024 ** 2 - 1], [24 * 1024 ** 2, size - 1],
  ]);
  assert.equal(writes.reduce((total, [, length]) => total + length, 0), size);
  assert.equal(progress[0].stage, "checking");
  const transfers = progress.filter(item => item.bytes !== undefined);
  assert.equal(transfers.at(-1).bytes, size);
  assert(transfers.every((item, i) => item.total === size && (!i || item.bytes >= transfers[i-1].bytes)));
  assert.deepEqual(renames, [["/sumstats", "/sumstats.tsv.gz"]]);
});

test("LD archive order is chromosome-numeric for local files and URLs", () => {
  const context = worker();
  const sorted = context.sortLDInputs([
    {file: {name: "chr_22.zip"}},
    {url: "https://example.org/chr_10.zip"},
    {file: {name: "chr_2.zip"}},
  ]);
  assert.deepEqual(
    [...sorted].map(source => source.file?.name || source.url.split("/").at(-1)),
    ["chr_2.zip", "chr_10.zip", "chr_22.zip"],
  );
});

test("filesystem failures cancel the reader and close the file", async () => {
  let cancelled = false;
  const stream = new ReadableStream({
    pull(controller) { controller.enqueue(new Uint8Array([1, 2])); },
    cancel() { cancelled = true; },
  });
  const FS = filesystem(() => { throw new Error("Out of memory"); });
  await assert.rejects(worker().stageInput({file: {size: 2, stream: () => stream}}, FS, "/input"), /Out of memory/);
  assert.equal(cancelled, true);
  assert.equal(FS.state.closed, 1);
  assert.equal(stream.locked, false);
});

test("empty inputs and HTTP errors remain actionable", async () => {
  const FS = filesystem();
  await assert.rejects(worker().stageInput({file: {size: 0}}, FS, "/input"), /empty/);
  const missing = worker(() => ({ok: false, status: 404}));
  await assert.rejects(missing.stageInput({url: "https://data.example/missing"}, FS, "/input"), /HTTP 404/);
  const empty = worker(() => ({ok: true, body: new ReadableStream({start(c) { c.close(); }})}));
  await assert.rejects(empty.stageInput({url: "https://data.example/empty"}, FS, "/input"), /empty/);
  assert.equal(FS.state.closed, 1);
});


test("summary staging detects compression across chunk boundaries and renames without rewriting", async () => {
  for (const local of [true, false]) {
    for (const bytes of [[0x1f, 0x8b, 8, 0], [67, 72, 82, 9], [0x1f]]) {
      const stream = new ReadableStream({start(controller) {
        controller.enqueue(new Uint8Array());
        for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
        controller.close();
      }});
      const received = [], renames = [];
      const FS = filesystem(chunk => { received.push(...chunk); return chunk.length; });
      FS.rename = (from, to) => {
        assert.equal(FS.state.closed, 1);
        renames.push([from, to]);
      };
      const context = worker(() => ({ok: true, body: stream}));
      // Neither a misleading upload name nor an extensionless URL controls compression.
      const source = local ? {file: {name: "misleading.gz", size: bytes.length, stream: () => stream}}
        : {url: "https://data.example/download?id=123"};
      const expected = "/sumstats-0" + (bytes[1] === 0x8b ? ".tsv.gz" : ".tsv");
      assert.equal(await context.stageInput(source, FS, "/sumstats-0", true), expected);
      assert.deepEqual(received, bytes);
      assert.deepEqual(renames, [["/sumstats-0", expected]]);
      assert.equal(FS.state.opened, 1);
      assert.equal(FS.state.bytes, bytes.length);
      assert.equal(stream.locked, false);
    }
  }
});

test("failed summary staging does not rename a partial input", async () => {
  const stream = new ReadableStream({start(controller) {
    controller.enqueue(new Uint8Array([0x1f, 0x8b]));
    controller.close();
  }});
  const FS = filesystem(() => { throw new Error("Out of memory"); });
  FS.rename = () => assert.fail("must not rename after failed staging");
  await assert.rejects(worker().stageInput({file: {size: 2, stream: () => stream}}, FS, "/sumstats-0", true), /Out of memory/);
});


test("Catalog resolution follows inspected LD metadata before downloading statistics", async () => {
  const context = worker(), calls = [], globals = new Map();
  const data = {options: {ld_genome_build: "GRCh38"}, ld: [{url: "hf://datasets/panel/EUR/chr_*.zip", stream: true}], sumstats: [{catalog: "GCST005195"}]};
  const pyodide = {globals, FS: {open() { assert.fail("Hosted metadata must not stage an archive"); }},
    runPython(code) { calls.push("metadata"); assert.match(code, /inspect_browser_ld_build/); assert.equal(globals.get("ld_expected_build"), "GRCh38"); return "GRCh38"; },
    async runPythonAsync(code) { calls.push("catalog"); assert.match(code, /genome_build=catalog_genome_build/); assert.equal(globals.get("catalog_genome_build"), "GRCh38"); return "https://example.org/build38.tsv.gz"; }};
  await context.prepareCatalogInputs(data, pyodide, "/run");
  assert.deepEqual(calls, ["metadata", "catalog"]);
  assert.equal(data.options.sumstats_genome_build, "GRCh38");
  assert.equal(data.sumstats[0].url, "https://example.org/build38.tsv.gz");
  assert.equal(globals.size, 0);
});

test("Conflicting LD metadata stops Catalog lookup", async () => {
  const context = worker();
  const data = {options: {ld_genome_build: "GRCh37"}, ld: [{url: "hf://datasets/panel/EUR/chr_*.zip", stream: true}], sumstats: [{catalog: "GCST005195"}]};
  const pyodide = {globals: new Map(), runPython() { throw new Error("conflicting genome builds"); }, runPythonAsync() { assert.fail("Catalog lookup must not run"); }};
  await assert.rejects(context.prepareCatalogInputs(data, pyodide, "/run"), /conflicting/);
});


test("A first archive staged for metadata is reused for fitting", async () => {
  const context = worker(), seen = [], removed = [];
  await context.fitLDArchives(
    [{file: {name: "chr1.zip"}, staged_path: "/run/ld-first"}],
    {open() { assert.fail("Do not stage the first archive twice"); }, unlink(path) { removed.push(path); }},
    "/run", async path => seen.push(path));
  assert.deepEqual(seen, ["/run/ld-first"]);
  assert(removed.includes("/run/ld-first"));
});


test("stale magenpy bundles fail during runtime setup with rebuild guidance", () => {
  const context = worker();
  const manifest = {pyodide_version: "0.28.3", wheels: ["./wheels/magenpy-0.2.5-cp313-cp313-pyodide_2025_0_wasm32.whl#sha256=test"]};
  assert.doesNotThrow(() => context.validateManifest(manifest));
  for (const wheels of [[], [manifest.wheels[0].replace("0.2.5", "0.2.3")], [manifest.wheels[0], manifest.wheels[0]]]) {
    assert.throws(() => context.validateManifest({...manifest, wheels}), /Rebuild/);
  }
});


test("runtime manifest and Python adapter bypass stale browser cache", async () => {
  const context = worker();
  context.fetch = async (url, options) => {
    assert.equal(options.cache, "no-store");
    return {ok: true, text: async () => "fresh contents"};
  };
  for (const path of ["runtime.json", "inference.py"]) {
    assert.equal(await context.localText(path), "fresh contents");
  }
});


test("package initialization reports wheel installation and import as separate stages", async () => {
  const context = worker(), events = [], installed = [], globals = new Map();
  context.self.postMessage = data => events.push(data);
  const manifest = {builtin_packages: ["micropip", "numpy"], wheels: ["./wheels/a.whl#sha256=one", "./wheels/b.whl#sha256=two"]};
  const pyodide = {globals,
    async loadPackage(names, options) {
      assert.deepEqual([...names], manifest.builtin_packages);
      options.messageCallback("Loaded numpy");
    },
    async runPythonAsync(code) {
      if (code.includes("micropip.install")) {
        assert.match(code, /deps=False/);
        installed.push(globals.get("package_url"));
      }
    },
  };
  await context.installRuntimePackages(pyodide, manifest);
  assert.deepEqual(installed, ["https://example.org/webapp/wheels/a.whl#sha256=one", "https://example.org/webapp/wheels/b.whl#sha256=two"]);
  assert.equal(globals.size, 0);
  const statuses = events.filter(e => e.type === "runtime-status").map(e => e.message);
  assert(statuses.includes("Loaded numpy"));
  assert(statuses.some(s => s.includes("Installing model package 2/2: b.whl")));
  assert.equal(statuses.at(-1), "Importing VIPRS and magenpy");
});

test("dependency and wheel failures retain the failing source and stop initialization", async () => {
  const context = worker(), globals = new Map();
  const pyodide = {globals,
    async loadPackage(_names, options) { options.errorCallback("Could not fetch numpy"); },
    async runPythonAsync() { assert.fail("must not install after a dependency error"); },
  };
  const manifest = {builtin_packages: ["numpy"], wheels: ["./wheels/broken.whl"]};
  await assert.rejects(context.installRuntimePackages(pyodide, manifest), /Could not fetch numpy/);
  pyodide.loadPackage = async () => {};
  pyodide.runPythonAsync = async code => { if (code.includes("micropip.install")) throw new Error("Wheel download failed"); };
  await assert.rejects(context.installRuntimePackages(pyodide, manifest), /Wheel download failed/);
  assert.equal(globals.size, 0);
  assert.equal(context.diagnostic(new Error("failure")).source, "https://example.org/webapp/wheels/broken.whl");
});


test("Catalog status identifies pending requests, updates elapsed wait, and clears timers", () => {
  const timers = new Map(), messages = [];
  let now = 0, nextTimer = 0;
  const context = worker(undefined, {
    performance: {now: () => now},
    setInterval(callback) { const id = ++nextTimer; timers.set(id, callback); return id; },
    clearInterval(id) { timers.delete(id); },
    self: {location: {href: "https://example.org/webapp/worker.js"}, postMessage(message) { messages.push(message); }},
  });
  const progress = context.catalogProgress("GCST90079133", "GRCh37");
  progress.report("request", "https://ftp.ebi.ac.uk/study/");
  assert.match(messages.at(-1).message, /Reading GWAS Catalog file list/);
  assert.match(messages.at(-1).detail, /study file list.*directory response.*GRCh37 \(0s\)/);
  now = 240000;
  [...timers.values()][0]();
  assert.match(messages.at(-1).detail, /\(240s\)/);
  assert.equal(messages.at(-1).quiet, true);
  progress.report("received", "https://ftp.ebi.ac.uk/study/");
  assert.equal(timers.size, 0);
  progress.report("request", "https://ftp.ebi.ac.uk/study/harmonised/study.h.tsv.gz-meta.yaml");
  assert.match(messages.at(-1).message, /Checking GWAS Catalog genome build/);
  assert.match(messages.at(-1).detail, /study.h.tsv.gz-meta.yaml.*metadata response.*\(0s\)/);
  progress.report("failed", "https://ftp.ebi.ac.uk/study/harmonised/study.h.tsv.gz-meta.yaml");
  assert.equal(timers.size, 0);
  progress.report("request", "https://ftp.ebi.ac.uk/study/harmonised/");
  assert.match(messages.at(-1).detail, /harmonised file list/);
  progress.stop();
  assert.equal(timers.size, 0);
});

test("Catalog lookup failures release request timers and Python callback globals", async () => {
  const timers = new Set(), globals = new Map();
  const context = worker(undefined, {setInterval(callback) { timers.add(callback); return callback; }, clearInterval(callback) { timers.delete(callback); }});
  const pyodide = {globals, runPython: () => "GRCh37", async runPythonAsync() {
    globals.get("catalog_progress_callback")("request", "https://ftp.ebi.ac.uk/study/");
    assert.equal(timers.size, 1);
    throw new Error("Catalog server unavailable");
  }};
  await assert.rejects(context.prepareCatalogInputs({options: {}, ld: [{url: "hf://datasets/panel/chr1.zip", stream: true}], sumstats: [{catalog: "GCST90079133"}]}, pyodide, "/run"), /server unavailable/);
  assert.equal(timers.size, 0);
  assert.equal(globals.size, 0);
});

test("stream progress reports received bytes with known and unknown totals", async () => {
  for (const total of [5, null]) {
    const updates = [];
    const context = worker(() => ({ok: true, headers: {get: () => total}, body: new ReadableStream({start(controller) {
      controller.enqueue(new Uint8Array([0,1]));
      controller.enqueue(new Uint8Array([2,3,4]));
      controller.close();
    }})}));
    await context.stageInput({url: "https://example.org/study.tsv.gz"}, filesystem(), "/sumstats", false, value => updates.push(value));
    assert.equal(updates[0].stage, "connecting");
    assert.deepEqual(updates.filter(item => item.bytes !== undefined).map(item => item.bytes), [0,2,5,5]);
    assert.equal(updates[1].total, total);
    assert.equal(updates.at(-1).total, 5);
  }
});

test("byte progress updates are throttled, remain precise, and finish at 100 percent", () => {
  const messages = [];
  let now = 0;
  const context = worker(undefined, {performance: {now: () => now}, self: {location: {href: "https://example.org/webapp/worker.js"}, postMessage(message) { messages.push(message); }}});
  const report = context.inputProgress("Downloading summary statistics", "File 1 of 1.");
  messages.length = 0;
  report({stage: "connecting"});
  report({bytes: 1, total: 10});
  assert.equal(messages.filter(item => item.type === "phase").length, 1);
  now = 600;
  report({bytes: 5, total: 10});
  assert.match(messages.at(-1).detail, /50%/);
  report({bytes: 10, total: 10});
  assert.match(messages.at(-1).detail, /100%/);
  now = 1200;
  report({bytes: 10});
  assert.match(messages.at(-1).detail, /received/);
  assert(messages.filter(item => item.type === "phase").every(message => message.quiet));
});


test("model progress uses exact chromosome totals and keeps the iteration limit separate", () => {
  const context = worker();
  const event = {stage: "fitting", state: "active", chromosome: "1", model: "VIPRS", completed: 0, total: 22, iteration: 40, max_iter: 1000, variants: 100000, archive_fraction: 0};
  const progress = context.modelProgress(event, 1, 1);
  assert.equal(progress.heading, "Fitting VIPRS · chromosome 1 (1/22)");
  assert.equal(progress.detail, "0/22 chromosomes complete");
  assert.match(progress.message, /iteration 40 \(limit 1000\)/);
  assert.equal(progress.completed, 0, "Iterations must not imply a fraction of chromosome completion");
  assert.equal(progress.total, 22);
  assert.match(progress.message, /100,000 variants/);
  const completed = context.modelProgress({...event, state: "done", completed: 1, archive_fraction: 1/22}, 1, 1);
  assert.equal(completed.state, "waiting");
  assert.equal(completed.completed, 1);
  assert(!completed.detail.includes("iteration"));
  assert(!completed.message.includes("iteration"));
  const last = context.modelProgress({...event, state: "done", chromosome: "22", completed: 22, archive_fraction: 1}, 1, 1);
  assert.equal(last.state, "done");
  assert.equal(last.heading, "Fitting VIPRS · chromosome 22 (22/22)");
});

test("unknown chromosome totals use explicitly labelled file progress rather than guessing", () => {
  const context = worker();
  const progress = context.modelProgress({stage: "fitting", state: "active", model: "VIPRSMix", chromosome: "21", completed: 0, total: null, archive_fraction: 0, iteration: 1, max_iter: 1000, variants: 48}, 1, 2);
  assert.equal(progress.heading, "Fitting VIPRSMix · chromosome 21");
  assert.match(progress.detail, /0 chromosome\(s\) complete · file 1\/2/);
  assert.equal(progress.total, 2);
  assert.equal(progress.completed, 0);
  const loading = context.modelProgress({stage: "ld", state: "active", detail: "Reading LD metadata."}, 1, 1);
  assert.equal(loading.total, null);
  assert.equal(loading.heading, "Loading LD reference");
});


test("cleanup tolerates Python globals that an early Catalog failure never created", () => {
  const context = worker();
  const values = new Map([["work_path", "/runs/run-1"]]);
  const globals = {
    has: name => values.has(name),
    delete(name) {
      if (!values.has(name)) throw new Error("KeyError: " + name);
      values.delete(name);
    },
  };
  context.deletePythonGlobals({globals}, "fit_progress_callback", "memory_callback",
    "request_json", "progress_callback", "stage_callback", "work_path");
  assert.equal(values.size, 0);
});


test("validated Catalog URLs skip Python discovery after checking actual LD build", async () => {
  const context = worker(), globals = new Map();
  const data = {options: {}, ld: [{url: "hf://datasets/panel/chr1.zip", stream: true}], sumstats: [{catalog: "GCST005195", catalog_files: {GRCh38: "https://example.org/build38.tsv.gz"}}]};
  const pyodide = {globals, runPython() { return "GRCh38"; }, runPythonAsync() { throw new Error("Discovery should be skipped"); }};
  await context.prepareCatalogInputs(data, pyodide, "/run");
  assert.equal(data.sumstats[0].url, "https://example.org/build38.tsv.gz");
  assert.equal(data.options.sumstats_genome_build, "GRCh38");
});
