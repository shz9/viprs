import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";

const workerSource = readFileSync(new URL("../docs/webapp/worker.js", import.meta.url), "utf8");

function worker(fetchInput = () => { throw new Error("Unexpected fetch"); }) {
  const context = vm.createContext({
    URL, performance,
    self: {location: {href: "https://example.org/webapp/worker.js"}, postMessage() {}},
    fetch: (url, options) => String(url).endsWith("/runtime.json")
      ? new Promise(() => {}) : fetchInput(url, options),
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
