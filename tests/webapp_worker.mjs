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
