import assert from "node:assert/strict";
import test from "node:test";
import {EXAMPLE, resolveDataURL, globMatcher, selectLocalFiles, expandDataURL} from "../docs/webapp/sources.mjs";
const base = "https://example.org/viprs/webapp/";

test("supplied HF URI resolves to the actual LD archive", () => {
  assert.equal(resolveDataURL(EXAMPLE.ld, base), "https://huggingface.co/datasets/shz9/ukb-ld/resolve/main/EUR/chr_22.zip");
});
test("GitHub blob URL resolves to compressed file bytes", () => {
  assert.equal(resolveDataURL(EXAMPLE.sumstats, base), "https://raw.githubusercontent.com/shz9/magenpy/master/magenpy/data/ukb_height_chr22.fastGWA.gz");
});
test("relative paths preserve the deployment prefix", () => {
  assert.equal(resolveDataURL("data/ld.zip", base), base + "data/ld.zip");
});
test("HF revisions are supported", () => {
  assert.equal(resolveDataURL("hf://datasets/shz9/ukb-ld@revision/EUR/chr_22.zip", base), "https://huggingface.co/datasets/shz9/ukb-ld/resolve/revision/EUR/chr_22.zip");
});
test("local paths and unsupported protocols fail before starting the worker", () => {
  for (const input of ["", "file:///private/data.zip", "C:\\data.zip", "javascript:alert(1)", "hf://models/owner/repo/file", "https://user:pass@example.org/file"])
    assert.throws(() => resolveDataURL(input, base));
});

test("glob matching respects directories, recursion, ranges and literal punctuation", () => {
  assert(globMatcher("EUR/chr_*.zip").test("EUR/chr_22.zip"));
  assert(!globMatcher("EUR/chr_*.zip").test("EUR/nested/chr_22.zip"));
  assert(globMatcher("**/chr_2[12].zip").test("chr_21.zip"));
  assert(globMatcher("**/chr_2[12].zip").test("EUR/deep/chr_22.zip"));
  assert(!globMatcher("**/chr_2[12].zip").test("EUR/chr_20.zip"));
  assert(globMatcher("height.v?.tsv").test("height.v1.tsv"));
  assert(!globMatcher("height.v?.tsv").test("heightXv1.tsv"));
  assert.throws(() => globMatcher("chr_[9-1].zip"), /Invalid/);
});

test("local patterns match folder-relative paths and deduplicate overlapping matches", () => {
  const files = ["chr22.tsv", "chr21.tsv", "LD/chr_21.zip", "LD/chr_22.zip"]
    .map(path => ({name: path.split("/").pop(), webkitRelativePath: "chosen/" + path}));
  assert.deepEqual(selectLocalFiles(files, "**/*.tsv\nchr21.tsv").map(f => f.name), ["chr21.tsv", "chr22.tsv"]);
  assert.equal(selectLocalFiles(files, "LD/chr_2[12].zip").length, 2);
  assert.throws(() => selectLocalFiles(files, "*.gz"), /No files match/);
  assert.throws(() => selectLocalFiles(files, "../*.tsv"), /relative/);
});

const response = (body, next) => ({ok: true, json: async () => body,
  headers: new Headers(next ? {link: "<" + next + '>; rel="next"'} : {})});

test("HF glob expands every listing page, ignores directories and sorts chromosomes", async () => {
  const calls = [];
  const urls = await expandDataURL("hf://datasets/shz9/ukb-ld@v1/EUR/chr_*.zip", base, async url => {
    calls.push(url);
    return calls.length === 1
      ? response([{type:"file",path:"EUR/chr_22.zip"}, {type:"directory",path:"EUR/chr_3.zip"}],
        "https://huggingface.co/api/datasets/shz9/ukb-ld/tree/v1/EUR?cursor=next")
      : response([{type:"file",path:"EUR/chr_2.zip"}, {type:"file",path:"AFR/chr_1.zip"}]);
  });
  assert(calls[0].includes("/tree/v1/EUR?recursive=true"));
  assert.deepEqual(urls.map(url => url.split("/").pop()), ["chr_2.zip", "chr_22.zip"]);
  assert(urls.every(url => url.includes("/resolve/v1/")));
});

test("GitHub glob resolves paths without downloading data, and rejects truncated listings", async () => {
  const url = "https://github.com/org/repo/blob/main/data/chr_*.gz";
  const urls = await expandDataURL(url, base, async api => {
    assert.equal(api, "https://api.github.com/repos/org/repo/git/trees/main?recursive=1");
    return response({tree:[{type:"blob",path:"data/chr_22.gz"}],truncated:false});
  });
  assert.deepEqual(urls, ["https://raw.githubusercontent.com/org/repo/main/data/chr_22.gz"]);
  await assert.rejects(expandDataURL(url, base, async () => response({truncated:true})), /incomplete/);
});

test("unsupported hosts, unmatched patterns, API failures and excessive matches fail clearly", async () => {
  await assert.rejects(expandDataURL("https://example.org/chr_*.zip", base), /require a Hugging Face/);
  const url = "hf://datasets/owner/repo/chr_*.zip";
  await assert.rejects(expandDataURL(url, base, async () => response([])), /No files match/);
  await assert.rejects(expandDataURL(url, base, async () => ({ok:false,status:429})), /HTTP 429/);
  await assert.rejects(expandDataURL(url, base, async () => response(Array.from({length:65}, (_,i) => ({type:"file",path:"chr_" + i + ".zip" })))), /more than 64/);
});

test("explicit URLs bypass listing APIs and retain query strings", async () => {
  const url = "https://example.org/chr_22.zip?download=1";
  assert.deepEqual(await expandDataURL(url, base, () => assert.fail("must not list")), [url]);
});
