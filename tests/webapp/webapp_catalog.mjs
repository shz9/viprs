import assert from "node:assert/strict";
import test from "node:test";
import {catalogDirectory, validateCatalog} from "../../docs/webapp/catalog.mjs";
const directory = catalogDirectory("GCST005195");
function transport(documents) {
  return async url => ({ok: url in documents, status: url in documents ? 200 : 404, text: async () => documents[url]});
}
test("accession validation and FTP grouping", () => {
  assert.match(directory, /GCST005001-GCST006000\/GCST005195\/$/);
  assert.throws(() => catalogDirectory("GCST0"), /range/);
  assert.throws(() => catalogDirectory("invalid"), /accession/);
});
test("discovers both builds with harmonised priority and excludes external links", async () => {
  const result = await validateCatalog("GCST005195", {fetcher: transport({
    [directory]: '<a href="harmonised/">h</a><a href="raw-build37.tsv.gz">raw</a><a href="https://elsewhere.org/build38.tsv.gz">external</a>',
    [directory + "harmonised/"]: '<a href="study.h.tsv.gz">h</a><a href="study.h.tsv.gz-meta.yaml">meta</a><a href="study-build37.f.tsv.gz">f</a>',
    [directory + "harmonised/study.h.tsv.gz-meta.yaml"]: 'genome_assembly: "hg38"',
  })});
  assert.equal(result.files.GRCh37, directory + "harmonised/study-build37.f.tsv.gz");
  assert.equal(result.files.GRCh38, directory + "harmonised/study.h.tsv.gz");
  assert.equal(result.complete, true);
});
test("unknown or unavailable metadata leaves discovery incomplete", async () => {
  const result = await validateCatalog("GCST005195", {fetcher: transport({[directory]: '<a href="study.tsv.gz">data</a><a href="study.tsv.gz-meta.yaml">meta</a>'})});
  assert.equal(result.complete, false);
  assert.deepEqual(result.files, {});
  assert.equal(result.warnings.length, 1);
});
test("missing accession and unsupported files give actionable errors", async () => {
  await assert.rejects(validateCatalog("GCST005195", {fetcher: transport({})}), /Check the GCST ID/);
  await assert.rejects(validateCatalog("GCST005195", {fetcher: transport({[directory]: '<a href="study.txt">data</a>'})}), /no supported TSV/);
});
test("timeout includes a stalled response body", async () => {
  await assert.rejects(validateCatalog("GCST005195", {timeout: 5, fetcher: async (url, {signal}) => ({ok: true, text: () => new Promise((resolve, reject) => signal.addEventListener("abort", () => reject(new Error("aborted"))))})}), /timed out/);
});

test("size checks use HEAD without reading study bodies", async () => {
  const url = directory + "study-build37.tsv.gz";
  const result = await validateCatalog("GCST005195", {fetcher: async (requested, options) => {
    if (requested === directory) return {ok: true, text: async () => '<a href="study-build37.tsv.gz">data</a>'};
    assert.equal(requested, url);
    assert.equal(options.method, "HEAD");
    return {ok: true, headers: new Headers({"content-length": "3500000000"}), text() { throw new Error("Must not read study body"); }};
  }});
  assert.equal(result.file_sizes.GRCh37, 3500000000);
});

test("unavailable size does not invalidate verified builds", async () => {
  for (const headers of [{}, {"content-length": "invalid"}]) {
    const result = await validateCatalog("GCST005195", {fetcher: async (url, options) => options.method === "HEAD"
      ? {ok: true, headers: new Headers(headers)}
      : {ok: true, text: async () => '<a href="study-build37.tsv.gz">data</a>'}});
    assert.equal(result.complete, true);
    assert.equal(result.file_sizes.GRCh37, null);
  }
  const result = await validateCatalog("GCST005195", {fetcher: async (url, options) => {
    if (options.method === "HEAD") throw new TypeError("Network error");
    return {ok: true, text: async () => '<a href="study-build37.tsv.gz">data</a>'};
  }});
  assert.equal(result.complete, true);
  assert.equal(result.file_sizes.GRCh37, null);
});
