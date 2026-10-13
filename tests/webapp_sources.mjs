import assert from "node:assert/strict";
import test from "node:test";
import {EXAMPLE, resolveDataURL} from "../docs/webapp/sources.mjs";
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
