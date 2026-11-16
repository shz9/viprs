import assert from "node:assert/strict";
import {mkdtempSync, realpathSync, rmSync, readFileSync, writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {spawnSync} from "node:child_process";
import test from "node:test";
import {getScoringGuide, genotypeFormats, PLINK_IMAGE, recommendedTool, platformGuidance} from "../../docs/webapp/scoring.js";

const runtimes = ["native", "docker", "apptainer", "singularity"];
test("all generated shell commands parse for supported formats, environments and runtimes", () => {
  for (const [genotype, format] of Object.entries(genotypeFormats)) {
    for (const tool of format.tools) for (const environment of ["generic", "ukb", "aou"]) for (const runtime of runtimes) for (const layout of ["combined", "chromosomes"]) {
      for (const [, , command] of getScoringGuide(tool, environment, genotype, runtime, layout).steps) {
        if (!command || command.startsWith("sampleset,")) continue;
        const checked = spawnSync("bash", ["-n"], {input: command, encoding: "utf8"});
        assert.equal(checked.status, 0, `${genotype}/${tool}/${environment}/${runtime}: ${checked.stderr}`);
      }
    }
  }
});

test("container scoring preserves paths with spaces and scoring arguments", () => {
  const directory = mkdtempSync(join(tmpdir(), "viprs scoring "));
  try {
    for (const runtime of ["docker", "apptainer", "singularity"]) {
      const guide = getScoringGuide("plink", "generic", "bed", runtime);
      const command = guide.steps.find(step => step[0] === "Calculate scores")[2];
      // Replace the runtime with an argument recorder: no container is launched.
      const checked = spawnSync("bash", ["-c", `GENOTYPE_DIR='/existing genotypes'\nTARGET_PREFIX="$GENOTYPE_DIR/cohort"\n${runtime}() { printf '%s\\n' "$@"; }\n${command}`], {cwd: directory, encoding: "utf8"});
      assert.equal(checked.status, 0, checked.stderr);
      const args = checked.stdout.trimEnd().split("\n");
      assert(args.includes(`${directory}:/work`) || args.includes(`${realpathSync(directory)}:/work`));
      const scoreIndex = args.indexOf("--score");
      assert.deepEqual(args.slice(scoreIndex, scoreIndex + 5), ["--score", "viprs-scoring.tsv", "2", "4", "7"]);
      assert(args.includes("--bfile") && args.includes("/existing genotypes/cohort"));
      assert(args.includes("/existing genotypes:/existing genotypes:ro"));
      if (runtime === "docker") {
        assert(args.includes(PLINK_IMAGE));
        assert.deepEqual(args.slice(args.indexOf("--entrypoint"), args.indexOf("--entrypoint") + 2), ["--entrypoint", "plink2"]);
      } else assert(args.includes("plink2.sif") && args.includes("plink2"));
      assert(!guide.steps.some(step => step[0] === "Optional: use PLINK 1.9"));
    }
  } finally { rmSync(directory, {recursive: true, force: true}); }
});

test("native setup reuses PLINK on PATH without downloading or changing files", () => {
  const command = getScoringGuide("plink").steps[0][2];
  const checked = spawnSync("bash", ["-c", `plink2() { printf 'existing plink2\\n'; }; curl() { return 99; };\n${command}`], {encoding: "utf8"});
  assert.equal(checked.status, 0, checked.stderr);
  assert.match(checked.stdout, /existing plink2/);
  assert(command.includes('"$(uname -sm)" = "Linux x86_64"'));
  assert(command.includes('export PATH="$HOME/.local/bin:$PATH"'));
});

test("workflow profiles follow the chosen runtime and use an explicit release", () => {
  for (const runtime of runtimes) {
    const guide = getScoringGuide("pgsc", "generic", "pgen", runtime);
    const commands = guide.steps.map(step => step[2] || "").join("\n");
    const profile = runtime === "native" ? "conda" : runtime;
    assert(commands.includes(`-profile test,${profile}`));
    assert(commands.includes(`-profile ${profile}`));
    assert(commands.includes("-r v2.2.0"));
    assert(!commands.includes("curl -fL"));
  }
});

test("RAP reuses its app while platform recommendations preserve format compatibility", () => {
  for (const runtime of runtimes) {
    const guide = getScoringGuide("plink", "ukb", "bgen", runtime);
    assert(guide.steps.some(step => step[0] === "Use RAP’s pre-installed PLINK 2"));
    assert(!guide.steps.some(step => step[2]?.includes("curl -fL") || step[2]?.includes("docker run")));
  }
  for (const environment of ["ukb", "aou"]) {
    for (const [genotype, format] of Object.entries(genotypeFormats)) assert(format.tools.includes(recommendedTool(genotype, environment)));
    assert.equal(recommendedTool("pgen", environment), environment === "aou" ? "pgsc" : "plink");
    assert.equal(recommendedTool("bgen", environment), "plink");
    assert(!JSON.stringify(platformGuidance(environment, "bgen")).match(/PRSice|preview|not a platform-installed app/));
  }
  assert.equal(platformGuidance("generic", "pgen"), null);
});


test("chromosome totals join by ID, sum score sums, and reject incomplete results", () => {
  for (const tool of ["plink", "score"]) {
    const directory = mkdtempSync(join(tmpdir(), "viprs totals "));
    try {
      const plink = tool === "plink", ext = plink ? "sscore" : "tsv";
      const header = plink ? "#FID IID ALLELE_CT NAMED_ALLELE_DOSAGE_SUM BETA_AVG BETA_SUM\n" : "SAMPLE viprs-pgs viprs-pgs_CNT\n";
      const row = (id, sum, count = 10) => plink ? `family ${id} 20 10 999 ${sum}\n` : `${id} ${sum} ${count}\n`;
      writeFileSync(join(directory, `chr21.${ext}`), header + row("001", 1) + row("002", 2));
      writeFileSync(join(directory, `chr22.${ext}`), header + row("002", 4) + row("001", 3));
      const guide = getScoringGuide(tool, "generic", plink ? "bed" : "vcf", "native", "chromosomes", [21, 22]);
      const command = guide.steps.find(step => step[0] === "Combine chromosome scores")[2];
      const run = (chromosomes = "21 22") => spawnSync("bash", ["-c", `CHROMOSOMES=(${chromosomes})\n${command}`], {env: {...process.env, SCORE_DIR: directory, PYTHONDONTWRITEBYTECODE: "1"}, encoding: "utf8"});
      const scored = run();
      assert.equal(scored.status, 0, scored.stderr);
      const output = readFileSync(join(directory, "viprs-total.tsv"), "utf8");
      assert(output.includes(plink ? "family\t001\t4" : "001\t4\t20"));
      assert(output.includes(plink ? "family\t002\t6" : "002\t6\t20"));
      assert.notEqual(run("21 22 20").status, 0, "Missing chromosome must fail");
      assert.notEqual(run("21 21").status, 0, "Duplicate chromosome must fail");
      for (const badRows of [row("001", 3), row("001", 3) + row("001", 4), row("001", "nan") + row("002", 4)]) {
        writeFileSync(join(directory, `chr22.${ext}`), header + badRows);
        assert.notEqual(run().status, 0, "Missing/duplicate IDs and invalid scores must fail");
      }
    } finally { rmSync(directory, {recursive: true, force: true}); }
  }
});

test("only chromosome-specific routes explain supported wildcard behavior", () => {
  const viprs = getScoringGuide("viprs", "generic", "bed", "native", "chromosomes", [21, 22]);
  assert(viprs.steps.some(step => step[2]?.includes('BED_FILES="$GENOTYPE_DIR/cohort_chr*.bed"')));
  assert(viprs.steps.some(step => step[2]?.includes('--bfile "$BED_FILES"')));
  assert(!viprs.steps.some(step => step[0] === "Combine chromosome scores"));
  const pgsc = getScoringGuide("pgsc", "generic", "pgen", "native", "chromosomes", [21, 22]);
  const sheet = pgsc.steps.find(step => step[2]?.startsWith("sampleset,"))[2];
  assert.equal(sheet.split("\n").length, 3);
  assert(sheet.includes("cohort_chr21,21,pfile") && sheet.includes("cohort_chr22,22,pfile"));
  for (const genotype of ["pgen", "bed", "vcf", "bgen"]) {
    const steps = getScoringGuide("plink", "generic", genotype, "docker", "chromosomes", [21, 22]).steps;
    assert(steps.some(step => step[2]?.includes("CHROMOSOMES=(21 22)")));
    assert(steps.find(step => step[0] === "Calculate scores")[2].includes('for chromosome in "${CHROMOSOMES[@]}"'));
    assert(!steps[0][2].includes("GENOTYPE_DIR"), "Runtime check must not require target path variables");
  }
});

test("public recipes use existing paths without speculative software recommendations", () => {
  for (const [genotype, format] of Object.entries(genotypeFormats)) for (const tool of format.tools) for (const environment of ["generic", "ukb", "aou"]) {
    const text = JSON.stringify(getScoringGuide(tool, environment, genotype));
    assert(!text.match(/PRSice|v3 preview|Stage the|Place target|Keep target|gcloud storage cp|not a platform-installed app/));
    assert(text.includes("absolute"));
  }
  const score = JSON.stringify(getScoringGuide("score", "generic", "vcf"));
  assert(score.includes("--sample-header SAMPLE"));
});
