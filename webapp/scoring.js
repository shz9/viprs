// Recipes contain short checklist items, commands and links for the selected inputs.
const sources = {
  plink: ["PLINK 2 scoring documentation", "https://www.cog-genomics.org/plink/2.0/score"],
  viprs: ["viprs_score documentation", "https://github.com/shz9/viprs/blob/master/bin/viprs_score"],
  pgsc: ["pgsc_calc custom scoring files", "https://pgsc-calc.readthedocs.io/en/2.2.0/how-to/calculate_custom.html"],
  score: ["BCFtools Score plugin documentation", "https://github.com/freeseek/score#usage"],
};
const installationLinks = {
  plink: ["Download PLINK 2", "https://www.cog-genomics.org/plink/2.0/"],
  viprs: ["Install VIPRS", "https://shz9.github.io/viprs/installation/"],
  pgsc: ["Install pgsc_calc and its dependencies", "https://pgsc-calc.readthedocs.io/en/2.2.0/getting-started.html"],
  score: ["Download and install the Score plugin", "https://github.com/freeseek/score#installation"],
};
export const genotypeFormats = {
  pgen: {tools: ["plink", "pgsc"], recommended: "plink", extensions: ["pgen", "pvar", "psam"], input: '--pfile "$TARGET_PREFIX"', schema: "pfile", hint: "Keep the PGEN, PVAR and PSAM files together with the same filename prefix."},
  bed: {tools: ["plink", "viprs", "pgsc"], recommended: "plink", extensions: ["bed", "bim", "fam"], input: '--bfile "$TARGET_PREFIX"', schema: "bfile", hint: "Keep the BED, BIM and FAM files together with the same filename prefix."},
  vcf: {tools: ["score", "plink", "pgsc"], recommended: "score", extensions: ["vcf.gz", "vcf.gz.tbi"], input: '--vcf "$TARGET_FILE"', schema: "vcf", hint: "Use a normalized, indexed, multi-sample VCF."},
  bgen: {tools: ["plink"], recommended: "plink", extensions: ["bgen", "sample"], input: '--bgen "$TARGET_FILE" ref-first --sample "$SAMPLE_FILE"', hint: "Provide the BGEN path and the path to its matching sample file."}
};
// Commands run in one Bash session. Genotype files stay at their original paths.
function chromosomeSetup(chromosomes) {
  return `CHROMOSOMES=(${chromosomes.join(" ")})\nSCORE_DIR=$(mktemp -d ./viprs-scores.XXXXXX)`;
}
function combineScores(tool) {
  const plink = tool === "plink";
  return `python3 - "$SCORE_DIR" "\u0024{CHROMOSOMES[@]}" <<'PYTHON'
import csv
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
chromosomes = sys.argv[2:]
if not chromosomes or len(set(chromosomes)) != len(chromosomes):
    raise ValueError("List each expected chromosome exactly once")
totals = {}
expected_header = None
for chromosome in chromosomes:
    path = root / f"chr{chromosome}.${plink ? "sscore" : "tsv"}"
    with path.open() as handle:
        header = handle.readline().lstrip("#").split()
        if expected_header is not None and header != expected_header:
            raise ValueError(f"Inconsistent columns: {path}")
        expected_header = header
        ids = ${plink ? '[name for name in ("FID", "IID", "SID") if name in header]' : '["SAMPLE"]'}
        scores = ${plink ? '[name for name in header if name.endswith("_SUM") and name != "NAMED_ALLELE_DOSAGE_SUM"]' : '[name for name in header if name != "SAMPLE" and not name.endswith("_CNT")]'}
        if ${plink ? '"IID" not in ids' : '"SAMPLE" not in header'} or len(scores) != 1:
            raise ValueError(f"Expected sample IDs and one score column: {path}")
        columns = scores${plink ? '' : ' + [name for name in header if name.endswith("_CNT")]'}
        rows = {}
        for line in handle:
            row = line.split()
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"Incomplete row: {path}")
            key = tuple(row[header.index(name)] for name in ids)
            values = [float(row[header.index(name)]) for name in columns]
            if key in rows or not all(math.isfinite(v) for v in values):
                raise ValueError(f"Duplicate ID or invalid score: {path}, {key}")
            rows[key] = values
        if not rows or (totals and rows.keys() != totals.keys()):
            raise ValueError(f"Empty file or different sample IDs: {path}")
        for key, values in rows.items():
            totals.setdefault(key, [[] for _ in columns])
            for terms, value in zip(totals[key], values):
                terms.append(value)
output = root / "viprs-total.tsv"
with output.open("w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\\t")
    writer.writerow(ids + ["PRS"]${plink ? '' : ' + [name for name in columns[1:]]'})
    for key, values in totals.items():
        writer.writerow([*key, *(format(math.fsum(v), ".12g") for v in values)])
print(output)
PYTHON`;
}
function softwareSteps(tool, genotype, format, layout, fittedChromosomes) {
  const source = sources[tool], install = installationLinks[tool], split = layout === "chromosomes";
  const directory = 'GENOTYPE_DIR=\'/absolute/path/to/your/genotypes\'';
  const session = "Replace the example paths with your dataset’s absolute paths. Run the commands in order in the same Bash session.";
  const chromosomes = "Set CHROMOSOMES to the chromosomes represented in your weights; include each exactly once. Edit the filename pattern to match your files.";
  if (tool === "plink") {
    const target = ["pgen", "bed"].includes(genotype) ? 'TARGET_PREFIX="$GENOTYPE_DIR/cohort"' : `TARGET_FILE="$GENOTYPE_DIR/cohort.${genotype === "vcf" ? "vcf.gz" : "bgen"}"`;
    const paths = [directory, split ? chromosomeSetup(fittedChromosomes) : target];
    if (genotype === "bgen") paths.push('SAMPLE_FILE="$GENOTYPE_DIR/cohort.sample"');
    const input = {pgen: '--pfile "$TARGET_PREFIX"', bed: '--bfile "$TARGET_PREFIX"', vcf: '--vcf "$TARGET_FILE"', bgen: '--bgen "$TARGET_FILE" ref-first --sample "$SAMPLE_FILE"'}[genotype];
    const checks = ["Confirm matching genome builds, variant IDs and effect alleles.", "Use the downloaded dosage weights without further variance standardization."];
    if (genotype === "vcf") checks.push('This uses GT hard calls. For imputed dosages, add dosage=DS immediately after "$TARGET_FILE".');
    if (genotype === "bgen") {
      checks.push("This assumes the reference allele is first in the BGEN file. Use ref-last if required by your dataset’s documentation.", "Set SAMPLE_FILE to the matching sample file’s absolute path.");
      if (split) checks.push("If sample files differ by chromosome, set SAMPLE_FILE inside the loop using the chromosome number.");
    }
    checks.push('For fewer than 50 samples, provide reference allele frequencies with --read-freq "$GENOTYPE_DIR/reference.afreq".');
    const command = `plink2 ${input} \\
  --score viprs-scoring.tsv 2 4 7 header-read cols=+scoresums list-variants \\
  --out ${split ? '"$SCORE_DIR/chr$chromosome"' : 'viprs-target'}`;
    const pattern = ["pgen", "bed"].includes(genotype) ? 'TARGET_PREFIX="$GENOTYPE_DIR/cohort_chr${chromosome}"' : `TARGET_FILE="$GENOTYPE_DIR/cohort_chr\u0024{chromosome}.${genotype === "vcf" ? "vcf.gz" : "bgen"}"`;
    const steps = [
      ["Set the target dataset path", ["Download Dosage scoring weights as viprs-scoring.tsv.", session, split ? "PLINK accepts one genotype file set per run, not a wildcard spanning multiple chromosomes." : ["pgen", "bed"].includes(genotype) ? "TARGET_PREFIX is the existing file path without .pgen or .bed. Companion files must share that prefix." : "TARGET_FILE is the full path, including the file extension.", ...(split ? [chromosomes] : [])], paths.join("\n"), []],
      ["Calculate scores", checks, split ? `(\nset -e\nfor chromosome in "\u0024{CHROMOSOMES[@]}"; do\n  ${pattern}\n  ${command.replaceAll("\n", "\n  ")}\ndone\n)` : command, [source]],
    ];
    if (!split && genotype === "bed") steps.push(["Optional: use PLINK 1.9", ["If you use PLINK 1.9, run this command instead of the PLINK 2 command."], 'plink --bfile "$TARGET_PREFIX" \\\n  --score viprs-scoring.tsv 2 4 7 header sum \\\n  --out viprs-target', [["Download PLINK 1.9", "https://www.cog-genomics.org/plink/1.9/"]]]);
    if (split) steps.push(["Combine chromosome scores", ["Run this after every chromosome finishes successfully. Python 3 is required.", "Sum BETA_SUM by sample ID, not row position; do not add the BETA_AVG chromosome averages. The command checks for missing files, duplicate IDs and inconsistent sample sets."], combineScores("plink"), [source]]);
    steps.push(["Review the results", [split ? "Read PRS in viprs-total.tsv inside the printed score directory." : genotype === "bed" ? "Read BETA_SUM in viprs-target.sscore (or the score sum in viprs-target.profile with PLINK 1.9)." : "Read BETA_SUM in viprs-target.sscore.", "Check the logs and .sscore.vars files for variant overlap and allele mismatches.", "Review missingness and validate scores in independent samples."], null, [source]]);
    return steps;
  }
  if (tool === "viprs") return [
    ["Install VIPRS", ["Install VIPRS in your Python environment, then check the scoring command."], "python -m pip install viprs\nviprs_score --help", [install]],
    ["Prepare the weights", ["Download Dosage scoring weights as viprs-scoring.tsv.", "The scorer requires the .fit extension."], "cp viprs-scoring.tsv viprs-scoring.fit", [source]],
    ["Set the target dataset path", [session, split ? "Edit BED_FILES to match your paths; * can appear in directory or file names. Keep the pattern quoted so viprs_score expands it." : "Use the full path to your existing BED file; BIM and FAM files must share its prefix."], `${directory}\nBED_FILES="$GENOTYPE_DIR/${split ? 'cohort_chr*.bed' : 'cohort.bed'}"`, [source]],
    ["Score the target cohort", ["Confirm matching genome builds, variant IDs and alleles.", ...(split ? ["Match one BED file per fitted chromosome, with identical participants in the same order.", "viprs_score sums chromosome contributions automatically. Check the log for complete chromosome coverage."] : [])], 'viprs_score -f viprs-scoring.fit \\\n  --bfile "$BED_FILES" \\\n  --output-file results/viprs-target', [source]],
    ["Review the results", ["Read sample identifiers and PRS from results/viprs-target.prs.", "Check SNP overlap and allele harmonization in the log; validate in independent samples."], null, [source]],
  ];
  if (tool === "pgsc") {
    const header = 'sampleset,path_prefix,chrom,format' + (genotype === "vcf" ? ',vcf_genotype_field' : '');
    const rows = split ? fittedChromosomes.map(chromosome => `Target,/absolute/path/to/your/genotypes/cohort_chr${chromosome},${chromosome},${format.schema}${genotype === "vcf" ? ',GT' : ''}`) : [`Target,/absolute/path/to/your/genotypes/cohort,,${format.schema}${genotype === "vcf" ? ',GT' : ''}`];
    return [
      ["Install and test the workflow", [], null, [install]],
      ["Set paths in samplesheet.csv", ["Save this text as samplesheet.csv. Replace each example with the existing absolute genotype path, excluding all file extensions (including .vcf.gz).", split ? "Include one row per fitted chromosome with the same sampleset name. List paths explicitly; samplesheet paths do not expand wildcards." : "Leave chrom empty for a file containing multiple chromosomes.", ...(genotype === "vcf" ? ["GT uses called genotypes. Use DS in the final column for imputed dosages."] : [])], [header, ...rows].join("\n"), [["Samplesheet reference", "https://pgsc-calc.readthedocs.io/en/2.2.0/how-to/samplesheet.html"]]],
      ["Calculate scores", ["Set --target_build to the genotype build; it must match the downloaded weights.", ...(split ? ["The workflow combines chromosome scores automatically for rows sharing the same sampleset name."] : [])], `nextflow run pgscatalog/pgsc_calc -r v2.2.0 \\
  -profile docker \\
  --input samplesheet.csv \\
  --scorefile viprs-pgs.txt \\
  --target_build BUILD \\
  --outdir results/pgsc`, [source]],
      ["Review the results", ["Inspect the matching reports and SUM scores under results/pgsc.", "Resolve build or allele mismatches and check variant coverage before interpreting scores.", "Validate in independent samples."], null, [source]],
    ];
  }
  const steps = [
    ["Install BCFtools and the Score plugin", ["Follow the installation guide, then check that +score is available.", "Set BCFTOOLS_PLUGINS to the plugin directory if needed."], "bcftools +score --help", [install]],
    ["Set the target dataset path", ["Download PGS Catalog format as viprs-pgs.txt.", session, "Use normalized, indexed VCFs with matching genome builds and chromosome labels. Preserve homozygous-reference calls.", split ? chromosomes : "TARGET_FILE is the full path to your existing .vcf.gz file."], `${directory}\n${split ? chromosomeSetup(fittedChromosomes) : 'TARGET_FILE="$GENOTYPE_DIR/cohort.vcf.gz"'}\nGENOTYPE_FIELD='DS'`, [source]],
    ["Calculate scores", ["Use DS for imputed ALT dosages or set GENOTYPE_FIELD to GT for called genotypes.", "The VCF must already be indexed. If needed, its owner can create the index with bcftools index -t /absolute/path/to/cohort.vcf.gz.", ...(split ? ["Score one VCF per chromosome using the loop; a wildcard cannot supply multiple target VCFs to a single invocation."] : [])], split ? `(\nset -e\nfor chromosome in "\u0024{CHROMOSOMES[@]}"; do\n  bcftools +score --use "$GENOTYPE_FIELD" --counts --sample-header SAMPLE \\\n    -o "$SCORE_DIR/chr$chromosome.tsv" "$GENOTYPE_DIR/cohort_chr\u0024{chromosome}.vcf.gz" -c PGS viprs-pgs.txt\ndone\n)` : 'bcftools +score --use "$GENOTYPE_FIELD" --counts --sample-header SAMPLE \\\n  -o viprs-scores.tsv "$TARGET_FILE" -c PGS viprs-pgs.txt', [source]],
  ];
  if (split) steps.push(["Combine chromosome scores", ["After every chromosome finishes, run this with Python 3 to sum scores and variant counts by sample ID.", "The command checks for missing files, duplicate IDs and inconsistent sample sets."], combineScores("score"), [source]]);
  steps.push(["Review the results", [split ? "Read PRS and the combined variant count in viprs-total.tsv inside the printed score directory." : "Read sample scores and variant counts from viprs-scores.tsv.", "Review per-sample missingness: the plugin skips missing genotypes rather than mean-imputing them.", "Check chromosome coverage and validate in independent samples."], null, [source]]);
  return steps;
}
const environments = {
  ukb: {
    source: ["UK Biobank RAP guide", "https://community.ukbiobank.ac.uk/hc/en-gb/articles/26120097962397-How-do-I-analyse-bulk-data-using-the-UKB-RAP-tools-library"],
    access: ["Open your RAP project", ["Use your approved project and upload WEIGHTS.", "Confirm the genotype release and genome build. Start with a small approved sample subset."]],
    finish: "Save participant-level results, software versions and commands within the approved RAP project."
  },
  aou: {
    source: ["All of Us genomic data guide", "https://support.researchallofus.org/hc/en-us/articles/49999549117588-How-the-All-of-Us-Genomic-and-multi-omics-data-are-organized"],
    access: ["Open your Workbench workspace", ["Use an approved Controlled Tier workspace and upload WEIGHTS.", "Find the current release’s genomic paths in its Data Dictionary."]],
    finish: "Save participant-level scores and genomic data within your approved Workbench workspace."
  }
};
// Verified BioContainers tag. Keep the runtime and score commands on the same image.
export const PLINK_IMAGE = "quay.io/biocontainers/plink2:2.0.0a7.post13--hea6131b_0";
const containerDocs = ["PLINK BioContainer", "https://bioconda.github.io/recipes/plink2/README.html"];
const platformSources = {
  ukb: ["RAP pre-installed applications", "https://community.ukbiobank.ac.uk/hc/en-gb/articles/19459498669981-Pre-loaded-software-in-the-UKB-RAP"],
  aou: ["pgsc_calc Workbench deployment", "https://pgsc-calc.readthedocs.io/en/latest/how-to/AOU.html"],
};
export function recommendedTool(genotype, environment) {
  const format = genotypeFormats[genotype];
  if (environment === "ukb") return "plink";
  return environment === "aou" && format.tools.includes("pgsc") ? "pgsc" : format.recommended;
}
export function platformGuidance(environment, genotype, tool = recommendedTool(genotype, environment)) {
  if (environment === "generic") return null;
  if (environment === "ukb") return {
    title: "Scoring on UK Biobank RAP",
    items: [tool === "plink" ? "Swiss Army Knife includes PLINK 2. Use mounted project paths to read genotype files in place." : "Run the selected software in a RAP compute session with project storage mounted."],
    links: [tool === "plink" ? platformSources.ukb : environments.ukb.source],
  };
  return {
    title: "Scoring on All of Us Workbench",
    items: [tool === "pgsc" ? "Use pgsc_calc with your custom scoring file. For batch execution, follow the Workbench deployment guide." : "Run the selected software on genotype files accessible to your workspace’s compute system."],
    links: tool === "pgsc" ? [platformSources.aou] : [environments.aou.source],
  };
}
function plinkSetup(execution) {
  if (execution === "docker") return ["Check Docker and the PLINK image", ["Confirm Docker is installed and its daemon is accessible on the compute system.", "The image supplies PLINK 2; no host PLINK installation is needed. These examples use the Linux amd64 image."], `docker --version
docker info
docker run --rm --platform linux/amd64 --entrypoint plink2 ${PLINK_IMAGE} --version`, [containerDocs]];
  if (["apptainer", "singularity"].includes(execution)) return [`Check ${execution === "apptainer" ? "Apptainer" : "Singularity"} and prepare PLINK`, ["Run the version check on your compute node; load your site’s module first if needed.", "Pull the pre-packaged Linux x86_64 image once. It includes PLINK 2."], `${execution} --version
${execution} pull plink2.sif docker://${PLINK_IMAGE}
${execution} exec --bind "$PWD:/work" --pwd /work plink2.sif plink2 --version`, [containerDocs]];
  return ["Check or install PLINK 2", ["An existing plink2 on PATH is reused. Otherwise this installs the official Linux x86_64 build without administrator access.", "Run this on your compute system. The PATH update applies to the current shell; other platforms have builds on the download page."], `if command -v plink2 >/dev/null 2>&1; then
  plink2 --version
elif [ "$(uname -sm)" = "Linux x86_64" ]; then
  mkdir -p "$HOME/.local/bin" "$HOME/.cache/viprs-plink2" &&
  curl -fL 'https://s3.amazonaws.com/plink2-assets/beta1/plink2_linux_x86_64_20261107.zip' \\
    -o "$HOME/.cache/viprs-plink2/plink2.zip" &&
  unzip -o "$HOME/.cache/viprs-plink2/plink2.zip" plink2 -d "$HOME/.local/bin" &&
  chmod u+x "$HOME/.local/bin/plink2" &&
  export PATH="$HOME/.local/bin:$PATH" &&
  plink2 --version
else
  printf '%s\\n' 'Choose the matching PLINK build from the download page.'
fi`, [installationLinks.plink]];
}
function genotypeSampleMount(steps, execution) {
  if (!steps.some(step => step[2]?.includes('--sample "$SAMPLE_FILE"'))) return "";
  return ` ${execution === "docker" ? "--volume" : "--bind"} "$(dirname "$SAMPLE_FILE"):$(dirname "$SAMPLE_FILE"):ro"`;
}
function applyExecution(steps, tool, execution, environment) {
  if (tool === "plink") {
    if (environment === "ukb") {
      steps.unshift(["Use RAP’s pre-installed PLINK 2", ["Swiss Army Knife already includes PLINK 2. Run this check inside the app; no download is required."], "plink2 --version", [platformSources.ukb]]);
      return steps;
    }
    steps.unshift(plinkSetup(execution));
    if (execution !== "native") {
      const prefix = execution === "docker"
        ? `docker run --rm --platform linux/amd64 --user "$(id -u):$(id -g)" \\
  --volume "$PWD:/work" --volume "$GENOTYPE_DIR:$GENOTYPE_DIR:ro"${genotypeSampleMount(steps, "docker")} \\
  --workdir /work --entrypoint plink2 ${PLINK_IMAGE}`
        : `${execution} exec --bind "$PWD:/work" --bind "$GENOTYPE_DIR:$GENOTYPE_DIR:ro"${genotypeSampleMount(steps, execution)} --pwd /work plink2.sif plink2`;
      for (const step of steps) if (/(^|\n)[ \t]*plink2 /.test(step[2] || "")) {
        step[2] = step[2].replace(/(^|\n)([ \t]*)plink2 /, (_, line, indent) => line + indent + prefix + " ");
        step[1].push("The genotype directory is mounted read-only at its existing path. Keep the weights in your working directory; outputs are written there.");
      }
      const optional = steps.findIndex(step => step[0] === "Optional: use PLINK 1.9");
      if (optional >= 0) steps.splice(optional, 1);
      steps.at(-1)[1][0] = steps.at(-1)[1][0].replace(" (or the score sum in viprs-target.profile with PLINK 1.9)", "");
    }
    return steps;
  }
  if (tool === "pgsc") {
    const profile = execution === "native" ? "conda" : execution;
    steps[0][0] = "Confirm the workflow runtime";
    steps[0][1] = ["Check Nextflow, Java and the selected runtime on the machine that will execute the workflow.", profile === "conda" ? "The conda profile creates the workflow’s software environments; PLINK does not need a separate installation." : "The workflow supplies tool containers, including its scoring software; no host PLINK installation is needed.", "The commands use pgsc_calc v2.2.0. Download PGS Catalog format as viprs-pgs.txt."];
    const check = profile === "docker" ? "docker --version\\ndocker info" : profile + " --version";
    steps[0][2] = `nextflow -version\njava -version\n${check.replaceAll("\\n", "\n")}\nnextflow run pgscatalog/pgsc_calc -r v2.2.0 -profile test,${profile}`;
    for (const step of steps.slice(1)) if (step[2]?.includes("nextflow run")) step[2] = step[2].replace("-profile docker", `-profile ${profile}`);
  }
  return steps;
}
export function getScoringGuide(tool, environment = "generic", genotype = "pgen", execution = "native", layout = "combined", fittedChromosomes = Array.from({length: 22}, (_, i) => String(i+1))) {
  const format = genotypeFormats[genotype];
  if (!format || !format.tools.includes(tool)) throw new Error("Choose software compatible with the genotype format.");
  if (!["native", "docker", "apptainer", "singularity"].includes(execution)) throw new Error("Choose an available runtime.");
  if (!["combined", "chromosomes"].includes(layout)) throw new Error("Choose how your genotype files are organized.");
  fittedChromosomes = fittedChromosomes.map(String);
  if (!fittedChromosomes.length || new Set(fittedChromosomes).size !== fittedChromosomes.length || fittedChromosomes.some(c => !/^(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)$/.test(c))) throw new Error("Invalid fitted chromosome list.");
  const steps = applyExecution(softwareSteps(tool, genotype, format, layout, fittedChromosomes), tool, execution, environment), platform = environments[environment];
  if (!platform) return {steps};
  const weightFile = ["pgsc", "score"].includes(tool) ? "viprs-pgs.txt" : "viprs-scoring.tsv";
  const access = [platform.access[0], platform.access[1].map(text => text.replaceAll("WEIGHTS", weightFile)), null, [platform.source]];
  const setup = environment === "ukb"
    ? ["Connect to your project data", [tool === "plink" ? "Open Swiss Army Knife and use its mounted project folder for the genotype inputs." : "Open a RAP compute session with the selected software and project storage mounted.", "Use the full /mnt/project/… path to the dataset. Download only the weights into your writable analysis directory.", "Choose an output folder and compute instance with enough memory and disk."], null, [["Access mounted RAP data", "https://dnanexus.gitbook.io/uk-biobank-rap/working-on-the-research-analysis-platform/accessing-data/accessing-bulk-data"]]]
    : ["Select your cohort data", ["Find the release paths in the workspace’s Data Dictionary and select your analysis cohort.", "For these commands, use an existing filesystem path accessible to the compute job. Cloud bucket URLs (gs://…) require a mount or the documented cloud-execution workflow.", "For VDS-derived files, apply cohort and genotype filters and preserve reference calls when exporting."], null, [platform.source, ["pgsc_calc cloud execution", "https://pgsc-calc.readthedocs.io/en/2.2.0/how-to/cloud.html"]]];
  return {steps: [access, setup, ...steps, ["Save the cohort results", [platform.finish], null, [platform.source]]]};
}
