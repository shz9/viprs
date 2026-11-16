"""Check input combinations and responsive UI without downloading inference packages.

Run with a Playwright Python environment. VIPRS_BROWSER_EXECUTABLE may specify
an installed Chromium browser. The worker is stubbed; inference is covered by
webapp_browser.py.
"""
import functools
import http.server
import itertools
import json
import os
import runpy
from pathlib import Path
import tempfile
import threading

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "build/wasm/browser-test"
SUMSTATS_FORMATS = sorted(["magenpy", "plink2", "plink1.9", "cojo", "fastgwa", "gwascatalog", "saige", "regenie", "ldak"])
FORMAT_HELP = runpy.run_path(str(ROOT / "tools/wasm/format_help.py"))["get_sumstats_format_help"]()
FORMAT_LABELS = {
    "magenpy": "Magenpy", "plink2": "PLINK 2", "plink1.9": "PLINK 1.9",
    "cojo": "GCTA-COJO", "fastgwa": "GCTA fastGWA", "gwascatalog": "GWAS Catalog",
    "saige": "SAIGE", "regenie": "REGENIE", "ldak": "LDAK / SumHer / MegaPRS",
}
MOCK_WORKER = """
self.postMessage({type: "ready", seconds: 0.01});
self.onmessage = ({data}) => {
  if (data.holdProgress) {
    self.postMessage({type: "phase", id: data.id, started: true, major: "fitting", message: "Fitting VIPRS"});
    for (const stage of ["sumstats", "ld", "harmonization"]) self.postMessage({type: "stage-progress", id: data.id, stage, state: "done", completed: 22, total: 22, detail: "22/22 chromosomes ready"});
    self.postMessage({type: "stage-progress", id: data.id, stage: "fitting", state: "active", completed: 1, total: 22,
      heading: "Fitting VIPRS · chromosome 2 (2/22)", message: "100,000 variants · iteration 17 (limit 1000).", detail: "1/22 chromosomes complete"});
    return;
  }
  for (const stage of ["sumstats", "ld", "harmonization", "fitting"]) self.postMessage({type: "stage-progress", id: data.id, stage, state: "done", completed: 1, total: 1, detail: "Complete"});
  self.postMessage({type: "log", id: data.id, message: "REQUEST " + JSON.stringify({
    sumstats: data.sumstats.map(s => s.url || s.file?.name || s.catalog),
    ld: data.ld.map(s => s.url || s.file.name), options: data.options
  })});
  const row = {SNP: "rs12345", CHR: 22, POS: 20000000, MAF: 0.2, GWAS_Z: 8, A1: "A", A2: "G", PIP: 0.98, BETA: 0.012, VAR_BETA: 0.0001};
  self.postMessage({type: "result", id: data.id, result: {
    rows: [row], chromosome_rows: {22: [row]}, plot_data: {rows: [row], total: 2, sampled: true},
    tsv: "CHR\\tSNP\\tPOS\\tA1\\tA2\\tPIP\\tBETA\\tVAR_BETA\\tMAF\\tGWAS_Z\\n22\\trs12345\\t20000000\\tA\\tG\\t0.98\\t0.012\\t0.0001\\t0.2\\t8\\n22\\trs99999\\t21000000\\tC\\tT\\t0.1\\t0.003\\t0.0002\\t0.3\\t2\\n",
    report: {model: data.options.model, components: data.options.components, variants: 2,
      chromosomes: ["22"], variants_per_chromosome: {22: 2}, iterations: 12,
      genome_build: "GRCh37", elbo: [-100, -60, -50],
      chromosome_diagnostics: {22: {elbo: [-100, -60, -50], iterations: 12, converged: true,
        variants: 1, fit_seconds: 125.4, wasm_heap_bytes_at_fit_end: 134217728}},
      converged: true, message: "UI fixture completed.",
      timings: {preparation_seconds: 65.8, fit_seconds: 125.4, run_seconds: 191.2}}
  }});
};
"""


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(ROOT / "docs")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as temporary, sync_playwright() as playwright:
            folders = {}
            for kind, name in (("sumstats", "height.tsv"), ("ld", "chr_22.zip")):
                folder = Path(temporary) / kind
                folder.mkdir()
                (folder / name).write_text("UI fixture")
                folders[kind] = folder
            browser = playwright.chromium.launch(executable_path=os.environ.get("VIPRS_BROWSER_EXECUTABLE"), headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            requests = []
            page.on("request", lambda request: requests.append(request.url))
            page.route("**/worker.js*", lambda route: route.fulfill(content_type="text/javascript", body=MOCK_WORKER))
            page.add_init_script("""
                const InferenceWorker = window.Worker;
                window.Worker = class extends InferenceWorker {
                    constructor(...args) { super(...args); window.progressWorker = this; }
                    postMessage(data) {
                        window.progressRequest = data;
                        super.postMessage({...data, holdProgress: Boolean(window.holdProgress)});
                    }
                };
            """)
            page.goto(f"http://127.0.0.1:{server.server_port}/webapp/")
            assert page.locator("#runtime-state").get_attribute("data-state") == "idle"
            assert len(page.workers) == 0
            assert page.locator("#sumstats-format option").all_text_contents() == [FORMAT_LABELS[alias] for alias in SUMSTATS_FORMATS]
            assert page.locator("#sumstats-format option").evaluate_all("options => options.map(option => option.value)") == SUMSTATS_FORMATS
            assert page.locator("#ld-source").input_value() == "hosted"
            assert page.locator("#sumstats-source").input_value() == "catalog"
            assert page.locator("#sumstats-source option").all_text_contents()[0] == "GWAS Catalog ID"
            assert "Example" not in page.locator("#sumstats-source").inner_text()
            assert page.locator("#sumstats-catalog-id").input_value() == ""
            assert page.locator("#sumstats-format").input_value() == "gwascatalog"
            assert page.locator("#sumstats-format").is_disabled()
            page.locator("#sumstats-format-info").click()
            assert page.locator("#sumstats-format-dialog").is_visible()
            assert "GWAS Catalog" in page.locator("#sumstats-format-title").inner_text()
            assert page.locator("#sumstats-format-separator").inner_text() == r"Tabs (\t)"
            assert "base_pair_location" in page.locator("#sumstats-format-example").inner_text()
            assert page.locator("#sumstats-format-docs").get_attribute("href").startswith("https://www.ebi.ac.uk/")
            page.keyboard.press("Escape")
            assert not page.locator("#sumstats-format-dialog").is_visible()
            assert page.locator("#sumstats-format-info").evaluate("el => el === document.activeElement")
            page.locator("#sumstats-format-info").click()
            page.mouse.click(2, 2)
            assert not page.locator("#sumstats-format-dialog").is_visible()
            page.locator("#sumstats-source").select_option("files")
            for alias in SUMSTATS_FORMATS:
                page.locator("#sumstats-format").select_option(alias)
                page.locator("#sumstats-format-info").click()
                assert page.locator("#sumstats-format-title").inner_text() == FORMAT_LABELS[alias]
                assert page.locator("#sumstats-format-example").text_content() == FORMAT_HELP[alias]["example"]
                assert page.locator("#sumstats-format-docs").get_attribute("target") == "_blank"
                page.locator("#sumstats-format-close").click()
            page.locator("#sumstats-source").select_option("catalog")
            assert len(page.workers) == 0
            assert not any("pyodide" in url or url.endswith("runtime.json") or url.endswith("worker.js") for url in requests)
            assert page.locator("#ld-hosted-path").text_content() == "hf://datasets/shz9/ukb-ld/EUR/chr_*.zip"
            assert page.locator("#ld-source option").all_text_contents()[0] == "Hosted matrices"
            assert page.locator("#ld-hosted-path").is_visible()
            page.locator("#ld-dataset-info").click()
            assert page.locator("#ld-dataset-dialog").is_visible()
            assert page.locator("#ld-panel-name").inner_text() == "UK Biobank"
            assert page.locator("#ld-genome-build").inner_text() == "GRCh37"
            assert "ukb-ld" in page.locator("#ld-panel-link").get_attribute("href")
            page.keyboard.press("Escape")
            assert not page.locator("#ld-dataset-dialog").is_visible()
            assert page.locator("#ld-dataset-info").evaluate("el => el === document.activeElement")
            page.locator("#ld-dataset-info").click()
            page.mouse.click(2, 2)
            assert not page.locator("#ld-dataset-dialog").is_visible()
            assert "GRCh37" in page.locator("#ld-genome-build").inner_text()
            assert "GRCh37" in page.locator("#catalog-build-hint").inner_text()
            page.locator("#ld-source").select_option("files")
            assert page.locator("#ld-custom-build-inputs").is_visible()
            assert "metadata first" in page.locator("#catalog-build-hint").inner_text()
            page.locator("#ld-custom-build").select_option("GRCh38")
            assert "GRCh38" in page.locator("#catalog-build-hint").inner_text()
            page.locator("#ld-custom-build").select_option("")
            page.locator("#ld-source").select_option("hosted")
            page.locator("#ld-cohort").select_option("hgdp-1kg")
            assert "/hgdp-1kg-ld/EUR/" in page.locator("#ld-hosted-path").text_content()
            assert "GRCh38" in page.locator("#ld-genome-build").inner_text()
            assert page.locator("#ld-snp-set").inner_text() == "MAF > 1%"
            page.locator("#ld-dataset-info").click()
            assert page.locator("#ld-panel-name").inner_text() == "HGDP + 1000 Genomes"
            assert "hgdp-1kg-ld" in page.locator("#ld-panel-link").get_attribute("href")
            page.locator("#ld-dataset-close").click()
            assert "GRCh38" in page.locator("#catalog-build-hint").inner_text()
            page.locator("#ld-hosted-inputs").screenshot(path=str(ARTIFACTS / "ld-hgdp-desktop.png"))
            page.locator("#ld-ancestry").select_option("AFR")
            assert "/hgdp-1kg-ld/AFR/" in page.locator("#ld-hosted-path").text_content()
            page.locator("#sumstats-catalog-id").fill("GCST005195")
            page.locator("#run").click()
            page.wait_for_function("!document.querySelector('#run').disabled")
            request = json.loads(page.locator("#log").text_content().split("REQUEST ")[-1].splitlines()[0])
            assert request["options"]["ld_genome_build"] == "GRCh38"
            page.locator("#sumstats-catalog-id").fill("")
            page.locator("#ld-cohort").select_option("ukb")
            page.locator("#ld-ancestry").select_option("EUR")
            assert page.locator(".sumstats-card").evaluate("(el) => el.getBoundingClientRect().left") < page.locator(".ld-card").evaluate("(el) => el.getBoundingClientRect().left")
            assert page.locator(".resource-links a").evaluate_all("links => links.every(a => a.target === '_blank' && a.rel.includes('noopener'))")
            assert "User guide" not in page.locator(".resource-links").inner_text()
            assert "Shadi Zabad" in page.locator("footer").inner_text()
            assert page.locator("label[for=model]").text_content() == "Prior on the genetic architecture"
            assert page.locator("#slab-curves .slab-curve").count() == 1
            assert page.locator("#prior-summary").text_content() == "1 spike + 1 slab"
            assert page.locator("#prior-plot title").count() == 0
            assert page.locator(".spike-mark path").get_attribute("stroke") is None
            assert page.locator(".spike-mark path").evaluate("el => getComputedStyle(el).stroke") != page.locator("#slab-curves .slab-curve").evaluate("el => getComputedStyle(el).stroke")
            assert page.locator(".spike-legend").evaluate("el => getComputedStyle(el).color") != page.locator(".slab-legend").evaluate("el => getComputedStyle(el).color")
            page.locator("#model").select_option("VIPRSMix")
            assert page.locator("#slab-curves .slab-curve").count() == 3
            assert page.locator("#components").get_attribute("max") == "9"
            page.locator("#components").fill("9")
            assert page.locator("#slab-curves .slab-curve").count() == 9
            assert len(set(page.locator("#slab-curves .slab-curve").evaluate_all("curves => curves.map(curve => curve.getAttribute('stroke'))"))) == 9
            assert "9 centered Gaussian slabs" in page.locator("#prior-plot-description").text_content()
            page.locator("#model").select_option("VIPRS")
            page.screenshot(path=str(ARTIFACTS / "ui-desktop.png"), full_page=True)
            page.locator("#ld-ancestry").select_option("AFR")
            assert "/AFR/chr_*.zip" in page.locator("#ld-hosted-path").text_content()
            assert page.locator(".genetics-figure input").count() == 0
            assert page.locator("#demo").count() == 1
            page.locator(".genetics-figure").hover()
            page.wait_for_function("new DOMMatrix(getComputedStyle(document.querySelector('.prs-threshold')).transform).m41 < -1")
            transforms = page.locator(".prs-threshold, .threshold-label, #tail-window").evaluate_all("els => els.map(el => getComputedStyle(el).transform)")
            assert len(set(transforms)) == 1
            page.locator("#page-title").hover()
            assert page.locator(".prs-threshold").evaluate("el => getComputedStyle(el).animationPlayState") == "paused"
            for kind, folder in folders.items():
                page.locator(f"#{kind}-source").select_option("files")
                page.locator(f"#{kind}-file").set_input_files(next(folder.iterdir()))
                page.locator(f"#{kind}-source").select_option("folder")
                page.locator(f"#{kind}-folder").set_input_files(folder)
                assert "1 file(s) matched" in page.locator(f"#{kind}-folder-summary").inner_text()
                page.locator(f"#{kind}-source").select_option("urls")
                page.locator(f"#{kind}-url").fill(f"https://example.org/{kind}.zip")
            page.locator("#sumstats-format").select_option("magenpy")
            for summary_source, ld_source in itertools.product(["files", "folder", "urls"], ["hosted", "files", "folder", "urls", "example"]):
                page.locator("#sumstats-source").select_option(summary_source)
                page.locator("#ld-source").select_option(ld_source)
                assert page.locator(f"#sumstats-{summary_source}-inputs").is_visible()
                assert page.locator(f"#ld-{ld_source}-inputs").is_visible()
                assert not page.locator("#sumstats-format").is_disabled()
                page.locator("#run").click()
                page.wait_for_function("!document.querySelector('#run').disabled")
                assert page.locator("#status-badge").text_content() == "Converged", (summary_source, ld_source, page.locator("#log").text_content())
                request = json.loads(page.locator("#log").text_content().split("REQUEST ")[-1].splitlines()[0])
                expected_sumstats = {"files": "height.tsv", "folder": "height.tsv", "urls": "https://example.org/sumstats.zip"}
                expected_ld = {"files": "chr_22.zip", "folder": "chr_22.zip", "urls": "https://example.org/ld.zip", "example": "hf://datasets/shz9/ukb-ld/EUR/chr_22.zip", "hosted": "hf://datasets/shz9/ukb-ld/AFR/chr_*.zip"}
                assert request["sumstats"] == [expected_sumstats[summary_source]]
                assert request["ld"] == [expected_ld[ld_source]]
                assert request["options"]["sumstats_format"] == "magenpy"
                assert request["options"]["tolerance"] == 1e-6
            page.locator("#sumstats-source").select_option("catalog")
            page.locator("#sumstats-catalog-id").fill("gcst005195")
            page.locator("#run").click()
            page.wait_for_function("!document.querySelector('#run').disabled")
            request = json.loads(page.locator("#log").text_content().split("REQUEST ")[-1].splitlines()[0])
            assert request["sumstats"] == ["GCST005195"]
            assert request["options"]["sumstats_format"] == "gwascatalog"
            assert "UI fixture completed." not in page.locator("#result-summary").text_content()
            assert page.locator("#status").text_content() == ""
            page.evaluate("window.holdProgress = true")
            page.locator("#run").click()
            page.wait_for_function("document.querySelector('#status').textContent.includes('chromosome 2 (2/22)')")
            assert page.locator("#activity-title").inner_text() == "Model fitting"
            assert page.locator("#run-progress progress").count() == 3
            assert page.locator("#progress-ld-row").count() == 0
            assert page.locator("#progress-fitting").evaluate("p => p.value === 1 && p.max === 22")
            assert "iteration" not in page.locator("#progress-fitting-detail").inner_text()
            assert "iteration 17" in page.locator("#status").inner_text()
            for width in [1440, 760, 390, 320]:
                page.set_viewport_size({"width": width, "height": 1000})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), width
                for bar in page.locator("#run-progress progress").all():
                    assert bar.bounding_box()["width"] > 0
                    assert bar.bounding_box()["width"] <= 500
                    assert bar.bounding_box()["height"] <= 4
                page.locator(".monitor").screenshot(path=str(ARTIFACTS / f"inference-progress-{width}.png"))
            page.evaluate("""window.progressWorker.onmessage({data: {type: 'stage-progress', id: window.progressRequest.id, stage: 'harmonization', state: 'active', detail: 'Matching chromosome 3 variants.'}})""")
            assert not page.locator("#progress-harmonization").evaluate("p => p.hasAttribute('value')")
            page.locator("#cancel").click()
            assert page.locator("#progress-harmonization-row").get_attribute("data-state") == "cancelled"
            page.evaluate("window.holdProgress = false")
            page.locator("#run").click()
            page.wait_for_function("!document.querySelector('#run').disabled")
            page.set_viewport_size({"width": 1440, "height": 1100})
            page.locator("#demo").click()
            assert page.locator("#ld-source").input_value() == "example"
            assert page.locator("#sumstats-source").input_value() == "urls"
            assert page.locator("#sumstats-url").input_value() == "https://raw.githubusercontent.com/shz9/magenpy/master/magenpy/data/ukb_height_chr22.fastGWA.gz"
            page.locator("#ld-source").select_option("hosted")
            page.locator("#sumstats-source").select_option("files")
            page.locator("#ld-ancestry").select_option("EUR")
            for width in [1440, 1000, 800, 760, 390, 320]:
                page.set_viewport_size({"width": width, "height": 1000})
                for cohort in ["ukb", "hgdp-1kg"]:
                    page.locator("#ld-cohort").select_option(cohort)
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (width, cohort)
            page.locator("#ld-hosted-inputs").screenshot(path=str(ARTIFACTS / "ld-hgdp-mobile.png"))
            page.locator("#ld-cohort").select_option("ukb")
            for width in [1440, 390, 320]:
                page.set_viewport_size({"width": width, "height": 844})
                page.locator("#sumstats-format-info").click()
                assert page.locator("#sumstats-format-dialog").is_visible()
                for _ in range(6):
                    page.keyboard.press("Tab")
                    # Chromium can move focus to browser chrome (activeElement
                    # becomes body); underlying form controls must stay inert.
                    assert page.evaluate("document.activeElement === document.body || document.querySelector('#sumstats-format-dialog').contains(document.activeElement)")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                bounds = page.locator("#sumstats-format-dialog").bounding_box()
                assert bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= width
                page.screenshot(path=str(ARTIFACTS / f"format-help-{width}.png"))
                page.locator("#sumstats-format-close").click()
            for width in [1440, 390, 320]:
                page.set_viewport_size({"width": width, "height": 844})
                page.locator("#ld-dataset-info").click()
                assert page.locator("#ld-dataset-dialog").is_visible()
                for _ in range(6):
                    page.keyboard.press("Tab")
                    assert page.evaluate("document.activeElement === document.body || document.querySelector('#ld-dataset-dialog').contains(document.activeElement)")
                bounds = page.locator("#ld-dataset-dialog").bounding_box()
                assert bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= width
                assert page.locator("#ld-dataset-dialog").evaluate("el => el.scrollWidth <= el.clientWidth")
                page.screenshot(path=str(ARTIFACTS / f"ld-details-{width}.png"))
                page.locator("#ld-dataset-close").click()
            page.set_viewport_size({"width": 390, "height": 844})
            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(ARTIFACTS / "ui-mobile.png"))
            page.locator("#ld-hosted-inputs").screenshot(path=str(ARTIFACTS / "ui-mobile-input.png"))
            page.set_viewport_size({"width": 1440, "height": 1100})
            assert page.locator("#result-rows tr").count() == 1
            page.locator("#results").screenshot(path=str(ARTIFACTS / "ui-results.png"))
            assert page.locator("#tab-summary").get_attribute("aria-selected") == "true"
            assert page.locator(".result-navigation #results-title").count() == 1
            assert page.locator(".result-navigation #download").count() == 1
            assert page.locator(".result-download").evaluate("el => getComputedStyle(el).borderTopStyle") == "dashed"
            assert page.locator("#fit-message").inner_text() == "All chromosomes converged successfully."
            assert page.locator("#elapsed").inner_text() == "03:11"
            assert "Input preparation 01:05 · Model fitting 02:05" in page.locator("#timing-summary").inner_text()
            assert page.locator("#fit-overview .fit-stat strong").all_text_contents()[:2] == ["02:05", "03:11"]
            assert "02:05" in page.locator("#runtime-chart").inner_text()
            assert page.locator("#diagnostic-rows td").nth(3).text_content() == "02:05"


            assert page.locator("#results-title").evaluate("el => getComputedStyle(el).fontSize === getComputedStyle(document.querySelector('.model-card h2')).fontSize")
            assert page.locator(".scoring-options select").evaluate_all("selects => selects.map(el => el.id)") == ["biobank-platform", "genotype-format", "genotype-layout", "scoring-tool", "execution-runtime"]
            assert page.locator("#elbo-chart svg").count() == 1
            elbo = page.locator("#elbo-chart svg")
            assert "2 fitted observations; first -60" in elbo.get_attribute("aria-label")
            assert "-100" not in elbo.text_content()
            assert "128.0 MiB" in page.locator("#diagnostic-rows").text_content()
            page.locator("#tab-summary").focus()
            page.keyboard.press("ArrowRight")
            assert page.locator("#tab-scoring").get_attribute("aria-selected") == "true"
            page.wait_for_function("document.querySelector('#download').hasAttribute('href')")
            dosage = page.locator("#download").evaluate("async a => await (await fetch(a.href)).text()")
            headers, values, *_ = [line.split("\t") for line in dosage.strip().split("\n")]
            import math
            assert math.isclose(float(values[headers.index("BETA")]), 0.012 / math.sqrt(0.32))
            assert math.isclose(float(values[headers.index("VAR_BETA")]), 0.0001 / 0.32)
            assert headers[6] == "BETA"
            page.select_option("#download-format", "pgs")
            pgs = page.locator("#download").evaluate("async a => await (await fetch(a.href)).text()")
            assert "#genome_build=GRCh37" in pgs and "effect_weight" in pgs
            page.select_option("#download-format", "native")
            page.locator("#tab-summary").click()
            page.locator("#tab-scoring").click()
            assert page.locator("#download-format").input_value() == "native"
            browser_context = page.context
            browser_context.grant_permissions(["clipboard-read", "clipboard-write"])
            page.locator("#scoring-guide section").filter(has=page.locator("h4", has_text="Calculate scores")).locator(".copy-command").click()
            assert 'plink2 --pfile "$TARGET_PREFIX"' in page.evaluate("navigator.clipboard.readText()")
            assert page.locator("#genotype-format").input_value() == "pgen"
            assert page.locator("#biobank-platform").input_value() == "generic"
            assert page.locator("#scoring-tool").input_value() == "plink"
            assert page.locator("#results a[download]").count() == 1
            assert page.locator("#tab-plots strong").inner_text() == "Visualization"
            assert page.locator(".category-icon[aria-hidden=true] svg").count() == 4
            assert page.locator("#tab-scoring strong").inner_text() == "Scoring instructions"
            assert page.locator("#biobank-platform option[value=generic]").inner_text() == "Linux/Unix Compute environment"
            assert page.locator("#tab-scoring strong").evaluate("el => parseFloat(getComputedStyle(el).fontSize)") == 16
            compatible = {"pgen": ["plink", "pgsc"], "bed": ["plink", "viprs", "pgsc"], "vcf": ["score", "plink", "pgsc"], "bgen": ["plink"]}
            expected_input = {"pgen": '--pfile "$TARGET_PREFIX"', "bed": '--bfile "$TARGET_PREFIX"', "vcf": '--vcf "$TARGET_FILE"', "bgen": '--bgen "$TARGET_FILE" ref-first --sample "$SAMPLE_FILE"'}
            for genotype, tools in compatible.items():
                page.select_option("#biobank-platform", "generic")
                page.select_option("#genotype-format", genotype)
                assert page.locator("#scoring-tool").input_value() == ("score" if genotype == "vcf" else "plink")
                enabled = page.locator("#scoring-tool option:enabled").evaluate_all("options => options.map(o => o.value)")
                assert set(enabled) == set(tools)
                for environment in ["generic", "ukb", "aou"]:
                    page.select_option("#biobank-platform", environment)
                    expected_tool = "plink" if environment == "ukb" else ("pgsc" if environment == "aou" and "pgsc" in tools else ("score" if genotype == "vcf" else "plink"))
                    assert page.locator("#scoring-tool").input_value() == expected_tool
                    assert page.locator("#platform-guidance").is_visible() == (environment != "generic")
                    for tool in tools:
                        page.select_option("#scoring-tool", tool)
                        assert page.locator("#view-scoring .scoring-guide").count() == 1
                        assert page.locator("#scoring-guide > .guide-source").count() == 0
                        assert page.locator("#scoring-guide .guide-step a").count() > 0
                        assert page.locator("#scoring-guide .guide-step > p").count() == 0
                        assert page.locator("#scoring-guide .guide-step ul").count() == page.locator("#scoring-guide .guide-step").count()
                        assert "also accepts" not in page.locator("#scoring-guide").inner_text()
                        commands = "\n".join(page.locator("#scoring-guide pre").all_text_contents())
                        assert {"plink": expected_input[genotype], "viprs": "viprs_score -f", "pgsc": "nextflow run", "score": "bcftools +score"}[tool] in commands
                        installation_urls = {"plink": "https://www.cog-genomics.org/plink/2.0/", "viprs": "https://shz9.github.io/viprs/installation/", "pgsc": "https://pgsc-calc.readthedocs.io/en/2.2.0/getting-started.html", "score": "https://github.com/freeseek/score#installation"}
                        if not (environment == "ukb" and tool == "plink"):
                            assert page.locator(f'#scoring-guide a[href="{installation_urls[tool]}"]').count() >= 1
                        else:
                            assert "pre-installed PLINK 2" in page.locator("#scoring-guide").inner_text()
                            assert "curl -fL" not in commands
                        if tool == "viprs":
                            assert page.locator("#scoring-guide section").filter(has=page.locator("h4", has_text="Score the target cohort")).locator("a[href*=viprs_score]").count() == 1
                        if environment == "aou":
                            assert "gcloud storage cp" not in commands
                            assert "existing filesystem path" in page.locator("#scoring-guide").inner_text()
            page.select_option("#genotype-format", "pgen")
            page.select_option("#biobank-platform", "generic")
            for tool in ["plink", "pgsc"]:
                page.select_option("#scoring-tool", tool)
                for runtime in ["native", "docker", "apptainer", "singularity"]:
                    page.select_option("#execution-runtime", runtime)
                    commands = "\n".join(page.locator("#scoring-guide pre").all_text_contents())
                    if tool == "plink":
                        if runtime == "native":
                            assert "curl -fL" in commands and 'export PATH="$HOME/.local/bin:$PATH"' in commands
                        elif runtime == "docker":
                            assert 'docker info' in commands and '--volume "$PWD:/work"' in commands
                            assert "quay.io/biocontainers/plink2:" in commands
                        else:
                            assert runtime + ' pull plink2.sif docker://' in commands
                            assert runtime + ' exec --bind "$PWD:/work"' in commands
                    else:
                        profile = "conda" if runtime == "native" else runtime
                        assert "-profile " + profile in commands
                        assert "-r v2.2.0" in commands
            # File layout selects one route and uses this run's fitted chromosomes.
            for genotype, tool in [("pgen", "plink"), ("bed", "viprs"), ("vcf", "score"), ("pgen", "pgsc")]:
                page.select_option("#genotype-format", genotype)
                page.select_option("#scoring-tool", tool)
                page.select_option("#genotype-layout", "chromosomes")
                guide = page.locator("#scoring-guide").inner_text()
                assert "absolute" in guide
                if tool in ["plink", "score"]:
                    assert "Combine chromosome scores" in guide
                    assert "CHROMOSOMES=(22)" in guide
                elif tool == "viprs":
                    assert "cohort_chr*.bed" in guide and "automatically" in guide
                else:
                    assert "cohort_chr22,22,pfile" in guide and "automatically" in guide
                page.select_option("#genotype-layout", "combined")
                assert "Combine chromosome scores" not in page.locator("#scoring-guide").inner_text()
            page.select_option("#genotype-format", "pgen")
            page.select_option("#execution-runtime", "native")
            page.select_option("#scoring-tool", "plink")
            page.select_option("#download-format", "report")
            report_export = page.locator("#download").evaluate("async a => await (await fetch(a.href)).json()")
            assert report_export["genome_build"] == "GRCh37"
            page.select_option("#download-format", "native")
            native_export = page.locator("#download").evaluate("async a => await (await fetch(a.href)).text()")
            assert "0.012" in native_export
            page.locator("#tab-table").click()
            assert page.locator("#result-rows").is_visible()
            page.fill("#table-rsid", "RS99999")
            page.wait_for_function("document.querySelector('#result-rows').textContent.includes('rs99999')")
            assert page.locator("#result-rows tr").count() == 1
            assert "0.003" in page.locator("#result-rows").inner_text()
            page.fill("#table-rsid", "rs12345, rs99999")
            page.wait_for_function("document.querySelectorAll('#result-rows tr').length === 2")
            page.select_option("#chromosome-filter", "22")
            assert page.locator("#result-rows tr").count() == 2
            page.fill("#table-rsid", "rs-not-present")
            page.wait_for_function("!document.querySelector('#table-empty').hidden")
            assert page.locator("#result-rows tr").count() == 0
            page.fill("#table-rsid", "")
            page.wait_for_function("document.querySelectorAll('#result-rows tr').length === 1")
            page.select_option("#chromosome-filter", "")

            page.evaluate("""() => {
                window.axisLabels = [];
                const original = CanvasRenderingContext2D.prototype.fillText;
                CanvasRenderingContext2D.prototype.fillText = function(text, x, y, ...args) {
                    if (this.canvas.id === 'weight-canvas') window.axisLabels.push({text, x, y, align: this.textAlign});
                    return original.call(this, text, x, y, ...args);
                };
            }""")
            page.locator("#tab-plots").click()
            assert page.locator("#weight-plot-kind").input_value() == "miami"
            assert page.locator("#weight-plot-kind option[value=miami]").inner_text() == "Miami · GWAS vs. posterior estimates"
            assert page.locator("#weight-plot-kind option[value=trumpet]").inner_text() == "Trumpet · MAF vs. effect size"
            assert page.locator("#weight-canvas").is_visible()
            labels = page.evaluate("window.axisLabels")
            upper = [label for label in labels if label["text"] == "PIP"][-1]
            lower = [label for label in labels if label["text"] == "GWAS Z²"][-1]
            assert upper["x"] == lower["x"] and upper["align"] == lower["align"] == "left"
            assert upper["y"] < lower["y"]
            chromosome_label = [label for label in labels if label["text"] == "Chromosome"][-1]
            assert lower["y"] >= chromosome_label["y"] + 20
            for width in [1440, 390, 320]:
                page.set_viewport_size({"width": width, "height": 1100})
                page.evaluate("window.axisLabels = []")
                page.select_option("#miami-metric", "BETA2")
                page.select_option("#miami-metric", "PIP")
                chart_labels = page.evaluate("window.axisLabels")
                lower_title = [label for label in chart_labels if label["text"] == "GWAS Z²"][-1]
                x_title = [label for label in chart_labels if label["text"] == "Chromosome"][-1]
                assert lower_title["y"] >= x_title["y"] + 20
                tick_labels = [label for label in chart_labels if label["align"] == "center" and label["text"] != "Chromosome"]
                assert all(lower_title["y"] >= tick["y"] + 40 for tick in tick_labels)
                page.locator("#weight-canvas").screenshot(path=str(ARTIFACTS / f"miami-labels-{width}.png"))
            page.set_viewport_size({"width": 1440, "height": 1100})
            for kind in ["trumpet", "miami", "locus"]:
                page.select_option("#weight-plot-kind", kind)
                assert page.locator("#weight-canvas").is_visible()
                assert page.locator("#weight-canvas").get_attribute("aria-label").startswith(kind)
                if kind == "trumpet":
                    assert page.locator("#trumpet-scale").input_value() == "dosage"
                    assert "matching the dosage scoring download" in page.locator("#weight-plot-description").inner_text()
                    page.evaluate("window.axisLabels = []")
                    page.select_option("#trumpet-scale", "standardized")
                    native_ticks = page.evaluate("window.axisLabels.filter(t => t.y === 44)")
                    assert math.isclose(float(native_ticks[-1]["text"]), 0.012, rel_tol=1e-4)
                    page.evaluate("window.axisLabels = []")
                    page.select_option("#trumpet-scale", "dosage")
                    dosage_ticks = page.evaluate("window.axisLabels.filter(t => t.y === 44)")
                    assert math.isclose(float(dosage_ticks[-1]["text"]), 0.012 / math.sqrt(0.32), rel_tol=1e-4)
                    page.locator("#results").screenshot(path=str(ARTIFACTS / "results-trumpet.png"))
                if kind == "miami":
                    page.select_option("#miami-metric", "BETA2")
                    assert "Posterior β² above, GWAS Z² below" in page.locator("#weight-plot-description").inner_text()
                    assert page.locator("label[for=miami-metric]").inner_text() == "Upper panel"
                    page.locator("#results").screenshot(path=str(ARTIFACTS / "results-miami.png"))
            ucsc = page.locator("#locus-browser").get_attribute("href")
            assert "genome.ucsc.edu/cgi-bin/hgTracks?db=hg19" in ucsc
            assert "position=chr22%3A" in ucsc
            assert page.locator("a[href*=ensembl]").count() == 0
            page.fill("#variant-id", "rs12345")
            page.locator("#variant-search button").click()
            assert "rs12345" in page.locator("#variant-detail").inner_text()
            page.fill("#locus-center", "30000000")
            page.locator("#locus-controls button").click()
            assert page.locator("#plot-empty").is_visible()
            for width in [1440, 760, 390, 320]:
                page.set_viewport_size({"width": width, "height": 1100})
                for view in ["summary", "scoring", "table", "plots"]:
                    page.locator(f"#tab-{view}").click()
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (width, view)
                    if view == "table":
                        chromosome = page.locator("#chromosome-filter").bounding_box()
                        search = page.locator("#table-rsid").bounding_box()
                        assert abs(chromosome["height"] - search["height"]) <= 1
                        if width > 600:
                            assert abs(chromosome["y"] - search["y"]) <= 1
                        else:
                            assert abs(chromosome["x"] - search["x"]) <= 1
                            assert abs(chromosome["width"] - search["width"]) <= 1
                    if width in [1440, 390]:
                        page.locator("#results").screenshot(path=str(ARTIFACTS / f"results-{view}-{width}.png"))
            page.set_viewport_size({"width": 1440, "height": 1100})
            # Exercise unavailable metadata and the fixed scoring-column contract.
            page.evaluate("""async () => {
              const {resultsPanel} = await import('./results.js');
              window.fixtureResultsPanel = resultsPanel;
              const row = {CHR: 22, SNP: 'no-position', A1: 'A', A2: 'G', PIP: 0.9, BETA: 0.01, VAR_BETA: 0.001, MAF: 0.2};
              window.edgeResult = {rows: [row], chromosome_rows: {22: [row]},
                tsv: 'CHR\\tSNP\\tA1\\tA2\\tPIP\\tBETA\\tVAR_BETA\\tMAF\\n22\\tno-position\\tA\\tG\\t0.9\\t0.01\\t0.001\\t0.2\\n',
                report: {chromosomes: ['22'], variants: 1, variants_per_chromosome: {22: 1}, iterations: 4, converged: false,
                  timings: {fit_seconds: 1, run_seconds: 2}}};
              window.edgeDownload = (_id, data, type) => URL.createObjectURL(new Blob([data], {type}));
              resultsPanel.show(window.edgeResult, window.edgeDownload);
            }""")
            assert "not recorded" in page.locator("#elbo-chart").inner_text()
            assert "Review fit" in page.locator("#fit-overview").inner_text()
            page.locator("#tab-scoring").click()
            page.wait_for_function("document.querySelector('#download').hasAttribute('href')")
            missing_pos = page.locator("#download").evaluate("async a => await (await fetch(a.href)).text()")
            assert missing_pos.splitlines()[0].split("\t")[6] == "BETA"
            page.select_option("#download-format", "pgs")
            assert page.locator("#download").get_attribute("aria-disabled") == "true"
            page.locator("#tab-plots").click()
            page.select_option("#weight-plot-kind", "locus")
            assert page.locator("#plot-empty").is_visible()
            assert page.locator("#locus-browser").is_hidden()
            page.evaluate("""() => {
                window.edgeResult.report.genome_build = 'GRCh38';
                window.edgeResult.rows[0].POS = 33840160;
                window.fixtureResultsPanel.show(window.edgeResult, window.edgeDownload);
            }""")
            page.locator("#tab-plots").click()
            page.select_option("#weight-plot-kind", "locus")
            assert "db=hg38" in page.locator("#locus-browser").get_attribute("href")
            page.evaluate("""() => {
                window.edgeResult.tsv = window.edgeResult.tsv.replace('MAF', 'UNKNOWN');
                window.fixtureResultsPanel.show(window.edgeResult, window.edgeDownload);
            }""")
            page.locator("#tab-scoring").click()
            assert page.locator("#download").get_attribute("aria-disabled") == "true"
            assert "unavailable" in page.locator("#scoring-availability").text_content()
            page.evaluate("""() => {
                Object.assign(window.edgeResult.report, {
                    chromosomes: ['21', '22'], converged: false,
                    message: 'chr21: Optimization converged; chr22: Maximum iterations reached',
                    chromosome_diagnostics: {
                        21: {converged: true, message: 'Optimization converged'},
                        22: {converged: false, message: 'Maximum iterations reached'}
                    }
                });
                window.fixtureResultsPanel.show(window.edgeResult, window.edgeDownload);
            }""")
            assert page.locator("#fit-message li").all_text_contents() == ["Chromosome 22: Maximum iterations reached"]
            assert "Optimization converged" not in page.locator("#fit-message").inner_text()
            page.evaluate("""() => {
                window.edgeResult.report.converged = true;
                window.edgeResult.report.chromosome_diagnostics[22].converged = true;
                window.fixtureResultsPanel.show(window.edgeResult, window.edgeDownload);
            }""")
            assert page.locator("#fit-message").inner_text() == "All chromosomes converged successfully."
            assert page.locator("#fit-message li").count() == 0
            page.emulate_media(reduced_motion="reduce")
            assert page.locator(".genetics-figure").evaluate("el => getComputedStyle(el).transitionDuration") == "0s"
            page.locator(".genetics-figure").hover()
            assert page.locator(".prs-threshold").evaluate("el => getComputedStyle(el).animationName") == "none"
            assert not errors, errors
            browser.close()
            print("Passed: format guides for each distinct format, keyboard/backdrop dismissal, mobile dialogs, 15 independent source combinations, GWAS Catalog input, hosted ancestry/glob, separate folders, example shortcut, threshold, table rendering, new-tab links, footer, reduced motion, four results views, keyboard tab navigation, dosage and PGS exports, command copying, all plots, locus lookup, and mobile results layouts.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
