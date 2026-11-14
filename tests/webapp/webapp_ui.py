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
from magenpy.parsers.sumstats_parsers import get_supported_sumstats_formats

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "build/wasm/browser-test"
SUMSTATS_FORMATS = sorted(get_supported_sumstats_formats(include_aliases=True))
FORMAT_HELP = runpy.run_path(str(ROOT / "tools/wasm/format_help.py"))["get_sumstats_format_help"]()
MOCK_WORKER = """
self.postMessage({type: "ready", seconds: 0.01});
self.onmessage = ({data}) => {
  self.postMessage({type: "log", id: data.id, message: "REQUEST " + JSON.stringify({
    sumstats: data.sumstats.map(s => s.url || s.file?.name || s.catalog),
    ld: data.ld.map(s => s.url || s.file.name), options: data.options
  })});
  const row = {SNP: "rs12345", CHR: 22, A1: "A", A2: "G", PIP: 0.98, BETA: 0.012, VAR_BETA: 0.0001};
  self.postMessage({type: "result", id: data.id, result: {
    rows: [row], chromosome_rows: {22: [row]}, tsv: "SNP\\tBETA\\nrs12345\\t0.012\\n",
    report: {model: data.options.model, components: data.options.components, variants: 1,
      chromosomes: ["22"], variants_per_chromosome: {22: 1}, iterations: 12,
      converged: true, message: "UI fixture completed.",
      timings: {preparation_seconds: 0.1, fit_seconds: 0.2, run_seconds: 0.3}}
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
            page.route("**/worker.js", lambda route: route.fulfill(content_type="text/javascript", body=MOCK_WORKER))
            page.goto(f"http://127.0.0.1:{server.server_port}/webapp/")
            assert page.locator("#runtime-state").get_attribute("data-state") == "idle"
            assert len(page.workers) == 0
            assert page.locator("#sumstats-format option").all_text_contents() == SUMSTATS_FORMATS
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
                assert FORMAT_HELP[alias]["name"] in page.locator("#sumstats-format-title").inner_text()
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
            page.emulate_media(reduced_motion="reduce")
            assert page.locator(".genetics-figure").evaluate("el => getComputedStyle(el).transitionDuration") == "0s"
            page.locator(".genetics-figure").hover()
            assert page.locator(".prs-threshold").evaluate("el => getComputedStyle(el).animationName") == "none"
            assert not errors, errors
            browser.close()
            print("Passed: format guides for every alias, keyboard/backdrop dismissal, mobile dialogs, 15 independent source combinations, GWAS Catalog input, hosted ancestry/glob, separate folders, example shortcut, threshold, table rendering, new-tab links, footer, reduced motion, and six viewport widths.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
