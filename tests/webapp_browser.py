"""Exercise real VIPRS WASM inference on the user-specified chromosome 22 datasets."""
import csv
import functools
import http.server
import json
import math
import os
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "build" / "wasm" / "browser-test"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def wait_for_run(page):
    page.wait_for_function("document.querySelector('#run').disabled === false", timeout=300000)
    if page.locator("#status-badge").text_content() == "Error":
        raise AssertionError(page.locator("#log").inner_text())
    assert page.locator("#results").is_visible()


def report_download(page, destination):
    with page.expect_download() as event:
        page.locator("#download-report").click()
    event.value.save_as(destination)
    return json.loads(destination.read_text())


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(QuietHandler, directory=os.environ.get("VIPRS_WEBAPP_ROOT", str(ROOT / "docs")))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=os.environ.get("VIPRS_BROWSER_EXECUTABLE"), headless=True)
            page = browser.new_page(accept_downloads=True)
            page.set_default_timeout(300000)
            page.set_viewport_size({"width": 1440, "height": 1000})
            errors, workers, requests = [], [], []
            page.on("worker", lambda worker: workers.append(worker))
            page.on("request", lambda request: requests.append(request.url))
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/webapp/")
            assert page.locator("#engine").count() == 0
            page.wait_for_function("document.querySelector('#runtime-state').dataset.state === 'ready'", timeout=180000)
            assert len(workers) == 1
            assert "Select your inputs" in page.locator("#status").inner_text()
            assert not any("chr_22.zip" in url or "fastGWA.gz" in url for url in requests)
            page.screenshot(path=str(ARTIFACTS / "initial.png"), full_page=True)
            print("Running the default chromosome 22 data through VIPRS WASM…", flush=True)
            page.locator("#run").click()
            assert page.locator("#metrics").is_hidden()
            assert "running" in page.locator("#activity").get_attribute("class")
            page.screenshot(path=str(ARTIFACTS / "running.png"), full_page=True)
            wait_for_run(page)
            report = report_download(page, ARTIFACTS / "run.json")
            assert report["variants"] > 10000
            assert report["model"] == "VIPRS"
            assert not report["runtime_reused"]
            assert report["timings"]["fit_seconds"] > 0
            assert all("wasm32-emscripten.so" in path for path in report["extensions"].values())
            assert len(report["elbo"]) > 3
            assert all(math.isfinite(v) for v in report["elbo"])
            with page.expect_download() as event:
                page.locator("#download").click()
            event.value.save_as(ARTIFACTS / "weights.tsv")
            with (ARTIFACTS / "weights.tsv").open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            assert len(rows) == report["variants"]
            assert all(math.isfinite(float(row["BETA"])) and 0 <= float(row["PIP"]) <= 1 for row in rows)
            print(f"Default run: {report['variants']} variants, {report['iterations']} iterations, converged={report['converged']}.", flush=True)
            page.set_viewport_size({"width": 1440, "height": 1000})
            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(ARTIFACTS / "desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(ARTIFACTS / "mobile.png"), full_page=True)
            page.set_viewport_size({"width": 1440, "height": 1000})
            # Missing inputs clear old results and preserve the initialized worker.
            page.locator("#source").select_option("files")
            page.locator("#run").click()
            page.wait_for_function("document.querySelector('#status-badge').textContent === 'Error'")
            assert page.locator("#status-badge").text_content() == "Error"
            assert page.locator("#results").is_hidden()
            # Exercise file inputs when the optional local test-data download exists.
            data_dir = ROOT / "build/wasm/test-data"
            if (data_dir / "chr_22.zip").exists():
                page.locator("#sumstats-file").set_input_files(data_dir / "ukb_height_chr22.fastGWA.gz")
                page.locator("#ld-file").set_input_files(data_dir / "chr_22.zip")
                page.locator("#run").click()
                wait_for_run(page)
                upload_report = report_download(page, ARTIFACTS / "upload-run.json")
                assert upload_report["variants"] == report["variants"]
                assert upload_report["elbo"] == report["elbo"]
                assert upload_report["runtime_reused"]
            assert len(workers) == 1
            assert sum(url.endswith("/runtime.json") for url in requests) == 1
            # Exercise both models across two chromosomes using real Zarr fixtures.
            multi = data_dir / "multi"
            assert multi.exists(), "Generate fixtures with: .venv/bin/python tests/webapp_fixtures.py"
            page.locator("#source").select_option("files")
            page.locator("#sumstats-format").select_option("magenpy")
            page.locator("#sumstats-file").set_input_files([multi / "chr21.tsv", multi / "chr22.tsv"])
            page.locator("#ld-file").set_input_files([multi / "chr21.zip", multi / "chr22.zip"])
            for model in ("VIPRS", "VIPRSMix"):
                page.locator("#model").select_option(model)
                page.locator("#run").click()
                wait_for_run(page)
                multi_report = report_download(page, ARTIFACTS / f"multi-{model}.json")
                assert multi_report["model"] == model
                assert multi_report["chromosomes"] == ["21", "22"]
                assert multi_report["variants"] == 96
                assert multi_report["runtime_reused"]
                assert multi_report["components"] == (3 if model == "VIPRSMix" else 1)
                page.locator("#chromosome-filter").select_option("21")
                assert page.locator("#result-rows tr").count() == 20
                assert set(page.locator("#result-rows tr td:nth-child(2)").all_text_contents()) == {"21"}
            page.locator("#sumstats-file").set_input_files(multi / "combined.tsv")
            page.locator("#ld-file").set_input_files(multi / "combined.zip")
            page.locator("#run").click()
            wait_for_run(page)
            combined_report = report_download(page, ARTIFACTS / "multi-combined.json")
            assert combined_report["elbo"] == multi_report["elbo"]
            # A worker-side input failure also leaves the runtime usable.
            page.locator("#ld-file").set_input_files(multi / "chr21.tsv")
            page.locator("#run").click()
            page.wait_for_function("document.querySelector('#status-badge').textContent === 'Error'")
            assert "zip" in page.locator("#status").inner_text().lower()
            assert len(workers) == 1
            page.locator("#model").select_option("VIPRS")
            page.locator("#sumstats-format").select_option("fastGWA")
            # Explicit hf:// and GitHub blob URL inputs use the same full datasets.
            page.locator("#source").select_option("urls")
            page.locator("#sumstats-url").fill("https://github.com/shz9/magenpy/blob/master/magenpy/data/ukb_height_chr22.fastGWA.gz")
            page.locator("#ld-url").fill("hf://datasets/shz9/ukb-ld/EUR/chr_22.zip")
            page.locator("#run").click()
            page.locator("#cancel").click()
            assert page.locator("#status-badge").text_content() == "Cancelled"
            page.locator("#run").click()
            wait_for_run(page)
            url_report = report_download(page, ARTIFACTS / "url-run.json")
            assert url_report["variants"] == report["variants"]
            assert url_report["elbo"] == report["elbo"]
            assert len(workers) == 2
            assert not url_report["runtime_reused"]
            # Initialization failures surface before a run and can be retried.
            retry_page = browser.new_page()
            retry_page.route("**/runtime.json", lambda route: route.fulfill(status=503, body="Unavailable"))
            retry_page.goto(f"http://127.0.0.1:{server.server_port}/webapp/")
            retry_page.wait_for_function("document.querySelector('#runtime-state').dataset.state === 'error'")
            assert retry_page.locator("#retry-runtime").is_visible()
            retry_page.unroute("**/runtime.json")
            retry_page.locator("#retry-runtime").click()
            retry_page.wait_for_function("document.querySelector('#runtime-state').dataset.state === 'ready'", timeout=180000)
            retry_page.close()
            assert not errors, errors
            browser.close()
            print("Passed: actual WASM extensions, real chr22 inference, eager initialization, runtime reuse, VIPRSMix, multiple chromosomes, downloads, uploads, cancellation/rerun, initialization retry, mobile layout.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
