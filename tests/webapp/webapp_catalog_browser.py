"""Check Catalog failures through Run, including real Pyodide and LD inspection.

Catalog responses are controlled; runtime packages and hosted LD metadata require
network access. VIPRS_BROWSER_EXECUTABLE may specify an installed Chromium.
"""
import functools
import http.server
import os
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def main():
    handler = functools.partial(QuietHandler, directory=str(ROOT / "docs"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=os.environ.get("VIPRS_BROWSER_EXECUTABLE"), headless=True,
            )
            context = browser.new_context()
            scenario = "mismatch"

            def catalog_response(route):
                if route.request.url.endswith("-meta.yaml"):
                    if scenario == "stalled":
                        return  # Leave the request pending until the app aborts it.
                    body = "genome_assembly: GRCh38\n"
                else:
                    body = (
                        '<a href="GCST90833078.tsv.gz">data</a>'
                        '<a href="GCST90833078.tsv.gz-meta.yaml">metadata</a>'
                    )
                route.fulfill(status=200, body=body, headers={"Access-Control-Allow-Origin": "*"})

            context.route("https://ftp.ebi.ac.uk/**", catalog_response)
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/webapp/")
            page.fill("#sumstats-catalog-id", "GCST90833078")
            for scenario, expected in (
                ("mismatch", "Change the LD matrix to a compatible GRCh38 panel"),
                ("stalled", "timed out after 15s"),
            ):
                page.click("#run")
                page.wait_for_function(
                    """expected => document.querySelector('#status').textContent.includes(expected)
                        && !document.querySelector('#controls').disabled""",
                    arg=expected, timeout=60000,
                )
                assert page.locator("#sumstats-error").is_visible()
                assert page.locator("#run").is_enabled()
                print(f"{scenario}: {page.locator('#status').inner_text()}", flush=True)
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
