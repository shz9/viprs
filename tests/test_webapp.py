"""Validation and numerical regression coverage for the standalone browser adapter."""
import importlib.util
import json
import os
import hashlib
from urllib.parse import urlsplit
import zipfile
from pathlib import Path

import pytest

ASSETS = Path(__file__).resolve().parents[1] / "docs" / "webapp"
spec = importlib.util.spec_from_file_location("viprs_browser", ASSETS / "inference.py")
browser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(browser)
OPTIONS = {"max_iter": 200, "tolerance": 1e-6}


@pytest.mark.parametrize("member", ["../outside", "/outside", "dir/../../outside", "dir\\outside"])
def test_archive_traversal_rejected(tmp_path, member):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, "bad")
    with pytest.raises(ValueError, match="unsafe"):
        browser.unpack_ld(archive, tmp_path / "out")
    assert not (tmp_path / "outside").exists()


def test_archive_store_discovery(tmp_path):
    archive = tmp_path / "ld.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("chr22/.zgroup", "{}")
        handle.writestr("chr22/matrix/.zgroup", "{}")
    assert browser.unpack_ld(archive, tmp_path / "out") == [str(tmp_path / "out/chr22")]


@pytest.mark.parametrize("options", [{"max_iter": 0}, {"max_iter": 3.5}, {"tolerance": float("nan")}, {"tolerance": 0}])
def test_invalid_options_rejected(options):
    with pytest.raises(ValueError):
        browser.validate_options(dict(OPTIONS, **options))


def test_bundle_contains_real_webassembly_extensions():
    if not (ASSETS / "runtime.json").exists():
        if os.environ.get("VIPRS_REQUIRE_WASM") == "1":
            pytest.fail("The WASM build must produce runtime.json before validation.")
        pytest.skip("Build or download the WASM bundle to validate compiled assets.")
    manifest = json.loads((ASSETS / "runtime.json").read_text())
    assert manifest["pyodide_version"] == "0.28.3"
    modules = []
    for wheel_url in manifest["wheels"]:
        parsed = urlsplit(wheel_url)
        path = ASSETS / parsed.path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == parsed.fragment.removeprefix("sha256=")
        with zipfile.ZipFile(path) as wheel:
            for name in wheel.namelist():
                if name.endswith(".so"):
                    assert name.endswith("cpython-313-wasm32-emscripten.so")
                    assert wheel.read(name)[:4] == b"\x00asm"
                    modules.append(name)
            if path.name.startswith("viprs-"):
                # The fitted model and its C++ kernel source are the actual repository implementation.
                for source in ["viprs/model/VIPRS.py", "viprs/model/VIPRSMix.py", "viprs/model/vi/e_step.hpp"]:
                    assert wheel.read(source) == (ASSETS.parents[1] / source).read_bytes()
    assert len(modules) == 5
    assert any("e_step_cpp" in name for name in modules)
    assert any("c_utils" in name for name in modules)


@pytest.mark.parametrize("options", [{"model": "unknown"}, {"model": "VIPRSMix", "components": 0},
                                     {"model": "VIPRSMix", "components": 2.5}, {"seed": -1}])
def test_model_options_rejected(options):
    with pytest.raises(ValueError):
        browser.validate_options(dict(OPTIONS, **options))


@pytest.fixture
def multichrom_inputs(tmp_path):
    from webapp_fixtures import make_multichrom_inputs
    return make_multichrom_inputs(tmp_path)


@pytest.mark.parametrize("model", ["VIPRS", "VIPRSMix"])
def test_multi_chromosome_fit(model, multichrom_inputs):
    summaries, archives, combined, combined_ld = multichrom_inputs
    options = dict(OPTIONS, model=model, components=3, sumstats_format="magenpy")
    rows, report = browser.native_fit(summaries, archives, options)
    assert report["chromosomes"] == ["21", "22"]
    assert report["variants_per_chromosome"] == {"21": 48, "22": 48}
    assert report["model"] == model
    assert report["components"] == (3 if model == "VIPRSMix" else 1)
    assert report["timings"]["fit_seconds"] > 0
    other_rows, other_report = browser.native_fit([combined], [combined_ld], options)
    assert other_report["elbo"] == report["elbo"]
    assert other_rows == rows
    output = json.loads(browser.serialize_result(rows, report))
    assert len(output["tsv"].splitlines()) == 97
    assert set(output["chromosome_rows"]) == {"21", "22"}
    assert all(str(row["CHR"]) == key for key, records in output["chromosome_rows"].items() for row in records)


def test_duplicate_and_missing_chromosomes_fail(multichrom_inputs):
    summaries, archives, _, _ = multichrom_inputs
    options = dict(OPTIONS, sumstats_format="magenpy")
    with pytest.raises(ValueError, match="More than one LD store"):
        browser.native_fit(summaries, [archives[0], archives[0]], options)
    with pytest.raises(ValueError, match="overlap on chromosome"):
        browser.native_fit([summaries[0], summaries[0]], archives, options)
    with pytest.raises(ValueError, match="missing for LD chromosome"):
        browser.native_fit([summaries[0]], archives, options)
