"""Validation and numerical regression coverage for the standalone browser adapter."""
import asyncio
import importlib.util
import json
import os
import hashlib
from urllib.parse import urlsplit
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ASSETS = Path(__file__).resolve().parents[1] / "docs" / "webapp"
spec = importlib.util.spec_from_file_location("viprs_browser", ASSETS / "inference.py")
browser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(browser)
OPTIONS = {}


def test_huggingface_compat_is_limited_to_emscripten(monkeypatch):
    modules = {name: SimpleNamespace(marker=name) for name in ("termios", "tty")}
    for name, module in modules.items():
        monkeypatch.setitem(browser.sys.modules, name, module)
    monkeypatch.setattr(browser.sys, "platform", "darwin")
    browser.ensure_huggingface_compat()
    assert all(browser.sys.modules[name] is module for name, module in modules.items())

    for name in modules:
        monkeypatch.delitem(browser.sys.modules, name)
    monkeypatch.setattr(browser.sys, "platform", "emscripten")
    browser.ensure_huggingface_compat()
    assert all(browser.sys.modules[name].__name__ == name for name in modules)


@pytest.mark.parametrize("member", ["../outside", "/outside", "dir/../../outside", "dir\\outside"])
@pytest.mark.parametrize("zip_separator", ["/", "\\"])
def test_archive_traversal_rejected(tmp_path, member, zip_separator, monkeypatch):
    # Exercise ZipInfo's Windows and POSIX name normalization on either host.
    monkeypatch.setattr(zipfile, "os", SimpleNamespace(**{**vars(os), "sep": zip_separator}))
    archive = tmp_path / "bad.zip"
    info = zipfile.ZipInfo(member)
    # Bypass writer normalization to construct the deliberately malformed ZIP.
    info.filename = member
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(info, "bad")
    with zipfile.ZipFile(archive) as handle:
        assert handle.infolist()[0].orig_filename == member
    with pytest.raises(ValueError, match="unsafe"):
        browser.unpack_ld(archive, tmp_path / "out")
    assert not (tmp_path / "outside").exists()
    assert not (tmp_path / "out").exists()


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


def test_browser_catalog_resolution_uses_shared_python_utility():
    from viprs.utils.data_utils import gwas_catalog_directory

    directory = gwas_catalog_directory("GCST005195")

    async def read_text(url):
        assert url == directory
        return '<a href="study-build37.tsv.gz">data</a>'

    assert asyncio.run(
        browser.resolve_browser_gwas_catalog("GCST005195", read_text)
    ) == (directory + "study-build37.tsv.gz")


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
    assert len(modules) == 4
    assert any("e_step_cpp" in name for name in modules)
    assert any("c_utils" in name for name in modules)
    assert not any("math_utils" in name for name in modules)


@pytest.mark.parametrize("options", [{"model": "unknown"}, {"model": "VIPRSMix", "components": 0},
                                     {"model": "VIPRSMix", "components": 2.5},
                                     {"model": "VIPRSMix", "components": 10}, {"seed": -1}])
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


@pytest.mark.parametrize("model_name", ["VIPRS", "VIPRSMix"])
def test_matches_direct_loader_and_fit(multichrom_inputs, tmp_path, model_name):
    from magenpy import GWADataLoader
    from viprs.model import VIPRS, VIPRSMix
    import numpy as np

    summaries, archives, _, _ = multichrom_inputs
    options = dict(OPTIONS, model=model_name, components=3, sumstats_format="magenpy")
    rows, report = browser.native_fit(summaries, archives, options)
    roots = [root for i, archive in enumerate(archives)
             for root in browser.unpack_ld(archive, tmp_path / f"direct-{i}")]
    gdl = GWADataLoader(sumstats_files=list(map(str, summaries)), ld_store_files=roots,
                       sumstats_format="magenpy", temp_dir=str(tmp_path / "temp"), threads=1)
    expected_rows = []
    elbo_histories = []
    expected_iterations = 0
    for chromosome, chromosome_gdl in sorted(gdl.split_by_chromosome().items()):
        model = (VIPRSMix(chromosome_gdl, K=3, threads=1)
                 if model_name == "VIPRSMix" else VIPRS(chromosome_gdl, threads=1))
        np.random.seed(0)
        model.fit(disable_pbar=True)
        expected_rows.extend(json.loads(
            model.to_table().to_json(orient="records", double_precision=15)
        ))
        elbo_histories.append(model.history["ELBO"])
        expected_iterations += model.optim_result.nit
        chromosome_gdl.cleanup()
    expected_elbo = [
        sum(history[min(i, len(history) - 1)] for history in elbo_histories)
        for i in range(max(map(len, elbo_histories)))
    ]
    assert rows == expected_rows
    assert report["elbo"] == expected_elbo
    assert report["iterations"] == expected_iterations


def test_reports_progress_after_each_chromosome(multichrom_inputs):
    summaries, archives, _, _ = multichrom_inputs
    progress = []
    browser.native_fit(
        summaries,
        archives,
        dict(OPTIONS, sumstats_format="magenpy"),
        progress_callback=lambda *event: progress.append(event),
    )
    assert progress == [("21", 1, 2), ("22", 2, 2)]


def test_optimization_defaults_match_viprs():
    import inspect
    from viprs.model import VIPRS

    defaults = inspect.signature(VIPRS.fit).parameters
    iterations, tolerance = browser.validate_options({})
    assert iterations == defaults["max_iter"].default == 1000
    assert tolerance == defaults["x_abs_tol"].default == defaults["f_abs_tol"].default


def test_large_catalog_parser_projects_and_compacts_columns(tmp_path):
    import numpy as np
    from magenpy.parsers.sumstats_parsers import get_sumstats_parser

    source = tmp_path / "catalog.tsv"
    source.write_text(
        "chromosome\tbase_pair_location\trs_id\teffect_allele\tother_allele\t"
        "beta\tstandard_error\teffect_allele_frequency\tp_value\tn\t"
        + "\t".join(f"unused_{index}" for index in range(10))
        + "\n22\t101\trs1\tA\tG\t0.1\t0.02\t0.3\t1e-6\t10000\t"
        + "\t".join("1.5" for _ in range(10))
        + "\n"
    )
    parser = browser._compact_sumstats_parser(
        browser._configure_sumstats_parser(
            get_sumstats_parser("gwascatalog")(None)
        )
    )

    table = parser.parse(source)

    assert set(table) == {
        "CHR", "POS", "SNP", "A1", "A2", "BETA", "SE", "MAF", "PVAL", "N"
    }
    assert not any(column.startswith("unused_") for column in table)
    assert table["POS"].dtype == np.int32
    assert str(table["A1"].dtype) == str(table["A2"].dtype) == "category"
    assert all(
        table[column].dtype == np.float32
        for column in ("BETA", "SE", "MAF", "PVAL", "N")
    )


def test_compact_categorical_alleles_harmonize_with_ld(
    multichrom_inputs, tmp_path
):
    import numpy as np
    from magenpy import GWADataLoader
    from magenpy.parsers.sumstats_parsers import get_sumstats_parser

    summaries, archives, _, _ = multichrom_inputs
    roots = [
        root
        for index, archive in enumerate(archives)
        for root in browser.unpack_ld(archive, tmp_path / f"compact-{index}")
    ]
    parser = browser._compact_sumstats_parser(
        get_sumstats_parser("magenpy")(None)
    )
    loader = GWADataLoader(
        ld_store_files=roots, temp_dir=str(tmp_path / "compact-temp"), threads=1
    )
    loader.read_summary_statistics(summaries, parser=parser)
    assert all(
        str(table.table["A1"].dtype) == str(table.table["A2"].dtype) == "category"
        for table in loader.sumstats_table.values()
    )
    loader.harmonize_data()

    assert loader.n_snps == 96
    assert all(
        np.array_equal(table.a1, np.full(len(table), "A"))
        and np.array_equal(table.a2, np.full(len(table), "G"))
        for table in loader.sumstats_table.values()
    )
    loader.cleanup()


def test_catalog_without_optional_rsid_is_keyed_from_ld(
    multichrom_inputs, tmp_path
):
    import pandas as pd

    summaries, archives, _, _ = multichrom_inputs
    position_only = []
    for index, summary in enumerate(summaries):
        table = pd.read_csv(summary, sep="\t")
        if index == 0:
            table.drop(columns="SNP", inplace=True)
        else:
            table["SNP"] = "#NA"
        path = tmp_path / f"position-only-{index}.tsv"
        table.to_csv(path, sep="\t", index=False)
        position_only.append(path)

    rows, report = browser.native_fit(
        position_only,
        archives,
        dict(OPTIONS, sumstats_format="magenpy"),
    )

    assert report["variants"] == 96
    assert {row["SNP"] for row in rows} == {
        f"test_chr{chromosome}_{index}"
        for chromosome in (21, 22)
        for index in range(48)
    }


@pytest.mark.parametrize("compressed", [False, True])
def test_staged_sumstats_are_read_without_copying(multichrom_inputs, tmp_path, monkeypatch, compressed):
    import gzip
    import shutil

    _, _, combined, combined_ld = multichrom_inputs
    options = dict(OPTIONS, sumstats_format="magenpy")
    expected_rows, _ = browser.native_fit(combined, combined_ld, options)
    staged = tmp_path / ("sumstats-0.tsv.gz" if compressed else "sumstats-0.tsv")
    contents = combined.read_bytes()
    staged.write_bytes(gzip.compress(contents) if compressed else contents)
    before = staged.stat()

    def forbid_copy(*args, **kwargs):
        pytest.fail("Staged summary statistics must not be copied")

    monkeypatch.setattr(shutil, "copyfile", forbid_copy)
    rows, _ = browser.native_fit(staged, combined_ld, options)
    assert rows == expected_rows
    assert staged.stat().st_ino == before.st_ino
    assert staged.stat().st_mtime_ns == before.st_mtime_ns
