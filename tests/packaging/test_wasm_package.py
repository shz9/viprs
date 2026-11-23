"""Publication must reject corrupt bundles and exclude stale, unlisted wheels."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "wasm_package", Path(__file__).resolve().parents[2] / "tools/wasm/package.py")
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "webapp"
    (root / "wheels").mkdir(parents=True)
    for name in packager.STATIC_FILES:
        (root / name).write_text("test static content")
    wheel = root / "wheels/test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in ("e_step_cpp", "c_utils", "score_cpp", "variant_cpp"):
            archive.writestr(f"{name}.cpython-313-wasm32-emscripten.so", b"\0asmtest")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (root / "runtime.json").write_text(json.dumps({
        "pyodide_version": "0.28.3", "wheels": [f"./wheels/test.whl#sha256={digest}"]}))
    (root / "build-info.json").write_text(json.dumps({
        "pyodide_version": "0.28.3", "wheel_sha256": {"test.whl": digest}}))
    return root


def test_package_only_manifest_assets_and_valid_checksum(assets, tmp_path):
    (assets / "wheels/stale.whl").write_bytes(b"stale")
    (assets / "unrelated.txt").write_text("exclude")
    output = packager.package(assets, tmp_path / "dist")
    before = output.read_bytes()
    with zipfile.ZipFile(output) as archive:
        assert "webapp/index.html" in archive.namelist()
        assert "webapp/results.js" in archive.namelist()
        assert "webapp/scoring.js" in archive.namelist()
        assert "webapp/wheels/test.whl" in archive.namelist()
        assert "webapp/LICENSE" in archive.namelist()
        assert not any("stale" in name or "unrelated" in name for name in archive.namelist())
    assert output.with_suffix(".sha256").read_text().split()[0] == hashlib.sha256(before).hexdigest()
    assert packager.package(assets, tmp_path / "dist").read_bytes() == before


def test_corrupt_wheel_is_not_published(assets, tmp_path):
    (assets / "wheels/test.whl").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        packager.package(assets, tmp_path / "dist")
    assert not (tmp_path / "dist").exists()


def test_manifest_cannot_include_outside_files(assets, tmp_path):
    manifest = json.loads((assets / "runtime.json").read_text())
    manifest["wheels"] = ["../../private.whl"]
    (assets / "runtime.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="local wheels"):
        packager.package(assets, tmp_path / "dist")


def test_inconsistent_provenance_is_not_published(assets, tmp_path):
    info = json.loads((assets / "build-info.json").read_text())
    info["wheel_sha256"] = {}
    (assets / "build-info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="provenance"):
        packager.package(assets, tmp_path / "dist")


def test_missing_static_asset_is_not_published(assets, tmp_path):
    (assets / "worker.js").unlink()
    with pytest.raises(FileNotFoundError):
        packager.package(assets, tmp_path / "dist")
    assert not (tmp_path / "dist").exists()


def test_package_includes_all_local_javascript_imports():
    """A missing module prevents the entire page from starting in release ZIPs."""
    root = Path(__file__).resolve().parents[2] / "docs/webapp"
    packaged = set(packager.STATIC_FILES)
    for name in packaged:
        if Path(name).suffix not in {".js", ".mjs"}:
            continue
        source = (root / name).read_text()
        imports = re.findall(
            r"^import\s+(?:[^\n;]*?\s+from\s+)?[\"'](\.[^\"']+)[\"']",
            source, re.MULTILINE,
        )
        imports += re.findall(r"importScripts\([\"'](\.[^\"']+)[\"']\)", source)
        for dependency in imports:
            path = (root / name).parent / dependency
            relative = path.resolve().relative_to(root.resolve()).as_posix()
            assert relative in packaged, f"{name} imports {relative}, missing from release archive"
