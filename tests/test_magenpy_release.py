"""Only a genuinely absent compatible release asset permits a source build."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import urllib.error
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "magenpy_release", Path(__file__).resolve().parents[1] / "tools/wasm/magenpy_release.py")
release_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_tools)


def wheel_bytes(version="0.2.2", wasm=True):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as wheel:
        wheel.writestr(zipfile.ZipInfo(f"magenpy-{version}.dist-info/METADATA"), f"Name: magenpy\nVersion: {version}\n")
        wheel.writestr(zipfile.ZipInfo(f"magenpy-{version}.dist-info/WHEEL"), f"Tag: {release_tools.WHEEL_TAG}\n")
        for module in release_tools.EXTENSIONS:
            wheel.writestr(zipfile.ZipInfo(module.replace(".", "/") + ".cpython-313-wasm32-emscripten.so"), b"\0asmtest" if wasm else b"native")
    return stream.getvalue()


@pytest.fixture
def release():
    data = wheel_bytes()
    name = f"magenpy-0.2.2-{release_tools.WHEEL_TAG}.whl"
    return {"version": "0.2.2", "tag": "v0.2.2", "commit": "a" * 40,
            "release_url": "https://github.com/shz9/magenpy/releases/tag/v0.2.2",
            "source_url": "https://codeload.github.com/shz9/magenpy/tar.gz/" + "a" * 40,
            "assets": [{"name": name, "id": 123, "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
                        "browser_download_url": "https://github.com/shz9/magenpy/releases/download/v0.2.2/" + name}]}


def test_latest_release_is_resolved_to_immutable_commit(monkeypatch):
    requested = []
    def read(url):
        requested.append(url)
        if url.endswith("releases/latest"):
            return json.dumps({"draft": False, "tag_name": "v0.2.2", "id": 1, "html_url": "release", "assets": []}).encode()
        return json.dumps({"sha": "a" * 40}).encode()
    monkeypatch.setattr(release_tools, "read_url", read)
    result = release_tools.resolve_release()
    assert requested[-1].endswith("commits/v0.2.2")
    assert result["source_url"].endswith("a" * 40)
    assert result["version"] == "0.2.2"


def test_existing_wasm_is_verified_and_reused(release, tmp_path, monkeypatch):
    downloads = []
    monkeypatch.setattr(release_tools, "read_url", lambda url: downloads.append(url) or wheel_bytes())
    info = release_tools.reuse_wheel(release, tmp_path)
    assert info["origin"] == "release-asset"
    assert info["source_commit"] == release["commit"]
    assert (tmp_path / info["wheel"]).read_bytes() == wheel_bytes()
    assert release_tools.reuse_wheel(release, tmp_path) == info
    assert len(downloads) == 1


def test_stale_cached_wheel_is_replaced(release, tmp_path, monkeypatch):
    (tmp_path / release["assets"][0]["name"]).write_bytes(b"previous build")
    monkeypatch.setattr(release_tools, "read_url", lambda _: wheel_bytes())
    assert release_tools.reuse_wheel(release, tmp_path)["origin"] == "release-asset"


@pytest.mark.parametrize("kind", ["absent", "native", "wrong-python", "wrong-version"])
def test_only_compatible_release_version_is_selected(release, tmp_path, monkeypatch, kind):
    if kind == "absent":
        release["assets"] = []
    else:
        name = release["assets"][0]["name"]
        release["assets"][0]["name"] = {
            "native": name.replace("pyodide_2025_0_wasm32", "manylinux_2_28_x86_64"),
            "wrong-python": name.replace("cp313", "cp312"),
            "wrong-version": name.replace("0.2.2", "0.2.1"),
        }[kind]
    monkeypatch.setattr(release_tools, "read_url", lambda _: pytest.fail("Unexpected download"))
    assert release_tools.reuse_wheel(release, tmp_path) is None


def test_checksum_sidecar_when_github_digest_is_missing(release, tmp_path, monkeypatch):
    asset = release["assets"][0]
    checksum = asset.pop("digest").removeprefix("sha256:")
    release["assets"].append({"name": asset["name"] + ".sha256", "browser_download_url": "https://example.test/checksum"})
    monkeypatch.setattr(release_tools, "read_url", lambda url: (f"{checksum}  {asset['name']}\n".encode()
        if url.endswith("checksum") else wheel_bytes()))
    assert release_tools.reuse_wheel(release, tmp_path)["wheel_sha256"] == checksum


def test_corrupt_download_fails_without_source_fallback(release, tmp_path, monkeypatch):
    monkeypatch.setattr(release_tools, "read_url", lambda _: b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        release_tools.reuse_wheel(release, tmp_path)
    assert not list(tmp_path.iterdir())


def test_download_failure_is_not_treated_as_absent_asset(release, tmp_path, monkeypatch):
    def fail(_):
        raise urllib.error.HTTPError("url", 403, "Forbidden", {}, None)
    monkeypatch.setattr(release_tools, "read_url", fail)
    with pytest.raises(urllib.error.HTTPError):
        release_tools.reuse_wheel(release, tmp_path)


@pytest.mark.parametrize("version,wasm", [("0.2.1", True), ("0.2.2", False)])
def test_valid_hash_cannot_hide_wrong_wheel_contents(release, tmp_path, monkeypatch, version, wasm):
    data = wheel_bytes(version, wasm)
    release["assets"][0]["digest"] = "sha256:" + hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(release_tools, "read_url", lambda _: data)
    with pytest.raises(ValueError, match="metadata|WebAssembly"):
        release_tools.reuse_wheel(release, tmp_path)
    assert not list(tmp_path.iterdir())


def test_duplicate_assets_are_rejected(release):
    release["assets"].append(dict(release["assets"][0]))
    with pytest.raises(ValueError, match="multiple compatible"):
        release_tools.select_wheel(release)


def test_build_skips_magenpy_staging_and_compiler_when_asset_exists(release, tmp_path, monkeypatch):
    # The build CLI needs tomllib (Python 3.11+); other resolver tests also run on 3.10.
    pytest.importorskip("tomllib")
    import sys
    tools_dir = Path(__file__).resolve().parents[1] / "tools/wasm"
    monkeypatch.syspath_prepend(str(tools_dir))
    build_spec = importlib.util.spec_from_file_location("viprs_wasm_build", tools_dir / "build.py")
    build = importlib.util.module_from_spec(build_spec)
    build_spec.loader.exec_module(build)
    monkeypatch.setattr(build, "BUILD", tmp_path / "build")
    monkeypatch.setattr(build, "WHEELS", tmp_path / "wheels")
    monkeypatch.setattr(build, "resolve_release", lambda _: release)
    monkeypatch.setattr(build, "reuse_wheel", release_tools.reuse_wheel)
    monkeypatch.setattr(release_tools, "read_url", lambda _: wheel_bytes())
    monkeypatch.setattr(build, "stage", lambda *_: pytest.fail("Released WASM must not be staged"))
    monkeypatch.setattr(build, "extract_release_source", lambda *_: pytest.fail("Released WASM must not fetch source"))
    monkeypatch.setattr(build.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Compiler must not run"))
    bundled = []
    monkeypatch.setattr(build, "bundle", lambda info: bundled.append(info))
    monkeypatch.setattr(sys, "argv", ["build.py", "--package", "magenpy"])
    monkeypatch.setattr(sys, "version_info", (3, 13))
    build.main()
    assert bundled[0]["magenpy"]["origin"] == "release-asset"
    assert (build.BUILD / "native-test-requirements.txt").read_text() == f"magenpy @ {release['source_url']}\n"
