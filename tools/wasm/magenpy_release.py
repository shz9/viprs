"""Resolve a magenpy release once and reuse a verified, ABI-compatible WASM asset."""
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
from urllib.parse import quote, urlsplit
import urllib.request
import zipfile
from email.parser import Parser

from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import Version

REPOSITORY = "shz9/magenpy"
WHEEL_TAG = "cp313-cp313-pyodide_2025_0_wasm32"
EXTENSIONS = ["magenpy.stats.ld.c_utils", "magenpy.stats.score.score_cpp",
              "magenpy.stats.variant.variant_cpp"]


def read_url(url):
    headers = {"User-Agent": "viprs-wasm-builder"}
    if urlsplit(url).netloc == "api.github.com":
        headers["Accept"] = "application/vnd.github+json"
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        return response.read()


def resolve_release(tag="latest"):
    endpoint = "latest" if tag == "latest" else "tags/" + quote(tag, safe="")
    release = json.loads(read_url(f"https://api.github.com/repos/{REPOSITORY}/releases/{endpoint}"))
    if release["draft"]:
        raise ValueError("Select a published magenpy release.")
    tag = release["tag_name"]
    version = str(Version(tag.removeprefix("v")))
    commit = json.loads(read_url(f"https://api.github.com/repos/{REPOSITORY}/commits/{quote(tag, safe='')}"))["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("GitHub returned an invalid release commit.")
    return {"tag": tag, "version": version, "commit": commit,
            "release_id": release["id"], "release_url": release["html_url"],
            "assets": release["assets"],
            "source_url": f"https://codeload.github.com/{REPOSITORY}/tar.gz/{commit}"}


def select_wheel(release):
    matches = []
    for asset in release["assets"]:
        if not asset["name"].endswith(".whl"):
            continue
        try:
            name, version, _, tags = parse_wheel_filename(asset["name"])
        except InvalidWheelFilename:
            continue
        if name == "magenpy" and version == Version(release["version"]) and WHEEL_TAG in map(str, tags):
            matches.append(asset)
    if len(matches) > 1:
        raise ValueError("The magenpy release has multiple compatible WASM wheels; cannot choose unambiguously.")
    return matches[0] if matches else None


def validate_wheel(path, version):
    with zipfile.ZipFile(path) as wheel:
        metadata_name, = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
        metadata = Parser().parsestr(wheel.read(metadata_name).decode())
        if metadata["Name"].lower() != "magenpy" or Version(metadata["Version"]) != Version(version):
            raise ValueError("The magenpy wheel metadata does not match the selected release.")
        wheel_metadata, = [n for n in wheel.namelist() if n.endswith(".dist-info/WHEEL")]
        if f"Tag: {WHEEL_TAG}" not in wheel.read(wheel_metadata).decode().splitlines():
            raise ValueError("The magenpy wheel has incompatible Python/Pyodide ABI metadata.")
        expected = {module.replace(".", "/") + ".cpython-313-wasm32-emscripten.so" for module in EXTENSIONS}
        actual = {n for n in wheel.namelist() if n.endswith(".so")}
        if actual != expected or any(wheel.read(n)[:4] != b"\0asm" for n in actual):
            raise ValueError("The magenpy wheel must contain its three actual WebAssembly extensions.")


def asset_checksum(asset, release):
    digest = asset.get("digest") or ""
    if digest.startswith("sha256:"):
        checksum = digest.removeprefix("sha256:")
    else:
        sidecar = next((a for a in release["assets"] if a["name"] == asset["name"] + ".sha256"), None)
        if sidecar is None:
            raise ValueError("The magenpy WASM asset needs a GitHub SHA-256 digest or a .whl.sha256 sidecar.")
        fields = read_url(sidecar["browser_download_url"]).decode().strip().split()
        if len(fields) != 2 or fields[1].lstrip("*") != asset["name"]:
            raise ValueError("Invalid magenpy wheel checksum sidecar.")
        checksum = fields[0]
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Invalid SHA-256 digest for the magenpy wheel.")
    return checksum


def reuse_wheel(release, wheels):
    asset = select_wheel(release)
    if asset is None:
        return None  # Only absence of a compatible asset permits the source fallback.
    checksum = asset_checksum(asset, release)
    target = Path(wheels) / asset["name"]
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
        data = read_url(asset["browser_download_url"])
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError("Downloaded magenpy WASM wheel failed SHA-256 verification.")
        temporary = target.with_suffix(".whl.part")
        temporary.write_bytes(data)
        try:
            validate_wheel(temporary, release["version"])
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    validate_wheel(target, release["version"])
    return {"version": release["version"], "wheel": target.name, "wheel_sha256": checksum,
            "extensions": EXTENSIONS, "origin": "release-asset",
            "release_tag": release["tag"], "release_url": release["release_url"],
            "source_commit": release["commit"], "asset_id": asset["id"],
            "asset_url": asset["browser_download_url"]}


def extract_release_source(release, build):
    archive = Path(build) / f"magenpy-{release['commit']}.tar.gz"
    if not archive.exists():
        data = read_url(release["source_url"])
        partial = archive.with_suffix(".gz.part")
        partial.write_bytes(data)
        partial.replace(archive)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    source = Path(build) / f"magenpy-{release['commit']}"
    if not source.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(build, filter="data")
    if not (source / "pyproject.toml").is_file():
        raise ValueError("The magenpy release archive does not contain the expected source tree.")
    return source, checksum
