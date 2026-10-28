"""Validate and archive the exact static application and its manifest-listed wheels."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
STATIC_FILES = (
    "index.html",
    "app.css",
    "app.js",
    "worker.js",
    "inference.py",
    "sources.mjs",
    "runtime.json",
    "build-info.json",
)


def package(assets, destination):
    assets, destination = Path(assets), Path(destination)
    manifest = json.loads((assets / "runtime.json").read_text())
    provenance = json.loads((assets / "build-info.json").read_text())
    if manifest["pyodide_version"] != provenance["pyodide_version"]:
        raise ValueError(
            "Runtime and build provenance have different Pyodide versions."
        )
    files = list(STATIC_FILES)
    checksums, modules = {}, []
    for url in manifest["wheels"]:
        parsed = urlsplit(url)
        path = PurePosixPath(parsed.path)
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.query
            or len(path.parts) != 2
            or path.parts[0] != "wheels"
            or path.suffix != ".whl"
        ):
            raise ValueError(f"Expected a local wheels/*.whl path: {url}")
        if path.name in checksums:
            raise ValueError(f"Duplicate wheel: {path.name}")
        wheel_path = assets / path
        checksum = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
        if parsed.fragment != f"sha256={checksum}":
            raise ValueError(f"Wheel checksum mismatch: {path.name}")
        checksums[path.name] = checksum
        with zipfile.ZipFile(wheel_path) as wheel:
            for name in wheel.namelist():
                if name.endswith(".so"):
                    if (
                        not name.endswith("cpython-313-wasm32-emscripten.so")
                        or wheel.read(name)[:4] != b"\0asm"
                    ):
                        raise ValueError(f"Expected a WebAssembly extension: {name}")
                    modules.append(name)
        files.append(path.as_posix())
    if len(modules) != 4 or not any("e_step_cpp" in name for name in modules):
        raise ValueError("Expected the VIPRS kernel and three magenpy WebAssembly extensions.")
    if checksums != provenance["wheel_sha256"]:
        raise ValueError("Wheel checksums differ from build provenance.")
    # Read everything before creating the archive, so missing sources fail early.
    contents = {f"webapp/{name}": (assets / name).read_bytes() for name in files}
    contents["webapp/LICENSE"] = (ROOT / "LICENSE").read_bytes()
    contents["README.txt"] = (
        "VIPRS browser application\n\n"
        "Serve this directory over HTTP, e.g. python -m http.server 8000, then open\n"
        "http://localhost:8000/webapp/. Do not open index.html via file://.\n"
        "Pyodide runtime packages and the example data require internet access.\n"
        "All four required wheels are included under webapp/wheels/.\n"
        "The runtime manifest pins their SHA-256 hashes; build-info.json records\n"
        "package versions, source commit/hashes, and compiler information.\n"
        "To install in the full documentation site, copy webapp/ into docs/.\n"
        "Documentation: https://shz9.github.io/viprs/web_application/\n"
    ).encode()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "viprs-webapp-wasm.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, data)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".sha256").write_text(f"{digest}  {archive.name}\n")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "docs/webapp")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/wasm")
    args = parser.parse_args()
    print(package(args.assets, args.output))
