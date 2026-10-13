"""Compile VIPRS and reuse the latest magenpy release's WASM wheel when available.

Run with the Python 3.13 environment described in requirements-build.txt.
All compatibility changes are applied to build/wasm copies, never to the source tree.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import urllib.request

from magenpy_release import resolve_release, reuse_wheel, extract_release_source, validate_wheel

ROOT = Path(__file__).resolve().parents[2]
PYODIDE_VERSION = "0.28.3"
PURE_WHEELS = {
    "zarr": ("2.18.7", "ac3dc4033e9ae4e9d7b5e27c97ea3eaf1003cc0a07f010bd83d5134bf8c4b223"),
    "fasteners": ("0.20", "9422c40d1e350e4259f509fb2e608d6bc43c0136f79a00db1b49046029d0b3b7"),
}
BUILTIN_PACKAGES = ["micropip", "numpy", "scipy", "pandas", "joblib", "tqdm",
                    "scikit-learn", "statsmodels", "numcodecs", "asciitree"]
BUILD = ROOT / "build" / "wasm"
WHEELS = ROOT / "docs" / "webapp" / "wheels"


def download(url, path):
    if not path.exists():
        print(f"Downloading {url}", flush=True)
        partial = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target)
        partial.replace(path)


def digest_tree(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def stage(name, source):
    target = BUILD / (name + "-source")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir()
    shutil.copytree(source / name, target / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.pyc", "*.cpp", "*.c"))
    for filename in ("LICENSE", "README.md"):
        shutil.copyfile(source / filename, target / filename)
    metadata = tomllib.loads((source / "pyproject.toml").read_text())["project"]
    version = metadata.get("version")
    if not version:
        tree = ast.parse((source / name / "_version.py").read_text())
        version = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets))
    dependencies = [dep for dep in metadata["dependencies"] if not dep.startswith("psutil")]
    if name == "viprs":
        # POSIX shared memory is only needed by multiprocess grid search, not VIPRS.fit.
        path = target / name / "model/gridsearch/HyperparameterSearch.py"
        text = path.read_text().replace("from multiprocessing import shared_memory\n", "")
        text = text.replace("    if shm_data is not None:\n        model.ld_data = {}", "    if shm_data is not None:\n        from multiprocessing import shared_memory\n\n        model.ld_data = {}", 1)
        text = text.replace("        if self.n_jobs > 1:\n", "        if self.n_jobs > 1:\n            from multiprocessing import shared_memory\n", 1)
        path.write_text(text)
    extensions = []
    for pyx in sorted((target / name).rglob("*.pyx")):
        relative = pyx.relative_to(target)
        extensions.append((str(relative.with_suffix("")).replace("/", "."), str(relative)))
    setup = f"""from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
extensions = [Extension(module, [path], language='c++', include_dirs=[np.get_include()],
                       define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                       extra_compile_args=['-O3', '-std=c++17']) for module, path in {extensions!r}]
setup(name={name!r}, version={version!r}, packages=find_packages(),
      description='WebAssembly build of {name} for summary-statistics inference',
      license_files=['LICENSE'], install_requires={dependencies!r},
      package_data={{{name!r}: ['config/*.ini', 'data/*.txt', 'data/*.gz', '**/*.pxd', '**/*.hpp']}},
      ext_modules=cythonize(extensions, compiler_directives={{'language_level': 3, 'boundscheck': False,
          'wraparound': False, 'nonecheck': False, 'cdivision': True}}), zip_safe=False)
"""
    (target / "setup.py").write_text(setup)
    (target / "pyproject.toml").write_text('[build-system]\nrequires = ["setuptools==80.9.0", "wheel==0.45.1", "Cython==3.1.2", "numpy==2.2.5"]\nbuild-backend = "setuptools.build_meta"\n')
    return target, {"version": version, "source_sha256": digest_tree(target / name), "extensions": [x[0] for x in extensions]}


def bundle(provenance):
    wheels = []
    for name in ("magenpy", "viprs"):
        version = provenance[name]["version"]
        path = WHEELS / provenance[name].get("wheel", f"{name}-{version}-cp313-cp313-pyodide_2025_0_wasm32.whl")
        if not path.exists():
            raise FileNotFoundError(f"Build the {name} WASM wheel before bundling: {path}")
        wheels.append(path)
    for name, (version, checksum) in PURE_WHEELS.items():
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as response:
            metadata = json.load(response)
        wheel = next(w for w in metadata["urls"] if w["filename"].endswith("none-any.whl"))
        target = WHEELS / wheel["filename"]
        download(wheel["url"], target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Checksum mismatch for {target}")
        wheels.append(target)
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(wheels)}
    manifest = {
        "pyodide_version": PYODIDE_VERSION,
        "builtin_packages": BUILTIN_PACKAGES,
        "wheels": [f"./wheels/{name}#sha256={digest}" for name, digest in artifacts.items()],
    }
    (WHEELS.parent / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    provenance["wheel_sha256"] = artifacts
    try:
        provenance["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        provenance["source_commit"] = None
    provenance["build_tools"] = (Path(__file__).parent / "requirements-build.txt").read_text().splitlines()
    (WHEELS.parent / "build-info.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Bundled real WASM extensions and pinned storage dependencies in docs/webapp/wheels.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--bundle-only", action="store_true")
    parser.add_argument("--package", choices=["magenpy", "viprs"])
    parser.add_argument("--magenpy-release", default="latest", help="GitHub release tag, or latest (default).")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("Build with Python 3.13 to match Pyodide 0.28.3.")
    BUILD.mkdir(parents=True, exist_ok=True)
    WHEELS.mkdir(parents=True, exist_ok=True)
    provenance_path = BUILD / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    if args.bundle_only:
        bundle(provenance)
        return
    provenance.update({"pyodide_version": PYODIDE_VERSION,
                       "compiler": "Emscripten 4.0.9; single-threaded; C++ kernels without BLAS/OpenMP"})
    if args.package != "viprs":
        release = resolve_release(args.magenpy_release)
        (BUILD / "magenpy-release.json").write_text(json.dumps(release, indent=2) + "\n")
        # Native regression tests use the same released source, even when WASM is reused.
        (BUILD / "native-test-requirements.txt").write_text(f"magenpy @ {release['source_url']}\n")
        print(f"Resolved magenpy {release['tag']} at {release['commit']}", flush=True)
        info = None if args.prepare_only else reuse_wheel(release, WHEELS)
        provenance.pop("magenpy_archive_sha256", None)
        provenance["magenpy_commit"] = release["commit"]
        if info is not None:
            print(f"Reusing released WASM wheel: {info['asset_url']}", flush=True)
        else:
            print("Staging the selected magenpy release source (no compatible asset, or prepare-only).", flush=True)
            upstream, checksum = extract_release_source(release, BUILD)
            target, info = stage("magenpy", upstream)
            from packaging.version import Version
            if Version(info["version"]) != Version(release["version"]):
                raise ValueError("magenpy source version does not match its release tag.")
            info.update({"origin": "source-build", "release_tag": release["tag"],
                         "release_url": release["release_url"], "source_commit": release["commit"],
                         "source_archive_sha256": checksum})
            if not args.prepare_only:
                subprocess.run(["pyodide", "build", str(target), "--outdir", str(WHEELS)], check=True)
                wheel = WHEELS / f"magenpy-{info['version']}-cp313-cp313-pyodide_2025_0_wasm32.whl"
                validate_wheel(wheel, info["version"])
                info.update({"wheel": wheel.name, "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()})
        if not args.prepare_only:
            provenance["magenpy"] = info
            provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    if args.package != "magenpy":
        target, info = stage("viprs", ROOT)
        print(f"Staged viprs {info['version']}: {info['extensions']}", flush=True)
        if not args.prepare_only:
            subprocess.run(["pyodide", "build", str(target), "--outdir", str(WHEELS)], check=True)
            provenance["viprs"] = info
            provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    if not args.prepare_only:
        bundle(provenance)


if __name__ == "__main__":
    main()
