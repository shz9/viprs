"""The browser build stays minimal and emits a SIMD-enabled VIPRS kernel."""

import importlib.util
from pathlib import Path
import sys

import pytest


pytest.importorskip("tomllib")


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "wasm"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("viprs_wasm_build", TOOLS / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_runtime_omits_optional_scientific_stacks():
    assert set(build.BUILTIN_PACKAGES) == {
        "micropip", "numpy", "pandas", "tqdm", "numcodecs", "asciitree",
        "fsspec", "httpx", "click", "packaging", "pyyaml",
        "typing-extensions",
    }
    assert {"scipy", "scikit-learn", "statsmodels", "joblib"}.isdisjoint(
        build.BUILTIN_PACKAGES
    )


def test_staged_viprs_wheel_is_inference_only_and_simd_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "BUILD", tmp_path)
    target, info = build.stage("viprs", ROOT)
    setup = (target / "setup.py").read_text()

    assert info["extensions"] == ["viprs.model.vi.e_step_cpp"]
    assert "-msimd128" in setup
    assert "math_utils.pyx" not in setup
    for dependency in ("scipy", "scikit-learn", "statsmodels", "joblib", "psutil"):
        assert dependency not in setup
    assert "VIPRSGrid" not in (target / "viprs/model/__init__.py").read_text()
