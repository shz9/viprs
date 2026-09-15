import runpy
from argparse import Namespace
from pathlib import Path

import magenpy.utils.system_utils as system_utils


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIT_CLI = runpy.run_path(PROJECT_ROOT / "bin/viprs_fit")


def test_check_args_accepts_hugging_face_ld_path(monkeypatch):
    hf_calls = []
    monkeypatch.setattr(
        system_utils,
        "glob_hf_path",
        lambda path: hf_calls.append(path) or ["hf://datasets/example/chr_1.zip"],
    )
    monkeypatch.setattr(system_utils, "get_filenames", lambda path, extension=None: [path])
    monkeypatch.setattr(system_utils, "is_path_writable", lambda path: True)

    FIT_CLI["check_args"](
        Namespace(
            ld_dir="hf://datasets/example/chr_*.zip",
            sumstats_path="sumstats",
            output_dir="output",
            temp_dir="temp",
            n_components=1,
            fix_sigma_epsilon=None,
            lambda_min=None,
            hyp_search="EM",
        )
    )

    assert hf_calls == ["hf://datasets/example/chr_*.zip"]
