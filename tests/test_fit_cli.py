import runpy
from argparse import Namespace
from pathlib import Path

import magenpy.utils.system_utils as system_utils
import pandas as pd


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


def test_write_effect_sizes_as_pgs_catalog(tmp_path):
    parameter_table = pd.DataFrame({
        "CHR": [1],
        "SNP": ["rs1"],
        "POS": [101],
        "A1": ["A"],
        "A2": ["G"],
        "BETA": [0.25],
    })
    output_prefix = tmp_path / "VIPRS_EM"

    output_file = FIT_CLI["write_effect_sizes"](
        parameter_table,
        output_prefix,
        Namespace(
            output_format="pgs_catalog",
            pgs_metadata="genome_build=GRCh37,pgs_name=VIPRS_height",
        ),
    )

    assert output_file == f"{output_prefix}.txt.gz"
    score_table = pd.read_csv(output_file, sep="\t", comment="#")
    assert score_table.columns.tolist() == [
        "rsID", "chr_name", "chr_position", "effect_allele", "other_allele",
        "effect_weight",
    ]
    assert score_table.loc[0, "effect_weight"] == 0.25
