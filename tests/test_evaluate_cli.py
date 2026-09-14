import importlib.machinery
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def load_evaluate_module():
    script = Path(__file__).parents[1] / "bin" / "viprs_evaluate"
    loader = importlib.machinery.SourceFileLoader("viprs_evaluate_cli", str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluate_cli():
    return load_evaluate_module()


def test_individual_evaluation_mode_is_backwards_compatible(evaluate_cli):
    parser = evaluate_cli.build_parser()
    args = parser.parse_args([
        "--prs-file", "scores.prs",
        "--phenotype-file", "phenotypes.txt",
        "--output-file", "results",
    ])

    assert evaluate_cli.get_evaluation_mode(args, parser) == "individual"


def test_individual_evaluation_still_runs_end_to_end(tmp_path):
    script = Path(__file__).parents[1] / "bin" / "viprs_evaluate"
    prs_file = tmp_path / "scores.prs"
    phenotype_file = tmp_path / "phenotypes.txt"
    output_file = tmp_path / "performance"

    pd.DataFrame({
        "FID": ["F1", "F2", "F3", "F4"],
        "IID": ["I1", "I2", "I3", "I4"],
        "PRS": [1.0, 2.0, 3.0, 4.0],
    }).to_csv(prs_file, sep="\t", index=False)
    pd.DataFrame([
        ["F1", "I1", 2.0],
        ["F2", "I2", 3.9],
        ["F3", "I3", 6.2],
        ["F4", "I4", 7.8],
    ]).to_csv(phenotype_file, sep="\t", index=False, header=False)

    process = subprocess.run([
        sys.executable, str(script),
        "--prs-file", str(prs_file),
        "--phenotype-file", str(phenotype_file),
        "--output-file", str(output_file),
        "--metrics", "Pearson_R", "R2",
    ], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr

    result = pd.read_csv(str(output_file) + ".eval", sep="\t").iloc[0]
    assert result["Sample size"] == 4
    assert result["Pearson_R"] > 0.99
    assert result["R2"] > 0.99


def test_summary_evaluation_requires_all_inputs(evaluate_cli, capsys):
    parser = evaluate_cli.build_parser()
    args = parser.parse_args([
        "--sumstats", "validation.sumstats",
        "--fit-files", "model.fit.gz",
        "--output-file", "results",
    ])

    with pytest.raises(SystemExit):
        evaluate_cli.get_evaluation_mode(args, parser)

    assert "--ld-dir" in capsys.readouterr().err


def test_individual_and_summary_inputs_are_mutually_exclusive(evaluate_cli, capsys):
    parser = evaluate_cli.build_parser()
    args = parser.parse_args([
        "--prs-file", "scores.prs",
        "--phenotype-file", "phenotypes.txt",
        "--sumstats", "validation.sumstats",
        "--fit-files", "model.fit.gz",
        "--ld-dir", "ld",
        "--output-file", "results",
    ])

    with pytest.raises(SystemExit):
        evaluate_cli.get_evaluation_mode(args, parser)

    assert "cannot be combined" in capsys.readouterr().err


def test_summary_evaluation_uses_pseudo_metrics(evaluate_cli, monkeypatch, tmp_path):
    import magenpy
    from viprs.eval import pseudo_metrics

    fit_file = tmp_path / "model.fit.gz"
    pd.DataFrame({
        "CHR": [1, 1],
        "SNP": ["rs1", "rs2"],
        "A1": ["A", "C"],
        "A2": ["G", "T"],
        "BETA_0": [0.1, 0.2],
        "BETA_1": [0.3, 0.4],
    }).to_csv(fit_file, sep="\t", index=False)

    class FakeSumstatsTable:
        def __init__(self):
            self.sample_size = None

        def set_sample_size(self, sample_size):
            self.sample_size = sample_size

    class FakeGWADataLoader:
        instance = None

        def __init__(self, ld_store_files, temp_dir, threads):
            self.ld_store_files = ld_store_files
            self.temp_dir = temp_dir
            self.threads = threads
            self.ld = {1: object()}
            self.sumstats_table = None
            self.harmonized = False
            self.cleaned_up = False
            FakeGWADataLoader.instance = self

        def read_summary_statistics(self, path, sumstats_format, parser):
            self.sumstats_args = (path, sumstats_format, parser)
            self.sumstats_table = {1: FakeSumstatsTable()}

        def harmonize_data(self):
            self.harmonized = True

        def cleanup(self):
            self.cleaned_up = True

    def fake_pseudo_pearson_r(gdl, effect_table):
        assert gdl.harmonized
        assert list(effect_table["SNP"]) == ["rs1", "rs2"]
        return np.array([0.2, -0.5])

    monkeypatch.setattr(magenpy, "GWADataLoader", FakeGWADataLoader)
    monkeypatch.setattr(pseudo_metrics, "pseudo_pearson_r", fake_pseudo_pearson_r)

    parser = evaluate_cli.build_parser()
    args = parser.parse_args([
        "--sumstats", "validation.sumstats",
        "--sumstats-format", "custom",
        "--custom-sumstats-mapper", "rsid=SNP,eff_allele=A1",
        "--gwas-sample-size", "10000",
        "--fit-files", str(fit_file),
        "--ld-dir", "ld/chr_*",
        "--output-file", str(tmp_path / "results"),
    ])

    result = evaluate_cli.evaluate_summary_statistics(args)

    assert result == {
        "Pseudo_Pearson_R_0": pytest.approx(0.2),
        "Pseudo_Pearson_R_1": pytest.approx(-0.5),
        "Pseudo_R2_0": pytest.approx(0.04),
        "Pseudo_R2_1": pytest.approx(0.25),
    }
    gdl = FakeGWADataLoader.instance
    assert gdl.sumstats_args[0] == "validation.sumstats"
    assert gdl.sumstats_args[1] is None
    assert gdl.sumstats_args[2] is not None
    assert gdl.sumstats_table[1].sample_size == 10000
    assert gdl.cleaned_up


def test_summary_evaluation_rejects_individual_metrics(evaluate_cli, tmp_path):
    parser = evaluate_cli.build_parser()
    args = parser.parse_args([
        "--sumstats", "validation.sumstats",
        "--fit-files", str(tmp_path / "model.fit.gz"),
        "--ld-dir", "ld",
        "--output-file", "results",
        "--metrics", "AUROC",
    ])

    with pytest.raises(ValueError, match="Unsupported summary-statistics metric"):
        evaluate_cli.evaluate_summary_statistics(args)
