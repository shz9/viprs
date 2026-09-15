import csv
import runpy
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = runpy.run_path(PROJECT_ROOT / "benchmarks/cli_benchmark.py")


def test_record_and_summarize_folds(tmp_path):
    raw_results = tmp_path / "folds.csv"
    for fold in range(1, 6):
        evaluation = tmp_path / f"fold-{fold}.eval"
        evaluation.write_text(f"Pseudo_R2\n{fold / 10}\n", encoding="utf-8")
        profile = tmp_path / f"fold-{fold}.prof"
        profile.write_text(
            f"Chromosome\tTotal_WallClockTime\n1\t{fold * 10}\n2\t{fold * 10}\n",
            encoding="utf-8",
        )
        BENCHMARK["record_fold"](
            Namespace(
                commit="a" * 40,
                date="2026-09-15",
                model="VIPRS",
                configurations="",
                phenotype="height",
                sumstats_url="https://example.com/height.tar.gz",
                ld_url="https://example.com/ld",
                fold=fold,
                evaluation_file=evaluation,
                profile_file=profile,
                output=raw_results,
            )
        )

    with raw_results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = BENCHMARK["summarize_folds"](rows)

    assert result == [
        {
            "Commit ID": "a" * 40,
            "Date": "2026-09-15",
            "Model": "VIPRS",
            "Configurations": "",
            "Phenotype": "height",
            "Accuracy": "0.300000",
            "Runtime": "30.00",
            "Sumstats": "https://example.com/height.tar.gz",
            "LD": "https://example.com/ld",
        }
    ]


def test_merge_history_replaces_rerun():
    current = [
        {
            "Commit ID": "abc",
            "Date": "2026-09-15",
            "Model": "VIPRS",
            "Configurations": "",
            "Phenotype": "height",
            "Accuracy": "0.2",
            "Runtime": "10.0",
            "Sumstats": "https://example.com/height.tar.gz",
            "LD": "https://example.com/ld",
        }
    ]
    old_copy = {**current[0], "Accuracy": "0.1", "Runtime": "20.0"}
    older_commit = {**current[0], "Commit ID": "def"}

    assert BENCHMARK["merge_history"](current, [old_copy, older_commit]) == [
        current[0],
        older_commit,
    ]
