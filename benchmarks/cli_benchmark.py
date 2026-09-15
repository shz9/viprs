#!/usr/bin/env python3
"""Record and summarize the continuous CLI benchmark results."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


IDENTITY_FIELDS = [
    "Commit ID",
    "Model",
    "Configurations",
    "Phenotype",
]
GROUP_FIELDS = [
    "Commit ID",
    "Date",
    "Model",
    "Configurations",
    "Phenotype",
    "Sumstats",
    "LD",
]
RESULT_FIELDS = [
    "Commit ID",
    "Date",
    "Model",
    "Configurations",
    "Phenotype",
    "Accuracy",
    "Runtime",
    "Sumstats",
    "LD",
]
FOLD_FIELDS = GROUP_FIELDS + ["Fold", "Accuracy", "Runtime"]


def read_tsv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def record_fold(args):
    evaluation = read_tsv(args.evaluation_file)
    profile = read_tsv(args.profile_file)
    if len(evaluation) != 1 or "Pseudo_R2" not in evaluation[0]:
        raise ValueError("Expected one evaluation row containing Pseudo_R2.")
    if not profile or "Total_WallClockTime" not in profile[0]:
        raise ValueError("Expected profiler output containing Total_WallClockTime.")

    row = {
        "Commit ID": args.commit,
        "Date": args.date,
        "Model": args.model,
        "Configurations": args.configurations,
        "Phenotype": args.phenotype,
        "Sumstats": args.sumstats_url,
        "LD": args.ld_url,
        "Fold": str(args.fold),
        "Accuracy": str(float(evaluation[0]["Pseudo_R2"])),
        # Total_WallClockTime is repeated for each chromosome in the profiler table.
        "Runtime": str(float(profile[0]["Total_WallClockTime"])),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FOLD_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_folds(raw_rows):
    grouped = defaultdict(list)
    for row in raw_rows:
        key = tuple(row[field] for field in GROUP_FIELDS)
        grouped[key].append(row)

    results = []
    for key, rows in grouped.items():
        folds = [int(row["Fold"]) for row in rows]
        if sorted(folds) != [1, 2, 3, 4, 5]:
            raise ValueError(
                f"Expected folds 1-5 exactly once for {key}, observed {sorted(folds)}."
            )
        result = dict(zip(GROUP_FIELDS, key))
        result["Accuracy"] = f"{sum(float(r['Accuracy']) for r in rows) / 5:.6f}"
        result["Runtime"] = f"{sum(float(r['Runtime']) for r in rows) / 5:.2f}"
        results.append(result)
    return sorted(results, key=lambda row: row["Model"])


def merge_history(current, history):
    current_keys = {
        tuple(row[field] for field in IDENTITY_FIELDS) for row in current
    }
    return current + [
        row
        for row in history
        if tuple(row[field] for field in IDENTITY_FIELDS) not in current_keys
    ]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, repository):
    lines = [
        "# VIPRS continuous benchmark results",
        "",
        "Accuracy is the mean test-set `Pseudo_R2` across five HEIGHT folds. Runtime is",
        "the mean fitting `Total_WallClockTime` across those folds, in seconds. Benchmarks",
        "run on GitHub-hosted `ubuntu-24.04` runners with Python 3.12 and one inference thread.",
        "Pull requests publish their proposed table in the workflow summary and as an artifact;",
        "pushes to the default branch also update this history.",
        "",
        "| Commit ID | Date | Model | Configurations | Phenotype | Accuracy | Runtime | Sumstats | LD |",
        "|---|---|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        commit = row["Commit ID"]
        commit_cell = f"[`{commit[:7]}`](https://github.com/{repository}/commit/{commit})"
        lines.append(
            "| "
            + " | ".join(
                [
                    commit_cell,
                    row.get("Date", ""),
                    row["Model"],
                    row["Configurations"],
                    row["Phenotype"],
                    row["Accuracy"],
                    f"{row['Runtime']} s",
                    f"[HEIGHT]({row['Sumstats']})" if row.get("Sumstats") else "",
                    f"[EUR]({row['LD']})" if row.get("LD") else "",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def summarize(args):
    raw_rows = []
    for path in sorted(Path(args.input_dir).glob("*.csv")):
        raw_rows.extend(read_csv_rows(path))
    if not raw_rows:
        raise ValueError(f"No fold result CSV files found in {args.input_dir}.")

    current = summarize_folds(raw_rows)
    rows = merge_history(current, read_csv_rows(args.history_csv))
    write_csv(args.output_csv, rows)

    markdown = markdown_table(rows, args.repository)
    output_markdown = Path(args.output_markdown)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(markdown, encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser("record", help="Record one fold's outputs.")
    record.add_argument("--commit", required=True)
    record.add_argument("--date", required=True)
    record.add_argument("--model", required=True)
    record.add_argument("--configurations", default="")
    record.add_argument("--phenotype", required=True)
    record.add_argument("--sumstats-url", required=True)
    record.add_argument("--ld-url", required=True)
    record.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    record.add_argument("--evaluation-file", required=True)
    record.add_argument("--profile-file", required=True)
    record.add_argument("--output", required=True)
    record.set_defaults(func=record_fold)

    summary = commands.add_parser("summarize", help="Average folds and update history.")
    summary.add_argument("--input-dir", required=True)
    summary.add_argument("--history-csv", required=True)
    summary.add_argument("--output-csv", required=True)
    summary.add_argument("--output-markdown", required=True)
    summary.add_argument("--repository", required=True)
    summary.set_defaults(func=summarize)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
