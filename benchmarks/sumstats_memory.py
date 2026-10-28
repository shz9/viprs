"""Generate and measure large GWAS-SSF parsing workloads.

Each ``read`` invocation should run in a fresh process so its peak RSS is
attributable to one parsing strategy. The generated file deliberately includes
ten optional float columns, matching the allocation shape seen in the browser.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import importlib.util
import json
from pathlib import Path
import resource
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil


SSF_COLUMNS = (
    "chromosome",
    "base_pair_location",
    "rsid",
    "effect_allele",
    "other_allele",
    "beta",
    "standard_error",
    "effect_allele_frequency",
    "p_value",
    "n",
)
EXTRA_COLUMNS = tuple(f"unused_{index}" for index in range(10))


class MemorySampler:
    def __init__(self, interval=0.01):
        self.process = psutil.Process()
        self.interval = interval
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self._stop.set()
        self._thread.join()


def generate(path: Path, rows: int, chunk_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    alleles = np.asarray(["A", "C", "G", "T"])
    started = time.perf_counter()
    with gzip.open(path, "wt", compresslevel=1, newline="") as output:
        for start in range(0, rows, chunk_rows):
            count = min(chunk_rows, rows - start)
            positions = np.arange(start + 1, start + count + 1, dtype=np.int64)
            table = {
                "chromosome": np.full(count, 22, dtype=np.int8),
                "base_pair_location": positions,
                "rsid": np.asarray([f"rs{value}" for value in positions], dtype=object),
                "effect_allele": alleles[rng.integers(0, 4, count)],
                "other_allele": alleles[rng.integers(0, 4, count)],
                "beta": rng.normal(0.0, 0.05, count),
                "standard_error": rng.uniform(0.005, 0.1, count),
                "effect_allele_frequency": rng.uniform(0.01, 0.99, count),
                "p_value": rng.uniform(0.0, 1.0, count),
                "n": rng.integers(10_000, 500_001, count),
            }
            table.update(
                {column: rng.normal(size=count) for column in EXTRA_COLUMNS}
            )
            pd.DataFrame(table).to_csv(
                output,
                sep="\t",
                index=False,
                header=start == 0,
                float_format="%.7g",
            )
            output.flush()
            completed = start + count
            print(
                f"generated {completed:,}/{rows:,} rows",
                file=sys.stderr,
                flush=True,
            )
    print(
        json.dumps(
            {
                "path": str(path),
                "rows": rows,
                "compressed_mib": path.stat().st_size / 1024**2,
                "seconds": time.perf_counter() - started,
            }
        )
    )


def pandas_options(strategy: str):
    if strategy == "browser":
        return {}
    if strategy == "full":
        return {}
    usecols = list(SSF_COLUMNS)
    if strategy == "projected":
        return {"usecols": usecols}
    if strategy == "app":
        dtype = {
            "base_pair_location": np.float64,
            "rsid": str,
            "effect_allele": "category",
            "other_allele": "category",
            **{column: np.float32 for column in SSF_COLUMNS[5:]},
        }
        return {"usecols": usecols, "dtype": dtype}
    if strategy in {"compact", "chunked", "filtered"}:
        dtype = {
            "chromosome": np.int8,
            "base_pair_location": np.int32,
            "rsid": str,
            "effect_allele": "category",
            "other_allele": "category",
            **{column: np.float32 for column in SSF_COLUMNS[5:]},
        }
        return {"usecols": usecols, "dtype": dtype}
    raise ValueError(f"Unknown strategy: {strategy}")


def read(path: Path, strategy: str, chunk_rows: int):
    options = pandas_options(strategy)
    sampler = MemorySampler()
    started = time.perf_counter()
    retained = None
    rows = 0
    with sampler:
        if strategy == "browser":
            inference_path = (
                Path(__file__).resolve().parents[1] / "docs/webapp/inference.py"
            )
            spec = importlib.util.spec_from_file_location(
                "viprs_browser_benchmark", inference_path
            )
            browser = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(browser)
            session = browser.BrowserFitSession.__new__(browser.BrowserFitSession)
            session.sumstats_paths = [path]
            session.options = {"sumstats_format": "gwascatalog"}
            tables = session._read_summary_statistics()
            if len(tables) != 1:
                raise RuntimeError(f"Expected one chromosome, found {list(tables)}")
            retained = next(iter(tables.values())).table
            rows = len(retained)
            retained_rows = rows
            dataframe_bytes = int(retained.memory_usage(deep=True).sum())
        elif strategy in {"chunked", "filtered"}:
            chunks = []
            for chunk in pd.read_csv(
                path, sep="\t", chunksize=chunk_rows, **options
            ):
                rows += len(chunk)
                if strategy == "filtered":
                    # Retain one in eight rows, approximating an eight-million-
                    # variant source filtered to a one-million-variant LD panel.
                    chunk = chunk.loc[(chunk.index.to_numpy() % 8) == 0]
                    chunks.append(chunk)
            if strategy == "filtered":
                retained = pd.concat(chunks, ignore_index=True)
                retained_rows = len(retained)
                dataframe_bytes = int(retained.memory_usage(deep=True).sum())
            else:
                retained_rows = 0
                dataframe_bytes = 0
        else:
            retained = pd.read_csv(path, sep="\t", **options)
            if strategy == "app":
                retained["base_pair_location"] = retained[
                    "base_pair_location"
                ].astype(np.int32)
            rows = len(retained)
            retained_rows = rows
            dataframe_bytes = int(retained.memory_usage(deep=True).sum())
    elapsed = time.perf_counter() - started
    result = {
        "strategy": strategy,
        "rows_read": rows,
        "rows_retained": retained_rows,
        "seconds": elapsed,
        "baseline_mib": sampler.baseline / 1024**2,
        "peak_rss_mib": sampler.peak / 1024**2,
        "incremental_peak_mib": (sampler.peak - sampler.baseline) / 1024**2,
        "dataframe_mib": dataframe_bytes / 1024**2,
        "resource_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "dtypes": {} if retained is None else {
            column: str(dtype) for column, dtype in retained.dtypes.items()
        },
    }
    print(json.dumps(result))
    del retained
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("path", type=Path)
    generate_parser.add_argument("--rows", type=int, default=8_000_000)
    generate_parser.add_argument("--chunk-rows", type=int, default=100_000)
    generate_parser.add_argument("--seed", type=int, default=20261028)
    read_parser = subparsers.add_parser("read")
    read_parser.add_argument("path", type=Path)
    read_parser.add_argument(
        "--strategy",
        choices=(
            "full",
            "projected",
            "browser",
            "app",
            "compact",
            "chunked",
            "filtered",
        ),
        required=True,
    )
    read_parser.add_argument("--chunk-rows", type=int, default=250_000)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.path, args.rows, args.chunk_rows, args.seed)
    else:
        read(args.path, args.strategy, args.chunk_rows)


if __name__ == "__main__":
    main()
