"""Run the compiled VIPRS package in Pyodide against summary statistics and LD."""

import csv
import io
import json
import math
import stat
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np


def validate_options(options):
    iterations = options.get("max_iter", 1000)
    tolerance = float(options.get("tolerance", 1e-6))
    if (
        isinstance(iterations, bool)
        or int(iterations) != iterations
        or not 3 <= iterations <= 5000
    ):
        raise ValueError("Maximum iterations must be an integer between 3 and 5000.")
    if not math.isfinite(tolerance) or not 1e-12 <= tolerance <= 0.01:
        raise ValueError("Tolerance must be between 1e-12 and 0.01.")
    if options.get("model", "VIPRS") not in ("VIPRS", "VIPRSMix"):
        raise ValueError("Select VIPRS or VIPRSMix.")
    components = options.get("components", 3)
    if options.get("model") == "VIPRSMix" and (
        isinstance(components, bool)
        or int(components) != components
        or not 1 <= components <= 10
    ):
        raise ValueError("Mixture components must be an integer between 1 and 10.")
    seed = options.get("seed", 0)
    if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be an integer from 0 to 4294967295.")
    return int(iterations), tolerance


def unpack_ld(archive_path, destination):
    """Validate ZIP members and find actual magenpy store roots."""
    destination = Path(destination)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > 50000:
            raise ValueError("LD ZIP exceeds the 50,000 entry limit.")
        seen = set()
        for member in members:
            # ZipInfo.filename normalizes backslashes on Windows. Validate the
            # original archive spelling before any platform-specific rewriting.
            original = member.orig_filename
            path = PurePosixPath(original)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in original
                or ":" in original
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError("LD ZIP contains an unsafe path or symbolic link.")
            path = PurePosixPath(member.filename)
            if path in seen:
                raise ValueError("LD ZIP contains duplicate paths.")
            seen.add(path)
        archive.extractall(destination)
    roots = sorted(
        p.parent
        for p in destination.rglob(".zgroup")
        if (p.parent / "matrix" / ".zgroup").is_file()
    )
    if not roots:
        raise ValueError(
            "No magenpy Zarr v2 LD stores found in the ZIP (expected .zgroup and matrix/.zgroup)."
        )
    return [str(p) for p in roots]


def native_fit(sumstats_paths, ld_paths, options):
    """Prepare all chromosomes, then hand the entire optimization to one fit() call."""
    import magenpy
    from magenpy import GWADataLoader
    from magenpy.stats.ld import c_utils

    from viprs import __version__
    from viprs.model import VIPRS, VIPRSMix
    from viprs.model.vi import e_step_cpp

    max_iter, tolerance = validate_options(options)
    sumstats_paths = (
        [sumstats_paths] if isinstance(sumstats_paths, (str, Path)) else sumstats_paths
    )
    ld_paths = [ld_paths] if isinstance(ld_paths, (str, Path)) else ld_paths
    preparation_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="viprs-browser-") as work:
        roots = [
            root
            for index, path in enumerate(ld_paths)
            for root in unpack_ld(path, Path(work) / f"ld-{index}")
        ]
        # The worker assigns compression-aware suffixes while staging inputs.
        gdl = GWADataLoader(
            sumstats_files=[str(path) for path in sumstats_paths],
            sumstats_format=options.get("sumstats_format", "fastGWA"),
            ld_store_files=roots,
            temp_dir=str(Path(work) / "temp"),
            threads=1,
        )

        model_name = options.get("model", "VIPRS")
        model = (
            VIPRSMix(
                gdl,
                K=int(options.get("components", 3)),
                threads=1,
            )
            if model_name == "VIPRSMix"
            else VIPRS(gdl, threads=1)
        )
        seed = int(options.get("seed", 0))
        np.random.seed(seed)
        fit_start = time.perf_counter()
        # The model owns its EM loop. No subclass, callback, stepping, or iteration messages.
        model.fit(
            max_iter=max_iter,
            x_abs_tol=tolerance,
            f_abs_tol=tolerance,
            disable_pbar=True,
        )
        fit_seconds = time.perf_counter() - fit_start
        if not model.optim_result.valid_optim_result:
            raise ValueError(
                f"{model_name} optimization failed: {model.optim_result.message}"
            )
        table = model.to_table()
        rows = json.loads(table.to_json(orient="records", double_precision=15))
        report = {
            "engine": f"{model_name} {__version__} (WebAssembly)",
            "model": model_name,
            "components": int(options.get("components", 3))
            if model_name == "VIPRSMix"
            else 1,
            "variants": len(rows),
            "chromosomes": list(map(str, sorted(gdl.ld))),
            "variants_per_chromosome": {str(c): int(n) for c, n in gdl.shapes.items()},
            "iterations": int(model.optim_result.nit),
            "converged": bool(model.optim_result.converged),
            "message": str(model.optim_result.message),
            "options": options,
            "seed": seed,
            "magenpy_version": magenpy.__version__,
            "extensions": {"viprs": e_step_cpp.__file__, "magenpy": c_utils.__file__},
            "elbo": [float(v) for v in model.history["ELBO"]],
            "timings": {
                "preparation_seconds": fit_start - preparation_start,
                "fit_seconds": fit_seconds,
            },
            "effect_scale": "Unmodified VIPRS posterior estimates; use the appropriate genotype scaling when scoring.",
        }
        return rows, report


def serialize_result(rows, report):
    columns = [
        c
        for c in ("CHR", "SNP", "POS", "A1", "A2", "PIP", "BETA", "VAR_BETA")
        if c in rows[0]
    ]
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=columns,
        delimiter="\t",
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    ranked = sorted(rows, key=lambda row: row["PIP"], reverse=True)
    chromosome_rows = {}
    for row in ranked:
        group = chromosome_rows.setdefault(str(row["CHR"]), [])
        if len(group) < 20:
            group.append(row)
    return json.dumps(
        {
            "rows": ranked[:20],
            "chromosome_rows": chromosome_rows,
            "tsv": output.getvalue(),
            "report": report,
        },
        allow_nan=False,
    )


def run_request(request_json):
    request = json.loads(request_json)
    rows, report = native_fit(
        request["sumstats_paths"], request["ld_paths"], request["options"]
    )
    return serialize_result(rows, report)
