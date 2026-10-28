"""Run the compiled VIPRS package in Pyodide against summary statistics and LD."""

import csv
import gc
import io
import json
import math
import stat
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np


def ensure_huggingface_compat():
    """Stub terminal-only modules unavailable inside a Pyodide web worker."""
    if sys.platform != "emscripten":
        return
    # huggingface_hub imports these modules eagerly for its interactive choice
    # prompt. A web worker has no TTY, so that code is unreachable, but Pyodide
    # does not provide the POSIX terminal modules and the import would otherwise
    # prevent HfFileSystem from loading.
    for name in ("termios", "tty"):
        sys.modules.setdefault(name, types.ModuleType(name))


async def resolve_browser_gwas_catalog(accession, read_text=None):
    """Resolve a Catalog accession using the browser's HTTP transport."""
    from viprs.utils.data_utils import resolve_gwas_catalog_async

    if read_text is None:
        from pyodide.http import pyfetch

        async def read_text(url):
            try:
                response = await pyfetch(url, credentials="omit")
            except Exception as error:
                raise ConnectionError(
                    "Could not reach the GWAS Catalog FTP server. "
                    "Check the connection."
                ) from error
            if response.status == 404:
                raise FileNotFoundError(
                    "No GWAS Catalog summary statistics were found for this accession."
                )
            if not 200 <= response.status < 300:
                raise ConnectionError(
                    f"Could not list GWAS Catalog files (HTTP {response.status})."
                )
            return await response.string()

    return await resolve_gwas_catalog_async(accession, read_text)


def ensure_linear_operator():
    """Provide the tiny scipy interface magenpy needs without loading SciPy."""
    try:
        from scipy.sparse.linalg import LinearOperator  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "scipy":
            raise

        class LinearOperator:
            def __init__(self, dtype, shape):
                self.dtype = np.dtype(dtype)
                self.shape = shape

            def matvec(self, vector):
                return self._matvec(vector)

            def rmatvec(self, vector):
                return self._rmatvec(vector)

            def dot(self, vector):
                return self.matvec(vector)

            def __matmul__(self, vector):
                return self.matvec(vector)

        scipy = types.ModuleType("scipy")
        sparse = types.ModuleType("scipy.sparse")
        linalg = types.ModuleType("scipy.sparse.linalg")
        scipy.__viprs_fallback__ = True
        scipy.sparse = sparse
        sparse.linalg = linalg
        linalg.LinearOperator = LinearOperator
        sys.modules.update(
            {"scipy": scipy, "scipy.sparse": sparse, "scipy.sparse.linalg": linalg}
        )
        return "minimal LinearOperator"
    return "SciPy"


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
        or not 1 <= components <= 9
    ):
        raise ValueError("Mixture components must be an integer between 1 and 9.")
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


def _compact_sumstats_parser(parser):
    """Restrict a parser to model-relevant columns and bounded numeric dtypes.

    Catalog GWAS-SSF files may contain many optional numeric columns. Leaving
    those columns to pandas' inference defeats the canonical dtype map and can
    make its BlockManager allocate another full-width float64 matrix while it
    constructs the DataFrame. Parser-declared raw fields are retained because
    format-specific post-processing may need them.
    """
    converters = parser.col_name_converter or {}
    position_columns = {"POS"} | {
        raw for raw, output in converters.items() if output == "POS"
    }

    def compact_dtypes(dtypes):
        return {
            column: (
                np.float32
                if np.dtype(dtype) == np.dtype(np.float64)
                and column not in position_columns
                else dtype
            )
            for column, dtype in dtypes.items()
        }

    parser.column_dtypes = compact_dtypes(parser.column_dtypes)
    parser.raw_column_dtypes = compact_dtypes(parser.raw_column_dtypes)
    # Alleles have a tiny vocabulary but otherwise become millions of Python
    # string references. Categories preserve pandas compatibility while storing
    # one small integer code per row.
    for column in ("A1", "A2"):
        if column in parser.column_dtypes:
            parser.column_dtypes[column] = "category"
    for column in ("REF", "ALT", "ALT1"):
        if column in parser.raw_column_dtypes:
            parser.raw_column_dtypes[column] = "category"
    retained_columns = (
        set(parser.get_standard_cols())
        | set(parser.column_dtypes)
        | set(parser.raw_column_dtypes)
        | set(converters)
    )
    parser.read_csv_kwargs["usecols"] = retained_columns.__contains__
    return parser


class BrowserFitSession:
    """Accumulate a run while keeping at most one LD archive loaded at a time."""

    def __init__(self, sumstats_paths, options):
        self.max_iter, self.tolerance = validate_options(options)
        self.options = options
        self.sumstats_paths = (
            [sumstats_paths]
            if isinstance(sumstats_paths, (str, Path))
            else list(sumstats_paths)
        )
        self.model_name = options.get("model", "VIPRS")
        self.seed = int(options.get("seed", 0))
        self.chromosome_rows = {}
        self.chromosome_results = {}
        self.preparation_seconds = 0.0
        self.fit_seconds = 0.0
        self.linear_operator = None
        self.sumstats_tables = self._read_summary_statistics()

    def _read_summary_statistics(self):
        """Parse summary statistics once, using bounded browser-friendly dtypes."""
        from magenpy import GWADataLoader
        from magenpy.parsers.sumstats_parsers import get_sumstats_parser

        # pandas otherwise consolidates every numeric GWAS-SSF column into a
        # second block. At eight million rows, optional float64 columns can add
        # hundreds of MiB to the parsing peak. Use compact storage only for large
        # staged inputs, preserving the established float64 numerics for ordinary
        # files and small regression fixtures.
        parser = get_sumstats_parser(
            self.options.get("sumstats_format", "fastgwa")
        )(None)
        compact = (
            sum(Path(path).stat().st_size for path in self.sumstats_paths)
            >= 64 * 1024**2
        )
        if compact:
            _compact_sumstats_parser(parser)
        with tempfile.TemporaryDirectory(prefix="viprs-browser-sumstats-") as work:
            loader = GWADataLoader(temp_dir=str(Path(work) / "temp"), threads=1)
            loader.read_summary_statistics(
                [str(path) for path in self.sumstats_paths],
                parser=parser,
                drop_duplicated=True,
            )
            tables = loader.sumstats_table or {}
            loader.sumstats_table = None
            loader.cleanup()
        if not tables:
            raise ValueError("No summary statistics could be read from the supplied inputs.")
        return {str(chromosome): table for chromosome, table in tables.items()}

    def fit_ld_archive(self, ld_path, remaining_archives=0, progress_callback=None):
        """Stream or extract, fit, and release one LD archive at a time."""
        self.linear_operator = self.linear_operator or ensure_linear_operator()
        if str(ld_path).startswith("hf://"):
            ensure_huggingface_compat()
        from magenpy import GWADataLoader
        from viprs.model import VIPRS, VIPRSMix

        preparation_start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="viprs-browser-ld-") as work:
            # GWADataLoader expands hf:// globs and delegates each store to
            # LDMatrix.from_path. Local browser uploads first need safe ZIP
            # extraction because Pyodide's filesystem does not expose them as
            # native ZipStore paths.
            ld_stores = (
                str(ld_path)
                if str(ld_path).startswith("hf://")
                else unpack_ld(ld_path, Path(work) / "stores")
            )
            gdl = GWADataLoader(
                ld_store_files=ld_stores,
                temp_dir=str(Path(work) / "temp"),
                threads=1,
            )
            selected = {}
            for chromosome in gdl.ld or {}:
                key = str(chromosome)
                if key in self.sumstats_tables:
                    selected[chromosome] = self.sumstats_tables.pop(key)
            if not selected:
                available = ", ".join(sorted(self.sumstats_tables, key=str)) or "none"
                raise ValueError(
                    "The LD chromosome does not occur in the summary statistics "
                    f"(remaining summary chromosomes: {available})."
                )
            # Keep parsed summaries across sequential local archives, while
            # GWADataLoader performs the actual cross-source harmonization.
            gdl.sumstats_table = selected
            gdl.harmonize_data()
            chromosome_loaders = gdl.split_by_chromosome()
            chromosomes = sorted(chromosome_loaders)
            self.preparation_seconds += time.perf_counter() - preparation_start

            for index, chromosome in enumerate(chromosomes):
                chromosome_key = str(chromosome)
                if chromosome_key in self.chromosome_results:
                    raise ValueError(
                        f"LD inputs contain chromosome {chromosome_key} more than once."
                    )
                chromosome_gdl = chromosome_loaders[chromosome]
                model = None
                fit_start = time.perf_counter()
                try:
                    variants = int(chromosome_gdl.n_snps)
                    np.random.seed(self.seed)
                    model = (
                        VIPRSMix(
                            chromosome_gdl,
                            K=int(self.options.get("components", 3)),
                            threads=1,
                        )
                        if self.model_name == "VIPRSMix"
                        else VIPRS(chromosome_gdl, threads=1)
                    )
                    model.fit(
                        max_iter=self.max_iter,
                        x_abs_tol=self.tolerance,
                        f_abs_tol=self.tolerance,
                        disable_pbar=True,
                    )
                    if not model.optim_result.valid_optim_result:
                        raise ValueError(
                            f"{self.model_name} optimization failed for chromosome "
                            f"{chromosome}: {model.optim_result.message}"
                        )
                    self.chromosome_rows[chromosome_key] = json.loads(
                        model.to_table().to_json(
                            orient="records", double_precision=15
                        )
                    )
                    self.chromosome_results[chromosome_key] = {
                        "elbo": [float(value) for value in model.history["ELBO"]],
                        "iterations": int(model.optim_result.nit),
                        "converged": bool(model.optim_result.converged),
                        "message": str(model.optim_result.message),
                        "variants": variants,
                    }
                finally:
                    self.fit_seconds += time.perf_counter() - fit_start
                    chromosome_gdl.cleanup()
                    del model
                    gc.collect()
                if progress_callback is not None:
                    completed = len(self.chromosome_results)
                    total = completed + len(chromosomes) - index - 1 + remaining_archives
                    progress_callback(chromosome_key, completed, total)
            gdl.cleanup()
            del gdl
        gc.collect()

    def finish(self):
        """Combine chromosome results after every LD archive has been released."""
        import magenpy
        from magenpy.stats.ld import c_utils

        from viprs import __version__
        from viprs.model.vi import e_step_cpp

        if not self.chromosome_results:
            raise ValueError("No chromosomes were fitted from the supplied LD inputs.")
        chromosomes = sorted(
            self.chromosome_results,
            key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
        )
        elbo_histories = [self.chromosome_results[c]["elbo"] for c in chromosomes]
        elbo = [
            sum(history[min(i, len(history) - 1)] for history in elbo_histories)
            for i in range(max(map(len, elbo_histories)))
        ]
        messages = [
            f"chr{chromosome}: {self.chromosome_results[chromosome]['message']}"
            for chromosome in chromosomes
        ]
        rows = [row for chromosome in chromosomes for row in self.chromosome_rows[chromosome]]
        report = {
            "engine": f"{self.model_name} {__version__} (WebAssembly)",
            "model": self.model_name,
            "components": int(self.options.get("components", 3))
            if self.model_name == "VIPRSMix"
            else 1,
            "variants": len(rows),
            "chromosomes": chromosomes,
            "variants_per_chromosome": {
                c: self.chromosome_results[c]["variants"] for c in chromosomes
            },
            "iterations": sum(self.chromosome_results[c]["iterations"] for c in chromosomes),
            "converged": all(self.chromosome_results[c]["converged"] for c in chromosomes),
            "message": "; ".join(messages),
            "options": self.options,
            "seed": self.seed,
            "magenpy_version": magenpy.__version__,
            "extensions": {"viprs": e_step_cpp.__file__, "magenpy": c_utils.__file__},
            "wasm_simd": bool(getattr(e_step_cpp, "check_simd_support", lambda: False)()),
            "blas": bool(e_step_cpp.check_blas_support()),
            "linear_operator": self.linear_operator,
            "elbo": elbo,
            "timings": {
                "preparation_seconds": self.preparation_seconds,
                "fit_seconds": self.fit_seconds,
            },
            "effect_scale": "Unmodified VIPRS posterior estimates; use the appropriate genotype scaling when scoring.",
        }
        return rows, report


def native_fit(sumstats_paths, ld_paths, options, progress_callback=None):
    """Native adapter mirroring the browser's sequential LD processing."""
    ld_paths = [ld_paths] if isinstance(ld_paths, (str, Path)) else list(ld_paths)
    session = BrowserFitSession(sumstats_paths, options)
    for index, ld_path in enumerate(ld_paths):
        session.fit_ld_archive(
            ld_path,
            remaining_archives=len(ld_paths) - index - 1,
            progress_callback=progress_callback,
        )
    return session.finish()


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


def run_request(request_json, progress_callback=None):
    request = json.loads(request_json)
    rows, report = native_fit(
        request["sumstats_paths"],
        request["ld_paths"],
        request["options"],
        progress_callback=progress_callback,
    )
    return serialize_result(rows, report)


_active_session = None


def start_request(request_json):
    """Start a browser run before any LD archive is staged."""
    global _active_session
    if _active_session is not None:
        raise RuntimeError("A browser fit is already active.")
    request = json.loads(request_json)
    _active_session = BrowserFitSession(
        request["sumstats_paths"], request["options"]
    )


def process_ld_archive(ld_path, remaining_archives=0, progress_callback=None):
    """Fit one staged archive; no reference to it survives this call."""
    if _active_session is None:
        raise RuntimeError("Start the browser fit before processing LD.")
    _active_session.fit_ld_archive(
        ld_path,
        remaining_archives=int(remaining_archives),
        progress_callback=progress_callback,
    )


def finish_request():
    """Serialize the active run and release its accumulated Python state."""
    global _active_session
    if _active_session is None:
        raise RuntimeError("No browser fit is active.")
    try:
        return serialize_result(*_active_session.finish())
    finally:
        _active_session = None
        gc.collect()


def abort_request():
    """Release a partially completed run after a worker-side failure."""
    global _active_session
    _active_session = None
    gc.collect()
