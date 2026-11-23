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


def _report_stage(stage):
    """Report precise failure context when hosted in the browser worker."""
    callback = globals().get("stage_callback")
    if callback is not None:
        callback(stage)


def _report_progress(stage, state="active", **fields):
    callback = globals().get("fit_progress_callback")
    if callback is not None:
        callback(json.dumps({"stage": stage, "state": state, **fields}))


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


async def resolve_browser_gwas_catalog(accession, read_text=None, genome_build="GRCh37", progress_callback=None, *, request_timeout=15):
    """Resolve a Catalog accession, bounding each listing/metadata request."""
    import asyncio
    import re
    from magenpy.utils.gwas_catalog import resolve_gwas_catalog

    if read_text is None:
        from js import AbortController
        from pyodide.http import pyfetch

        async def read_text(url):
            controller = AbortController.new()
            try:
                try:
                    response = await pyfetch(url, credentials="omit", signal=controller.signal)
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
                        f"Could not read GWAS Catalog files (HTTP {response.status})."
                    )
                return await response.string()
            finally:
                # Also cancel browser I/O when wait_for cancels this coroutine,
                # including a response whose headers arrived but body stalled.
                controller.abort()

    transport = read_text
    available_builds = set()
    metadata_failures = []

    def unavailable():
        if metadata_failures:
            return ConnectionError(
                f"Could not verify {genome_build} summary statistics for {accession}: "
                "GWAS Catalog metadata could not be read. Retry the lookup or provide "
                "a summary-statistics URL with a known genome build. "
                f"Last failure: {metadata_failures[-1]}"
            )
        aliases = "GRCh37/hg19" if genome_build == "GRCh37" else "GRCh38/hg38"
        message = f"This accession has no {aliases} summary-statistics file. "
        if available_builds:
            builds = ", ".join(sorted(available_builds))
            message += f"Available genome build: {builds}. Change the LD matrix to a compatible {builds} panel, or choose another study. "
        else:
            message += "Choose another study or an LD matrix matching the summary-statistics genome build. "
        return ValueError(message + "No automatic liftover is performed.")

    async def read_text(url):
        if progress_callback is not None:
            progress_callback("request", url)
        try:
            try:
                text = await asyncio.wait_for(transport(url), timeout=request_timeout)
            except asyncio.TimeoutError as error:
                raise TimeoutError(
                    f"GWAS Catalog request timed out after {request_timeout:g}s: {url}"
                ) from error
        except BaseException as error:
            if isinstance(error, Exception) and url.endswith("-meta.yaml"):
                metadata_failures.append(str(error))
            if progress_callback is not None:
                progress_callback("failed", url)
            raise
        if progress_callback is not None:
            progress_callback("received", url)
        if url.endswith("-meta.yaml"):
            match = re.search(r"(?:^|\n)\s*genome_assembly\s*:\s*[\"']?([^\s#\"']+)", text, re.I)
            if match:
                build = normalize_genome_build(match.group(1))
                if build in ("GRCh37", "GRCh38"):
                    available_builds.add(build)
        return text

    from magenpy.utils.model_utils import normalize_genome_build

    genome_build = normalize_genome_build(genome_build)
    if genome_build not in ("GRCh37", "GRCh38"):
        raise ValueError("Select or detect GRCh37/hg19 or GRCh38/hg38 before GWAS Catalog lookup.")
    if genome_build == "GRCh38":
        # magenpy 0.2.5 only resolves GRCh37. Reuse its directory/link parsing
        # for the missing GRCh38 path; retain its resolver below for GRCh37.
        from urllib.parse import unquote, urlsplit
        from magenpy.utils.gwas_catalog import gwas_catalog_directory, _listing_links

        directory = gwas_catalog_directory(accession)
        links = _listing_links(await read_text(directory), directory)
        if directory + "harmonised/" in links:
            links += _listing_links(await read_text(directory + "harmonised/"), directory + "harmonised/")
        candidates = sorted(
            (url for url in links if re.search(r"\.tsv(?:\.gz)?$", urlsplit(url).path, re.I)),
            key=lambda url: (not url.endswith(".h.tsv.gz"), not url.endswith(".f.tsv.gz"), url),
        )
        for url in candidates:
            match = re.search(r"(?:grch|build(?:grch)?|hg)(37|38|19)(?![0-9])", unquote(urlsplit(url).path), re.I)
            if match:
                if normalize_genome_build(match.group(1)) == genome_build:
                    return url
                continue
            if url + "-meta.yaml" not in links:
                continue
            try:
                metadata = await read_text(url + "-meta.yaml")
            except Exception:
                continue
            match = re.search(r"(?:^|\n)\s*genome_assembly\s*:\s*[\"']?([^\s#\"']+)", metadata, re.I)
            if match and normalize_genome_build(match.group(1)) == genome_build:
                return url
        raise unavailable()

    # magenpy's resolver is synchronous. Replay it against fetched documents
    # so it owns discovery/build selection while browser I/O stays asynchronous.
    # BaseException lets cache misses escape its optional-metadata error handler.
    class FetchRequired(BaseException):
        pass

    documents = {}

    def cached_text(url):
        if url not in documents:
            raise FetchRequired(url)
        value = documents[url]
        if isinstance(value, Exception):
            raise value
        return value

    while True:
        try:
            return resolve_gwas_catalog(accession, cached_text)
        except ValueError as error:
            # Preserve accession validation errors before any I/O.
            if not documents:
                raise
            raise unavailable() from error
        except FetchRequired as pending:
            url = pending.args[0]
            try:
                documents[url] = await read_text(url)
            except Exception as error:
                documents[url] = error


def inspect_browser_ld_build(path, expected_build=None):
    """Read LD metadata before downloading build-specific Catalog statistics."""
    from magenpy import LDMatrix
    from magenpy.utils.model_utils import normalize_genome_build, validate_harmonization_builds

    expected_build = normalize_genome_build(expected_build)
    if str(path).startswith(("hf://", "http://", "https://")):
        if str(path).startswith("hf://"):
            ensure_huggingface_compat()
            if any(char in str(path) for char in "*?["):
                from magenpy.utils.system_utils import glob_hf_path
                paths = glob_hf_path(path)
                if not paths:
                    raise ValueError("No LD matrices match the selected panel.")
                path = paths[0]
        ld = LDMatrix.from_path(path)
        try:
            builds = [normalize_genome_build(ld.genome_build)]
        finally:
            ld.release()
            close = getattr(ld.store, "close", None)
            if close is not None:
                close()
    else:
        import zarr
        # Inspect nested or multi-chromosome ZIPs without extracting LD chunks.
        with zarr.ZipStore(str(path), mode="r") as store:
            names = set(store.keys())
            roots = [name[:-len(".zgroup")] for name in names
                     if name.endswith(".zgroup") and name[:-len(".zgroup")] + "matrix/.zgroup" in names]
            matrices = {root: LDMatrix(zarr.open_group(store=store, path=root.rstrip("/"), mode="r")) for root in roots}
            if not matrices:
                raise ValueError("No magenpy LD stores found in the selected ZIP.")
            validate_harmonization_builds({"LD": matrices})
            builds = [normalize_genome_build(ld.genome_build) for ld in matrices.values()]
    known = {build for build in builds if build is not None}
    if expected_build is not None and known and known != {expected_build}:
        raise ValueError(f"Selected genome build {expected_build} conflicts with LD metadata: {sorted(known)}.")
    build = next(iter(known), expected_build)
    if build not in ("GRCh37", "GRCh38"):
        raise ValueError("LD metadata does not specify GRCh37 or GRCh38. Select the genome build for your custom LD input before GWAS Catalog lookup.")
    return build


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


def _ensure_ld_harmonization_schema(sumstats, ld, chromosome):
    """Give magenpy an SNP-keyed table, inferring optional rsIDs from LD."""
    table = sumstats.table
    if "SNP" in table:
        missing_snp = table["SNP"].isna() | table["SNP"].isin(
            ("", "#NA", "NA", "N/A")
        )
    else:
        missing_snp = None

    if (missing_snp is None or missing_snp.any()) and "POS" not in table:
        raise ValueError(
            f"Summary statistics for chromosome {chromosome} contain neither "
            "magenpy's SNP column nor a POS column that can be matched to LD."
        )

    if missing_snp is None or missing_snp.any():
        reference = ld.to_snp_table(col_subset=("POS", "SNP"))
        reference = reference.dropna(subset=("POS", "SNP")).drop_duplicates(
            subset="POS", keep=False
        )
        # magenpy owns position-based ID inference. Preserve supplied IDs when
        # only part of the input needs inference, and exclude ambiguous positions.
        original = table["SNP"].copy() if missing_snp is not None else None
        reference["CHR"] = chromosome
        sumstats.infer_snp_id(reference, allow_na=True)
        if original is not None:
            table["SNP"] = np.where(missing_snp, table["SNP"], original)

    valid_snp = table["SNP"].notna() & ~table["SNP"].isin(
        ("", "#NA", "NA", "N/A")
    )
    if not valid_snp.all():
        table = table.loc[valid_snp]
    from magenpy.parsers.sumstats_parsers import SumstatsParser

    sumstats.table = SumstatsParser.drop_na_from_essential_cols(table)
    sumstats.drop_duplicates()

    if len(sumstats.table) == 0:
        raise ValueError(
            f"No chromosome {chromosome} summary-statistics positions matched "
            "the selected LD reference. Check that both inputs use the same genome build."
        )


class BrowserFitSession:
    """Accumulate a run while keeping at most one LD archive loaded at a time."""

    def __init__(self, sumstats_paths, options):
        _report_stage("validation")
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
        _report_stage("sumstats")
        self.sumstats_tables = self._read_summary_statistics()

    def _read_summary_statistics(self):
        """Parse summary statistics once using magenpy's optimized parser."""
        from magenpy import SumstatsTable
        from magenpy.utils.chromosome_utils import get_present_chromosomes
        from magenpy.parsers.sumstats_parsers import get_sumstats_parser
        import pandas as pd

        # Project wide Catalog files with magenpy's column-selection API.
        # Specialized formats retain their derived fields (e.g. LDAK Z/SAIGE N).
        parser_type = get_sumstats_parser(self.options.get("sumstats_format", "fastgwa"))
        tables = {}
        for path in self.sumstats_paths:
            parser = parser_type(None)
            columns = None
            if parser.format_name in ("magenpy", "ssf"):
                header = pd.read_csv(str(path), nrows=0, **parser.read_csv_kwargs).columns
                converter = parser.col_name_converter or {}
                available = {converter.get(column, column) for column in header}
                columns = sorted(available.intersection(parser.column_dtypes))
            # The parser exposes drop_na; SumstatsTable.from_file does not.
            # LD arrives later, so retain missing IDs until they can be inferred.
            parsed = parser.parse(str(path), col_subset=columns, drop_na=False)
            if "CHR" not in parsed.columns:
                raise ValueError(
                    f"The selected {self.options.get('sumstats_format', 'fastgwa')} format "
                    "did not identify a chromosome column. Check the format and file headers."
                )
            table = SumstatsTable(parsed, genome_build=self.options.get("sumstats_genome_build"))
            chromosomes = get_present_chromosomes(table.table["CHR"], dropna=False)
            if len(chromosomes) == 1 and pd.notna(chromosomes[0]):
                tables[chromosomes[0]] = table
            else:
                tables.update(table.split_by_chromosome())
        if not tables:
            raise ValueError("No summary statistics could be read from the supplied inputs.")
        return tables

    def fit_ld_archive(self, ld_path, remaining_archives=0, progress_callback=None):
        """Stream or extract, fit, and release one LD archive at a time."""
        _report_stage("ld")
        _report_progress("ld", detail="Reading LD reference metadata.")
        self.linear_operator = self.linear_operator or ensure_linear_operator()
        if str(ld_path).startswith("hf://"):
            ensure_huggingface_compat()
        from magenpy import GWADataLoader
        from viprs.model import VIPRS, VIPRSMix

        preparation_start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="viprs-browser-ld-") as work:
            # GWADataLoader expands hf:// globs and delegates each store to
            # LDMatrix.from_path. Extract local uploads to support archives
            # containing nested or multiple stores; magenpy's local ZipStore
            # reader requires a single store at the archive root.
            ld_stores = (
                str(ld_path)
                if str(ld_path).startswith("hf://")
                else unpack_ld(ld_path, Path(work) / "stores")
            )
            gdl = GWADataLoader(
                ld_store_files=ld_stores,
                genome_build=self.options.get("ld_genome_build"),
                temp_dir=str(Path(work) / "temp"),
                threads=1,
            )
            from magenpy.utils.model_utils import normalize_genome_build, validate_harmonization_builds

            validate_harmonization_builds({"LD": gdl.ld or {}})
            known_builds = {normalize_genome_build(ld.genome_build) for ld in (gdl.ld or {}).values()} - {None}
            if known_builds:
                self.options["ld_genome_build"] = next(iter(known_builds))
            _report_stage("harmonization")
            _report_progress("harmonization", detail="Matching chromosomes, variant IDs and alleles.")
            selected = {}
            for chromosome in gdl.ld or {}:
                if chromosome in self.sumstats_tables:
                    selected[chromosome] = self.sumstats_tables.pop(chromosome)
            if not selected:
                available = ", ".join(map(str, sorted(self.sumstats_tables, key=str))) or "none"
                raise ValueError(
                    "The LD chromosome does not occur in the summary statistics "
                    f"(remaining summary chromosomes: {available})."
                )
            # Keep parsed summaries across sequential local archives, while
            # GWADataLoader performs the actual cross-source harmonization.
            for chromosome, sumstats in selected.items():
                _ensure_ld_harmonization_schema(
                    sumstats, gdl.ld[chromosome], chromosome
                )
            gdl.sumstats_table = selected
            gdl.harmonize_data()
            chromosome_loaders = gdl.split_by_chromosome()
            from magenpy.utils.chromosome_utils import chromosome_sort_key

            chromosomes = sorted(chromosome_loaders, key=chromosome_sort_key)
            self.preparation_seconds += time.perf_counter() - preparation_start
            completed_before = len(self.chromosome_results)
            total_chromosomes = completed_before + len(chromosomes) if remaining_archives == 0 else None
            _report_progress("harmonization", "done", completed=completed_before + len(chromosomes),
                             total=total_chromosomes, archive_fraction=1,
                             detail=f"Matched {len(chromosomes)} chromosome(s) in this file.")

            for index, chromosome in enumerate(chromosomes):
                chromosome_key = str(chromosome)
                if chromosome_key in self.chromosome_results:
                    raise ValueError(
                        f"LD inputs contain chromosome {chromosome_key} more than once."
                    )
                chromosome_gdl = chromosome_loaders[chromosome]
                model = None
                _report_stage("ld")
                fit_start = time.perf_counter()
                try:
                    variants = int(chromosome_gdl.n_snps)
                    completed = len(self.chromosome_results)
                    progress_fields = {"chromosome": chromosome_key, "completed": completed,
                                       "total": total_chromosomes, "model": self.model_name,
                                       "variants": variants, "archive_fraction": index / len(chromosomes)}
                    _report_progress("ld", **progress_fields,
                                     detail=f"Loading chromosome {chromosome_key} LD matrix into memory.")
                    last_iteration_update = -math.inf

                    def report_iteration(fitted_model):
                        nonlocal last_iteration_update
                        iteration = max(0, len(fitted_model.history["ELBO"]) - 1)
                        now = time.perf_counter()
                        if iteration == 0 or now - last_iteration_update >= 0.5:
                            last_iteration_update = now
                            _report_progress("fitting", **progress_fields, iteration=iteration,
                                             max_iter=self.max_iter, quiet=True)
                        return None

                    np.random.seed(self.seed)
                    model = (
                        VIPRSMix(
                            chromosome_gdl,
                            K=int(self.options.get("components", 3)),
                            tracked_params=[report_iteration],
                            threads=1,
                        )
                        if self.model_name == "VIPRSMix"
                        else VIPRS(chromosome_gdl, threads=1, tracked_params=[report_iteration])
                    )
                    _report_progress("ld", "done", **{**progress_fields, "completed": completed + 1,
                                     "archive_fraction": (index + 1) / len(chromosomes)},
                                     detail=f"Chromosome {chromosome_key} LD loaded.")
                    _report_stage("inference")
                    _report_progress("fitting", **progress_fields, iteration=0, max_iter=self.max_iter)
                    model.fit(
                        max_iter=self.max_iter,
                        x_abs_tol=self.tolerance,
                        f_abs_tol=self.tolerance,
                        disable_pbar=True,
                    )
                    chromosome_fit_seconds = time.perf_counter() - fit_start
                    if not model.optim_result.valid_optim_result:
                        raise ValueError(
                            f"{self.model_name} optimization failed for chromosome "
                            f"{chromosome}: {model.optim_result.message}"
                        )
                    _report_stage("results")
                    table = model.to_table()
                    # These sources have already been harmonized to the fitted LD order.
                    maf = chromosome_gdl.ld[chromosome].maf
                    if maf is not None:
                        table["MAF"] = np.minimum(maf, 1.0 - maf)
                    summaries = chromosome_gdl.sumstats_table[chromosome]
                    try:
                        table["GWAS_Z"] = summaries.z_score
                    except KeyError:
                        pass
                    if "P" in summaries.table:
                        table["GWAS_P"] = summaries.table["P"].to_numpy()
                    self.chromosome_rows[chromosome_key] = json.loads(
                        table.to_json(orient="records", double_precision=15)
                    )
                    del table, summaries, maf
                    memory_callback = globals().get("memory_callback")
                    self.chromosome_results[chromosome_key] = {
                        "elbo": [float(value) for value in model.history["ELBO"]],
                        "iterations": int(model.optim_result.nit),
                        "converged": bool(model.optim_result.converged),
                        "message": str(model.optim_result.message),
                        "variants": variants,
                        "fit_seconds": chromosome_fit_seconds,
                        "wasm_heap_bytes_at_fit_end": int(memory_callback()) if memory_callback else None,
                    }
                finally:
                    self.fit_seconds += time.perf_counter() - fit_start
                    chromosome_gdl.cleanup()
                    del model
                    gc.collect()
                completed = len(self.chromosome_results)
                _report_progress("fitting", "done", **{**progress_fields, "completed": completed,
                                 "archive_fraction": (index + 1) / len(chromosomes)},
                                 iteration=int(self.chromosome_results[chromosome_key]["iterations"]),
                                 max_iter=self.max_iter)
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
        from magenpy.utils.chromosome_utils import chromosome_sort_key, normalize_chromosome

        from viprs import __version__
        from viprs.model.vi import e_step_cpp

        if not self.chromosome_results:
            raise ValueError("No chromosomes were fitted from the supplied LD inputs.")
        chromosomes = sorted(
            self.chromosome_results,
            key=lambda value: chromosome_sort_key(normalize_chromosome(value)),
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
            "genome_build": self.options.get("ld_genome_build"),
            "extensions": {"viprs": e_step_cpp.__file__, "magenpy": c_utils.__file__},
            "wasm_simd": bool(getattr(e_step_cpp, "check_simd_support", lambda: False)()),
            "blas": bool(e_step_cpp.check_blas_support()),
            "linear_operator": self.linear_operator,
            "elbo": elbo,
            "chromosome_diagnostics": self.chromosome_results,
            "timings": {
                "preparation_seconds": self.preparation_seconds,
                "fit_seconds": self.fit_seconds,
            },
            "effect_scale": "BETA is on standardized genotypes. For dosage scoring divide BETA by sqrt(2*MAF*(1-MAF)) using the fitted LD reference frequencies. Scores remain on a standardized phenotype scale; no intercept is fitted.",
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
        for c in ("CHR", "SNP", "POS", "A1", "A2", "PIP", "BETA", "VAR_BETA", "MAF", "GWAS_Z", "GWAS_P")
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
    # Bound browser plot payloads while preserving the strongest posterior and
    # marginal signals on every chromosome. Complete estimates stay in the TSV.
    limit = 30000
    if len(rows) <= limit:
        plot_rows = rows
    else:
        chosen = set(np.linspace(0, len(rows) - 1, limit // 2, dtype=int).tolist())
        for chromosome in report["chromosomes"]:
            indices = [i for i, row in enumerate(rows) if str(row["CHR"]) == chromosome]
            budget = max(1, (limit // 2) // len(report["chromosomes"]) // 2)
            chosen.update(sorted(indices, key=lambda i: rows[i]["PIP"], reverse=True)[:budget])
            chosen.update(sorted(indices, key=lambda i: abs(rows[i].get("GWAS_Z") or 0), reverse=True)[:budget])
        plot_rows = [rows[i] for i in sorted(chosen)]
    return json.dumps(
        {
            "plot_data": {"rows": plot_rows, "total": len(rows), "sampled": len(plot_rows) < len(rows)},
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
