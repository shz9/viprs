# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2026-10-28

### Added

- Added a browser application that runs the real VIPRS and VIPRSMix inference code locally in a Pyodide
Web Worker, supports local and hosted summary statistics and LD references, and exports posterior tables
and detailed JSON run reports without uploading user data to a server.
- Added browser input support for files, folders, URL and glob patterns, hosted UK Biobank LD panels,
GWAS Catalog accessions, and multi-chromosome analyses, with chromosome-aware ordering, secure ZIP
extraction, input validation, progress reporting, cancellation, and a bundled chromosome 22 example.
- Added a reproducible WebAssembly build and packaging toolchain for VIPRS and magenpy, including SIMD-enabled
Cython/C++ kernels, pinned runtime dependencies, wheel and source checksums, build provenance, release-asset
verification and reuse, and source-build fallback when a compatible magenpy wheel is unavailable.
- Added continuous WebAssembly builds, browser integration tests, release archives, and documentation deployment
that publishes only a tested application bundle from the same source commit.
- Added comprehensive tests for browser input resolution and streaming, WebAssembly artifacts and release
selection, archive safety, multi-chromosome inference, user-interface behavior, and parity with native VIPRS.

### Changed

- Optimized the VIPRSMix E-step by accumulating component responsibilities, posterior moments, and ELBO terms
inside the native kernel, using thread-local double-precision reductions and cached sufficient statistics to
avoid repeated genome-wide scans during M-step and ELBO calculations.
- Added WebAssembly SIMD implementations of the inference kernel's dot-product and update operations while
preserving the existing native and OpenMP execution paths.
- Updated `viprs_fit` to obtain all supported summary-statistics formats and aliases from magenpy's parser
registry instead of maintaining a fixed list, and raised the minimum magenpy version to 0.2.3.
- Centralized GRCh37 GWAS Catalog accession resolution in `data_utils` and made accessions valid
summary-statistics inputs in both `viprs_fit` and the browser application.
- Improved the browser application's performance and memory use by range-streaming hosted Hugging Face LD,
downloading large range-capable GWAS Catalog files in parallel, parsing summary statistics once, using compact
numeric storage for large inputs, and releasing chromosome data and local LD archives incrementally.
- Improved the browser interface with clearer source selection, model configuration and prior visualization,
runtime and convergence status, timing summaries, result filtering, download controls, responsive styling,
and more actionable validation and network errors.
- Refreshed continuous benchmark results for the release changes.

### Fixed

- Fixed summary-statistics train/validation splitting to use the sample sizes returned by magenpy, avoid
compounding sample-size reductions across repeated splits, preserve mixture-model dimensions, and keep the
model-wide sample size synchronized when restoring or replacing inputs.
- Fixed `viprs_fit` to use `--validation-sumstats-format` when reading validation summary statistics rather
than reusing the training-data format.
- Fixed VIPRSMix parameter updates to preserve the configured floating-point precision and skipped negligible
LD updates in mixture and grid E-step kernels.
- Fixed browser and WebAssembly build behavior across Windows and POSIX path conventions, along with runtime,
URL-resolution, and browser-test failures discovered during application development.

## [0.1.5] - 2026-10-07

### Added

- Added a summary-statistics evaluation mode to `viprs_evaluate`, with support for independent GWAS
summary statistics, LD reference panels, fitted VIPRS effect sizes, custom summary-statistic formats,
and the `Pseudo_Pearson_R` and `Pseudo_R2` metrics.
- Added a continuous benchmarking workflow that fits VIPRS and VIPRSMix across five folds, records
prediction accuracy, runtime, and peak memory usage, and publishes CSV and Markdown benchmark histories.
- Added a reusable CLI benchmark utility and regression tests for benchmark result recording and aggregation.
- Added PGS Catalog scoring-file format 2.0 import and export, including metadata validation, compressed
output, selectable effect columns, and `viprs_fit --output-format pgs_catalog` support.
- Added `LDPredInfGrid`, which tunes the LDpred-inf penalty using PUMAS summary-statistic splitting and
pseudo-validation, then optionally refits the selected model on the full GWAS.
- Added allele-frequency-dependent priors to LDpred-inf through the `alpha` parameter, with optional joint
search over alpha values and penalty factors in `LDPredInfGrid`.
- Added CLI documentation for `viprs_fit`, `viprs_evaluate`, and `viprs_score`, plus an FAQ covering required
summary-statistic fields, allele conventions, genome builds, variant matching, and VIPRSMix selection.
- Added broader unit and integration coverage for VIPRSMix, E-step implementations, CLI argument handling,
summary-statistics evaluation, PGS Catalog files, and LDpred-inf model fitting and selection.

### Changed

- Reparameterized effect-size quantities during VIPRS optimization using a sample-size-based scale to improve
numerical stability, while restoring parameters to their original scale after fitting and in public outputs.
- Simplified model selection in `viprs_fit`: `--n-components 1` selects VIPRS, while values greater than one
select VIPRSMix, removing the need for a separate model flag.
- Refactored `viprs_evaluate` into explicit individual-level and summary-statistics workflows with stricter
input validation and support for multiple fitted models in one effect-size table.
- Updated VIPRSMix initialization, fixed-parameter handling, component precision updates, prior summaries,
posterior moment calculations, and theta-table reporting to consistently support scalar, vector-valued,
and chromosome-indexed parameters.
- Optimized mixed-precision E-step operations to avoid temporary converted arrays and skip negligible
posterior-mean updates.
- Updated continuous benchmarks to download their input archives directly instead of depending on
Hugging Face, and refreshed the recorded benchmark results.
- Reimplemented `LDPredInf` using standardized marginal effects, chromosome-wise MINRES solves, current
`magenpy` LD interfaces, solver convergence checks, and scalar or chromosome-specific heritability.
- Extended Hugging Face LD-path handling to CLI preflight validation and validation LD panels.
- Included chromosome and position columns in inferred parameter tables to support portable scoring-file
formats and position-based harmonization.
- Improved error logging when writing inferred parameters and hyperparameter tables.

### Fixed

- Fixed VIPRSMix prior initialization and heritability accounting, including component precision ratios,
mixture probabilities, null probabilities, average prior variance, and chromosome-specific parameters.
- Fixed VIPRSMix E-step calculations for the null component and regularization terms, including vector-valued
`lambda_min` inputs and cached variational precisions.
- Fixed numerical precision loss in VIPRSMix posterior second-moment calculations by accumulating in
double precision.
- Fixed LDpred-inf compatibility with current `magenpy`, use of unstandardized GWAS effects, unchecked solver
termination, unsupported chromosome-specific heritability, and unnecessary genome-wide block matrices.
- Fixed Windows compatibility in CLI evaluation tests.

## [0.1.4] - 2026-07-04

### Added

- Added cloud-streamable UK Biobank LD matrix paths hosted on Hugging Face, allowing users to pass
`hf://datasets/shz9/ukb-ld/<POP>/chr_*.zip` directly to `viprs_fit` without pre-downloading archives.
- Added `uv` and `Apptainer` installation examples to the documentation.
- Added a DockerHub publishing workflow and modernized the CLI Docker image for release as
`shadizabad/viprs`.
- Added `uv.lock` for reproducible development environments.
- Added tests to ensure runtime package version metadata stays synchronized.

### Changed

- Restricted supported Python versions to Python 3.10 through 3.13, inclusive.
- Centralized package version metadata in `viprs/_version.py` and reused it across runtime and packaging metadata.
- Updated `pyproject.toml` metadata and dependencies to match the package's runtime, optional, test, and docs
requirements.
- Split wheel build/testing and PyPI publishing into separate GitHub Actions workflows.
- Updated GitHub Actions runners and wheel builds to use current Linux, Windows, Intel macOS, and Apple Silicon
macOS targets.
- Modernized the CLI Dockerfile to use `python:3.11-slim-bookworm`, OCI labels, configurable install targets,
and installation smoke tests.
- Updated LD matrix documentation to put provenance and QC information before download/streaming instructions.

### Fixed

- Fixed the binary `f1` metric so it fits a logistic model on top of PRS values and thresholds predicted
probabilities, instead of passing continuous PRS values directly to `sklearn.metrics.f1_score`.
- Made BLAS discovery in `setup.py` tolerate missing system `pkg-config` and fall back to the existing
no-BLAS build path.

## [0.1.3] - 2025-04-22

### Changed

- Fixed bugs in `VIPRSGridSearch` and `VIPRSBMA` models, specifically how they were handling `_log_var_tau`, 
and the hyperparameters objects after selecting best models or performing model averaging.
- Fixed bug in how `viprs_fit` handles validation `gdl`s when the user passes genotype data.
- Updated interfaces in `HyperparameterSearch` script to make it more flexible and efficient. Primarily, 
I added shared memory object for the LD matrix to avoid redundant memory usage when fitting multiple
models in parallel. (** WORK IN PROGRESS **).
- Updated implementation of `pseudo_r2` to use square of pseudo correlation coefficient instead. The previous 
implementation can be problematic with highly sparsified LD matrices.
- Updated implementation of `VIPRSGrid` to be better integrated with the `VIPRS` class. The new implementation
also allows for fitting the grid in a `pathwise` fashion (now default behavior), where we use 
parameter estimates from previous grid points as warm-start initialization for the current grid point.
- Removed `VIPRSGridSearch` and `VIPRSBMA` classes for now. These functions are implemented in `grid_utils.py` instead
and they can be applied generically to any `VIPRSGrid` model.

### Added

- Added `viprs-cli-example.ipynb` notebook to demonstrate how to use the `viprs` commandline interface.
- Added documentation page for Downloading LD matrices.
- Added new utility function `combine_coefficient_tables` to combine the output from multiple VIPRS models.
- Added more thorough tests for the various models + CLI scripts.
- Added `PeakMemoryProfiler` to `viprs_fit` to more accurately track peak memory usage. Temporary solution, 
this will be moved to `magenpy` later on.
- Added support for splitting GWAS sumstats to training/validation sets and exposed appropriate interfaces 
in the base class `BayesPRSModel`.
- Added `IterationConditionCounter` class to keep track of the number of consecutive iterations 
where a certain condition is met. This is used to monitors convergence of the optimization routine.

## [0.1.2] - 2024-12-25

### Changed

- Fixed bug in implementation of `.fit` method of VIPRS models. Specifically, 
there was an issue with the `continued=True` flag not working because the `OptimizeResult`
object wasn't refreshed.
- Replaced `print` statements with `logging` where appropriate (still needs some more work).
- Updated way we measure peak memory in `viprs_fit`
- Updated `dict_concat` to just return the element if there's a single entry.
- Refactored pars of `VIPRS` to cache some recurring computations.
- Updated `VIPRSBMA` & `VIPRSGridSearch` to only consider models that
successfully converged.
- Fixed bug in `psuedo_metrics` when extracting summary statistics data.
- Streamlined evaluation code.
- Refactored code to slightly reduce import/load time.
- Fixed bug in `viprs_evaluate`

### Added

- Added SNP position to output table from VIPRS objects.
- Added measure of time taken to prepare data in `viprs_fit`.
- Added option to keep long-range LD regions in `viprs_fit`.
- Added convergence check based on parameter values.
- Added `min_iter` parameter to `.fit` methods to ensure CAVI is run for at least `min_iter` iterations.
- Added separate method for initializing optimization-related objects.
- Added regularization penalty `lambda_min`.
- Added Spearman R and residualized R-Squared metrics to continuous metrics.

## [0.1.1] - 2024-04-24

### Changed

- Fixed bugs in the E-Step benchmarking script.
- Re-wrote the logic for finding BLAS libraries in the `setup.py` script. :crossed_fingers:
- Fixed bugs in CI / GitHub Actions scripts.

### Added

- `Dockerfile`s for both `cli` and `jupyter` modes.

## [0.1.0] - 2024-04-05

A large scale restructuring of the code base to improve efficiency and usability.

### Changed

- Moved plotting script to its own separate module.
- Updated some method names / commandline flags to be consistent throughout.
- Updated the `VIPRS` class to allow for more flexibility in the optimization process.
- Removed the `VIPRSAlpha` model for now. This will be re-implemented in the future, 
using better interfaces / data structures.
- Moved all hyperparameter search classes/models to their own directory.
- Restructured the `viprs_fit` commandline script to make the code cleaner, 
do better sanity checking, and introduce process parallelism over chromosomes.

### Added

- Basic integration testing with `pytest` and GitHub workflows.
- Documentation for the entire package using `mkdocs`.
- Integration testing / automating building with GitHub workflows.
- New self-contained implementation of E-Step in `Cython` and `C++`.
  - Uses `OpenMP` for parallelism across chunks of variants.
  - Allows for de-quantization on the fly of the LD matrix.
  - Uses BLAS linear algebra operations where possible.
  - Allows model fitting with only 
- Benchmarking scripts (`benchmark_e_step.py`) to compare computational performance of different implementations.
- Added functionality to allow the user to track time / memory utilization in `viprs_fit`.
- Added `OptimizeResult` class to keep track of the info/parameters of EM optimization.
- New evaluation metrics
  - `pseudo_metrics` has been moved to its own module to allow for more flexibility in evaluation.
  - New evaluation metrics for binary traits: `nagelkerke_r2`, `mcfadden_r2`, 
  `cox_snell_r2` `liability_r2`, `liability_probit_r2`, `liability_logit_r2`.
  - New function to compute standard errors / test statistics for all R-Squared metrics.

## [0.0.4] - 2022-09-07

### Changed

- Removed the `--fast-math` compiler flag due to concerns about 
numerical precision (e.g. [Beware of fast-math](https://simonbyrne.github.io/notes/fastmath/)).

## [0.0.3] - 2022-09-06

### Added

- New implementation for the e-step in `VIPRS`, where we multiply with the rows of the
LD matrix only once.
- Added support for deterministic annealing in the `VIPRS` optimization.
- Added support for `pseudo_validation` as a metric for choosing models. Now, the
`VIPRS` class has a method called `pseudo_validate`.
- New implementations for grid-based models: `VIPRSGrid`, `VIPRSGridSearch`, `VIPRSBMA`.
- New python implementation of the `LDPredinf` model, using the `viprs`/`magenpy` 
data structures.
- MIT license for the software.

### Changed

- Corrected implementation of Mean Squared Error (MSE) metric.
- Changed the `c_utils.pyx` script to be `math_utils.pyx`.
- Updated documentation in `README` to follow latest APIs.

## [0.0.2] - 2022-06-28

### Changed

- Updating the dependency structure between `viprs` and `magenpy`.

## [0.0.1] - 2022-06-28

### Added

- Refactoring the code in the  `viprs` repository and re-organizing it into a python package.
- Added a module to compute predictive performance metrics.
- Added commandline scripts to allow users to access some of the functionalities of `viprs` without 
necessarily having to write python code.
- Added the estimate of the posterior variance to the output from the module.  

### Changed

- Updated plotting script.
- Updated implementation of `VIPRSMix`, `VIPRSAalpha`, etc. to inherit most 
of their functionalities from the base `VIPRS` class.
- Cleaned up implementation of hyperparameter search modules.
