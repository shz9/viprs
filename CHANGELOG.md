# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2024-06-03

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

