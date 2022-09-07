# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

