# `viprs`: Variational Inference of Polygenic Risk Scores

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/viprs.svg)](https://pypi.python.org/pypi/viprs/)
[![PyPI version fury.io](https://badge.fury.io/py/viprs.svg)](https://pypi.python.org/pypi/viprs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Linux CI](https://github.com/shz9/viprs/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/shz9/viprs/actions/workflows/ci-linux.yml)
[![MacOS CI](https://github.com/shz9/viprs/actions/workflows/ci-osx.yml/badge.svg)](https://github.com/shz9/viprs/actions/workflows/ci-osx.yml)
[![Windows CI](https://github.com/shz9/viprs/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/shz9/viprs/actions/workflows/ci-windows.yml)
[![Docs Build](https://github.com/shz9/viprs/actions/workflows/ci-docs.yml/badge.svg)](https://github.com/shz9/viprs/actions/workflows/ci-docs.yml)
[![Binary wheels](https://github.com/shz9/viprs/actions/workflows/wheels.yml/badge.svg)](https://github.com/shz9/viprs/actions/workflows/wheels.yml)


[![Downloads](https://static.pepy.tech/badge/viprs)](https://pepy.tech/project/viprs)
[![Downloads](https://static.pepy.tech/badge/viprs/month)](https://pepy.tech/project/viprs)


`viprs` is a python package that implements variational inference techniques to estimate the posterior distribution 
of variant effect sizes conditional on the GWAS summary statistics. The package is designed to be fast and accurate, 
and to provide a variety of options for the user to customize the inference process.
Highlighted features:

* The coordinate ascent algorithms are written in `C/C++` and `cython` for improved speed and efficiency.
* The code is written in object-oriented form, allowing the user to extend and 
experiment with existing implementations.
* Different priors on the effect size: Spike-and-slab, Sparse mixture, etc.
* We also provide scripts for different hyperparameter tuning strategies, including: 
Grid search, Bayesian optimization, Bayesian model averaging.
* Easy and straightforward interfaces for computing PRS from fitted models.
* Implementation for a wide variety of evaluation metrics for both binary and continuous phenotypes.


### Helpful links

- [Documentation](https://shz9.github.io/viprs/)
- [Citation / BibTeX records](./CITATION.md)
- [Report issues/bugs](https://github.com/shz9/viprs/issues)