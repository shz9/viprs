# `viprs`: Variational Inference of Polygenic Risk Scores

`viprs` is a python package that implements scripts and utilities for running variational inference 
algorithms on genome-wide association study (GWAS) data for the purposes polygenic risk estimation. 

Highlighted features:

- The coordinate ascent algorithms are written in `cython` for improved speed and efficiency.
- The code is written in object-oriented form, allowing the user to extend and experiment with existing implementations.
- We provide scripts running the model with different priors on the effect size: `VIPRSAlpha`, `VIPRSMix`, etc.
- We also provide scripts for different hyperparameter tuning strategies, including: `Grid search`, `Bayesian optimization`, `Bayesian model averaging`.

## Installation

We are working on adding the source code into `pypi` in the near future. In the meantime, you can manually install it as follows:

```
git clone https://github.com/shz9/viprs.git
git submodule update --recursive --remote
pip install -r requirements.txt
pip install -r optional-requirements.txt
```

## Getting started

TODO

## Download LD matrices

To run the `viprs` software, you may need access to pre-computed LD matrices. Here, we provide access to some pre-computed LD matrices
from the UK Biobank (UKB):

- LD matrices derived from the "White British" cohort in the UKB: [ukb_eur_50k_windowed_ld.tar.gz](https://doi.org/10.5281/zenodo.6529229)

## Citations

Shadi Zabad, Simon Gravel, Yue Li. **Fast and Accurate Bayesian Polygenic Risk Modeling with Variational Inference**. (2022)
