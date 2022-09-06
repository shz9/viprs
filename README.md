# `viprs`: Variational Inference of Polygenic Risk Scores

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/viprs.svg)](https://pypi.python.org/pypi/viprs/)
[![PyPI version fury.io](https://badge.fury.io/py/viprs.svg)](https://pypi.python.org/pypi/viprs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`viprs` is a python package that implements scripts and utilities for running variational inference 
algorithms on genome-wide association study (GWAS) data for the purposes polygenic risk estimation. 

Highlighted features:

- The coordinate ascent algorithms are written in `cython` for improved speed and efficiency.
- The code is written in object-oriented form, allowing the user to extend and experiment with existing implementations.
- We provide scripts for fitting the model with different priors on the effect size: `VIPRSAlpha`, `VIPRSMix`, etc.
- We also provide scripts for different hyperparameter tuning strategies, including: Grid search, Bayesian optimization, Bayesian model averaging.

**NOTE**: The codebase is still in active development and some of interfaces or data structures will be 
replaced or modified in future releases.

## Table of Contents

- [Installation](#Installation)
- [Getting started](#getting-started)
- [Features and Configurations](#features-and-configurations)
  - [(1) Evaluating `viprs` with training and test data](#1-evaluating-viprs-with-training-and-test-data)
  - [(2) Grid search and validation](#2-grid-search-and-validation)
- [Commandline scripts](#commandline-scripts)
- [Download LD matrices](#download-ld-matrices)
- [Frequently Asked Questions (FAQs)](#frequently-asked-questions-faqs)
- [Citations](#citations) 


## Installation

`viprs` is now available on the python package index `pypi` and 
can be minimally installed using the package installer `pip`:

```shell
pip install viprs
```

To access the full functionalities of `viprs`, however, it is recommended that 
you install the full list of dependencies:

```shell
pip install viprs[full]
```

To use `viprs` on a shared computing cluster, we recommend installing it in a 
`python` virtual environment. For example:

```shell
module load python/3.8
python -m venv viprs_env
source viprs_env/bin/activate
pip install --upgrade pip
pip install viprs
```

Finally, if you wish to install the package from source, 
you can directly clone the GitHub repository and install it locally 
as follows:

```
git clone https://github.com/shz9/viprs.git
cd viprs
make install
```

## Getting started

`viprs` is a `python` package for fitting Bayesian Polygenic Risk Score (PRS) models to summary statistics 
derived from Genome-wide Association Studies (GWASs). To showcase the interfaces and functionalities of the package 
as well as the data structures that power it, we will start with a simple example. 

Generally, summary statistics-based PRS methods require access to **(1)** GWAS summary statistics for the 
trait of interest and **(2)** Linkage-Disequilibrium (LD) matrices from an appropriately-matched reference panel (e.g. 
the 1000G dataset). For the first item, we will use summary statistics for standing height from the `fastGWA` catalogue. 
For the second item, we will use genotype data on chromosome 22 for a subset of 378 European samples from the 
1000G project. This small dataset is shipped with the python package `magenpy`.

To start, let's import the required `python` modules:

```python
import magenpy as mgp
import viprs as vp
```

Then, we will use `magenpy` to read the 1000G genotype dataset and *automatically* match it with the GWAS 
summary statistics from `fastGWA`:

```python
# Load genotype and GWAS summary statistics data (chromosome 22):
gdl = mgp.GWADataLoader(bed_files=mgp.tgp_eur_data_path(),
                        sumstats_files=mgp.ukb_height_fastGWA_path(),
                        sumstats_format="fastGWA")
```

Once the genotype and summary statistics data are read by `magenpy`, we can go ahead and compute 
the LD (or SNP-by-SNP correlation) matrix:

```python
# Compute LD using the shrinkage estimator (Wen and Stephens 2010):
gdl.compute_ld("shrinkage",
               output_dir="temp",
               genetic_map_ne=11400, # effective population size (Ne)
               genetic_map_sample_size=183,
               threshold=1e-3)
```

Because of the small sample size of the reference panel, here we recommend using the `shrinkage` estimator 
for LD from Wen and Stephens (2010). The shrinkage estimator results in compact and sparse LD matrices that are 
more robust than the sample LD. The estimator requires access to information about the genetic map, such as 
the position of each SNP in centi Morgan, the effective population size, and the sample size used to 
estimate the genetic map.

Given the LD information from the reference panel, we can next fit the VIPRS model to the summary statistics data:

```python
# Initialize VIPRS, passing it the GWADataLoader object
v = vp.VIPRS(gdl)
# Invoke the .fit() method to obtain posterior estimates
v.fit()
```

Once the model converges, we can generate PRS estimates for height for the 1000G samples by simply 
invoking the `.predict()` method:

```python
v.predict()
# array([ 0.01944202,  0.00597704,  0.07329462, ..., 0.06666187,  0.05251297,  0.00359018])
```

To examine posterior estimates for the model parameters, you can simply invoke the `.to_table()` method:

```python
v.to_table()
#       CHR         SNP A1 A2       PIP          BETA      VAR_BETA
# 0       22    rs131538  A  G  0.006107 -5.955517e-06  1.874619e-08
# 1       22   rs9605903  C  T  0.005927  5.527188e-06  1.774252e-08
# 2       22   rs5746647  G  T  0.005015  1.194178e-07  1.120063e-08
# 3       22  rs16980739  T  C  0.008331 -1.335695e-05  3.717944e-08
# 4       22   rs9605923  A  T  0.006181  6.334971e-06  1.979157e-08
# ...    ...         ... .. ..       ...           ...           ...
# 15930   22   rs8137951  A  G  0.006367 -6.880591e-06  2.059650e-08
# 15931   22   rs2301584  A  G  0.179406 -7.234545e-04  2.597197e-06
# 15932   22   rs3810648  G  A  0.008000  1.222151e-05  3.399927e-08
# 15933   22   rs2285395  A  G  0.005356  3.004282e-06  1.349082e-08
# 15934   22  rs28729663  A  G  0.005350 -2.781053e-06  1.351239e-08
#
# [15935 rows x 7 columns]
```

Here, `PIP` is the **P**osterior **I**nclusion **P**robability under the variational density, while 
`BETA` and `VAR_BETA` are the posterior mean and variance for the effect size, respectively. 
For the purposes of prediction, we only need the `BETA` column. You can also examine the 
inferred hyperparameters of the model by invoking the `.to_theta_table()` method:

```python
v.to_theta_table()
#           Parameter     Value
# 0  Residual_variance  0.994231
# 1       Heritability  0.005736
# 2  Proportion_causal  0.015887
# 3         sigma_beta  0.000021
```

Note that here, the SNP heritability only considers the contribution of variants on 
chromosome 22.

## Features and Configurations 

### (1) Evaluating `viprs` with training and test data

Here, we will go through an artificial example on simulated 
phenotypes using a subset of 387 individuals from the 1000 Genomes dataset. To begin, let's use the python 
package `magenpy` to simulate a quantitative trait where 1% of the variants 
are causal and the SNP heritability set to `0.8`:

```python
g_sim = mgp.GWASimulator(mgp.tgp_eur_data_path(),
                         pi=(.99, .01),
                         h2=0.8)
g_sim.simulate()
g_sim.to_phenotype_table()
#          FID      IID  phenotype
# 0    HG00096  HG00096  -1.787443
# 1    HG00097  HG00097   0.282521
# 2    HG00099  HG00099   0.532999
# 3    HG00100  HG00100   0.781203
# 4    HG00101  HG00101   0.081900
# ..       ...      ...        ...
# 373  NA20815  NA20815   0.898588
# 374  NA20818  NA20818   1.549989
# 375  NA20819  NA20819  -0.259188
# 376  NA20826  NA20826  -0.044187
# 377  NA20828  NA20828   2.254222
#
# [378 rows x 3 columns]
```

To obtain a realistic setting where the training data is separate from the test data, next we are going to 
randomly split the samples into 80% training and 20% testing:

```python
training_gdl, test_gdl = g_sim.split_by_samples(proportions=[.8, .2])
print("Number of training samples:", training_gdl.sample_size)
# Number of training samples: 314
print("Number of test samples:", test_gdl.sample_size)
# Number of test samples: 64
```

Using the training data, we will compute LD matrices and perform GWAS 
on the simulated phenotype:

```python
# Compute LD using the shrinkage estimator:
training_gdl.compute_ld('shrinkage',
                        output_dir='temp',
                        genetic_map_ne=11400, # effective population size (Ne)
                        genetic_map_sample_size=183,
                        threshold=1e-3)
# Perform GWAS on the simulated phenotype:
training_gdl.perform_gwas()
```

And that is all we need to fit the `VIPRS` method to the training data! To perform model fit,
simply pass the `g_sim` object to `VIPRS` and then invoke the `.fit()` method:

```python
v = vp.VIPRS(training_gdl)
v.fit()
```

If everything works as expected, the method will converge reasonably fast and you should be able to 
use object to perform predictions on the test set. Once convergence is achieved, you can compute 
polygenic risk scores (PRSs) for the test samples. This can be simply done by invoking the `.predict()` method and 
passing the `test_gdl` object as an argument, as follows:

```python
test_prs = v.predict(test_gdl)
```

For the sake of comparison, we will also include the PRS based on the marginal effect sizes from GWAS:

```python
naive_prs = test_gdl.predict(beta={c: ss.marginal_beta for c, ss in training_gdl.sumstats_table.items()})
```

To evaluate the accuracy of the polygenic score, you can use some of the provided metrics with the `viprs` package. 
For quantitative traits, it is customary to report the prediction R^2, or the proportion 
of variance explained by the PRS:

```python
from viprs.eval.metrics import r2 
r2(test_prs, test_gdl.sample_table.phenotype)
# 0.171429
```

The prediction R^2 here is much lower than we would expect, given our simulation 
scenario. This is again largely due to the small sample size that we are working with and 
the noise this induces on the estimates of LD, GWAS summary statistics, etc. For reference, 
we can compare this to the R^2 from the "naive" PRS estimator:

```python
r2(naive_prs, test_gdl.sample_table.phenotype)
# 0.169567
```

In this artificial setting, `viprs` results in minor improvements in prediction accuracy 
on the test set compared to the naive PRS estimator. However, as we show in our manuscript, this 
improvement is more pronounced when we use larger sample sizes for both GWAS and the LD reference panel.

### (2) Grid search and validation

In our manuscript, we found that VIPRS paired with grid search with a held-out validation set 
performs favorably compared to the vanilla model. Here, we will illustrate how to use the grid search functionalities 
implemented in the `viprs` package. To start, let's use the same simulation scenario as before:

```python
g_sim = mgp.GWASimulator(mgp.tgp_eur_data_path(),
                         pi=(.99, .01),
                         h2=0.8)
g_sim.simulate()
g_sim.to_phenotype_table()
#         FID      IID  phenotype
# 0    HG00096  HG00096  -0.283120
# 1    HG00097  HG00097   0.060302
# 2    HG00099  HG00099  -0.571485
# 3    HG00100  HG00100   1.145548
# 4    HG00101  HG00101   0.535409
# ..       ...      ...        ...
# 373  NA20815  NA20815   0.214542
# 374  NA20818  NA20818  -0.986338
# 375  NA20819  NA20819  -0.056678
# 376  NA20826  NA20826   0.119752
# 377  NA20828  NA20828   0.889052
# 
# [378 rows x 3 columns]
```

Then, we will split this small dataset into 70% training, 15% validation, and 15% testing:

```python
training_gdl, validation_gdl, test_gdl = g_sim.split_by_samples(proportions=[.7, .15, .15])
print("Number of training samples:", training_gdl.sample_size)
# Number of training samples: 251
print("Number of validation samples:", validation_gdl.sample_size)
# Number of validation samples: 66
print("Number of test samples:", test_gdl.sample_size)
# Number of test samples: 61
```

As before, we will compute the LD and perform GWAS on the training samples:

```python
# Compute LD using the shrinkage estimator:
training_gdl.compute_ld('shrinkage',
                        output_dir='temp',
                        genetic_map_ne=11400, # effective population size (Ne)
                        genetic_map_sample_size=183,
                        threshold=1e-3)
# Perform GWAS on the simulated phenotype:
training_gdl.perform_gwas()
```

For a simple baseline, we'll examine the performance of the standard VIPRS model 
without any grid search:

```python
v = vp.VIPRS(training_gdl).fit()
```

Once converged, for this simulation scenario, the standard VIPRS results in the following prediction R^2:

```python
from viprs.eval.metrics import r2
r2(v.predict(test_gdl), test_gdl.sample_table.phenotype)
0.168073
```

To see how grid search would perform in comparison, first we construct a grid 
for some of the hyperparameters of interest, including `pi` and `sigma_epsilon`, the 
proportion of causal variants and the residual variance, respectively:

```python
# Create a grid:
grid = vp.HyperparameterGrid()
# Generate a grid for pi using 5 equidistant grid points:
grid.generate_pi_grid(steps=5, n_snps=training_gdl.n_snps)
# Generate a grid for sigma epsilon using 5 equidistant grid points:
grid.generate_sigma_epsilon_grid(steps=5)
```

This will generate a grid of 5x5=25 hyperparameter settings that we will search over. 
To fit the VIPRS model with this grid, we can simply use the `GridSearch` class, passing it 
the training data as well as the grid that we just created:

```python
vgv_gs = vp.VIPRSGridSearch(training_gdl, grid)
vgv_gs = vgv_gs.fit()
vgv.select_best_model(validation_gdl=validation_gdl, criterion='validation')
```

Here, in order to make use of the held-out validation data, we pass it the `GridSearch` object and 
we also specify that the objective that we wish to optimize is the `validation` R^2. When we call `.fit()` on 
the `GridSearch` object, it will fit 25 different VIPRS models, each with a different hyperparameter setting 
and then choose the one with the best predictive performance on the held-out validation set.

Given this "best" model, we can using to make predictions and test prediction accuracy on the held-out test set:

```python
r2(vgv.predict(test_gdl), test_gdl.sample_table.phenotype)
# 0.167938
```

In this case, the prediction R^2 is actually marginally lower than standard VIPRS model, which can happen in some 
cases, as illustrated in the simulation results in our manuscript. As our results showed, the main advantage of 
the `GridSearch` module will be when applied on real phenotypes or when there are some mismatches between 
the LD reference panel and the GWAS summary statistics.

## Commandline scripts

If you are not comfortable programming in `python` and would like to access some of the functionalities 
of `viprs` with minimal interaction with `python` code, we packaged a number of commandline 
scripts that can be useful for some downstream applications.

The binaries that are currently supported are:

1) `viprs_fit`: For performing model fit and estimating the posterior for the effect sizes.
2) `viprs_score`: For generating polygenic scores for new test samples.
3) `viprs_evaluate`: For evaluating the predictive performance of PRS methods.

Once you install `viprs` via `pip`, these scripts will be added to the system `PATH` 
and you can invoke them directly from the commandline, as follows:

```shell
$ viprs_fit -h

        **********************************************
                    _____
            ___   _____(_)________ ________________
            __ | / /__  / ___  __ \__  ___/__  ___/
            __ |/ / _  /  __  /_/ /_  /    _(__  )
            _____/  /_/   _  .___/ /_/     /____/
                          /_/
        Variational Inference of Polygenic Risk Scores
        Version: 0.0.2 | Release date: June 2022
        Author: Shadi Zabad, McGill University
        **********************************************
        < Fit VIPRS model to GWAS summary statistics >
    
usage: viprs_fit [-h] -l LD_DIR -s SUMSTATS_PATH --output-file OUTPUT_FILE [--sumstats-format {fastGWA,custom,plink,magenpy,COJO}] [--snp SNP] [--a1 A1] [--n-per-snp N_PER_SNP]
                 [--z-score Z_SCORE] [--beta BETA] [--se SE] [--temp-dir TEMP_DIR] [--validation-bed VALIDATION_BED] [--validation-pheno VALIDATION_PHENO]
                 [--validation-keep VALIDATION_KEEP] [-m {VIPRS,VIPRSMix,VIPRSAlpha}] [--annealing-schedule {harmonic,linear,geometric}] [--annealing-steps ANNEALING_STEPS]
                 [--initial-temperature INITIAL_TEMPERATURE] [--n-components N_COMPONENTS] [--prior-mult PRIOR_MULT] [--hyp-search {EM,BMA,GS,BO}]
                 [--grid-metric {validation,pseudo_validation,ELBO}] [--opt-params OPT_PARAMS] [--pi-grid PI_GRID] [--pi-steps PI_STEPS] [--sigma-epsilon-grid SIGMA_EPSILON_GRID]
                 [--sigma-epsilon-steps SIGMA_EPSILON_STEPS] [--sigma-beta-grid SIGMA_BETA_GRID] [--sigma-beta-steps SIGMA_BETA_STEPS] [--h2-informed-grid] [--compress] [--genomewide]
                 [--backend {xarray,plink}] [--max-attempts MAX_ATTEMPTS] [--use-multiprocessing] [--n-jobs N_JOBS]

Commandline arguments for fitting the VIPRS models

optional arguments:
  -h, --help            show this help message and exit
  -l LD_DIR, --ld-panel LD_DIR
                        The path to the directory where the LD matrices are stored. Can be a wildcard of the form ld/chr_*
  -s SUMSTATS_PATH, --sumstats SUMSTATS_PATH
                        The summary statistics directory or file. Can be a wildcard of the form sumstats/chr_*
  --output-file OUTPUT_FILE
                        The output file where to store the inference results. Only include the prefix, the extensions will be added automatically.
  --sumstats-format {fastGWA,custom,plink,magenpy,COJO}
                        The format for the summary statistics file(s).
  --snp SNP             The column name for the SNP rsID in the summary statistics file (custom formats).
  --a1 A1               The column name for the effect allele in the summary statistics file (custom formats).
  --n-per-snp N_PER_SNP
                        The column name for the sample size per SNP in the summary statistics file (custom formats).
  --z-score Z_SCORE     The column name for the z-score in the summary statistics file (custom formats).
  --beta BETA           The column name for the beta (effect size estimate) in the summary statistics file (custom formats).
  --se SE               The column name for the standard error in the summary statistics file (custom formats).
  --temp-dir TEMP_DIR   The temporary directory where to store intermediate files.
  --validation-bed VALIDATION_BED
                        The BED files containing the genotype data for the validation set. You may use a wildcard here (e.g. "data/chr_*.bed")
  --validation-pheno VALIDATION_PHENO
                        A tab-separated file containing the phenotype for the validation set. The expected format is: FID IID phenotype (no header)
  --validation-keep VALIDATION_KEEP
                        A plink-style keep file to select a subset of individuals for the validation set.
  -m {VIPRS,VIPRSMix,VIPRSAlpha}, --model {VIPRS,VIPRSMix,VIPRSAlpha}
                        The PRS model to fit
  --annealing-schedule {harmonic,linear,geometric}
                        The type of schedule for updating the temperature parameter in deterministic annealing.
  --annealing-steps ANNEALING_STEPS
                        The number of deterministic annealing steps to perform.
  --initial-temperature INITIAL_TEMPERATURE
                        The initial temperature for the deterministic annealing procedure.
  --n-components N_COMPONENTS
                        The number of non-null Gaussian mixture components to use with the VIPRSMix model (i.e. excluding the spike component).
  --prior-mult PRIOR_MULT
                        Prior multipliers on the variance of the non-null Gaussian mixture component.
  --hyp-search {EM,BMA,GS,BO}
                        The strategy for tuning the hyperparameters of the model. Options are EM (Expectation-Maximization), GS (Grid search), BO (Bayesian Optimization), and BMA
                        (Bayesian Model Averaging).
  --grid-metric {validation,pseudo_validation,ELBO}
                        The metric for selecting best performing model in grid search.
  --opt-params OPT_PARAMS
                        The hyperparameters to tune using GridSearch/BMA/Bayesian optimization (comma-separated).Possible values are pi, sigma_beta, and sigma_epsilon. Or a combination
                        of them.
  --pi-grid PI_GRID     A comma-separated grid values for the hyperparameter pi (see also --pi-steps).
  --pi-steps PI_STEPS   The number of steps for the (default) pi grid. This will create an equidistant grid between 1/M and (M-1)/M on a log10 scale, where M is the number of SNPs.
  --sigma-epsilon-grid SIGMA_EPSILON_GRID
                        A comma-separated grid values for the hyperparameter sigma_epsilon (see also --sigma-epsilon-steps).
  --sigma-epsilon-steps SIGMA_EPSILON_STEPS
                        The number of steps for the (default) sigma_epsilon grid.
  --sigma-beta-grid SIGMA_BETA_GRID
                        A comma-separated grid values for the hyperparameter sigma_beta (see also --sigma-beta-steps).
  --sigma-beta-steps SIGMA_BETA_STEPS
                        The number of steps for the (default) sigma_beta grid.
  --h2-informed-grid    Construct a grid for sigma_epsilon/sigma_beta based on informed estimates of the trait heritability.
  --compress            Compress the output files
  --genomewide          Fit all chromosomes jointly
  --backend {xarray,plink}
                        The backend software used for computations on the genotype matrix.
  --max-attempts MAX_ATTEMPTS
                        The maximum number of model restarts (in case of optimization divergence issues).
  --use-multiprocessing
                        Use multiprocessing where applicable. For now, this mainly affects the GridSearch/Bayesian Model Averaging implementations.
  --n-jobs N_JOBS       The number of processes/threads to launch for the hyperparameter search (default is 1, but we recommend increasing this depending on system capacity).
```

```shell
$ viprs_score -h

        **********************************************
                    _____
            ___   _____(_)________ ________________
            __ | / /__  / ___  __ \__  ___/__  ___/
            __ |/ / _  /  __  /_/ /_  /    _(__  )
            _____/  /_/   _  .___/ /_/     /____/
                          /_/
        Variational Inference of Polygenic Risk Scores
        Version: 0.0.2 | Release date: June 2022
        Author: Shadi Zabad, McGill University
        **********************************************
        < Compute polygenic scores for test samples >

usage: viprs_score [-h] -f FIT_FILES --bed-files BED_FILES --output-file OUTPUT_FILE [--temp-dir TEMP_DIR] [--keep KEEP] [--extract EXTRACT] [--backend {xarray,plink}]
                   [--n-threads N_THREADS] [--compress]

Commandline arguments for generating polygenic scores

optional arguments:
  -h, --help            show this help message and exit
  -f FIT_FILES, --fit-files FIT_FILES
                        The path to the file(s) with the output parameter estimates from VIPRS. You may use a wildcard here (e.g. "prs/chr_*.fit")
  --bed-files BED_FILES
                        The BED files containing the genotype data. You may use a wildcard here (e.g. "data/chr_*.bed")
  --output-file OUTPUT_FILE
                        The output file where to store the polygenic scores (with no extension).
  --temp-dir TEMP_DIR   The temporary directory where to store intermediate files.
  --keep KEEP           A plink-style keep file to select a subset of individuals for the test set.
  --extract EXTRACT     A plink-style extract file to select a subset of SNPs for scoring.
  --backend {xarray,plink}
                        The backend software used for computations with the genotype matrix.
  --n-threads N_THREADS
                        The number of threads to use for computations.
  --compress            Compress the output file
```

```shell
$ viprs_evaluate -h

     **************************************************
                    _____
            ___   _____(_)________ ________________
            __ | / /__  / ___  __ \__  ___/__  ___/
            __ |/ / _  /  __  /_/ /_  /    _(__  )
            _____/  /_/   _  .___/ /_/     /____/
                          /_/
        Variational Inference of Polygenic Risk Scores
        Version: 0.0.2 | Release date: June 2022
        Author: Shadi Zabad, McGill University
     **************************************************
       < Evaluate the predictive performance of PRS >

usage: viprs_evaluate [-h] --prs-file PRS_FILE --phenotype-file PHENO_FILE [--phenotype-likelihood {binomial,gaussian}] --output-file OUTPUT_FILE [--covariate-file COVARIATE_FILE]

Commandline arguments for evaluating polygenic score estimates

optional arguments:
  -h, --help            show this help message and exit
  --prs-file PRS_FILE   The path to the PRS file (expected format: FID IID PRS, tab-separated)
  --phenotype-file PHENO_FILE
                        The path to the phenotype file. The expected format is: FID IID phenotype (no header), tab-separated.
  --phenotype-likelihood {binomial,gaussian}
                        The phenotype likelihood ("gaussian" for continuous, "binomial" for case-control).
  --output-file OUTPUT_FILE
                        The output file where to store the evaluation metrics (with no extension).
  --covariate-file COVARIATE_FILE
                        A file with covariates for the samples included in the analysis. This tab-separated file should not have a header and the first two columns should be the FID
                        and IID of the samples.
```


## Download LD matrices

To run the `viprs` software, you may need access to pre-computed LD matrices. Here, we provide access to pre-computed LD matrices
from the UK Biobank (UKB) dataset:

- LD matrices derived from the "White British" cohort (N=50,000) in the UKB: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7036625.svg)](https://doi.org/10.5281/zenodo.7036625)


You can also use the `data_utils` module from the `viprs` package to download those 
data sources. For example, to download the UKB LD matrices listed above, you can do 
the following:

```python
import viprs as vp

# Download all matrices from Zenodo:
vp.download_ukb_wb_ld_matrix('data/ld/')

# Download LD matrices for chromosomes 21 and 22:
vp.download_ukb_wb_ld_matrix('data/ld/', chromosome=[21, 22])

```

This will download the LD matrix to a sub-folder in the working directory called `data/ld`. We will add more utilities 
like this in the future to streamline PRS model training and testing.

## Frequently Asked Questions (FAQs)

- **How do I create my own LD matrices and store them in formats compatible with `viprs`?**

You can use the software package [magenpy](https://github.com/shz9/magenpy) to compute LD matrices 
from any reference genotype dataset. `magenpy` provides interfaces for computing LD matrices 
using 3 commonly used LD estimators: `shrinkage`, `windowed` (or banded), and `block
`-based LD matrices. Check the relevant documentation [here](https://github.com/shz9/magenpy#3-calculating-ld-matrices).


## Citations

Shadi Zabad, Simon Gravel, Yue Li. **Fast and Accurate Bayesian Polygenic Risk Modeling with Variational Inference**. (2022)

```bibtex
@article {
    Zabad2022.05.10.491396,
    author = {Zabad, Shadi and Gravel, Simon and Li, Yue},
    title = {Fast and Accurate Bayesian Polygenic Risk Modeling with Variational Inference},
    elocation-id = {2022.05.10.491396},
    year = {2022},
    doi = {10.1101/2022.05.10.491396},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2022/05/11/2022.05.10.491396},
    journal = {bioRxiv}
}
```
