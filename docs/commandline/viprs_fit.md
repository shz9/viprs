Fit VIPRS model to GWAS summary statistics (`viprs_fit`)
---

The `viprs_fit` script is used to fit the variational PRS model to the GWAS summary statistics and to estimate the 
posterior distribution of the variant effect sizes. The script provides a variety of options for the user to 
customize the inference process, including the choice of prior distributions and the choice of 
optimization algorithms.

A full listing of the options available for the `viprs_fit` script can be found by running the following command in your terminal:

```bash
viprs_fit -h
```

Which outputs the following help message:

```bash

        **********************************************
                    _____
            ___   _____(_)________ ________________
            __ | / /__  / ___  __ \__  ___/__  ___/
            __ |/ / _  /  __  /_/ /_  /    _(__  )
            _____/  /_/   _  .___/ /_/     /____/
                          /_/
        Variational Inference of Polygenic Risk Scores
        Version: 0.1.0 | Release date: April 2024
        Author: Shadi Zabad, McGill University
        **********************************************
        < Fit VIPRS models to GWAS summary statistics >
    
usage: viprs_fit [-h] -l LD_DIR -s SUMSTATS_PATH --output-dir OUTPUT_DIR [--output-file-prefix OUTPUT_PREFIX] [--temp-dir TEMP_DIR]
                 [--sumstats-format {fastgwa,plink,ssf,plink2,custom,cojo,saige,magenpy,plink1.9,gwas-ssf,gwascatalog}]
                 [--custom-sumstats-mapper CUSTOM_SUMSTATS_MAPPER] [--custom-sumstats-sep CUSTOM_SUMSTATS_SEP] [--gwas-sample-size GWAS_SAMPLE_SIZE]
                 [--validation-bed VALIDATION_BED] [--validation-pheno VALIDATION_PHENO] [--validation-keep VALIDATION_KEEP]
                 [--validation-ld-panel VALIDATION_LD_PANEL] [--validation-sumstats VALIDATION_SUMSTATS_PATH]
                 [--validation-sumstats-format {fastgwa,plink,ssf,plink2,custom,cojo,saige,magenpy,plink1.9,gwas-ssf,gwascatalog}] [-m {VIPRSMix,VIPRS}]
                 [--float-precision {float32,float64}] [--use-symmetric-ld] [--n-components N_COMPONENTS] [--h2-est H2_EST] [--h2-se H2_SE]
                 [--hyp-search {GS,EM,BMA,BO}] [--grid-metric {validation,ELBO,pseudo_validation}] [--pi-grid PI_GRID] [--pi-steps PI_STEPS]
                 [--sigma-epsilon-grid SIGMA_EPSILON_GRID] [--sigma-epsilon-steps SIGMA_EPSILON_STEPS] [--genomewide] [--backend {xarray,plink}] [--n-jobs N_JOBS]
                 [--threads THREADS] [--output-profiler-metrics] [--seed SEED]

Commandline arguments for fitting VIPRS models to GWAS summary statistics

optional arguments:
  -h, --help            show this help message and exit
  -l LD_DIR, --ld-panel LD_DIR
                        The path to the directory where the LD matrices are stored. Can be a wildcard of the form ld/chr_*
  -s SUMSTATS_PATH, --sumstats SUMSTATS_PATH
                        The summary statistics directory or file. Can be a wildcard of the form sumstats/chr_*
  --output-dir OUTPUT_DIR
                        The output directory where to store the inference results.
  --output-file-prefix OUTPUT_PREFIX
                        A prefix to append to the names of the output files (optional).
  --temp-dir TEMP_DIR   The temporary directory where to store intermediate files.
  --sumstats-format {fastgwa,plink,ssf,plink2,custom,cojo,saige,magenpy,plink1.9,gwas-ssf,gwascatalog}
                        The format for the summary statistics file(s).
  --custom-sumstats-mapper CUSTOM_SUMSTATS_MAPPER
                        A comma-separated string with column name mappings between the custom summary statistics format and the standard format expected by
                        magenpy/VIPRS. Provide only mappings for column names that are different, in the form of:--custom-sumstats-mapper
                        rsid=SNP,eff_allele=A1,beta=BETA
  --custom-sumstats-sep CUSTOM_SUMSTATS_SEP
                        The delimiter for the summary statistics file with custom format.
  --gwas-sample-size GWAS_SAMPLE_SIZE
                        The overall sample size for the GWAS study. This must be provided if the sample size per-SNP is not in the summary statistics file.
  --validation-bed VALIDATION_BED
                        The BED files containing the genotype data for the validation set. You may use a wildcard here (e.g. "data/chr_*.bed")
  --validation-pheno VALIDATION_PHENO
                        A tab-separated file containing the phenotype for the validation set. The expected format is: FID IID phenotype (no header)
  --validation-keep VALIDATION_KEEP
                        A plink-style keep file to select a subset of individuals for the validation set.
  --validation-ld-panel VALIDATION_LD_PANEL
                        The path to the directory where the LD matrices for the validation set are stored. Can be a wildcard of the form ld/chr_*
  --validation-sumstats VALIDATION_SUMSTATS_PATH
                        The summary statistics directory or file for the validation set. Can be a wildcard of the form sumstats/chr_*
  --validation-sumstats-format {fastgwa,plink,ssf,plink2,custom,cojo,saige,magenpy,plink1.9,gwas-ssf,gwascatalog}
                        The format for the summary statistics file(s) for the validation set.
  -m {VIPRSMix,VIPRS}, --model {VIPRSMix,VIPRS}
                        The type of PRS model to fit to the GWAS data
  --float-precision {float32,float64}
                        The float precision to use when fitting the model.
  --use-symmetric-ld    Use the symmetric form of the LD matrix when fitting the model.
  --n-components N_COMPONENTS
                        The number of non-null Gaussian mixture components to use with the VIPRSMix model (i.e. excluding the spike component).
  --h2-est H2_EST       The estimated heritability of the trait. If available, this value can be used for parameter initialization or hyperparameter grid search.
  --h2-se H2_SE         The standard error for the heritability estimate for the trait. If available, this value can be used for parameter initialization or
                        hyperparameter grid search.
  --hyp-search {GS,EM,BMA,BO}
                        The strategy for tuning the hyperparameters of the model. Options are EM (Expectation-Maximization), GS (Grid search), BO (Bayesian
                        Optimization), and BMA (Bayesian Model Averaging).
  --grid-metric {validation,ELBO,pseudo_validation}
                        The metric for selecting best performing model in grid search.
  --pi-grid PI_GRID     A comma-separated grid values for the hyperparameter pi (see also --pi-steps).
  --pi-steps PI_STEPS   The number of steps for the (default) pi grid. This will create an equidistant grid between 1/M and (M-1)/M on a log10 scale, where M is
                        the number of SNPs.
  --sigma-epsilon-grid SIGMA_EPSILON_GRID
                        A comma-separated grid values for the hyperparameter sigma_epsilon (see also --sigma-epsilon-steps).
  --sigma-epsilon-steps SIGMA_EPSILON_STEPS
                        The number of steps (unique values) for the sigma_epsilon grid.
  --genomewide          Fit all chromosomes jointly
  --backend {xarray,plink}
                        The backend software used for computations on the genotype matrix.
  --n-jobs N_JOBS       The number of processes to launch for the hyperparameter search (default is 1, but we recommend increasing this depending on system
                        capacity).
  --threads THREADS     The number of threads to use in the E-Step of VIPRS.
  --output-profiler-metrics
                        Output the profiler metrics that measure runtime, memory usage, etc.
  --seed SEED           The random seed to use for the random number generator.
```