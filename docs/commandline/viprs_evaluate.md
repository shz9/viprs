Evaluate Predictive Performance of PRS (`viprs_evaluate`)
---

The `viprs_evaluate` script supports two evaluation modes:

* **Individual-level evaluation** compares computed scores (`--prs-file`) with observed phenotypes
  (`--phenotype-file`). Continuous and binary phenotype metrics are supported, with optional covariate adjustment.
* **Summary-statistics evaluation** compares inferred effects from VIPRS `.fit` files (`--fit-files`) with
  independent validation/test GWAS summary statistics (`--sumstats`) using an LD reference (`--ld-dir`). This
  pseudo-evaluation mode computes `Pseudo_Pearson_R` and `Pseudo_R2` using standardized marginal effects and LD.

Choose exactly one mode. Individual-level inputs cannot be combined with
`--sumstats`, `--fit-files`, or `--ld-dir` in the same invocation.

## Individual-level evaluation

This mode requires scores and observed phenotypes for overlapping individuals. It
does not require genotype data or an LD panel.

### Score file

`--prs-file` must be a whitespace-delimited table with a header and the columns
`FID`, `IID`, and `PRS`. This is the format written by [`viprs_score`](viprs_score.md):

```text
FID    IID    PRS
F1     I1     0.0182
F2     I2    -0.0047
```

### Phenotype file

`--phenotype-file` expects the first two columns to be `FID` and `IID`. By default,
the phenotype is read from the third column (zero-based index `2`). Use
`--phenotype-col` when it is located elsewhere. A headerless file is the simplest
form:

```text
F1    I1    172.4
F2    I2    168.1
```

A header with the literal names `FID`, `IID`, and `phenotype` is also accepted.
Missing phenotypes are dropped. The score and phenotype rows need not be in the same
order: samples are joined using both IDs, and only their intersection is evaluated.

By default, `--phenotype-likelihood infer` recognizes quantitative traits and binary
traits encoded as `0`/`1` or PLINK-style `1`/`2`. You can set the likelihood
explicitly with `--phenotype-likelihood gaussian` or
`--phenotype-likelihood binomial`.

### Optional covariates and sample filtering

`--covariates-file` accepts a whitespace-delimited table whose first two columns are
`FID` and `IID` and whose remaining columns are covariates. It may have a header; in
a headerless file, the covariates are assigned generated names. Samples without a
row in every supplied input are excluded by the ID joins.

Use `--keep` to evaluate a subset of samples. The keep file must be headerless and
contain either one `IID` column or two tab-delimited `FID IID` columns. Covariates
are used by adjusted metrics such as `Incremental_R2`,
`R2_residualized_target`, and `Partial_Correlation`.

### Example

```bash
viprs_evaluate \
    --prs-file output/scores.prs \
    --phenotype-file data/phenotypes.txt \
    --metrics Pearson_R R2 \
    --output-file output/test_performance
```

## Summary-statistics (pseudo) evaluation

This mode estimates predictive accuracy without individual-level phenotypes. It
requires all three of the following inputs:

1. `--fit-files`: one genome-wide VIPRS effect table or a quoted wildcard matching
   the chromosome-specific pieces of one model. Each table must contain `CHR`,
   `SNP`, `A1`, `A2`, and `BETA`. Multiple effect columns named `BETA_0`, `BETA_1`,
   and so on are also supported.
2. `--sumstats`: independent validation or test GWAS summary statistics. Use the
   correct `--sumstats-format`; custom inputs require
   `--custom-sumstats-mapper` and may set `--custom-sumstats-sep`. See the
   [`viprs_fit` input guide](viprs_fit.md#summary-statistics-input) for the required
   variant, allele, sample-size, and association columns.
3. `--ld-dir`: an LD reference covering the same population and variants as the
   evaluation summary statistics. This can be a local Zarr directory/wildcard or a
   cloud path such as `hf://datasets/shz9/ukb-ld/EUR/chr_*.zip` when
   `huggingface_hub` is installed.

The fit table, summary statistics, and LD reference are harmonized by variant and
alleles. They should use compatible variant identifiers, genome builds, and allele
definitions. Use summary statistics from samples independent of the training GWAS;
otherwise, the estimated out-of-sample accuracy may be optimistic.

The evaluation summary statistics must contain per-variant sample sizes (`N`). If
they do not, provide their study-wide GWAS sample size with
`--gwas-sample-size`. This value describes the evaluation GWAS, not the training
sample used to fit the PRS.

### Example

```bash
viprs_evaluate \
    --sumstats data/validation.fastgwa \
    --sumstats-format fastgwa \
    --fit-files 'output/viprs_fit/*.fit.gz' \
    --ld-dir 'data/ld/chr_*' \
    --output-file output/pseudo_performance
```

Summary-statistics mode supports `Pseudo_Pearson_R` and `Pseudo_R2`. The former
estimates the correlation between the PRS and phenotype from the independent GWAS;
the latter is its squared value. Select either or both with, for example,
`--metrics Pseudo_Pearson_R Pseudo_R2`.

## Output and metric selection

Both modes write a one-row, tab-separated `<output-file>.eval` file and a `<output-file>.log` file. If a `.fit`
table contains several inferred-effect columns (`BETA_0`, `BETA_1`, ...), the output metrics are numbered in the
same order (for example, `Pseudo_R2_0`, `Pseudo_R2_1`, ...).

Individual-level mode supports continuous metrics such as `Pearson_R`, `Spearman_R`,
`MSE`, and `R2`, along with covariate-adjusted metrics, and binary metrics such as
`AUROC`, `AUPRC`, `Avg_Precision`, `F1_Score`, and several pseudo-R-squared measures.
Pass the desired names after `--metrics`. Supplying a short explicit list is useful
when building a stable evaluation pipeline.

## Command reference

A full listing of the options available for the `viprs_evaluate` script can be found by running the
following command in your terminal:

```bash
viprs_evaluate -h
```

Which outputs the following help message:

```text

          **********************************************
                     _____
             ___   _____(_)________ ________________
             __ | / /__  / ___  __ \__  ___/__  ___/
             __ |/ / _  /  __  /_/ /_  /    _(__  )
             _____/  /_/   _  .___/ /_/     /____/
                           /_/
          Variational Inference of Polygenic Risk Scores
           Version: 0.1.6 | Release date: October 2026
            Author: Shadi Zabad, University of Oxford
          **********************************************
          < Evaluate Prediction Accuracy of PRS Models >

usage: viprs_evaluate [-h] [--prs-file PRS_FILE] [--phenotype-file PHENO_FILE] [--phenotype-col PHENO_COL]
                      [--phenotype-likelihood {binomial,infer,gaussian}] [--keep KEEP]
                      [--covariates-file COVARIATES_FILE] [-s SUMSTATS_PATH] [-f FIT_FILES] [-l LD_DIR]
                      [--sumstats-format {custom,plink1.9,fastgwa,plink2,cojo,gwas-ssf,gwascatalog,magenpy,ssf,saige,plink}]
                      [--custom-sumstats-mapper CUSTOM_SUMSTATS_MAPPER] [--custom-sumstats-sep CUSTOM_SUMSTATS_SEP]
                      [--gwas-sample-size GWAS_SAMPLE_SIZE] [--temp-dir TEMP_DIR] [--threads THREADS] --output-file
                      OUTPUT_FILE [--metrics METRICS [METRICS ...]] [--log-level {INFO,WARNING,ERROR,DEBUG,CRITICAL}]

Commandline arguments for evaluating polygenic scores

options:
  -h, --help            show this help message and exit
  --prs-file PRS_FILE   The path to the PRS file (expected format: FID IID PRS, tab-separated).
  --phenotype-file PHENO_FILE
                        The path to the phenotype file. The expected format is: FID IID phenotype (no header), tab-
                        separated.
  --phenotype-col PHENO_COL
                        The column index for the phenotype in the phenotype file (0-based index).
  --phenotype-likelihood {binomial,infer,gaussian}
                        The phenotype likelihood ("gaussian" for continuous, "binomial" for case-control). If not set,
                        it will be inferred automatically from the phenotype file.
  --keep KEEP           A plink-style keep file to select a subset of individuals for the evaluation.
  --covariates-file COVARIATES_FILE
                        A file with covariates for the samples included in the analysis. This tab-separated file
                        should not have a header and the first two columns should be the FID and IID of the samples.
  -s SUMSTATS_PATH, --sumstats SUMSTATS_PATH, --sumstats-file SUMSTATS_PATH
                        The validation/test summary statistics file(s). This can be a file, directory, or wildcard
                        matching multiple files.
  -f FIT_FILES, --fit-files FIT_FILES
                        The VIPRS .fit file(s) containing inferred effect sizes. This can be a file, directory, or
                        wildcard matching per-chromosome files.
  -l LD_DIR, --ld-dir LD_DIR
                        The path to LD matrices for the validation/test dataset. This can be a directory or wildcard
                        of the form ld/chr_*.
  --sumstats-format {custom,plink1.9,fastgwa,plink2,cojo,gwas-ssf,gwascatalog,magenpy,ssf,saige,plink}
                        The format of the validation/test summary statistics file(s).
  --custom-sumstats-mapper CUSTOM_SUMSTATS_MAPPER
                        Comma-separated column mappings for custom summary statistics, e.g.
                        rsid=SNP,eff_allele=A1,beta=BETA.
  --custom-sumstats-sep CUSTOM_SUMSTATS_SEP
                        The delimiter for custom-format summary statistics (default: tab).
  --gwas-sample-size GWAS_SAMPLE_SIZE
                        The validation/test GWAS sample size. Required when it is absent from the summary statistics.
  --temp-dir TEMP_DIR   The temporary directory where intermediate files are stored.
  --threads THREADS     The number of threads to use when reading summary statistics and LD data.
  --output-file OUTPUT_FILE
                        The output file where evaluation metrics are stored (with no extension).
  --metrics METRICS [METRICS ...]
                        The evaluation metrics to compute. Summary-statistics evaluation supports Pseudo_Pearson_R and
                        Pseudo_R2. By default, all metrics relevant to the selected evaluation mode are computed.
  --log-level {INFO,WARNING,ERROR,DEBUG,CRITICAL}
                        The logging level for the console output.
```
