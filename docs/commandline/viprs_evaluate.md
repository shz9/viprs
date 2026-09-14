Evaluate Predictive Performance of PRS (`viprs_evaluate`)
---

The `viprs_evaluate` script supports two evaluation modes:

* **Individual-level evaluation** compares computed scores (`--prs-file`) with observed phenotypes
  (`--phenotype-file`). Continuous and binary phenotype metrics are supported, with optional covariate adjustment.
* **Summary-statistics evaluation** compares inferred effects from VIPRS `.fit` files (`--fit-files`) with
  independent validation/test GWAS summary statistics (`--sumstats`) using an LD reference (`--ld-dir`). This
  pseudo-evaluation mode computes `Pseudo_Pearson_R` and `Pseudo_R2` using standardized marginal effects and LD.

## Individual-level evaluation

```bash
viprs_evaluate \
    --prs-file output/scores.prs \
    --phenotype-file data/phenotypes.txt \
    --output-file output/test_performance
```

## Summary-statistics (pseudo) evaluation

```bash
viprs_evaluate \
    --sumstats data/validation.fastgwa \
    --sumstats-format fastgwa \
    --fit-files 'output/viprs_fit/*.fit.gz' \
    --ld-dir 'data/ld/chr_*' \
    --output-file output/pseudo_performance
```

The summary statistics must contain per-variant sample sizes. If they do not, provide the overall validation GWAS
sample size with `--gwas-sample-size`. Custom summary-statistics formats are supported with
`--sumstats-format custom`, `--custom-sumstats-mapper`, and optionally `--custom-sumstats-sep`.

Both modes write a one-row, tab-separated `<output-file>.eval` file and a `<output-file>.log` file. If a `.fit`
table contains several inferred-effect columns (`BETA_0`, `BETA_1`, ...), the output metrics are numbered in the
same order (for example, `Pseudo_R2_0`, `Pseudo_R2_1`, ...).

A full listing of the options available for the `viprs_evaluate` script can be found by running the
following command in your terminal:

```bash
viprs_evaluate -h
```
