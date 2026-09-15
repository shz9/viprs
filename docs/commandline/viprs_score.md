Compute Polygenic Scores using inferred variant effect sizes (`viprs_score`)
---

The `viprs_score` script is used to compute the polygenic risk scores (PRS) for a set of individuals 
using the estimated variant effect sizes from the `viprs_fit` script. This is the script that generates 
the PRS per individual.

## Required inputs

### Effect-size files

Pass one or more coefficient files with `-f`/`--fit-files`. The `.fit` or
`.fit.gz` output written by [`viprs_fit`](viprs_fit.md) can be used directly.
The files are whitespace-delimited tables and must contain:

| Column | Meaning |
|:-------|:--------|
| `CHR` | Chromosome containing the variant |
| `SNP` | Variant identifier, normally an rsID |
| `A1` | Effect allele for `BETA` |
| `A2` | Other/reference allele |
| `BETA` | Inferred effect used as the scoring weight |

Other columns produced by `viprs_fit`, such as `POS`, `PIP`, and `VAR_BETA`, may
remain in the table but are not used to calculate the score. When supplying
third-party weights, use one row per variant and a single `BETA` column with its
effect expressed relative to `A1`.

The input may be one genome-wide file or several files, for example one per
chromosome. Quote wildcard expressions so that `viprs_score`, rather than the shell,
receives the pattern:

```bash
--fit-files 'output/viprs_fit/chr_*.fit.gz'
```

All matched files are concatenated. A pattern should therefore identify the pieces
of one fitted model only; do not mix files from different models, traits, or folds.

### Target genotypes

`--bfile` reads PLINK 1 binary genotype data. Each dataset consists of a matching
`.bed`, `.bim`, and `.fam` trio with the same filename prefix:

```text
test.bed    # binary genotype calls
test.bim    # chromosome, variant ID, position, and alleles
test.fam    # family and individual IDs
```

You can pass either the prefix/`.bed` path for one dataset or a quoted wildcard for
chromosome-split data:

```bash
--bfile test
--bfile 'test/chr_*.bed'
```

Chromosome-split files should describe the same individuals. The `FID` and `IID`
values from the `.fam` file are carried into the score output.

### Optional sample and variant filters

`--keep` accepts a headerless, tab-delimited PLINK-style sample file containing
either one `IID` column or two `FID IID` columns. `--extract` accepts a headerless
file with one `SNP` identifier per line. These filters are applied to the target
genotypes before scoring.

## Variant alignment and allele handling

Coefficient rows and target genotypes are aligned by chromosome and `SNP`. The
alleles are then harmonized: `A1` is treated as the effect allele, and the sign of
`BETA` is changed when the target genotype encodes the two alleles in the opposite
order. The score is calculated over variants shared by the coefficient and genotype
files; variants without an available weight do not contribute.

For reliable matching, use the same variant identifiers and genome build in the fit
and target data. If variants have been renamed or lifted to another build, update the
`.bim` file or coefficient table consistently before scoring. Strand-ambiguous and
otherwise incompatible allele encodings should be resolved during data preparation.

Conceptually, the score is `PRS_i = sum_j(G_ij * BETA_j)`, where `G_ij` is
individual `i`'s effect-allele dosage and `BETA_j` is the fitted effect for variant
`j`.

## Example and output

```bash
viprs_score \
    --fit-files 'output/viprs_fit/*.fit.gz' \
    --bfile 'test/chr_*.bed' \
    --keep test_samples.keep \
    --output-file output/test_scores \
    --compress
```

`--output-file` is a prefix, not a complete filename. The command above writes
`output/test_scores.prs.gz`; without `--compress`, it writes
`output/test_scores.prs`. The output is tab-delimited and contains one row per
retained individual:

```text
FID    IID    PRS
F1     I1     0.0182
F2     I2    -0.0047
```

A companion `<output-file>.log` records the arguments and progress messages. The
`.prs` output can be passed directly to the individual-level mode of
[`viprs_evaluate`](viprs_evaluate.md).

## Command reference

A full listing of the options available for the `viprs_score` script can be found by running the 
following command in your terminal:

```bash
viprs_score -h
```

Which outputs the following help message:

```bash
          **********************************************
                     _____                              
             ___   _____(_)________ ________________    
             __ | / /__  / ___  __ \__  ___/__  ___/    
             __ |/ / _  /  __  /_/ /_  /    _(__  )     
             _____/  /_/   _  .___/ /_/     /____/      
                           /_/                          
                                                        
          Variational Inference of Polygenic Risk Scores
            Version: 0.1.4 | Release date: July 2026    
              Author: Shadi Zabad, McGill University    
          **********************************************
          < Compute Polygenic Scores for Test Samples > 

usage: viprs_score [-h] -f FIT_FILES --bfile BED_FILES --output-file OUTPUT_FILE [--temp-dir TEMP_DIR] [--keep KEEP] [--extract EXTRACT]
                   [--backend {xarray,plink,magenpy,bed-reader}] [--threads THREADS] [--compress] [--log-level {WARNING,CRITICAL,DEBUG,INFO,ERROR}]

Commandline arguments for computing polygenic scores

options:
  -h, --help            show this help message and exit
  -f FIT_FILES, --fit-files FIT_FILES
                        The path to the file(s) with the output parameter estimates from VIPRS. You may use a wildcard here if fit files are stored per-
                        chromosome (e.g. "prs/chr_*.fit")
  --bfile BED_FILES     The BED files containing the genotype data. You may use a wildcard here (e.g. "data/chr_*.bed")
  --output-file OUTPUT_FILE
                        The output file where to store the polygenic scores (with no extension).
  --temp-dir TEMP_DIR   The temporary directory where to store intermediate files.
  --keep KEEP           A plink-style keep file to select a subset of individuals for the test set.
  --extract EXTRACT     A plink-style extract file to select a subset of SNPs for scoring.
  --backend {xarray,plink,magenpy,bed-reader}
                        The backend software used for computations with the genotype matrix.
  --threads THREADS     The number of threads to use for computations.
  --compress            Compress the output file
  --log-level {WARNING,CRITICAL,DEBUG,INFO,ERROR}
                        The logging level for the console output.

```
