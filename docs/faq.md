# Frequently asked questions

## Summary statistics

### What columns does `viprs_fit` require?

The summary statistics need four kinds of information:

- A variant identifier: `SNP` (usually an rsID), or the pair `CHR` and `POS`.
- Sample size: a per-variant `N` column, or a study-wide sample size passed with
  `--gwas-sample-size`.
- Alleles: `A1` for the effect/tested allele and `A2` for the reference or other
  allele.
- A signed measure from which the standardized marginal effect can be obtained:
  `Z`, `BETA` with `SE`, or `PVAL`/`CHISQ` accompanied by a signed statistic such
  as `BETA`, `Z`, or `OR`.

A p-value on its own does not encode effect direction. See the
[`viprs_fit` summary-statistics guide](commandline/viprs_fit.md#summary-statistics-input)
for the canonical names and a custom-format example.

### How do PLINK 2 columns map to VIPRS columns?

The common mappings are `#CHROM` to `CHR`, `ID` to `SNP`, `OBS_CT` to `N`,
`T_STAT`/`Z_STAT` to `Z`, and `P` to `PVAL`. In a simple biallelic table where the
alternate allele was tested, `ALT` maps to `A1` and `REF` maps to `A2`.

For native PLINK 2 output, use `--sumstats-format plink2`. The parser uses PLINK 2's
`A1` column as the tested/effect allele and infers `A2` from `REF` and `ALT` (or
`ALT1`), which also handles rows where the tested allele is not `ALT`.

### Does VIPRS use `A1` as the effect allele?

Yes. `BETA`, `Z`, and other signed association statistics must be expressed with
respect to `A1`. `A2` is the reference or other allele. During harmonization, VIPRS
checks the alleles against the LD reference and changes the sign of signed statistics
when it detects an allele swap.

### How are variants matched between summary statistics and the LD reference?

The underlying [`merge_snp_tables`](https://github.com/shz9/magenpy/blob/master/magenpy/utils/model_utils.py)
function first matches on `SNP` when both tables contain variant IDs. Otherwise, it
matches on the `CHR`/`POS` pair. It then checks `A1` and `A2`, corrects valid allele
swaps, and excludes incompatible variants.

### Which genome build should I use?

The precomputed LD panels currently distributed for VIPRS use GRCh37/hg19 positions.
Summary statistics matched by `CHR` and `POS` must therefore use GRCh37/hg19 as well.
Lift over coordinates from another build before fitting. Even when matching by rsID,
make sure the variant definitions and alleles agree with the LD panel.

## Models

### What is VIPRSMix, and how do I select it?

VIPRSMix uses a sparse mixture prior with several non-null Gaussian components of
different scales and a spike at zero, conceptually similar to SBayesR. The
`--n-components` value counts the non-null components only. The default value of one
fits VIPRS, while any value greater than one automatically fits VIPRSMix; for example,
`--n-components 4` selects a four-component VIPRSMix model.

The Supplementary Material of [Zabad et al. (2023)](citation.md) reports experiments
with this prior.
