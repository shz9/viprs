from magenpy import GWADataLoader
import numpy as np


def _match_variant_stats(test_gdl, prs_beta_table):
    """
    Match the standardized marginal betas from the validation set to the inferred PRS effect sizes.
    This function takes a `GWADataLoader` object from the validation set (with matched LD matrices and
    GWAS summary statistics) and a PRS table object and returns a tuple of three arrays:

    #. The standardized marginal betas from the validation set
    #. The inferred PRS effect sizes
    #. The LD-weighted PRS effect sizes (q)

    :param test_gdl: A `GWADataLoader` object from the validation or test set.
    :param prs_beta_table: A pandas DataFrame with the PRS effect sizes. Must contain
    the columns: CHR, SNP, A1, A2, BETA.

    :return: A tuple of three arrays: (1) The standardized marginal betas from the validation set,
    (2) The inferred PRS effect sizes, (3) The LD-weighted PRS effect sizes (q).
    """

    # Sanity checks:
    assert isinstance(test_gdl, GWADataLoader), "The test/validation set must be an instance of GWADataLoader."
    assert test_gdl.ld is not None, "The test/validation set must have LD matrices initialized."
    assert test_gdl.sumstats_table is not None, "The test/validation set must have summary statistics initialized."

    from magenpy.utils.model_utils import merge_snp_tables

    validation_tab = test_gdl.to_snp_table(col_subset=['CHR', 'SNP', 'A1', 'A2', 'STD_BETA'],
                                           per_chromosome=True)

    required_cols = ['CHR', 'SNP', 'A1', 'A2']
    for col in required_cols:
        assert col in prs_beta_table.columns, f"The PRS effect sizes table must contain a column named {col}."

    validation_beta = []
    prs_beta = []
    ld_weighted_beta = []

    if 'BETA' in prs_beta_table.columns:
        beta_cols = ['BETA']
    else:
        beta_cols = [col for col in prs_beta_table.columns if 'BETA' in col and 'VAR' not in col]
        assert len(beta_cols) > 0, "The PRS effect sizes table must contain a column named BETA or BETA_0, BETA_1, etc."

    per_chrom_prs_tables = dict(tuple(prs_beta_table.groupby('CHR')))

    for chrom, tab in validation_tab.items():

        if chrom not in per_chrom_prs_tables:
            continue

        c_df = merge_snp_tables(tab,
                                per_chrom_prs_tables[chrom],
                                how='left',
                                signed_statistics=beta_cols)

        test_gdl.ld[chrom].load(dtype=np.float32)

        validation_beta.append(tab['STD_BETA'].values)
        prs_beta.append(c_df[beta_cols].fillna(0.).values)
        ld_weighted_beta.append(test_gdl.ld[chrom].dot(prs_beta[-1]))

        test_gdl.ld[chrom].release()

    return (np.concatenate(validation_beta),
            np.concatenate(prs_beta, axis=0),
            np.concatenate(ld_weighted_beta))


def pseudo_r2(test_gdl, prs_beta_table):
    """
    Compute the R-Squared metric (proportion of variance explained) for a given
    PRS using standardized marginal betas from an independent test set.
    Here, we follow the pseudo-validation procedures outlined in Mak et al. (2017) and
    Yang and Zhou (2020), where the proportion of phenotypic variance explained by the PRS
    in an independent validation cohort can be approximated with:

    R2(PRS, y) ~= 2*r'b - b'Sb

    Where `r` is the standardized marginal beta from a validation/test set,
    `b` is the posterior mean for the effect size of each variant and `S` is the LD matrix.

    :param test_gdl: An instance of `GWADataLoader` with the summary statistics table initialized.
    :param prs_beta_table: A pandas DataFrame with the PRS effect sizes. Must contain
    the columns: CHR, SNP, A1, A2, BETA.
    """

    std_beta, prs_beta, q = _match_variant_stats(test_gdl, prs_beta_table)

    rb = np.sum((prs_beta.T * std_beta).T, axis=0)
    bsb = np.sum(prs_beta*q, axis=0)

    return 2*rb - bsb


def pseudo_pearson_r(test_gdl, prs_beta_table):
    """
    Perform pseudo-validation of the inferred effect sizes by comparing them to
    standardized marginal betas from an independent validation set. Here, we follow the pseudo-validation
    procedures outlined in Mak et al. (2017) and Yang and Zhou (2020), where
    the correlation between the PRS and the phenotype in an independent validation
    cohort can be approximated with:

    Corr(PRS, y) ~= r'b / sqrt(b'Sb)

    Where `r` is the standardized marginal beta from a validation set,
    `b` is the posterior mean for the effect size of each variant and `S` is the LD matrix.

    :param test_gdl: An instance of `GWADataLoader` with the summary statistics table initialized.
    :param prs_beta_table: A pandas DataFrame with the PRS effect sizes. Must contain
    the columns: CHR, SNP, A1, A2, BETA.
    """

    std_beta, prs_beta, q = _match_variant_stats(test_gdl, prs_beta_table)

    rb = np.sum((prs_beta.T * std_beta).T, axis=0)
    bsb = np.sum(prs_beta * q, axis=0)

    return rb / np.sqrt(bsb)

