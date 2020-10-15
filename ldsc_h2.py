import os
import pandas as pd
import numpy as np
from ldsc.ldscore.regressions import Hsq


def read_ld_scores(ld_score_dir, w_ld_score_dir, chrom=None, count_file='M_5_50'):

    if chrom is None:
        chrom = range(1, 23)

    ld_df = []
    M_tot = None

    for c in chrom:

        ref_df = pd.read_csv(os.path.join(ld_score_dir, f'baseline.{c}.l2.ldscore.gz'), sep="\t")
        M = pd.read_csv(os.path.join(ld_score_dir, f'baseline.{c}.l2.{count_file}'),
                        header=None, sep="\t").values

        ldc_cols = [c for c in ref_df.columns if c[-2:] == 'L2']
        m_ref_df = ref_df[['CHR', 'SNP'] + ldc_cols]

        # The LD Score Weights file:
        w_df = pd.read_csv(os.path.join(w_ld_score_dir, f'weights.hm3_noMHC.{c}.l2.ldscore.gz'), sep="\t")

        try:
            w_df = w_df[['SNP', 'baseL2']]
            w_df = w_df.rename(columns={'baseL2': 'w_baseL2'})
        except Exception as e:
            w_df = w_df[['SNP', 'L2']]
            w_df = w_df.rename(columns={'L2': 'w_baseL2'})

        ld_df.append(pd.merge(m_ref_df, w_df, on='SNP'))

        if M_tot is None:
            M_tot = M
        else:
            M_tot += M

    return pd.concat(ld_df), M_tot


def perform_ldsc_regression(gdl,
                            ld_score_dir="data/1000G_EUR_Phase3_baseline",
                            w_ld_score_dir="data/1000G_Phase3_weights_hm3_no_MHC"):

    chrom_to_read = [v['CHR'] for k, v in gdl.genotypes.items()]

    ld_scores, M = read_ld_scores(ld_score_dir, w_ld_score_dir, chrom=chrom_to_read)
    ld_score_colnames = [c for c in ld_scores.columns if c[-2:] == 'L2' and c != 'w_baseL2']

    sum_stats = pd.DataFrame({
        'SNP': np.vstack([v['G'].snp.values for k, v in gdl.genotypes.items()])[0],
        'CHISQ': np.vstack([v.compute()**2 for c, v in gdl.get_z_scores().items()])[0]
    })
    sum_stats['N'] = gdl.N

    m_ss_df = pd.merge(ld_scores, sum_stats)

    reg = Hsq(m_ss_df[['CHISQ']].values,
              m_ss_df[ld_score_colnames].values,
              m_ss_df[['w_baseL2']].values,
              m_ss_df[['N']].values,
              M,
              old_weights=True)

    univar_reg = Hsq(m_ss_df[['CHISQ']].values,
                     m_ss_df[ld_score_colnames[:1]].values,
                     m_ss_df[['w_baseL2']].values,
                     m_ss_df[['N']].values,
                     M[:1, :1],
                     old_weights=True)

    return {
        'S-LDSC': {'Estimate': reg.tot, 'SE': reg.tot_se},
        'LDSC': {'Estimate': univar_reg.tot, 'SE': univar_reg.tot_se}
    }

