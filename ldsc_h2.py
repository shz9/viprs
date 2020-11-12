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


class LDSCHeritability(object):

    def __init__(self,
                 gdl,
                 ld_score_dir="data/1000G_EUR_Phase3_baseline",
                 w_ld_score_dir="data/1000G_Phase3_weights_hm3_no_MHC",
                 univariate=False):

        self.gdl = gdl

        chrom_to_read = [v['CHR'] for k, v in self.gdl.genotypes.items()]

        self.ld_scores, self.M = read_ld_scores(ld_score_dir, w_ld_score_dir, chrom=chrom_to_read)
        self.ld_score_colnames = [c for c in self.ld_scores.columns if c[-2:] == 'L2' and c != 'w_baseL2']

        self.sumstats_table = gdl.to_sumstats_table()

        self.univariate = univariate
        self.heritability, self.heritability_se = None, None

    def fit(self):

        ss_table = self.gdl.to_sumstats_table()
        ss_table['CHISQ'] = ss_table['Z']**2

        m_ss_df = pd.merge(self.ld_scores, ss_table, on='SNP')

        if self.univariate:
            reg = Hsq(m_ss_df[['CHISQ']].values,
                      m_ss_df[self.ld_score_colnames[:1]].values,
                      m_ss_df[['w_baseL2']].values,
                      m_ss_df[['N']].values,
                      self.M[:1, :1],
                      old_weights=True)
        else:
            reg = Hsq(m_ss_df[['CHISQ']].values,
                      m_ss_df[self.ld_score_colnames].values,
                      m_ss_df[['w_baseL2']].values,
                      m_ss_df[['N']].values,
                      self.M,
                      old_weights=True)

        self.heritability = reg.tot
        self.heritability_se = reg.tot_se

    def get_heritability(self):
        return self.heritability
