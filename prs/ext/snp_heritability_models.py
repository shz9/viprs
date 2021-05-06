import pandas as pd
#from external.ldsc.ldscore.regressions import Hsq
from ..utils import delete_temp_files, run_shell_script


def read_ld_scores(ld_score_prefix, w_ld_score_prefix, chrom=None, count_file='M_5_50'):

    if chrom is None:
        chrom = range(1, 23)

    ld_df = []
    M_tot = None

    for c in chrom:

        try:
            ref_df = pd.read_csv(ld_score_prefix + f'.{c}.l2.ldscore.gz', sep="\t")
            M = pd.read_csv(ld_score_prefix + f'.{c}.l2.{count_file}',
                            header=None, sep="\t").values

            ldc_cols = [c for c in ref_df.columns if c[-2:] == 'L2']
            m_ref_df = ref_df[['CHR', 'SNP'] + ldc_cols]

            # The LD Score Weights file:
            w_df = pd.read_csv(w_ld_score_prefix + f'.{c}.l2.ldscore.gz', sep="\t")
        except FileNotFoundError:
            continue

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

    if len(ld_df) < 1:
        raise Exception(f"Could not find the LD Scores at {ld_score_prefix}")

    return pd.concat(ld_df), M_tot


class LDSCHeritability(object):

    def __init__(self,
                 gdl,
                 ld_window_cm=1.,
                 ref_ld_scores=None,
                 w_ld_scores=None,
                 univariate=True):

        self.gdl = gdl
        self.ld_window_cm = ld_window_cm
        self.ref_ld_scores = ref_ld_scores
        self.w_ld_scores = w_ld_scores
        self.univariate = univariate

        self.ld_scores = None
        self.M = None
        self.ld_score_colnames = None

        self.get_ld_scores()

        self.heritability = None
        self.heritability_se = None

    def get_ld_scores(self):

        if self.ref_ld_scores is None:

            for bf, chr in zip(self.gdl.bed_files, self.gdl.chromosomes):

                ldsc_cmd = f"""
                        python ldsc/ldsc.py \
                        --bfile {bf} \
                        --yes-really \
                        --l2 \
                        --ld-wind-cm {self.ld_window_cm} \
                        --out temp/ldsc/ldscores/baseline.{chr}
                """

                run_shell_script(ldsc_cmd)

            self.ref_ld_scores = f"temp/ldsc/ldscores/baseline"
            self.w_ld_scores = f"temp/ldsc/ldscores/baseline"

        self.ld_scores, self.M = read_ld_scores(self.ref_ld_scores, self.w_ld_scores)
        self.ld_score_colnames = [c for c in self.ld_scores.columns if c[-2:] == 'L2' and c != 'w_baseL2']

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

        return self

    def get_heritability(self):
        return self.heritability


class HERegression(object):

    def __init__(self,
                 gdl):

        self.gdl = gdl
        self.heritability = None

        self.compute_grm()

    def compute_grm(self):
        gcta_cmd = f"""
            ../external/gcta/gcta64  --bfile {self.gdl.bed_files[22].replace('.bed', '')}  \
            --autosome  --make-grm  --out temp/HEReg/grm
        """
        run_shell_script(gcta_cmd)

    def fit(self):

        phen_table = self.gdl.to_phenotype_table()

        phen_table.to_csv(f"temp/HEReg/{self.gdl.phenotype_id}.phen", index=False, sep=" ")

        gcta_cmd = f"""
            ../external/gcta/gcta64 --HEreg \
            --grm temp/HEReg/grm \
            --pheno temp/HEReg/{self.gdl.phenotype_id}.phen \
            --out temp/HEReg/{self.gdl.phenotype_id}
        """

        run_shell_script(gcta_cmd)

        he_res = pd.read_csv(f"temp/HEReg/{self.gdl.phenotype_id}.HEreg",
                             skiprows=1, nrows=2, sep="\s+", index_col=0)
        self.heritability = he_res.loc['V(G)/Vp', 'Estimate']

        delete_temp_files(f"temp/HEReg/{self.gdl.phenotype_id}")

        return self

    def get_heritability(self):
        return self.heritability
