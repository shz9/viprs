import numpy as np
import pandas as pd
import os
import glob
from prs.utils import delete_temp_files, run_shell_script
from prs.eval.evaluation import evaluate_predictive_performance
from prs.src.PRSModel import PRSModel


class TrueBetaPRS(PRSModel):
    def __init__(self, gdl):
        super().__init__(gdl)

    def fit(self):
        self.pip = self.gdl.pis
        self.inf_beta = self.gdl.betas
        return self


class MarginalBetaPRS(PRSModel):

    def __init__(self, gdl):
        super().__init__(gdl)

    def fit(self):
        self.pip = {c: 1. - p for c, p in self.gdl.p_values.items()}
        self.inf_beta = self.gdl.beta_hats
        return self


class PLINK_PT(PRSModel):

    def __init__(self, gdl,
                 clump_pval_threshold=1.,
                 clump_r2_threshold=.1,
                 clump_kb_threshold=250,
                 pval_thresholds=(0.001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)):

        super().__init__(gdl)

        self.clump_pval_threshold = clump_pval_threshold
        self.clump_r2_threshold = clump_r2_threshold
        self.clump_kb_threshold = clump_kb_threshold
        self.pval_thresholds = pval_thresholds

        self.best_threshold = None
        self.best_r2 = None

    def fit(self):

        ss_tables = self.gdl.to_sumstats_table(per_chromosome=True)
        beta_hat = self.gdl.beta_hats
        p_vals = self.gdl.p_values

        clump_beta = beta_hat.copy()

        temp_files = []

        for bfile, ss_table, beta_key in zip(self.gdl.bed_files, ss_tables, beta_hat.keys()):

            clumped_prefix = f"temp/plink/{self.gdl.phenotype_id}_{os.path.basename(bfile)}"
            temp_files.append(clumped_prefix)

            ss_table.to_csv(f"{clumped_prefix}.sumstats", index=False, sep="\t")

            clump_cmd = f"""
                plink \
                --bfile {bfile} \
                --clump-p1 {self.clump_pval_threshold} \
                --clump-r2 {self.clump_r2_threshold} \
                --clump-kb {self.clump_kb_threshold} \
                --clump {clumped_prefix}.sumstats \
                --clump-snp-field SNP \
                --clump-field PVAL \
                --out {clumped_prefix}
            """

            run_shell_script(clump_cmd)

            retained_snps = pd.read_csv(f"{clumped_prefix}.clumped", sep="\s+")
            clump_beta[beta_key][~clump_beta[beta_key].index.isin(retained_snps['SNP'])] = 0.

        # For the remaining SNPs, find the best p-value threshold:

        for pt in self.pval_thresholds:

            self.inf_beta = {}

            for i, beta in clump_beta.items():
                new_beta = beta.copy()
                new_beta[p_vals[i] > pt] = 0.
                self.inf_beta[i] = new_beta

            if all([not self.inf_beta[i].any() for i in self.inf_beta]):
                continue

            r2 = evaluate_predictive_performance(self.predict_phenotype(test=False),
                                                 self.gdl.phenotypes[self.gdl.train_idx])['R2']

            if self.best_threshold is None:
                self.best_threshold = pt
                self.best_r2 = r2
            elif r2 > self.best_r2:
                self.best_threshold = pt
                self.best_r2 = r2

        self.inf_beta = {}

        for i, beta in clump_beta.items():
            new_beta = beta.copy()
            new_beta[p_vals[i] > self.best_threshold] = 0.
            self.inf_beta[i] = new_beta

        # Delete temporary files:
        for f_pattern in temp_files:
            delete_temp_files(f_pattern)

        return self


class LASSOSum(PRSModel):

    def __init__(self, gdl):
        super().__init__(gdl)

    def fit(self):

        ss_tables = self.gdl.to_sumstats_table()
        ss_tables = ss_tables[['SNP', 'A1', 'A2', 'BETA', 'PVAL', 'N']]

        ss_tables.to_csv(f"temp/lassosum/{self.gdl.phenotype_id}.sumstats", index=False, sep=" ")

        lassosum_cmd = f"""
            Rscript run_lassosum.R \
            temp/lassosum/{self.gdl.phenotype_id}.sumstats \
            {self.gdl.bed_files[0]} \
            {self.gdl.bed_files[0]} \
            temp/lassosum/{self.gdl.phenotype_id}
        """

        run_shell_script(lassosum_cmd)

        snp_effects = pd.read_csv(f"temp/lassosum/{self.gdl.phenotype_id}.snpEffect", sep="\s+")

        self.inf_beta = {}

        for i in self.gdl.beta_hats:
            self.inf_beta[i] = pd.Series(np.zeros_like(self.gdl.beta_hats[i]),
                                         index=self.gdl.beta_hats[i].index)
            self.inf_beta[i][snp_effects['snp']] = snp_effects['effectSize'].values

        delete_temp_files(f"temp/lassosum/{self.gdl.phenotype_id}")

        return self


class SBayesR(PRSModel):
    def __init__(self, gdl,
                 ldm=None,
                 pi=(0.95, 0.02, 0.02, 0.01),
                 gamma=(0.0, 0.01, 0.1, 1),
                 burn_in=2000,
                 chain_length=10000,
                 out_freq=100):
        super().__init__(gdl)

        self.ldm = ldm
        self.pi = pi
        self.gamma = gamma
        self.burn_in = burn_in
        self.chain_length = chain_length
        self.out_freq = out_freq
        self.heritability = None

        if self.ldm is None:
            self.create_ldm()

    def create_ldm(self):
        sbayesr_ld_cmd = f"""
            ../external/gctb_2.0/gctb --bfile {self.gdl.bed_files[22].replace('.bed', '')} \
            --make-sparse-ldm --out temp/sbayesr/ldmat
        """
        run_shell_script(sbayesr_ld_cmd)

        self.ldm = "temp/sbayesr/ldmat.ldm.sparse"

    def fit(self):

        ss_tables = self.gdl.to_sumstats_table()
        ss_tables = ss_tables[['SNP', 'A1', 'A2', 'MAF', 'BETA', 'SE', 'PVAL', 'N']]
        ss_tables.columns = ['SNP', 'A1', 'A2', 'freq', 'b', 'se', 'p', 'n']

        ss_tables.to_csv(f"temp/sbayesr/pheno.ma", index=False, sep=" ")

        sbayesr_cmd = f"""
            ../external/gctb_2.0/gctb --sbayes R \
                 --ldm {self.ldm} \
                 --pi {','.join(map(str, self.pi))} \
                 --gamma {','.join(map(str, self.gamma))} \
                 --gwas-summary temp/sbayesr/pheno.ma \
                 --chain-length {self.chain_length} \
                 --burn-in {self.burn_in} \
                 --out-freq {self.out_freq} \
                 --out temp/sbayesr/test
        """

        run_shell_script(sbayesr_cmd)

        snp_effects = pd.read_csv(f"temp/sbayesr/test.snpRes", sep="\s+")

        self.inf_beta = {}
        self.pip = {}

        for i in self.gdl.beta_hats:
            self.inf_beta[i] = pd.Series(np.zeros_like(self.gdl.beta_hats[i]),
                                         index=self.gdl.beta_hats[i].index)
            self.inf_beta[i][snp_effects['Name']] = snp_effects['A1Effect'].values

            self.pip[i] = pd.Series(np.zeros_like(self.gdl.beta_hats[i]),
                                    index=self.gdl.beta_hats[i].index)
            self.pip[i][snp_effects['Name']] = snp_effects['PIP'].values

        params = pd.read_csv(f"temp/sbayesr/test.parRes", sep="\s+", skiprows=1)
        self.heritability = params.loc['hsq', 'Mean']

        # Delete temporary files:
        delete_temp_files(f"temp/sbayesr/test")

        return self

    def get_heritability(self):
        return self.heritability


class LDPred(PRSModel):

    def __init__(self, gdl, ld_radius=2000, pi=0.1):

        super().__init__(gdl)
        self.ld_radius = ld_radius
        self.pi = pi

    def fit(self):

        ss_tables = self.gdl.to_sumstats_table()
        ss_tables = ss_tables[['CHR', 'SNP', 'POS', 'A1', 'A2', 'MAF', 'BETA', 'SE', 'PVAL', 'N']]

        ss_tables.to_csv(f"temp/LDPred/{self.gdl.phenotype_id}.sumstats", index=False, sep=" ")

        coord_cmd = f"""
            ldpred coord \
            --gf={self.gdl.bed_files[0]} \
            --ssf=temp/LDPred/{self.gdl.phenotype_id}.sumstats \
            --N={ss_tables['N'][0]} \
            --ssf-format CUSTOM \
            --rs SNP \
            --pos POS \
            --pval PVAL \
            --eff BETA \
            --reffreq MAF \
            --eff_type BLUP \
            --out=temp/LDPred/coord.hdf5
        """

        run_shell_script(coord_cmd)

        gibbs_cmd = f"""
            ldpred gibbs \
            --cf temp/LDPred/coord.hdf5 \
            --f {self.pi} \
            --ldr {self.ld_radius} \
            --ldf temp/LDPred/ldfile \
            --out temp/LDPred/{self.gdl.phenotype_id}
        """

        run_shell_script(gibbs_cmd)

        f = glob.glob(f"temp/LDPred/{self.gdl.phenotype_id}_LDpred_*.txt")[0]
        snp_effects = pd.read_csv(f, sep="\s+")

        self.inf_beta = {}
        self.pip = {}

        for i in self.gdl.beta_hats:
            self.inf_beta[i] = pd.Series(np.zeros_like(self.gdl.beta_hats[i]),
                                         index=self.gdl.beta_hats[i].index)
            self.inf_beta[i][snp_effects['sid']] = snp_effects['ldpred_beta'].values

        delete_temp_files("temp/LDPred/coord.hdf5")
        delete_temp_files(f"temp/LDPred/ldfile")
        delete_temp_files(f"temp/LDPred/{self.gdl.phenotype_id}")

        return self
