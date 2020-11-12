import numpy as np
import pandas as pd
import os
import subprocess
from evaluation import evaluate_predictive_performance


class PRSModel(object):

    def __init__(self, gdl):

        self.gdl = gdl  # Gwas Data Loader
        self.inf_beta = None

    def fit(self):
        raise NotImplementedError

    def get_heritability(self):
        raise NotImplementedError

    def predict_phenotype(self, test=True):

        if self.inf_beta is None:
            raise Exception("Inferred betas are None. Call `.fit()` first.")

        if test:
            index = self.gdl.test_idx
        else:
            index = self.gdl.train_idx

        prs = np.zeros_like(index, dtype=float)

        for c in self.gdl.genotypes:
            prs += np.dot(self.gdl.genotypes[c]['G'][index, :], self.inf_beta[c])

        return prs


class MarginalBetaPRS(PRSModel):

    def __init__(self, gdl):
        super().__init__(gdl)

    def fit(self):
        self.inf_beta = self.gdl.get_beta_hat()

    def get_heritability(self):
        return None


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
        beta_hat = self.gdl.get_beta_hat()
        p_vals = self.gdl.get_p_values()

        clump_beta = beta_hat.copy()

        for bfile, ss_table, beta_key in zip(self.gdl.bed_files, ss_tables, beta_hat.keys()):

            clumped_prefix = f"{self.gdl.phenotype_id}_{os.path.basename(bfile)}"
            ss_table.to_csv(f"temp/{clumped_prefix}.sumstats", index=False, sep="\t")

            clump_cmd = f"""
                plink \
                --bfile {bfile} \
                --clump-p1 {self.clump_pval_threshold} \
                --clump-r2 {self.clump_r2_threshold} \
                --clump-kb {self.clump_kb_threshold} \
                --clump temp/{clumped_prefix}.sumstats \
                --clump-snp-field SNP \
                --clump-field PVAL \
                --out temp/{clumped_prefix}
            """

            result = subprocess.run(clump_cmd, shell=True, capture_output=True)

            if result.stderr:
                raise subprocess.CalledProcessError(
                    returncode=result.returncode,
                    cmd=result.args,
                    stderr=result.stderr
                )

            retained_snps = pd.read_csv(f"temp/{clumped_prefix}.clumped", sep="\s+")

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

            r2 = evaluate_predictive_performance(self.gdl.phenotypes[self.gdl.train_idx],
                                                 self.predict_phenotype(test=False))['R2']

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

    def get_heritability(self):
        return None


class LASSOSum(PRSModel):

    def __init__(self, gdl):
        super().__init__(gdl)

    def fit(self):
        raise NotImplementedError

    def get_heritability(self):
        return None


class SBayesR(PRSModel):
    def __init__(self, gdl,
                 pi=(0.95, 0.02, 0.02, 0.01),
                 gamma=(0.0, 0.01, 0.1, 1),
                 burn_in=2000,
                 chain_length=10000,
                 out_freq=100):
        super().__init__(gdl)

        self.pi = pi
        self.gamma = gamma
        self.burn_in = burn_in
        self.chain_length = chain_length
        self.out_freq = out_freq
        self.heritability = None

    def fit(self):

        ss_tables = self.gdl.to_sumstats_table()
        ss_tables = ss_tables[['SNP', 'A1', 'A2', 'MAF', 'BETA', 'SE', 'PVAL', 'N']]
        ss_tables.columns = ['SNP', 'A1', 'A2', 'freq', 'b', 'se', 'p', 'n']

        ss_tables.to_csv(f"temp/sbayesr/{self.gdl.phenotype_id}.ma")

        sbayesr_cmd = f"""
            gctb --sbayes R \
                 --ldm ../ldm/sparse/chr22/1000G_eur_chr22.ldm.sparse \
                 --pi {','.join(map(str, self.pi))} \
                 --gamma {','.join(map(str, self.gamma))} \
                 --gwas-summary temp/sbayesr/{self.gdl.phenotype_id}.ma \
                 --chain-length {self.chain_length} \
                 --burn-in {self.burn_in} \
                 --out-freq {self.out_freq} \
                 --out temp/sbayesr/{self.gdl.phenotype_id}
        """

        result = subprocess.run(sbayesr_cmd, shell=True, capture_output=True)

        if result.stderr:
            raise subprocess.CalledProcessError(
                returncode=result.returncode,
                cmd=result.args,
                stderr=result.stderr
            )

        snp_effects = pd.read_csv(f"temp/sbayesr/{self.gdl.phenotype_id}.snpRes", sep="\s+")

        params = pd.read_csv(f"temp/sbayesr/{self.gdl.phenotype_id}.parRes", sep="\s+", skiprows=1)
        self.heritability = params.loc['hsq', 'Mean']

    def get_heritability(self):
        return self.heritability
