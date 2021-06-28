# cython: linetrace=False
# cython: profile=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True

cimport cython
import pandas as pd
import numpy as np
cimport numpy as np

cdef class PRSModel:

    def __init__(self, gdl):

        self.N = gdl.sample_size
        self.M = gdl.M
        self.gdl = gdl  # An instance of GWASDataLoader
        self.shapes = self.gdl.shapes

        # Inferred model parameters:
        self.pip = None  # Posterior inclusion probability
        self.inf_beta = None  # Inferred beta

    cpdef fit(self):
        raise NotImplementedError

    cpdef get_proportion_causal(self):
        return None

    cpdef get_heritability(self):
        return None

    cpdef get_pip(self):
        return self.pip

    cpdef get_inf_beta(self):
        return self.inf_beta

    cpdef predict(self, gdl=None):

        if self.inf_beta is None:
            raise Exception("Inferred betas are not set. Call `.fit()` first.")

        if gdl is None:
            gdl = self.gdl

        prs = np.zeros(gdl.N, dtype=float)

        for c in gdl.genotypes:
            prs += np.dot(gdl.genotypes[c]['G'], self.inf_beta[c])

        return prs

    cpdef read_inferred_params(self, f_names):

        if isinstance(f_names, str):
            f_names = [f_names]

        eff_table = []

        for f_name in f_names:
            eff_table.append(pd.read_csv(f_name, sep="\t"))

        eff_table = pd.concat(eff_table)

        self.pip = {}
        self.inf_beta = {}

        for c, c_size in self.shapes.items():

            c_df = pd.DataFrame({'CHR': c,
                                 'SNP': self.gdl.genotypes[c]['G'].variant.values})
            c_df = c_df.merge(eff_table, how='left').fillna(0.)

            self.pip[c] = c_df['PIP'].values
            self.inf_beta[c] = c_df['BETA'].values

    cpdef write_inferred_params(self, f_name):

        dfs = []

        snps = self.gdl.snps
        ref_allel = self.gdl.ref_alleles
        alt_allel = self.gdl.alt_alleles

        for c, betas in self.inf_beta.items():
            dfs.append(
                pd.DataFrame({'CHR': c,
                              'SNP': snps[c],
                              'A1': alt_allel[c],
                              'A2': ref_allel[c],
                              'PIP': self.pip[c],
                              'BETA': betas})
            )

        dfs = pd.concat(dfs)
        dfs.to_csv(f_name, sep="\t", index=False)
