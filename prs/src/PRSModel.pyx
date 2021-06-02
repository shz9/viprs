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
            raise Exception("Inferred betas are None. Call `.fit()` first.")

        if gdl is None:
            gdl = self.gdl

        prs = np.zeros(gdl.N, dtype=float)

        for c in gdl.genotypes:
            prs += np.dot(gdl.genotypes[c]['G'], self.inf_beta[c])

        return prs

    cpdef read_inferred_params(self, f_name):

        df = pd.read_csv(f_name, sep="\t")
        self.pip = {}
        self.inf_beta = {}

        # TODO: make sure that the SNP order in the
        # read file is the same as in the genotype matrix...
        for c, c_size in self.shapes.items():
            self.pip[c] = df.loc[df['CHR'] == c, 'PIP']
            self.inf_beta[c] = df.loc[df['CHR'] == c, 'BETA']

    cpdef write_inferred_params(self, f_name):

        dfs = []
        for c, betas in self.inf_beta.items():
            dfs.append(
                pd.DataFrame({'CHR': c,
                              'SNP': self.gdl.snps[c],
                              'PIP': self.pip[c],
                              'BETA': betas})
            )

        dfs = pd.concat(dfs)
        dfs.to_csv(f_name, sep="\t", index=False)
