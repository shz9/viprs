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
import numpy as np
cimport numpy as np

cdef class PRSModel:

    def __init__(self, gdl):

        self.N = gdl.sample_size
        self.M = gdl.M
        self.gdl = gdl  # Gwas Data Loader
        self.pip = None
        self.inf_beta = None

    cpdef fit(self):
        raise NotImplementedError

    cpdef get_heritability(self):
        return None

    cpdef get_pip(self):
        return self.pip

    cpdef get_inf_beta(self):
        return self.inf_beta

    cpdef predict_phenotype(self):

        if self.inf_beta is None:
            raise Exception("Inferred betas are None. Call `.fit()` first.")

        index = self.gdl.test_idx
        prs = np.zeros_like(index, dtype=float)

        for c in self.gdl.genotypes:
            prs += np.dot(self.gdl.genotypes[c]['G'][index, :], self.inf_beta[c])

        return prs

    cpdef write_inferred_params(self, f_name):
        """
        TODO: Write a function to export the inferred parameters to file.
        :return:
        """
        pass