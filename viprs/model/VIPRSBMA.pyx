# cython: linetrace=False
# cython: profile=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True

from .VIPRSGrid cimport VIPRSGrid
import numpy as np
from viprs.utils.math_utils cimport softmax

cdef class VIPRSBMA(VIPRSGrid):


    def __init__(self, gdl, grid, fix_params=None, load_ld='auto', tracked_theta=None, verbose=True, threads=1):
        """
        :param gdl: An instance of `GWADataLoader`
        :param grid: An instance of `HyperparameterGrid`
        :param fix_params: A dictionary of hyperparameters with their fixed values.
        :param load_ld: A flag that specifies whether to load the LD matrix to memory (Default: `auto`).
        :param tracked_theta: A list of hyperparameters to track throughout the optimization procedure. Useful
        for debugging/model checking. Currently, we allow the user to track the following:
            - The proportion of causal variants (`pi`).
            - The heritability ('heritability').
            - The residual variance (`sigma_epsilon`).
        :param verbose: Verbosity of the information printed to standard output
        :param threads: The number of threads to use (experimental)
        """

        super().__init__(gdl, grid=grid, fix_params=fix_params, load_ld=load_ld,
                         tracked_theta=tracked_theta, verbose=verbose, threads=threads)

    cpdef average_models(self, normalization='softmax'):
        """
        Use Bayesian model averaging to obtain final weights for each predictor.
        We average the weights by using the final ELBO for each model.
        
        :param normalization: The normalization scheme for the final ELBOs. Options are (`softmax`, `sum`). 
        """

        assert normalization in ('softmax', 'sum')

        elbos = self.history['ELBO'][len(self.history['ELBO']) - 1]

        if normalization == 'softmax':
            elbos = softmax(elbos)
        elif normalization == 'sum':
            # Correction for negative ELBOs:
            elbos = elbos - elbos.min() + 1.
            elbos /= elbos.sum()

        if self.verbose:
            print("Averaging PRS models with weights:", np.array(elbos))

        for param in (self.pip, self.post_mean_beta, self.post_var_beta,
                      self.var_gamma, self.var_mu, self.var_sigma,
                      self.eta, self.zeta, self.q):

            for c in param:
                param[c] = (param[c]*elbos).sum(axis=1)

        return self
