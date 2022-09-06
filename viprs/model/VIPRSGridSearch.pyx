# cython: linetrace=False
# cython: profile=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True

import numpy as np
cimport numpy as np

from .VIPRSGrid cimport VIPRSGrid

cdef class VIPRSGridSearch(VIPRSGrid):


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

    cpdef select_best_model(self, validation_gdl=None, criterion='ELBO'):
        """
        From the grid of models that were fit to the data, select the best 
        model according to the specified `criterion`.
        
        :param validation_gdl: An instance of `GWADataLoader` containing data from the validation set.  
        :param criterion: The criterion for selecting the best model. 
        Options are: (`ELBO`, `validation`, `pseudo_validation`)
        """

        assert criterion in ('ELBO', 'validation', 'pseudo_validation')

        if criterion == 'ELBO':
            best_model_idx = np.argmax(self.history['ELBO'][len(self.history['ELBO']) - 1])
        elif criterion == 'validation':

            assert validation_gdl is not None
            assert validation_gdl.sample_table is not None
            assert validation_gdl.sample_table.phenotype is not None

            from viprs.eval.metrics import r2

            prs = self.predict(gdl=validation_gdl)
            prs_r2 = [r2(prs[:, i], validation_gdl.sample_table.phenotype) for i in range(self.n_models)]
            self.validation_result['Validation_R2'] = prs_r2
            best_model_idx = np.argmax(prs_r2)
        elif criterion == 'pseudo_validation':

            pseudo_corr = self.pseudo_validate(validation_gdl=validation_gdl)
            self.validation_result['Pseudo_Validation_Corr'] = pseudo_corr
            best_model_idx = np.argmax(np.nan_to_num(pseudo_corr, nan=-1., neginf=-1., posinf=-1.))

        if self.verbose:
            print(f"> Based on the {criterion} criterion, selected model: {best_model_idx}")
            print("> Model details:\n")
            print(self.validation_result.iloc[best_model_idx, :])

        # Update the variational parameters and their dependencies:
        for param in (self.pip, self.post_mean_beta, self.post_var_beta,
                      self.var_gamma, self.var_mu, self.var_sigma,
                      self.eta, self.zeta, self.q):

            for c in param:
                param[c] = param[c][:, best_model_idx]

        # Update sigma epsilon:
        self.sigma_epsilon = self.sigma_epsilon[best_model_idx]

        # Update sigma beta:
        if isinstance(self.sigma_beta, dict):
            for c in self.sigma_beta:
                self.sigma_beta[c] = self.sigma_beta[c][:, best_model_idx]
        else:
            self.sigma_beta = self.sigma_beta[best_model_idx]

        # Update pi

        if isinstance(self.pi, dict):
            for c in self.pi:
                self.pi[c] = self.pi[c][:, best_model_idx]
        else:
            self.pi = self.pi[best_model_idx]

        # Set the number of models to 1:
        self.n_models = 1

        return self
