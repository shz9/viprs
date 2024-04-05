import numpy as np
from .VIPRSGrid import VIPRSGrid


class VIPRSGridSearch(VIPRSGrid):
    """
    The `VIPRSGridSearch` class is an extension of the `VIPRSGrid` class that
    implements grid search for the `VIPRS` models. The grid search procedure
    fits multiple models to the data, each with different hyperparameters,
    and selects the best model based on user-defined criteria.

    The criteria supported are:

    * `ELBO`: The model with the highest ELBO is selected.
    * `validation`: The model with the highest R^2 on the validation set is selected.
    * `pseudo_validation`: The model with the highest pseudo-validation R^2 is selected.

    Note that the `validation` and `pseudo_validation` criteria require the user to provide
    validation data in the form of paired genotype/phenotype data or external GWAS summary
    statistics.

    """

    def __init__(self,
                 gdl,
                 grid,
                 **kwargs):
        """
        Initialize the `VIPRSGridSearch` model.

        :param gdl: An instance of `GWADataLoader`
        :param grid: An instance of `HyperparameterGrid`
        :param kwargs: Additional keyword arguments to pass to the parent `VIPRSGrid` class.
        """

        super().__init__(gdl, grid=grid, **kwargs)

    def select_best_model(self, validation_gdl=None, criterion='ELBO'):
        """
        From the grid of models that were fit to the data, select the best 
        model according to the specified `criterion`. If the criterion is the ELBO,
        the model with the highest ELBO will be selected. If the criterion is
        validation or pseudo-validation, the model with the highest R^2 on the
        validation set will be selected.
        
        :param validation_gdl: An instance of `GWADataLoader` containing data from the validation set.
        Must be provided if criterion is `validation` or `pseudo_validation`.
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

            from viprs.eval.continuous_metrics import r2

            prs = self.predict(test_gdl=validation_gdl)
            prs_r2 = [r2(prs[:, i], validation_gdl.sample_table.phenotype) for i in range(self.n_models)]
            self.validation_result['Validation_R2'] = prs_r2
            best_model_idx = np.argmax(prs_r2)
        elif criterion == 'pseudo_validation':

            pseudo_corr = self.pseudo_validate(validation_gdl, metric='r2')
            self.validation_result['Pseudo_Validation_Corr'] = pseudo_corr
            best_model_idx = np.argmax(np.nan_to_num(pseudo_corr, nan=-1., neginf=-1., posinf=-1.))

        if int(self.verbose) > 1:
            print(f"> Based on the {criterion} criterion, selected model: {best_model_idx}")
            print("> Model details:\n")
            print(self.validation_result.iloc[best_model_idx, :])

        # Update the variational parameters and their dependencies:
        for param in (self.pip, self.post_mean_beta, self.post_var_beta,
                      self.var_gamma, self.var_mu, self.var_tau,
                      self.eta, self.zeta, self.q):

            for c in param:
                param[c] = param[c][:, best_model_idx]

        # Update sigma epsilon:
        self.sigma_epsilon = self.sigma_epsilon[best_model_idx]

        # Update sigma beta:
        if isinstance(self.tau_beta, dict):
            for c in self.tau_beta:
                self.tau_beta[c] = self.tau_beta[c][:, best_model_idx]
        else:
            self.tau_beta = self.tau_beta[best_model_idx]

        # Update pi

        if isinstance(self.pi, dict):
            for c in self.pi:
                self.pi[c] = self.pi[c][:, best_model_idx]
        else:
            self.pi = self.pi[best_model_idx]

        # Set the number of models to 1:
        self.n_models = 1

        return self
