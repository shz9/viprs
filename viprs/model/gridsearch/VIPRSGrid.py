import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ..VIPRS import VIPRS
from ...utils.exceptions import OptimizationDivergence

# Set up the logger:
import logging
logger = logging.getLogger(__name__)


class VIPRSGrid(VIPRS):
    """
    A class to fit the `VIPRS` model to data using a grid of hyperparameters.
    Instead of having a single set of hyperparameters, we simultaneously fit
    multiple models with different hyperparameters and compare their performance
    at the end. The models with different hyperparameters are fit serially and in
    a pathwise manner, meaning that fit one model at a time and use its inferred parameters
    to initialize the next model.

    The class inherits all the basic attributes from the [VIPRS][viprs.model.VIPRS.VIPRS] class.

    :ivar grid_table: A pandas table containing the hyperparameters for each model.
    :ivar validation_result: A pandas table summarizing the performance of each model.
    :ivar optim_results: A list of optimization results for each model.
    :ivar n_models: The number of models to fit.

    """

    def __init__(self,
                 gdl,
                 grid,
                 **kwargs):
        """
        Initialize the `VIPRS` model with a grid of hyperparameters.

        :param gdl: An instance of `GWADataLoader`
        :param grid: An instance of `HyperparameterGrid`
        :param kwargs: Additional keyword arguments to pass to the parent `VIPRS` class.
        """

        self.grid_table = grid.to_table()

        # Placeholders:
        self.n_models = len(self.grid_table)
        self.validation_result = None
        self.optim_results = None

        self._reset_search()

        super().__init__(gdl, **kwargs)

    def _reset_search(self):
        """
        Reset the grid search object. This might be useful after
        fitting the model and performing model selection/BMA, to start over.
        """
        self.n_models = len(self.grid_table)
        assert self.n_models > 1, "Grid search requires at least 2 models."
        self.validation_result = None
        self.optim_results = []

    @property
    def models_to_keep(self):
        """
        :return: A boolean array indicating which models have converged successfully.
        """
        return np.logical_or(~self.terminated_models, self.converged_models)

    @property
    def converged_models(self):
        """
        :return: A boolean array indicating which models have converged successfully.
        """
        return np.array([optr.success for optr in self.optim_results])

    @property
    def terminated_models(self):
        """
        :return: A boolean array indicating which models have terminated.
        """
        return np.array([optr.stop_iteration for optr in self.optim_results])

    @property
    def valid_terminated_models(self):
        """
        :return: A boolean array indicating which models have terminated without error.
        """
        return np.array([optr.valid_optim_result for optr in self.optim_results])

    def to_validation_table(self):
        """
        :return: The validation table summarizing the performance of each model.
        :raises ValueError: if the validation result is not set.
        """

        if self.validation_result is None or len(self.validation_result) < 1:
            raise ValueError("Validation result is not set!")

        return pd.DataFrame(self.validation_result)

    def write_validation_result(self, v_filename, sep="\t"):
        """
        After performing hyperparameter search, write a table
        that records that value of the objective for each combination
        of hyperparameters.
        :param v_filename: The filename for the validation table.
        :param sep: The separator for the validation table
        """

        v_df = self.to_validation_table()
        v_df.to_csv(v_filename, index=False, sep=sep)

    def init_optim_meta(self):
        """
        Initialize the various quantities/objects to keep track of the optimization process.
         This method initializes the "history" object (which keeps track of the objective + other
         hyperparameters requested by the user), in addition to the OptimizeResult objects.
        """
        super().init_optim_meta()

        # Reset the OptimizeResult objects:
        self.optim_results = []

    def fit(self,
            pathwise=True,
            **fit_kwargs):
        """
        Fit the VIPRS model to the data using a grid of hyperparameters.
        The method fits multiple models with different hyperparameters and compares their performance
        at the end. By default, the models with different hyperparameters are fit serially and
        in a pathwise manner, meaning that fit one model at a time and use its inferred
        parameters to initialize the next model. The user can also fit the models independently by
        setting `pathwise=False`.

        :param pathwise: Whether to fit the models in a pathwise manner. Default is `True`.
        :param fit_kwargs: Additional keyword arguments to pass to fit method of the parent `VIPRS` class.

        :return: An instance of the `VIPRSGrid` class.
        """

        if self.n_models == 1:
            return super().fit(**fit_kwargs)

        # -----------------------------------------------------------------------
        # Setup the parameters that need to be tracked:

        var_gamma = {c: np.empty((size, self.n_models), dtype=self.float_precision)
                     for c, size in self.shapes.items()}
        var_mu = {c: np.empty((size, self.n_models), dtype=self.float_precision)
                  for c, size in self.shapes.items()}
        var_tau = {c: np.empty((size, self.n_models), dtype=self.float_precision)
                   for c, size in self.shapes.items()}
        q = {c: np.empty((size, self.n_models), dtype=self.float_precision)
             for c, size in self.shapes.items()}

        sigma_epsilon = np.empty(self.n_models, dtype=self.float_precision)
        pi = np.empty(self.n_models, dtype=self.float_precision)
        sigma_g = np.empty(self.n_models, dtype=self.float_precision)
        tau_beta = np.empty(self.n_models, dtype=self.float_precision)

        elbos = np.empty(self.n_models, dtype=self.float_precision)

        # -----------------------------------------------------------------------

        # Get a list of fixed hyperparameters from the grid table:
        params = self.grid_table.to_dict(orient='records')
        orig_threads = self.threads
        optim_results = []
        history = []

        # If the model is fit over a single chromosome, append this information to the
        # tqdm progress bar:
        if len(self.shapes) == 1:
            chrom, num_snps = list(self.shapes.items())[0]
            desc = f"Grid search | Chromosome {chrom} ({num_snps} variants)"
        else:
            desc = None

        disable_pbar = fit_kwargs.pop('disable_pbar', False)
        restart = not pathwise

        with logging_redirect_tqdm(loggers=[logger]):

            # Set up the progress bar for grid search:
            pbar = tqdm(range(self.n_models),
                        total=self.n_models,
                        disable=disable_pbar,
                        desc=desc)

            for i in pbar:

                # Fix the new set of hyperparameters:
                self.set_fixed_params(params[i])

                # Perform model fit:
                super().fit(continued=i > 0 and not restart,
                            disable_pbar=True,
                            **fit_kwargs)

                # Save the optimization result:
                optim_results.append(copy.deepcopy(self.optim_result))
                # Reset the optimization result:
                self.optim_result.reset()
                self.threads = orig_threads

                elbos[i] = self.history['ELBO'][-1]

                pbar.set_postfix({'ELBO': f"{self.history['ELBO'][-1]:.4f}",
                                  'Models Terminated': f"{i+1}/{self.n_models}"})

                # Update the saved parameters:
                for c in self.shapes:
                    var_gamma[c][:, i] = self.var_gamma[c]
                    var_mu[c][:, i] = self.var_mu[c]
                    var_tau[c][:, i] = self.var_tau[c]
                    q[c][:, i] = self.q[c]

                sigma_epsilon[i] = self.sigma_epsilon
                pi[i] = self.pi
                sigma_g[i] = self._sigma_g
                tau_beta[i] = self.tau_beta

        # Update the total number of iterations:
        self.optim_result.nit = np.sum([optr.nit for optr in self.optim_results])
        self.optim_results = optim_results

        # -----------------------------------------------------------------------
        # Update the object attributes:
        self.var_gamma = var_gamma
        self.var_mu = var_mu
        self.var_tau = var_tau
        self.q = q
        self.eta = self.compute_eta()
        self.zeta = self.compute_zeta()
        self._log_var_tau = {c: np.log(self.var_tau[c]) for c in self.var_tau}

        # Update posterior moments:
        self.update_posterior_moments()

        # Hyperparameters:
        self.sigma_epsilon = sigma_epsilon
        self.pi = pi
        self._sigma_g = sigma_g
        self.tau_beta = tau_beta

        # -----------------------------------------------------------------------

        # Population the validation result:
        self.validation_result = self.grid_table.copy()
        self.validation_result['ELBO'] = elbos
        self.validation_result['Converged'] = self.converged_models
        self.validation_result['Optimization_message'] = [optr.message for optr in self.optim_results]

        return self
