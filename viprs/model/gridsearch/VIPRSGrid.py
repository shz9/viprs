import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

from viprs.model.VIPRS import VIPRS
from ..vi.e_step import e_step_grid
from ..vi.e_step_cpp import cpp_e_step_grid
from viprs.utils.compute_utils import dict_mean
from viprs.utils.OptimizeResult import OptimizeResult


class VIPRSGrid(VIPRS):
    """
    A class to fit the `VIPRS` model to data using a grid of hyperparameters.
    Instead of having a single set of hyperparameters, we simultaneously fit
    multiple models with different hyperparameters and compare their performance
    at the end. This class is generic and does not support any model selection or
    averaging schemes.

    The class inherits all the basic attributes from the [VIPRS][viprs.model.VIPRS.VIPRS] class.

    !!! seealso "See Also"
        * [VIPRSGridSearch][viprs.model.gridsearch.VIPRSGridSearch.VIPRSGridSearch]
        * [VIPRSBMA][viprs.model.gridsearch.VIPRSBMA.VIPRSBMA]

    :ivar grid_table: A pandas table containing the hyperparameters for each model.
    :ivar n_models: The number of models to fit.
    :ivar shapes: A dictionary containing the shapes of the data matrices.

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
        self.n_models = len(self.grid_table)
        assert self.n_models > 1

        grid_params = {c: self.grid_table[c].values for c in self.grid_table.columns}

        if 'fix_params' not in kwargs:
            kwargs['fix_params'] = grid_params
        else:
            kwargs['fix_params'].update(grid_params)

        if 'lambda_min' in grid_params:
            kwargs['lambda_min'] = 0.

        # Make sure that the matrices are in Fortran order:
        kwargs['order'] = 'F'

        super().__init__(gdl, **kwargs)

        self.shapes = {c: (shp, self.n_models)
                       for c, shp in self.shapes.items()}
        self.Nj = {c: Nj[:, None].astype(self.float_precision, order=self.order) for c, Nj in self.Nj.items()}

        if not np.isscalar(self.lambda_min):
            self.lambda_min = self.lambda_min[:, None]

        self.optim_results = [OptimizeResult() for _ in range(self.n_models)]

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

    def initialize_theta(self, theta_0=None):
        """
        Initialize the global hyperparameters of the model.
        :param theta_0: A dictionary of initial values for the hyperparameters theta
        """

        super().initialize_theta(theta_0=theta_0)

        if np.isscalar(self.pi):
            self.pi *= np.ones(shape=(self.n_models, ), dtype=self.float_precision)

        if np.isscalar(self.tau_beta):
            self.tau_beta *= np.ones(shape=(self.n_models, ), dtype=self.float_precision)

        if np.isscalar(self.sigma_epsilon):
            self.sigma_epsilon *= np.ones(shape=(self.n_models, ), dtype=self.float_precision)

        if np.isscalar(self._sigma_g):
            self._sigma_g *= np.ones(shape=(self.n_models, ), dtype=self.float_precision)

        if 'lambda_min' in self.fix_params:
            self.lambda_min = self.fix_params['lambda_min'].astype(self.float_precision)

    def init_optim_meta(self):
        """
        Initialize the various quantities/objects to keep track of the optimization process.
         This method initializes the "history" object (which keeps track of the objective + other
         hyperparameters requested by the user), in addition to the OptimizeResult objects.
        """
        super().init_optim_meta()

        # Reset the OptimizeResult objects:
        for optr in self.optim_results:
            optr.reset()

    def e_step(self):
        """
        Run the E-Step of the Variational EM algorithm.
        Here, we update the variational parameters for each variant using coordinate
        ascent optimization techniques. The coordinate ascent procedure is run on all the models
        in the grid simultaneously. The update equations are outlined in
        the Supplementary Material of the following paper:

        > Zabad S, Gravel S, Li Y. Fast and accurate Bayesian polygenic risk modeling with variational inference.
        Am J Hum Genet. 2023 May 4;110(5):741-761. doi: 10.1016/j.ajhg.2023.03.009.
        Epub 2023 Apr 7. PMID: 37030289; PMCID: PMC10183379.
        """

        active_model_idx = np.where(~self.terminated_models)[0].astype(np.int32)

        for c, shapes in self.shapes.items():

            # Get the priors:
            tau_beta = self.get_tau_beta(c)
            pi = self.get_pi(c)

            # Updates for tau variational parameters:
            # NOTE: Here, we compute the variational sigma in-place to avoid the need
            # to change the order of the resulting matrix or its float precision:
            np.add(self.Nj[c]*(1. + self.lambda_min) / self.sigma_epsilon, tau_beta,
                   out=self.var_tau[c])
            np.log(self.var_tau[c], out=self._log_var_tau[c])

            # Compute some quantities that are needed for the per-SNP updates:
            mu_mult = self.Nj[c] / (self.var_tau[c] * self.sigma_epsilon)
            u_logs = np.log(pi) - np.log(1. - pi) + .5 * (np.log(tau_beta) - self._log_var_tau[c])

            if self.use_cpp:
                cpp_e_step_grid(self.ld_left_bound[c],
                                self.ld_indptr[c],
                                self.ld_data[c],
                                self.std_beta[c],
                                self.var_gamma[c],
                                self.var_mu[c],
                                self.eta[c],
                                self.q[c],
                                self.eta_diff[c],
                                u_logs,
                                0.5 * self.var_tau[c],
                                mu_mult,
                                self.dequantize_scale,
                                active_model_idx,
                                self.threads,
                                self.use_blas,
                                self.low_memory)
            else:

                e_step_grid(self.ld_left_bound[c],
                            self.ld_indptr[c],
                            self.ld_data[c],
                            self.std_beta[c],
                            self.var_gamma[c],
                            self.var_mu[c],
                            self.eta[c],
                            self.q[c],
                            self.eta_diff[c],
                            u_logs,
                            0.5 * self.var_tau[c],
                            mu_mult,
                            active_model_idx,
                            self.threads,
                            self.use_blas,
                            self.low_memory)

        self.zeta = self.compute_zeta()

    def to_theta_table(self):
        """
        :return: A `pandas` DataFrame containing information about the hyperparameters of the model.
        """

        if self.n_models == 1:
            return super(VIPRSGrid, self).to_theta_table()

        sig_e = self.sigma_epsilon
        h2 = self.get_heritability()
        pi = self.get_proportion_causal()

        if isinstance(self.tau_beta, dict):
            taus = dict_mean(self.tau_beta, axis=0)
        else:
            taus = self.tau_beta

        theta_table = []

        for m in range(self.n_models):

            theta_table += [
                {'Model': m, 'Parameter': 'Residual_variance', 'Value': sig_e[m]},
                {'Model': m, 'Parameter': 'Heritability', 'Value': h2[m]},
                {'Model': m, 'Parameter': 'Proportion_causal', 'Value': pi[m]},
                {'Model': m, 'Parameter': 'sigma_beta', 'Value': taus[m]}
            ]

            if 'lambda_min' in self.fix_params:
                theta_table.append({'Model': m, 'Parameter': 'lambda_min', 'Value': self.lambda_min[m]})

        if np.isscalar(self.lambda_min):
            theta_table.append({'Model': 'ALL', 'Parameter': 'lambda_min', 'Value': self.lambda_min})

        return pd.DataFrame(theta_table)

    def to_validation_table(self):
        """
        :return: The validation table summarizing the performance of each model.
        :raises ValueError: if the validation result is not set.
        """
        if self.validation_result is None:
            raise ValueError("Validation result is not set!")
        elif len(self.validation_result) < 1:
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

    def fit(self,
            max_iter=1000,
            theta_0=None,
            param_0=None,
            continued=False,
            min_iter=3,
            f_abs_tol=1e-6,
            x_abs_tol=1e-6,
            drop_r_tol=0.01,
            patience=5):
        """
        A convenience method to fit all the models in the grid using the Variational EM algorithm.

        :param max_iter: Maximum number of iterations. 
        :param theta_0: A dictionary of values to initialize the hyperparameters
        :param param_0: A dictionary of values to initialize the variational parameters
        :param continued: If true, continue the model fitting for more iterations.
        :param min_iter: The minimum number of iterations to run before checking for convergence.
        :param f_abs_tol: The absolute tolerance threshold for the objective (ELBO).
        :param x_abs_tol: The absolute tolerance threshold for the variational parameters.
        :param drop_r_tol: The relative tolerance for the drop in the ELBO to be considered as a red flag. It usually
        happens around convergence that the objective fluctuates due to numerical errors. This is a way to
        differentiate such random fluctuations from actual drops in the objective.
        :param patience: The maximum number of times the objective is allowed to drop before termination.
        """

        if not continued:
            self.initialize(theta_0, param_0)
            start_idx = 1
        else:
            start_idx = len(self.history['ELBO']) + 1
            for i, optr in enumerate(self.optim_results):
                optr.update(self.history['ELBO'][i], increment=False)

        patience = patience*np.ones(self.n_models)

        logging.debug("> Performing model fit...")
        if self.threads > 1:
            logging.debug(f"> Using up to {self.threads} threads.")

        # If the model is fit over a single chromosome, append this information to the
        # tqdm progress bar:
        if len(self.shapes) == 1:
            chrom, shape = list(self.shapes.items())[0]
            desc = f"Chromosome {chrom} ({shape[0]} variants)"
        else:
            desc = None

        # Progress bar:
        pbar = tqdm(range(start_idx, start_idx + max_iter),
                    disable=not self.verbose,
                    desc=desc)

        for i in pbar:

            terminated_models_i = self.terminated_models

            if np.all(terminated_models_i):

                # If converged, update the progress bar before exiting:
                models_converged = self.converged_models.sum()

                if self.models_to_keep.sum() > 0:
                    best_elbo = self.history['ELBO'][-1][self.models_to_keep].max()
                else:
                    best_elbo = np.nan

                pbar.set_postfix({
                    'Best ELBO': f"{best_elbo:.4f}",
                    'Models Converged': f"{models_converged}/{self.n_models}"
                })

                pbar.n = i - 1
                pbar.total = i - 1
                pbar.refresh()
                pbar.close()

                pbar.close()
                break

            self.update_theta_history()

            self.e_step()
            self.m_step()

            self.history['ELBO'].append(self.elbo(sum_axis=0))
            h2 = self.get_heritability()

            if i > 1:

                # Get the current and previous ELBO values
                curr_elbo = self.history['ELBO'][-1]
                prev_elbo = self.history['ELBO'][-2]

                for m in np.where(~terminated_models_i)[0]:

                    if not np.isfinite(curr_elbo[m]):
                        self.optim_results[m].update(curr_elbo[m],
                                                     stop_iteration=True,
                                                     success=False,
                                                     message=f'The objective (ELBO) is undefined ({curr_elbo[m]}).')
                    elif self.sigma_epsilon[m] <= 0.:
                        self.optim_results[m].update(curr_elbo[m],
                                                     stop_iteration=True,
                                                     success=False,
                                                     message='Optimization is halted (sigma_epsilon <= 0).')
                    elif h2[m] > 1. or h2[m] < 0.:
                        self.optim_results[m].update(curr_elbo[m],
                                                     stop_iteration=True,
                                                     success=False,
                                                     message='Value of estimated heritability is out of bounds.')
                    elif (i > min_iter) and np.isclose(prev_elbo[m], curr_elbo[m], atol=f_abs_tol, rtol=0.):
                        self.optim_results[m].update(curr_elbo[m],
                                                     stop_iteration=True,
                                                     success=True,
                                                     message='Objective (ELBO) converged successfully.')
                    elif (i > min_iter) and max([np.max(np.abs(diff[:, m]))
                                                 for diff in self.eta_diff.values()]) < x_abs_tol:
                        self.optim_results[m].update(curr_elbo[m],
                                                     stop_iteration=True,
                                                     success=True,
                                                     message='Variational parameters converged successfully.')

                    # Check to see if the objective drops due to numerical instabilities:
                    elif curr_elbo[m] < prev_elbo[m] and not np.isclose(curr_elbo[m],
                                                                        prev_elbo[m],
                                                                        atol=0.,
                                                                        rtol=drop_r_tol):
                        patience[m] -= 1

                        if patience[m] == 0:
                            self.optim_results[m].update(curr_elbo[m],
                                                         stop_iteration=True,
                                                         success=False,
                                                         message='Optimization is halted '
                                                                 'due to numerical instabilities.')
                        else:
                            self.optim_results[m].update(curr_elbo[m])

                    else:
                        self.optim_results[m].update(curr_elbo[m])

                # -----------------------------------------------------------------------

            # Update the progress bar:
            if self.models_to_keep.sum() > 0:
                best_elbo = self.history['ELBO'][-1][self.models_to_keep].max()
            else:
                best_elbo = np.nan

            pbar.set_postfix({
                'Best ELBO': f"{best_elbo:.4f}",
                'Models Terminated': f"{self.terminated_models.sum()}/{self.n_models}"
            })

        # Update posterior moments:
        self.update_posterior_moments()

        # Inspect the optimization results:
        for m, optr in enumerate(self.optim_results):
            if not optr.stop_iteration:
                optr.update(self.history['ELBO'][-1][m],
                            stop_iteration=True,
                            success=False,
                            message="Maximum iterations reached without convergence.\n"
                                    "You may need to run the model for more iterations.")

        # Inform the user about potential issues:
        if int(self.verbose) > 1:

            if self.models_to_keep.sum() > 0:
                logging.info(f"> Optimization is complete for all {self.n_models} models.")
                logging.info(f"> {np.sum(self.models_to_keep)} model(s) converged successfully.")
            else:
                logging.error("> All models failed to converge. Please check the optimization results.")

        self.validation_result = self.grid_table.copy()
        self.validation_result['ELBO'] = [optr.fun for optr in self.optim_results]
        self.validation_result['Converged'] = self.converged_models
        self.validation_result['Optimization_message'] = [optr.message for optr in self.optim_results]

        return self
