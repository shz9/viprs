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
import pandas as pd
import warnings
from tqdm import tqdm

from .VIPRS cimport VIPRS
from viprs.utils.math_utils cimport elementwise_add_mult, sigmoid, clip
from viprs.utils.math_utils import bernoulli_entropy
from viprs.utils.compute_utils import dict_mean


cdef class VIPRSGrid(VIPRS):


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

        self.grid_table = grid.to_table()
        self.n_models = len(self.grid_table)
        assert self.n_models > 1

        grid_params = {c: self.grid_table[c].values for c in self.grid_table.columns}

        if fix_params is None:
            fix_params = grid_params
        else:
            fix_params.update(grid_params)

        super().__init__(gdl, fix_params=fix_params, load_ld=load_ld,
                         tracked_theta=tracked_theta, verbose=verbose, threads=threads)

        self.shapes = {c: (shp, self.n_models)
                       for c, shp in self.shapes.items()}
        self.stop_iteration = None
        self.Nj = {c: Nj[:, None] for c, Nj in self.Nj.items()}

    cpdef initialize_theta(self, theta_0=None):
        """
        Initialize the global hyper-parameters
        :param theta_0: A dictionary of initial values for the hyperparameters theta
        """

        self.stop_iteration = [False for _ in range(self.n_models)]

        super(VIPRSGrid, self).initialize_theta(theta_0=theta_0)

        try:
            if self.pi.shape != (self.n_models, ):
                self.pi *= np.ones(shape=(self.n_models, ))
        except AttributeError:
            self.pi *= np.ones(shape=(self.n_models,))

        try:
            if self.sigma_beta.shape != (self.n_models, ):
                self.sigma_beta *= np.ones(shape=(self.n_models, ))
        except AttributeError:
            self.sigma_beta *= np.ones(shape=(self.n_models,))

        try:
            if self.sigma_epsilon.shape != (self.n_models, ):
                self.sigma_epsilon *= np.ones(shape=(self.n_models, ))
        except AttributeError:
            self.sigma_epsilon *= np.ones(shape=(self.n_models,))

    cpdef e_step(self):
        """
        In the E-step, update the variational parameters for each SNP 
        in a coordinate-wise fashion.
        """

        # Define memoryviews objects for fast access
        cdef:
            unsigned int j, m, m_idx, start, end,
            double u_j, eta_diff
            double[::1] std_beta, Dj  # Inputs
            double[::1, :] var_gamma, var_mu, var_sigma  # Variational parameters
            double[::1, :] mu_mult, u_logs, recip_sigma  # Helpers + other quantities that we need inside the for loop
            double[::1, :] eta, q  # Properties of proposed distribution
            long[:, ::1] ld_bound
            long [:] not_converged = np.array([i for i in range(self.n_models) if not self.stop_iteration[i]])

        for c, shapes in self.shapes.items():

            # Get the priors:
            sigma_beta = self.get_sigma_beta(c)
            pi = self.get_pi(c)

            # Updates for sigma_beta variational parameters:
            self.var_sigma[c] = self.sigma_epsilon / (
                    self.Nj[c] + self.sigma_epsilon / sigma_beta
            )

            # Compute some quantities that are needed for the per-SNP updates:
            mu_mult = np.asfortranarray(self.Nj[c]*self.var_sigma[c]/self.sigma_epsilon)
            u_logs = np.asfortranarray(np.log(pi / (1. - pi)) + .5*np.log(self.var_sigma[c] / sigma_beta))
            recip_sigma = np.asfortranarray(.5/self.var_sigma[c])

            # Set the numpy vectors into memoryviews for fast access:
            std_beta = self.std_beta[c]
            var_gamma = np.asfortranarray(self.var_gamma[c])
            var_mu = np.asfortranarray(self.var_mu[c])
            eta = np.asfortranarray(self.eta[c])
            ld_bound = self.ld_bounds[c]
            q = np.asfortranarray(self.q[c])

            for j, Dj in enumerate(self.ld[c]):

                start, end = ld_bound[:, j]

                for m_idx in range(not_converged.shape[0]):

                    m = not_converged[m_idx]

                    # Compute the variational mu beta:
                    var_mu[j, m] = mu_mult[j, m]*(std_beta[j] - q[j, m])

                    # Compute the variational gamma:
                    u_j = u_logs[j, m] + recip_sigma[j, m]*var_mu[j, m]*var_mu[j, m]
                    var_gamma[j, m] = clip(sigmoid(u_j), 1e-8, 1. - 1e-8)

                    # Compute the difference between the new and old values for the posterior mean:
                    eta_diff = var_gamma[j, m] * var_mu[j, m] - eta[j, m]

                    # Update the q factors for all neighboring SNPs that are in LD with SNP j
                    elementwise_add_mult(q[start: end, m], Dj, eta_diff)
                    # Operation above updates the q factor for SNP j, so we correct that here:
                    q[j, m] = q[j, m] - eta_diff

                    # Update the posterior mean:
                    eta[j, m] = eta[j, m] + eta_diff

            # Convert memoryviews back to numpy arrays:
            self.var_gamma[c] = np.asarray(var_gamma)
            self.var_mu[c] = np.asarray(var_mu)
            self.q[c] = np.asarray(q)
            self.eta[c] = np.asarray(eta)
            self.zeta[c] = self.var_gamma[c]*(self.var_mu[c]**2 + self.var_sigma[c])

    cpdef to_theta_table(self):
        """
        Output the values for the hyperparameters (theta) to a pandas table.
        """

        if self.n_models == 1:
            return super(VIPRSGrid, self).to_theta_table()

        sig_e = self.sigma_epsilon
        h2 = self.get_heritability()
        pi = self.get_proportion_causal()

        if isinstance(self.sigma_beta, dict):
            sigmas = dict_mean(self.sigma_beta, axis=0)
        else:
            sigmas = self.sigma_beta

        theta_table = []

        for m in range(self.n_models):

            theta_table += [
                {'Model': m, 'Parameter': 'Residual_variance', 'Value': sig_e[m]},
                {'Model': m, 'Parameter': 'Heritability', 'Value': h2[m]},
                {'Model': m, 'Parameter': 'Proportion_causal', 'Value': pi[m]},
                {'Model': m, 'Parameter': 'sigma_beta', 'Value': sigmas[m]}
            ]

        return pd.DataFrame(theta_table)

    def to_validation_table(self):
        """
        Summarize the validation results in a pandas table.
        """
        if self.validation_result is None:
            raise Exception("Validation result is not set!")
        elif len(self.validation_result) < 1:
            raise Exception("Validation result is not set!")

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

    cpdef fit(self, max_iter=1000, theta_0=None, param_0=None,
              continued=False, f_abs_tol=1e-3, x_abs_tol=1e-8, max_elbo_drops=10,
              annealing_schedule='linear', annealing_steps=0, init_temperature=5.):
        """
        Fit the model parameters to data.

        :param max_iter: Maximum number of iterations. 
        :param theta_0: A dictionary of values to initialize the hyperparameters
        :param param_0: A dictionary of values to initialize the variational parameters
        :param continued: If true, continue the model fitting for more iterations.
        :param f_abs_tol: The absolute tolerance threshold for the objective (ELBO)
        :param x_abs_tol: The absolute tolerance threshold for the parameters (gamma)
        :param max_elbo_drops: The maximum number of times the objective is allowed to drop before termination.
        :param annealing_schedule: The schedule for the variational annealing procedure 
        (Options: 'linear', 'harmonic', 'geometric').
        :param annealing_steps: The number of annealing steps to perform before turning back to 
        optimizing the original variational objective.
        :param init_temperature: The initial temperature for the annealing procedure (T > 1).
        """

        if not continued:
            self.initialize(theta_0, param_0)
            start_idx = 1
        else:
            start_idx = len(self.history['ELBO']) + 1

        if self.load_ld:

            if self.verbose:
                print("> Loading LD matrices into memory...")

            self.gdl.load_ld()

        if annealing_schedule is not None and annealing_steps > 0:

            assert annealing_schedule in ('linear', 'harmonic', 'geometric')
            assert init_temperature > 1.

            self.inv_temperature = 1./init_temperature

            for i in tqdm(range(annealing_steps + 1, 1, -1),
                          total=annealing_steps,
                          desc="Annealing steps: "):

                self.e_step()
                self.m_step()

                if annealing_schedule == 'linear':
                    delta = (1. - 1./init_temperature) / (annealing_steps - 1)
                    self.inv_temperature = 1./init_temperature + delta*(annealing_steps - i)
                elif annealing_schedule == 'harmonic':
                    delta = (init_temperature - 1) / (annealing_steps - 1)
                    self.inv_temperature = 1./(1. + delta*(i - 1))
                elif annealing_schedule == 'geometric':
                    delta = init_temperature**(1./ (annealing_steps - 1)) - 1
                    self.inv_temperature = (1 + delta)**(-(i - 1))

            self.inv_temperature = 1.

        elbo_dropped_count = np.zeros(self.n_models)

        if self.verbose:
            print("> Performing model fit...")
            print(f"> Using up to {self.threads} threads.")

        for i in tqdm(range(start_idx, start_idx + max_iter), disable=not self.verbose):

            if all(self.stop_iteration):
                break

            self.update_theta_history()

            self.e_step()
            self.m_step()

            self.history['ELBO'].append(self.elbo(sum_axis=0))
            h2 = self.get_heritability()

            if i > 1:

                for m in range(self.n_models):

                    if self.stop_iteration[m]:
                        continue

                    curr_elbo = self.history['ELBO'][i - 1][m]
                    prev_elbo = self.history['ELBO'][i - 2][m]

                    if curr_elbo < prev_elbo:
                        elbo_dropped_count[m] += 1
                        warnings.warn(f"Iteration {i} | Model {m}: ELBO dropped from {prev_elbo:.6f} "
                                      f"to {curr_elbo:.6f}.")

                        if elbo_dropped_count[m] > max_elbo_drops:
                            warnings.warn(f"The optimization for model {m} is halted due to numerical instabilities!")
                            self.stop_iteration[m] = True

                        continue

                    if np.isclose(prev_elbo, curr_elbo, atol=f_abs_tol, rtol=0.):
                        print(f"Model {m} converged at iteration {i} || ELBO: {curr_elbo:.6f}")
                        self.stop_iteration[m] = True
                    elif all([
                        np.allclose(bernoulli_entropy(v[:, m]), bernoulli_entropy(self.pip[c][:, m]),
                                    atol=x_abs_tol, rtol=0.)
                        for c, v in self.var_gamma.items()
                    ]):
                        print(f"Model {m} converged at iteration {i} | ELBO: {curr_elbo:.6f}")
                        self.stop_iteration[m] = True

                    if abs((curr_elbo - prev_elbo) / prev_elbo) > 1. and abs(curr_elbo - prev_elbo) > 1e3:
                        warnings.warn(f"Stopping at iteration {i} for model {m}: "
                                      f"The optimization algorithm is not converging!\n"
                                      f"Previous ELBO: {prev_elbo:.6f} | "
                                      f"Current ELBO: {curr_elbo:.6f}")
                        self.stop_iteration[m] = True
                    elif h2[m] >= 1.:
                        warnings.warn(f"Stopping at iteration {i} for model {m}: "
                                      f"The optimization algorithm is not converging!\n"
                                      f"Value of estimated heritability exceeded 1.")
                        self.stop_iteration[m] = True

            self.update_posterior_moments()

        if self.verbose:
            print(f"> Optimization is complete for all {self.n_models} models.")

        self.validation_result = self.grid_table.copy()
        self.validation_result['ELBO'] = self.history['ELBO'][len(self.history['ELBO']) - 1]

        return self