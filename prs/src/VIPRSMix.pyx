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
from libc.math cimport log, exp

from .PRSModel cimport PRSModel
from .exceptions import OptimizationDivergence
from .c_utils cimport dot, mt_sum, elementwise_add_mult, clip
from .utils import dict_mean, dict_sum, dict_concat, dict_repeat, dict_elementwise_dot


cdef class VIPRSMix(PRSModel):

    def __init__(self, gdl, K=1, prior_multipliers=None, fix_params=None, load_ld=True, verbose=True, threads=1):
        """
        :param gdl: An instance of GWAS data loader
        :param K: The number of components in the mixture prior
        :param prior_multipliers: Multiplier for the prior on the effect size (vector of size K)
        :param fix_params: A dictionary of parameters with their fixed values.
        :param load_ld: A flag that specifies whether to load the LD matrix to memory.
        :param verbose: Verbosity of the information printed to standard output
        :param threads: The number of threads to use (experimental)
        """

        super().__init__(gdl)

        # Sanity checks:
        assert K > 0  # Check that there are at least 1 causal component

        if prior_multipliers is not None:
            assert len(prior_multipliers) == K
            self.d = np.array(prior_multipliers)

        # Populate the fields:
        self.K = K
        self.shapes = {c: (shp, self.K) for c, shp in self.shapes.items()}

        # Global hyperparameters:
        self.pi = {}
        self.sigma_beta = {}

        # Variational parameters:
        self.var_gamma = {}
        self.var_mu_beta = {}
        self.var_sigma_beta = {}

        # Properties of proposed distribution:
        self.mean_beta = {}  # E[B] = \gamma*\mu_beta
        self.mean_beta_sq = {}  # E[B^2] = \gamma*(\mu_beta^2 + \sigma_beta^2)

        # q-factor (keeps track of LD-related terms)
        self.q = {}

        # ---------- Inputs to the model: ----------

        # LD-related quantities:
        self.load_ld = load_ld
        self.ld = self.gdl.get_ld_matrices()
        self.ld_bounds = self.gdl.get_ld_boundaries()

        # Standardized betas:
        self.std_beta = self.gdl.compute_xy_per_snp()

        # ---------- General properties: ----------

        self.threads = threads
        self.fix_params = fix_params or {}

        self.verbose = verbose
        self.history = {}

    cpdef initialize(self, theta_0=None):
        """
        A convenience method to initialize all the objects associated with the model.
        :param theta_0: A dictionary of initial values for the hyperparameters theta
        """

        if self.verbose:
            print("> Initializing model parameters")

        self.initialize_theta(theta_0)
        self.initialize_variational_params()
        self.init_history()

    cpdef init_history(self):
        """
        Initialize the history object to track various quantities.
        (For now, we track the ELBO only).
        """

        self.history = {
            'ELBO': []
        }

    cpdef initialize_theta(self, theta_0=None):
        """
        Initialize the global hyper-parameters
        :param theta_0: A dictionary of initial values for the hyperparameters theta
        """

        if theta_0 is not None and self.fix_params is not None:
            theta_0.update(self.fix_params)
        elif self.fix_params is not None:
            theta_0 = self.fix_params
        elif theta_0 is None:
            theta_0 = {}

        # ----------------------------------------------
        # (1) Initialize pi from a uniform
        if 'pis' in theta_0:
            init_pi = theta_0['pis']
        elif 'pi' in theta_0:
            init_pi = theta_0['pi']*np.random.dirichlet(np.ones(self.K))
        else:
            init_pi = np.random.dirichlet(np.ones(self.K + 1))[:self.K]

        # If pi is not a dict, convert to a dictionary of per-SNP values:
        if not isinstance(init_pi, dict):
            self.pi = dict_repeat(init_pi, self.shapes)
        else:
            self.pi = init_pi

        # For the remaining steps, estimate the mean value of pi:
        mean_pi = self.get_proportion_causal()
        # And the mixture proportions:
        mix_pi = dict_mean(self.pi, axis=1)

        # ----------------------------------------------
        # (2) Initialize sigma_epsilon and sigma_beta
        # Assuming that the genotype and phenotype are normalized,
        # these two quantities are conceptually linked.
        # The initialization routine here assumes that:
        # Var(y) = h2 + sigma_epsilon
        # Where, by assumption, Var(y) = 1,
        # And h2 ~= pi*M*sigma_beta

        if 'sigma_epsilon' not in theta_0:

            if 'sigma_betas' in theta_0:

                # If sigma_betas are given, use them to initialize sigma_epsilon

                init_sigma_beta = theta_0['sigma_betas']

                if not isinstance(init_sigma_beta, dict):
                    mean_sigma_beta = init_sigma_beta
                    self.sigma_beta = dict_repeat(init_sigma_beta, self.shapes)
                else:
                    mean_sigma_beta = dict_mean(init_sigma_beta)
                    self.sigma_beta = init_sigma_beta

                self.sigma_epsilon = clip(1. - dict_sum(dict_elementwise_dot(self.sigma_beta, self.pi)), 1e-6, 1. - 1e-6)

            elif 'sigma_beta' in theta_0:
                # NOTE: Here, we assume the provided `sigma_beta` is a scalar.
                # This is different from `sigma_betas`

                assert self.d is not None

                self.sigma_beta = dict_repeat(theta_0['sigma_beta'] * self.d, self.shapes)
                # Use the provided sigma_beta to initialize sigma_epsilon.
                # First, we derive a naive estimate of the heritability, based on the following equation:
                # h2g/M = \sum_k pi_k \sigma_k
                # Where the per-SNP heritability is defined by the sum over the mixtures.

                # Step (1): Given the provided sigma and associated multipliers,
                # obtain a naive estimate of the heritability:
                h2g_estimate = (theta_0['sigma_beta'] * self.d*self.M*mix_pi).sum()
                # Step (2): Set sigma_epsilon to 1 - h2g_estimate:
                self.sigma_epsilon = clip(1. - h2g_estimate, 1e-6, 1. - 1e-6)

            else:
                # If neither sigma_beta nor sigma_epsilon are given,
                # then initialize using the SNP heritability estimate based on summary statistics

                try:
                    naive_h2g = clip(self.gdl.estimate_snp_heritability(), 1e-6, 1. - 1e-6)
                except Exception as e:
                    naive_h2g = np.random.uniform(low=.001, high=.999)

                self.sigma_epsilon = 1. - naive_h2g

                if self.d is None:
                    mult = 10**np.arange(-(self.K - 1), 1).astype(float)
                else:
                    mult = self.d

                sigma_beta_estimate = naive_h2g / (self.M * (mix_pi * mult).sum())
                self.sigma_beta = dict_repeat(sigma_beta_estimate * mult, self.shapes)
        else:

            # If sigma_epsilon is given, use it in the initialization

            self.sigma_epsilon = theta_0['sigma_epsilon']

            # Initialize sigma_betas
            if 'sigma_betas' in theta_0:
                init_sigma_beta = theta_0['sigma_betas']
            elif 'sigma_beta' in theta_0:
                init_sigma_beta = theta_0['sigma_beta']
            else:
                # If not provided, initialize using sigma_epsilon value
                if self.d is None:
                    mult = 10**np.arange(-(self.K - 1), 1).astype(float)
                else:
                    mult = self.d

                init_sigma_beta = mult*(1. - self.sigma_epsilon) / (self.M * (mix_pi * mult).sum())

            if not isinstance(init_sigma_beta, dict):
                self.sigma_beta = dict_repeat(init_sigma_beta, self.shapes)
            else:
                self.sigma_beta = init_sigma_beta

    cpdef initialize_variational_params(self):
        """
        Initialize the variational parameters.
        """

        self.var_mu_beta = {}
        self.var_sigma_beta = {}
        self.var_gamma = {}

        for c, c_size in self.shapes.items():

            self.var_gamma[c] = self.pi[c].copy()
            self.var_mu_beta[c] = np.random.normal(scale=np.sqrt(self.sigma_beta[c]), size=c_size)
            self.var_sigma_beta[c] = self.sigma_beta[c].copy()

            self.mean_beta[c] = (self.var_gamma[c]*self.var_mu_beta[c]).sum(axis=1)
            self.mean_beta_sq[c] = (self.var_gamma[c]*(self.var_mu_beta[c]**2 + self.var_sigma_beta[c])).sum(axis=1)

    cpdef e_step(self):
        """
        In the E-step, update the variational parameters for each SNP 
        in a coordinate-wise fashion.
        """

        # Initialize memoryviews objects for fast access
        cdef:
            unsigned int j, k, start, end, j_idx
            double mu_beta_j, gamma_denom
            double[:, ::1] pi, sigma_beta  # Per-SNP priors
            double[:, ::1] var_gamma, var_mu_beta, var_sigma_beta  # Variational parameters
            double[::1] std_beta, Dj  # Inputs
            double[::1] mean_beta, mean_beta_sq, q  # Properties of proposed distribution
            double[::1] N  # Sample size per SNP
            long[:, ::1] ld_bound

        for c, c_size in self.shapes.items():

            # Updates for sigma_beta variational parameters:
            self.var_sigma_beta[c] = self.sigma_epsilon / (self.Nj[c][:, None] + self.sigma_epsilon / self.sigma_beta[c])

            # Set the numpy vectors into memoryviews for fast access:
            pi = self.pi[c]
            sigma_beta = self.sigma_beta[c]
            std_beta = self.std_beta[c]
            var_gamma = self.var_gamma[c]
            var_mu_beta = self.var_mu_beta[c]
            var_sigma_beta = self.var_sigma_beta[c]
            mean_beta = self.mean_beta[c]
            mean_beta_sq = self.mean_beta_sq[c]
            ld_bound = self.ld_bounds[c]
            N = self.Nj[c]
            q = np.zeros(shape=c_size[0])

            for j, Dj in enumerate(self.ld[c]):

                start, end = ld_bound[:, j]
                j_idx = j - start

                # The numerator for all the mu_beta updates:
                mu_beta_j = (std_beta[j] - dot(Dj, mean_beta[start: end], self.threads) + Dj[j_idx]*mean_beta[j])
                # The denominator for normalizing the gammas (start with the null component):
                gamma_denom = 1. - np.sum(pi[j])

                for k in range(self.K):
                    # Compute the mu beta for component `k`
                    var_mu_beta[j, k] = mu_beta_j / (1. + self.sigma_epsilon / (N[j] * sigma_beta[j, k]))
                    # Compute the unnormalized gamma for component `k`
                    var_gamma[j, k] = exp(log(pi[j, k]) + .5*log(var_sigma_beta[j, k] / sigma_beta[j, k]) +
                                        (.5/var_sigma_beta[j, k])*var_mu_beta[j, k]*var_mu_beta[j, k])
                    # Add the gamma to the sum
                    gamma_denom += var_gamma[j, k]

                # Normalize the gammas and update the beta statistics:
                mean_beta[j] = 0.
                mean_beta_sq[j] = 0.

                for k in range(self.K):
                    var_gamma[j, k] = clip(var_gamma[j, k] / gamma_denom, 1e-6, 1. - 1e-6)

                    mean_beta[j] += var_gamma[j, k]*var_mu_beta[j, k]
                    mean_beta_sq[j] += var_gamma[j, k] * (var_mu_beta[j, k] * var_mu_beta[j, k] + var_sigma_beta[j, k])

                # Update the q factor:
                if j_idx > 0:
                    # Update the q factor for snp j by adding the contribution of previous SNPs.
                    q[j] = dot(Dj[:j_idx], mean_beta[start: j], self.threads)
                    # Update the q factors for all previously updated SNPs that are in LD with SNP j
                    q[start: j] = elementwise_add_mult(q[start: j], Dj[:j_idx], mean_beta[j], self.threads)

            self.var_gamma[c] = np.array(var_gamma)
            self.var_mu_beta[c] = np.array(var_mu_beta)

            self.mean_beta[c] = np.array(mean_beta)
            self.mean_beta_sq[c] = np.array(mean_beta_sq)
            self.q[c] = np.array(q)

    cpdef update_pi(self):
        """
        Update the prior mixture proportions
        """

        if 'pis' not in self.fix_params:

            pis = []

            for k in range(self.K):
                pis.append(dict_sum({c: g[:, k] for c, g in self.var_gamma.items()}))

            pi_estimate = np.array(pis)

            if 'pi' in self.fix_params:
                # If the user provides an estimate for the proportion of causal variants,
                # update the pis such that the proportion of SNPs in the null component becomes 1. - pi.
                pi_estimate = self.fix_params['pi']*pi_estimate / pi_estimate.sum()
            else:
                pi_estimate /= self.M

            # Clip and normalize:
            all_pis = np.concatenate(pi_estimate, [1. - pi_estimate.sum()])
            pi_estimate = np.clip(all_pis, 1./self.M, 1. - 1./self.M)
            pi_estimate /= np.sum(pi_estimate)

            # Set pi to the new estimate:
            self.pi = dict_repeat(pi_estimate[:-1], self.shapes)

    cpdef update_sigma_beta(self):
        """
        Update the prior variance on the effect size, sigma_beta
        """

        if 'sigma_betas' not in self.fix_params:

            if self.d is None:

                sigma_betas = []

                for k in range(self.K):

                    sigma_betas.append(
                        dict_sum(
                            {c: (self.var_gamma[c][:, k]*(self.var_mu_beta[c][:, k]**2 +
                                                          self.var_sigma_beta[c][:, k])).sum()
                             for c in self.shapes}
                        ) / dict_sum({c: g[:, k] for c, g in self.var_gamma.items()})
                    )

                sigma_beta_estimate = np.array(sigma_betas)
            else:
                # If a list of multipliers is provided,
                # estimate the global sigma_beta and then multiply it
                # by the per-component multiplier to get the final sigma_betas.
                sigma_beta_estimate = dict_sum(
                            {c: (self.var_gamma[c]*(self.var_mu_beta[c]**2 +
                                                          self.var_sigma_beta[c]) / self.d).sum()
                             for c in self.shapes}
                        ) / dict_sum(self.var_gamma)
                sigma_beta_estimate = self.d*sigma_beta_estimate

            # Clip value:
            sigma_beta_estimate = np.clip(sigma_beta_estimate, 1e-12, 1. - 1e-12)
            # Set sigma_beta to the new estimate
            self.sigma_beta = dict_repeat(sigma_beta_estimate, self.shapes)

    cpdef update_sigma_epsilon(self):
        """
        Update the residual variance, sigma_epsilon.
        """

        if 'sigma_epsilon' not in self.fix_params:

            sig_eps = 0.

            for c, c_size in self.shapes.items():

                sig_eps += (
                        - 2. * dot(self.mean_beta[c], self.std_beta[c], self.threads) +
                        mt_sum(self.mean_beta_sq[c], self.threads) +
                        dot(self.mean_beta[c], self.q[c], self.threads)
                )

            self.sigma_epsilon = clip(1. + sig_eps, 1e-12, 1. - 1e-12)

    cpdef m_step(self):
        """
        In the M-step, update the global hyperparameters of the model.
        """

        self.update_pi()
        self.update_sigma_beta()
        self.update_sigma_epsilon()

    cpdef objective(self):
        """
        Compute the objective, Evidence Lower-BOund (ELBO)
        from summary statistics.
        """

        # Concatenate the dictionary items for easy computation:
        var_gamma = dict_concat(self.var_gamma)
        null_gamma = 1. - var_gamma.sum(axis=1)  # The gamma for the null component
        var_mu_beta = dict_concat(self.var_mu_beta)
        var_sigma_beta = dict_concat(self.var_sigma_beta)

        pi = dict_concat(self.pi)
        null_pi = 1. - pi.sum(axis=1)  # The pi for the null component
        sigma_beta = dict_concat(self.sigma_beta)

        q = dict_concat(self.q)
        mean_beta = dict_concat(self.mean_beta)
        mean_beta_sq = dict_concat(self.mean_beta_sq)

        std_beta = dict_concat(self.std_beta)

        # The ELBO is made of two components:
        elbo = 0.  # (1) log of joint density

        # -----------------------------------------------
        # (1) Compute the log of the joint density:

        #
        # (1.1) The following terms are an expansion of ||Y - X\beta||^2
        #
        # -N/2log(2pi*sigma_epsilon)
        elbo -= .5 * self.N * log(2 * np.pi * self.sigma_epsilon)

        # -Y'Y/(2*sigma_epsilon), where we assume Y'Y = N
        elbo -= .5*(self.N/self.sigma_epsilon)

        # + (1./sigma_epsilon)*\beta*(XY), where we assume XY = N\hat{\beta}
        elbo += (self.N/self.sigma_epsilon)*dot(mean_beta, std_beta, self.threads)

        # (-1/2sigma_epsilon)\beta'X'X\beta, where we assume X_j'X_j = N
        # Note that the q factor is equivalent to X'X\beta (excluding diagonal)
        elbo -= .5*(self.N/self.sigma_epsilon)*(dot(mean_beta, q, self.threads) +
                                                  mt_sum(mean_beta_sq, self.threads))

        elbo -= (var_gamma * np.log(var_gamma / pi)).sum()
        elbo -= (null_gamma * np.log(null_gamma / null_pi)).sum()

        elbo += .5 * (var_gamma * (1. + np.log(var_sigma_beta / sigma_beta) -
                                   (var_mu_beta ** 2 + var_sigma_beta) / sigma_beta)).sum()

        return elbo

    cpdef get_proportion_causal(self):
        """
        Estimate the proportion of causal variants for the trait.
        NOTE: This assumes that the last component is null
        """
        return dict_mean({c: pis.sum(axis=1) for c, pis in self.pi.items()})

    cpdef get_heritability(self):
        """
        Estimate the heritability of the trait
        """

        sigma_g = np.sum([
            mt_sum(self.mean_beta_sq[c], self.threads) +
            dot(self.q[c], self.mean_beta[c], self.threads)
            for c in self.shapes
        ])

        h2g = sigma_g / (sigma_g + self.sigma_epsilon)

        return h2g

    cpdef to_theta_table(self):
        """
        Generate a table of the hyperparameters
        """

        theta_table = {
            'Parameter': [f'pi_{i+1}' for i in range(self.K)] +
                         [f'sigma_beta_{i+1}' for i in range(self.K)] +
                         ['Proportion_causal', 'Residual_variance', 'Heritability'],
            'Value': list(dict_mean(self.pi, axis=0)) +
                     list(dict_mean(self.sigma_beta, axis=0)) +
                     [self.get_proportion_causal(), self.sigma_epsilon, self.get_heritability()]
        }

        return pd.DataFrame(theta_table)

    cpdef write_inferred_theta(self, f_name):
        """
        Write the inferred (and fixed) hyperparameters to file.
        :param f_name: The file name
        """

        # Write the table to file:
        try:
            self.to_theta_table().to_csv(f_name, sep="\t", index=False)
        except Exception as e:
            raise e

    cpdef fit(self, max_iter=1000, theta_0=None, continued=False, ftol=1e-5, xtol=1e-5, max_elbo_drops=10):
        """
        Fit the model parameters to data.

        :param max_iter: Maximum number of iterations. 
        :param theta_0: A dictionary of values to initialize the hyperparameters
        :param continued: If true, continue the model fitting for more iterations.
        :param ftol: The tolerance threshold for the objective (ELBO)
        :param xtol: The tolerance threshold for the parameters (mean beta)
        :param max_elbo_drops: The maximum number of times the objective is allowed to drop.
        """

        if not continued:
            self.initialize(theta_0)

        if self.load_ld:
            self.gdl.load_ld()

        elbo_dropped_count = 0
        converged = False

        if self.verbose:
            print("> Performing model fit...")
            print(f"> Using up to {self.threads} threads.")

        for i in tqdm(range(1, max_iter + 1), disable=not self.verbose):
            self.e_step()
            self.m_step()

            self.history['ELBO'].append(self.objective())

            if i > 1:

                curr_elbo = self.history['ELBO'][i - 1]
                prev_elbo = self.history['ELBO'][i - 2]

                if curr_elbo < prev_elbo:
                    elbo_dropped_count += 1
                    warnings.warn(f"Iteration {i}: ELBO dropped from {prev_elbo:.6f} "
                                  f"to {curr_elbo:.6f}.")

                if abs(curr_elbo - prev_elbo) <= ftol:
                    print(f"Converged at iteration {i} | ELBO: {curr_elbo:.6f}")
                    break
                elif max([np.abs(v - self.inf_beta[c]).max() for c, v in self.mean_beta.items()]) <= xtol:
                    print(f"Converged at iteration {i} | ELBO: {curr_elbo:.6f}")
                    break
                elif elbo_dropped_count > max_elbo_drops:
                    warnings.warn("The optimization is halted due to numerical instabilities!")
                    break

                if i > 2:
                    if abs((curr_elbo - prev_elbo) / prev_elbo) > 1. and abs(curr_elbo - prev_elbo) > 1e3:
                        raise OptimizationDivergence(f"Stopping at iteration {i}: "
                                                     f"The optimization algorithm is not converging!\n"
                                                     f"Previous ELBO: {prev_elbo:.6f} | "
                                                     f"Current ELBO: {curr_elbo:.6f}")
                    elif self.get_heritability() >= 1.:
                        raise OptimizationDivergence(f"Stopping at iteration {i}: "
                                                     f"The optimization algorithm is not converging!\n"
                                                     f"Value of estimated heritability exceeded 1.")

            self.pip = {c: v.sum(axis=1) for c, v in self.var_gamma.items()}
            self.inf_beta = {c: v.copy() for c, v in self.mean_beta.items()}

        if i == max_iter:
            warnings.warn("Max iterations reached without convergence. "
                          "You may need to run the model for more iterations.")

        if self.verbose:
            print(f"> Final ELBO: {self.history['ELBO'][i-1]:.6f}")
            print(f"> Estimated heritability: {self.get_heritability():.6f}")
            print(f"> Estimated proportion of causal variants: {self.get_proportion_causal():.6f}")

        return self
