# cython: linetrace=False
# cython: profile=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True

import pandas as pd
import numpy as np
cimport numpy as np

from magenpy.stats.h2.ldsc import simple_ldsc
from .VIPRS cimport VIPRS
from viprs.utils.math_utils cimport elementwise_add_mult, clip, softmax
from viprs.utils.compute_utils import dict_sum, dict_mean


cdef class VIPRSMix(VIPRS):

    cdef public:
        int K  # The number of mixture components
        object d  # Multipliers for the prior on the effect size

    def __init__(self, gdl, K=1, prior_multipliers=None, fix_params=None,
                 load_ld='auto', tracked_theta=None, verbose=True, threads=1):

        """
        :param gdl: An instance of GWAS data loader
        :param K: The number of causal (i.e. non-null) components in the mixture prior (minimum 1).
        :param prior_multipliers: Multiplier for the prior on the effect size (vector of size K)
        :param fix_params: A dictionary of parameters with their fixed values.
        :param load_ld: A flag that specifies whether to load the LD matrix to memory.
        :param tracked_theta: A list of hyperparameters to track throughout the optimization procedure. Useful
        for debugging/model checking. Currently, we allow the user to track the following:
            - The proportion of causal variants (`pi`).
            - The heritability ('heritability').
            - The residual variance (`sigma_epsilon`).
        :param verbose: Verbosity of the information printed to standard output
        :param threads: The number of threads to use (experimental)
        """

        super().__init__(gdl, fix_params=fix_params, load_ld=load_ld,
                         tracked_theta=tracked_theta, verbose=verbose, threads=threads)

        # Sanity checks:
        assert K > 0  # Check that there are at least 1 causal component

        if prior_multipliers is not None:
            assert len(prior_multipliers) == K
            self.d = np.array(prior_multipliers)

        # Populate relevant fields:
        self.K = K
        self.shapes = {c: (shp, self.K) for c, shp in self.shapes.items()}
        self.Nj = {c: Nj[:, None] for c, Nj in self.Nj.items()}

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
            self.pi = theta_0['pis']
        else:
            if 'pi' in theta_0:
                overall_pi = theta_0['pi']
            else:
                overall_pi = np.random.uniform(low=max(0.005, 1. / self.n_snps), high=.1)

            self.pi = overall_pi*np.random.dirichlet(np.ones(self.K))

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

                self.sigma_beta = theta_0['sigma_betas']

                self.sigma_epsilon = clip(1. - np.dot(self.sigma_beta, self.pi),
                                          1e-12, 1. - 1e-12)

            elif 'sigma_beta' in theta_0:
                # NOTE: Here, we assume the provided `sigma_beta` is a scalar.
                # This is different from `sigma_betas`

                assert self.d is not None

                self.sigma_beta = theta_0['sigma_beta'] * self.d
                # Use the provided sigma_beta to initialize sigma_epsilon.
                # First, we derive a naive estimate of the heritability, based on the following equation:
                # h2g/M = \sum_k pi_k \sigma_k
                # Where the per-SNP heritability is defined by the sum over the mixtures.

                # Step (1): Given the provided sigma and associated multipliers,
                # obtain a naive estimate of the heritability:
                h2g_estimate = (self.sigma_beta*self.n_snps*self.pi).sum()
                # Step (2): Set sigma_epsilon to 1 - h2g_estimate:
                self.sigma_epsilon = clip(1. - h2g_estimate, 1e-12, 1. - 1e-12)

            else:
                # If neither sigma_beta nor sigma_epsilon are given,
                # then initialize using the SNP heritability estimate based on summary statistics

                try:
                    naive_h2g = clip(simple_ldsc(self.gdl), 1e-3, 1. - 1e-3)
                except Exception as e:
                    naive_h2g = np.random.uniform(low=.001, high=.999)

                self.sigma_epsilon = 1. - naive_h2g

                if self.d is None:
                    mult = 10**np.arange(-(self.K - 1), 1).astype(float)
                else:
                    mult = self.d

                self.sigma_beta = mult*naive_h2g / (self.n_snps * (self.pi * mult).sum())
        else:

            # If sigma_epsilon is given, use it in the initialization

            self.sigma_epsilon = theta_0['sigma_epsilon']

            # Initialize sigma_betas
            if 'sigma_betas' in theta_0:
                self.sigma_beta = theta_0['sigma_betas']
            elif 'sigma_beta' in theta_0:
                self.sigma_beta = np.repeat(theta_0['sigma_beta'], self.K)
            else:
                # If not provided, initialize using sigma_epsilon value
                if self.d is None:
                    mult = 10**np.arange(-(self.K - 1), 1).astype(float)
                else:
                    mult = self.d

                self.sigma_beta = mult*(1. - self.sigma_epsilon) / (self.n_snps * (self.pi * mult).sum())

    cpdef e_step(self):
        """
        In the E-step, update the variational parameters for each SNP 
        in a coordinate-wise fashion.
        """

        # Initialize memoryviews objects for fast access
        cdef:
            unsigned int j, k, start, end
            double mu_beta_j, eta_diff
            double[::1] u_j, log_null_pi
            double[::1] std_beta, Dj  # Inputs
            double[:, ::1] var_gamma, var_mu, var_sigma  # Variational parameters
            double[:, ::1] mu_mult, u_logs, recip_sigma  # Helpers + other quantities that we need inside the for loop
            double[::1] eta, q  # Properties of proposed distribution
            long[:, ::1] ld_bound

        for c, shapes in self.shapes.items():

            # Get the priors:
            sigma_beta = self.get_sigma_beta(c)
            pi = self.get_pi(c)

            if isinstance(self.pi, dict):
                log_null_pi = np.log(1. - self.pi[c].sum(axis=1))
            else:
                log_null_pi = np.repeat(np.log(1. - self.pi.sum()), shapes[0])

            # Updates for sigma_beta variational parameters:
            self.var_sigma[c] = self.sigma_epsilon / (
                    self.inv_temperature*(self.Nj[c] + self.sigma_epsilon / sigma_beta)
            )

            # Compute some quantities that are needed for the per-SNP updates:
            mu_mult = self.inv_temperature*self.Nj[c] * self.var_sigma[c] / self.sigma_epsilon
            u_logs = np.log(pi) + .5 * np.log(self.var_sigma[c] / sigma_beta)
            recip_sigma = .5 / self.var_sigma[c]

            # Set the numpy vectors into memoryviews for fast access:
            std_beta = self.std_beta[c]
            var_gamma = self.var_gamma[c]
            var_mu = self.var_mu[c]
            var_sigma = self.var_sigma[c]
            eta = self.eta[c]
            ld_bound = self.ld_bounds[c]
            q = self.q[c]
            u_j = np.zeros(shape=self.K + 1)

            for j, Dj in enumerate(self.ld[c]):

                start, end = ld_bound[:, j]

                # Compute the variational mu betas and gammas:

                # This is the shared component across all mixtures
                mu_beta_j = std_beta[j] - q[j]

                for k in range(self.K):
                    # Compute the mu beta for component `k`
                    var_mu[j, k] = mu_beta_j * mu_mult[j, k]
                    # Compute the unnormalized gamma for component `k`
                    u_j[k] = self.inv_temperature*(u_logs[j, k] + recip_sigma[j, k]*var_mu[j, k]*var_mu[j, k])

                # Normalize the variational gammas:
                u_j[k+1] = self.inv_temperature*log_null_pi[j]
                u_j = softmax(u_j)

                # Compute the difference between the new and old values for the posterior mean:
                eta_diff = -eta[j]

                for k in range(self.K):
                    var_gamma[j, k] = clip(u_j[k], 1e-8, 1. - 1e-8)
                    eta_diff += var_gamma[j, k]*var_mu[j, k]

                # Update the q factors for all neighboring SNPs that are in LD with SNP j
                elementwise_add_mult(q[start: end], Dj, eta_diff)
                # Operation above updates the q factor for SNP j, so we correct that here:
                q[j] = q[j] - eta_diff

                # Update the posterior mean:
                eta[j] = eta[j] + eta_diff

            self.var_gamma[c] = np.asarray(var_gamma)
            self.var_mu[c] = np.asarray(var_mu)
            self.eta[c] = np.asarray(eta)
            self.q[c] = np.array(q)

        self.zeta = self.compute_zeta()

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
                pi_estimate /= self.n_snps

            # Clip and normalize:
            pi_estimate = np.concatenate([pi_estimate, [1. - pi_estimate.sum()]])
            pi_estimate /= np.sum(pi_estimate)

            # Set pi to the new estimate:
            self.pi = pi_estimate[:len(pi_estimate)-1]

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
                            {c: (self.var_gamma[c][:, k]*(self.var_mu[c][:, k]**2 +
                                                          self.var_sigma[c][:, k])).sum()
                             for c in self.shapes}
                        ) / dict_sum({c: g[:, k] for c, g in self.var_gamma.items()})
                    )

                sigma_beta_estimate = np.array(sigma_betas)
            else:
                # If a list of multipliers is provided,
                # estimate the global sigma_beta and then multiply it
                # by the per-component multiplier to get the final sigma_betas.
                sigma_beta_estimate = dict_sum(
                            {c: (self.var_gamma[c]*(self.var_mu[c]**2 +
                                                    self.var_sigma[c]) / self.d).sum()
                             for c in self.shapes}
                        ) / dict_sum(self.var_gamma)
                sigma_beta_estimate = self.d*sigma_beta_estimate

            # Clip values and set sigma_beta to the new estimate:
            self.sigma_beta = np.clip(sigma_beta_estimate, 1e-12, 1. - 1e-12)

    cpdef get_null_pi(self, chrom=None):

        pi = self.get_pi(chrom=chrom)

        if isinstance(pi, dict):
            return {c: 1. - c_pi.sum(axis=1) for c, c_pi in pi.items()}
        else:
            return 1. - np.sum(pi)

    cpdef get_proportion_causal(self):
        """
        Get the proportion of causal variants for the trait.
        """
        if isinstance(self.pi, dict):
            dict_mean({c: pis.sum(axis=1) for c, pis in self.pi.items()})
        else:
            return np.sum(self.pi)

    cpdef get_average_effect_size_variance(self):
        """
        Get the average per-SNP variance for the prior mixture components
        """

        avg_sigma = super(VIPRSMix, self).get_average_effect_size_variance()

        try:
            return avg_sigma.sum()
        except Exception:
            return avg_sigma

    cpdef compute_pip(self):
        """
        Compute the posterior inclusion probability
        """
        return {c: np.clip(gamma.sum(axis=1), a_min=1e-8, a_max=1. - 1e-8) for c, gamma in self.var_gamma.items()}

    cpdef compute_eta(self):
        """
        Compute the mean for the effect size under the variational posterior.
        """
        return {c: (v * self.var_mu[c]).sum(axis=1) for c, v in self.var_gamma.items()}

    cpdef compute_zeta(self):
        """
        Compute the expectation of the squared effect size under the variational posterior.
        """
        return {c: (v * (self.var_mu[c] ** 2 + self.var_sigma[c])).sum(axis=1)
                for c, v in self.var_gamma.items()}

    cpdef to_theta_table(self):

        table = super(VIPRSMix, self).to_theta_table()

        extra_theta = []

        if isinstance(self.pi, dict):
            pis = list(dict_mean(self.pi, axis=0))
        else:
            pis = self.pi

        for i in range(self.K):
            extra_theta.append({'Parameter': f'pi_{i + 1}', 'Value': pis[i]})

        return pd.concat([table, pd.DataFrame(extra_theta)])
