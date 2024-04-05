import pandas as pd
import numpy as np

from magenpy.stats.h2.ldsc import simple_ldsc
from .VIPRS import VIPRS
from .vi.e_step import e_step_mixture
from .vi.e_step_cpp import cpp_e_step_mixture
from ..utils.compute_utils import dict_sum, dict_mean


class VIPRSMix(VIPRS):
    """
    A class for the Variational Inference for Polygenic Risk Scores (VIPRS) model
    parametrized with the sparse mixture prior on the effect sizes. The class inherits
    many of the methods and attributes from the `VIPRS` class unchanged. However,
    there are many important updates and changes to the model, including the dimensionality
    of the arrays representing the variational parameters.

    Details for the algorithm can be found in the Supplementary Material of the following paper:

    > Zabad S, Gravel S, Li Y. Fast and accurate Bayesian polygenic risk modeling with variational inference.
    Am J Hum Genet. 2023 May 4;110(5):741-761. doi: 10.1016/j.ajhg.2023.03.009.
    Epub 2023 Apr 7. PMID: 37030289; PMCID: PMC10183379.

    :ivar K: The number of causal (i.e. non-null) components in the mixture prior (minimum 1). When `K=1`, this
    effectively reduces `VIPRSMix` to the `VIPRS` model.
    :ivar d: Multiplier for the prior on the effect size (vector of size K).

    """

    def __init__(self,
                 gdl,
                 K=1,
                 prior_multipliers=None,
                 **kwargs):

        """
        :param gdl: An instance of `GWADataLoader`
        :param K: The number of causal (i.e. non-null) components in the mixture prior (minimum 1). When `K=1`, this
            effectively reduces `VIPRSMix` to the `VIPRS` model.
        :param prior_multipliers: Multiplier for the prior on the effect size (vector of size K).
        :param kwargs: Additional keyword arguments to pass to the VIPRS model.
        """

        # Make sure that the matrices follow the C-contiguous order:
        kwargs['order'] = 'C'

        super().__init__(gdl, **kwargs)

        # Sanity checks:
        assert K > 0  # Check that there is at least 1 causal component
        self.K = K

        if prior_multipliers is not None:
            assert len(prior_multipliers) == K
            self.d = np.array(prior_multipliers).astype(self.float_precision)
        else:
            self.d = 2**np.linspace(-min(K - 1, 7), 0, K).astype(self.float_precision)

        # Populate/update relevant fields:
        self.shapes = {c: (shp, self.K) for c, shp in self.shapes.items()}
        self.Nj = {c: Nj[:, None].astype(self.float_precision, order=self.order) for c, Nj in self.Nj.items()}

    def initialize_theta(self, theta_0=None):
        """
        Initialize the global hyperparameters of the model
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

            if 'tau_betas' in theta_0:

                # If tau_betas are given, use them to initialize sigma_epsilon

                self.tau_beta = theta_0['tau_betas']

                self.sigma_epsilon = np.clip(1. - np.dot(1./self.tau_beta, self.pi),
                                             a_min=self.float_resolution,
                                             a_max=1. - self.float_resolution)

            elif 'tau_beta' in theta_0:
                # NOTE: Here, we assume the provided `tau_beta` is a scalar.
                # This is different from `tau_betas`

                assert self.d is not None

                self.tau_beta = theta_0['tau_beta'] * self.d
                # Use the provided tau_beta to initialize sigma_epsilon.
                # First, we derive a naive estimate of the heritability, based on the following equation:
                # h2g/M = \sum_k pi_k \tau_k
                # Where the per-SNP heritability is defined by the sum over the mixtures.

                # Step (1): Given the provided tau_beta and associated multipliers,
                # obtain a naive estimate of the heritability:
                h2g_estimate = (self.n_snps*self.pi/self.tau_beta).sum()
                # Step (2): Set sigma_epsilon to 1 - h2g_estimate:
                self.sigma_epsilon = np.clip(1. - h2g_estimate,
                                             a_min=self.float_resolution,
                                             a_max=1. - self.float_resolution)

            else:
                # If neither sigma_beta nor sigma_epsilon are given,
                # then initialize using the SNP heritability estimate based on summary statistics

                try:
                    naive_h2g = np.clip(simple_ldsc(self.gdl), 1e-3, 1. - 1e-3)
                except Exception as e:
                    naive_h2g = np.random.uniform(low=.001, high=.999)

                self.sigma_epsilon = 1. - naive_h2g

                global_tau = (self.n_snps * np.dot(1./self.d, self.pi) / naive_h2g)

                self.tau_beta = self.d*global_tau
        else:

            # If sigma_epsilon is given, use it in the initialization

            self.sigma_epsilon = theta_0['sigma_epsilon']

            # Initialize tau_betas
            if 'tau_betas' in theta_0:
                self.tau_beta = theta_0['tau_betas']
            elif 'tau_beta' in theta_0:
                self.tau_beta = np.repeat(theta_0['tau_beta'], self.K)
            else:
                # If not provided, initialize using sigma_epsilon value
                global_tau = (self.n_snps * np.dot(1./self.d, self.pi) / (1. - self.sigma_epsilon))

                self.tau_beta = self.d * global_tau

        # Cast all the hyperparameters to conform to the precision set by the user:
        self.sigma_epsilon = np.dtype(self.float_precision).type(self.sigma_epsilon)
        self.tau_beta = np.dtype(self.float_precision).type(self.tau_beta)
        self.pi = np.dtype(self.float_precision).type(self.pi)

    def e_step(self):
        """
        Run the E-Step of the Variational EM algorithm.
        Here, we update the variational parameters for each variant using coordinate
        ascent optimization techniques. The update equations are outlined in
        the Supplementary Material of the following paper:

        > Zabad S, Gravel S, Li Y. Fast and accurate Bayesian polygenic risk modeling with variational inference.
        Am J Hum Genet. 2023 May 4;110(5):741-761. doi: 10.1016/j.ajhg.2023.03.009.
        Epub 2023 Apr 7. PMID: 37030289; PMCID: PMC10183379.
        """

        for c, shapes in self.shapes.items():

            # Get the priors:
            tau_beta = self.get_tau_beta(c)
            pi = self.get_pi(c)

            # Updates for tau variational parameters:
            self.var_tau[c] = (self.Nj[c] / self.sigma_epsilon) + tau_beta

            if isinstance(self.pi, dict):
                log_null_pi = (np.log(1. - self.pi[c].sum(axis=1)))
            else:
                log_null_pi = np.ones_like(self.eta[c])*np.log(1. - self.pi.sum())

            # Compute some quantities that are needed for the per-SNP updates:
            mu_mult = self.Nj[c] / (self.var_tau[c] * self.sigma_epsilon)
            u_logs = np.log(pi) - np.log(1. - pi) + .5 * (np.log(tau_beta) - np.log(self.var_tau[c]))

            if self.use_cpp:
                cpp_e_step_mixture(self.ld_left_bound[c],
                                   self.ld_indptr[c],
                                   self.ld_data[c],
                                   self.std_beta[c],
                                   self.var_gamma[c],
                                   self.var_mu[c],
                                   self.eta[c],
                                   self.q[c],
                                   self.eta_diff[c],
                                   log_null_pi,
                                   u_logs,
                                   0.5*self.var_tau[c],
                                   mu_mult,
                                   self.dequantize_scale,
                                   self.threads,
                                   self.use_blas,
                                   self.low_memory)
            else:
                e_step_mixture(self.ld_left_bound[c],
                               self.ld_indptr[c],
                               self.ld_data[c],
                               self.std_beta[c],
                               self.var_gamma[c],
                               self.var_mu[c],
                               self.eta[c],
                               self.q[c],
                               self.eta_diff[c],
                               log_null_pi,
                               u_logs,
                               0.5*self.var_tau[c],
                               mu_mult,
                               self.threads,
                               self.use_blas,
                               self.low_memory)

        self.zeta = self.compute_zeta()

    def update_pi(self):
        """
        Update the prior mixing proportions `pi`
        """

        if 'pis' not in self.fix_params:

            pi_estimate = dict_sum(self.var_gamma, axis=0)

            if 'pi' in self.fix_params:
                # If the user provides an estimate for the total proportion of causal variants,
                # update the pis such that the proportion of SNPs in the null component becomes 1. - pi.
                pi_estimate = self.fix_params['pi']*pi_estimate / pi_estimate.sum()
            else:
                pi_estimate /= self.n_snps

            # Set pi to the new estimate:
            self.pi = pi_estimate

    def update_tau_beta(self):
        """
        Update the prior precision (inverse variance) for the effect sizes, `tau_beta`
        """

        if 'tau_betas' not in self.fix_params:

            # If a list of multipliers is provided,
            # estimate the global sigma_beta and then multiply it
            # by the per-component multiplier to get the final sigma_betas.

            zetas = sum(self.compute_zeta(sum_axis=0).values())

            tau_beta_estimate = np.sum(self.pi)*self.m / np.dot(self.d, zetas)
            tau_beta_estimate = self.d*tau_beta_estimate

            self.tau_beta = np.clip(tau_beta_estimate, a_min=1., a_max=None)

    def get_null_pi(self, chrom=None):
        """
        Get the proportion of SNPs in the null component
        :param chrom: If provided, get the mixing proportion for the null component on a given chromosome.
        :return: The value of the mixing proportion for the null component
        """

        pi = self.get_pi(chrom=chrom)

        if isinstance(pi, dict):
            return {c: 1. - c_pi.sum(axis=1) for c, c_pi in pi.items()}
        else:
            return 1. - np.sum(pi)

    def get_proportion_causal(self):
        """
        :return: The proportion of variants in the non-null components.
        """
        if isinstance(self.pi, dict):
            dict_mean({c: pis.sum(axis=1) for c, pis in self.pi.items()})
        else:
            return np.sum(self.pi)

    def get_average_effect_size_variance(self):
        """
        :return: The average per-SNP variance for the prior mixture components
        """

        avg_sigma = super().get_average_effect_size_variance()

        try:
            return avg_sigma.sum()
        except Exception:
            return avg_sigma

    def compute_pip(self):
        """
        :return: The posterior inclusion probability
        """
        return {c: gamma.sum(axis=1) for c, gamma in self.var_gamma.items()}

    def compute_eta(self):
        """
        :return: The mean for the effect size under the variational posterior.
        """
        return {c: (v * self.var_mu[c]).sum(axis=1) for c, v in self.var_gamma.items()}

    def compute_zeta(self, sum_axis=1):
        """
        :return: The expectation of the squared effect size under the variational posterior.
        """
        return {c: (v * (self.var_mu[c] ** 2 + (1./self.var_tau[c]))).sum(axis=sum_axis)
                for c, v in self.var_gamma.items()}

    def to_theta_table(self):
        """
        :return: A `pandas` DataFrame containing information about the estimated hyperparameters of the model.
        """

        table = super().to_theta_table()

        extra_theta = []

        if isinstance(self.pi, dict):
            pis = list(dict_mean(self.pi, axis=0))
        else:
            pis = self.pi

        for i in range(self.K):
            extra_theta.append({'Parameter': f'pi_{i + 1}', 'Value': pis[i]})

        return pd.concat([table, pd.DataFrame(extra_theta)])
