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
from tqdm import tqdm
from scipy.stats import invgamma
from libc.math cimport log, sqrt
from .PRSModel cimport PRSModel
from .c_utils cimport dot, sigmoid, elementwise_add_mult, clip
from .run_stats cimport RunStats, RunStatsVec


cdef class GibbsPRS(PRSModel):

    def __init__(self, gdl,
                 beta_prior=(10., 1e4), sigma_beta_prior=(1e-4, 1e-4), sigma_epsilon_prior=(1., 1.),
                 fix_params=None, load_ld=True):
        """
        :param gdl: An instance of GWAS data loader
        """

        super().__init__(gdl)

        self.beta_prior = beta_prior
        self.sigma_beta_prior = sigma_beta_prior
        self.sigma_epsilon_prior = sigma_epsilon_prior

        self.q = {}
        self.s_beta = {}
        self.rs_beta = {}
        self.s_gamma = {}
        self.rs_gamma = {}

        self.load_ld = load_ld
        self.ld = self.gdl.get_ld_matrices()
        self.ld_bounds = self.gdl.get_ld_boundaries()
        self.beta_hat = self.gdl.compute_xy_per_snp()

        self.fix_params = fix_params or {}

        self.initialize()

    cpdef initialize(self):

        print("> Initializing model parameters")

        self.initialize_theta()
        self.initialize_local_params()
        self.initialize_running_stats()

    cpdef initialize_running_stats(self):

        self.rs_beta = {c: RunStatsVec(c_size) for c, c_size in self.shapes.items()}
        self.rs_gamma = {c: RunStatsVec(c_size) for c, c_size in self.shapes.items()}

        self.rs_pi = RunStats()
        self.rs_sigma_beta = RunStats()
        self.rs_sigma_epsilon = RunStats()
        self.rs_h2g = RunStats()

    cpdef initialize_theta(self):

        if 'sigma_beta' not in self.fix_params:
            self.sigma_beta = np.random.uniform(low=1e-6, high=.1)
        else:
            self.sigma_beta = self.fix_params['sigma_beta']

        if 'sigma_epsilon' not in self.fix_params:
            self.sigma_epsilon = np.random.uniform(low=.5, high=1.)
        else:
            self.sigma_epsilon = self.fix_params['sigma_epsilon']

        if 'pi' not in self.fix_params:
            self.pi = np.random.uniform(low=1. / self.M, high=.5)
        else:
            self.pi = self.fix_params['pi']

    cpdef initialize_local_params(self):

        self.s_beta = {}
        self.s_gamma = {}

        for c, c_size in self.shapes.items():

            self.s_gamma[c] = np.random.binomial(1, self.pi, size=c_size).astype(np.float64)
            self.s_beta[c] = np.random.normal(scale=np.sqrt(self.sigma_beta), size=c_size)

    cpdef sample_local_parameters(self):
        """

        :return:
        """

        cdef:
            unsigned int j, start, end, j_idx
            double u_j, mu_beta_j
            double[::1] prod, s_beta, s_gamma, beta_hat, s_unif, s_norm, q, Dj
            long[:, ::1] ld_bound
            # The denominator for the mu_beta updates:
            double denom = (1. + self.sigma_epsilon / (self.N * self.sigma_beta))
            # The log(pi) for the gamma updates
            double logodds_pi = log(self.pi / (1. - self.pi))
            # The variance of the conditional distribution:
            double s_var = self.sigma_epsilon / (self.N + self.sigma_epsilon / self.sigma_beta)
            double sqrt_s_var = sqrt(s_var)
            double sqrt_prior_var = sqrt(self.sigma_beta)

        for c, c_size in self.shapes.items():
            beta_hat = self.beta_hat[c]
            s_beta = self.s_beta[c]
            s_gamma = self.s_gamma[c]
            ld_bound = self.ld_bounds[c]
            s_unif = np.random.uniform(size=c_size)
            s_norm = np.random.normal(size=c_size)
            q = np.zeros(shape=c_size)

            prod = np.multiply(s_gamma, s_beta)

            for j, Dj in enumerate(self.ld[c]):

                start, end = ld_bound[:, j]
                j_idx = j - start

                mu_beta_j = (beta_hat[j] - dot(Dj, prod[start: end]) +
                                  Dj[j_idx]*prod[j]) / denom

                u_j = (logodds_pi + .5*log(s_var / self.sigma_beta) +
                       (.5/s_var)*mu_beta_j*mu_beta_j)

                if s_unif[j] > sigmoid(u_j):
                    s_gamma[j] = 1.
                    s_beta[j] = mu_beta_j + sqrt_s_var * s_norm[j]
                else:
                    s_gamma[j] = 0.
                    s_beta[j] = sqrt_prior_var*s_norm[j]

                prod[j] = s_gamma[j]*s_beta[j]

                if j_idx > 0:
                    # Update the q factor for snp i by adding the contribution of previous SNPs.
                    q[j] = dot(Dj[:j_idx], prod[start: j])

                    if prod[j] != 0.:
                        # Update the q factors for all previously updated SNPs that are in LD with SNP j
                        q[start: j] = elementwise_add_mult(q[start: j], Dj[:j_idx], prod[j])

            self.q[c] = np.array(q)

    cpdef sample_pi(self):

        # Update pi:
        if 'pi' not in self.fix_params:

            n_causal = np.sum([
                np.sum(self.s_gamma[c])
                for c in self.shapes
            ])

            self.pi = np.random.beta(self.beta_prior[0] + n_causal,
                                     self.beta_prior[1] + (self.M - n_causal))

    cpdef sample_sigma_beta(self):

        if 'sigma_beta' not in self.fix_params:

            n_causal = np.sum([
                np.sum(self.s_gamma[c])
                for c in self.shapes
            ])

            sum_beta_sq = np.sum([
                np.sum(self.s_gamma[c]*self.s_beta[c]**2)
                for c in self.shapes
            ])

            self.sigma_beta = invgamma(self.sigma_beta_prior[0] + .5*n_causal,
                                       loc=0.,
                                       scale=self.sigma_beta_prior[1] + .5*sum_beta_sq).rvs()

    cpdef sample_sigma_epsilon(self):

        if 'sigma_epsilon' not in self.fix_params:

            sigma_g = 0.
            ssr = 0.

            for c, c_size in self.shapes.items():
                beta_hat = self.beta_hat[c]
                prod = np.multiply(self.s_gamma[c], self.s_beta[c])
                q = self.q[c]

                sigma_g += (
                        np.dot(prod, prod) +
                        np.dot(prod, q)
                )

                ssr += - 2. * np.dot(prod, beta_hat)

            ssr = clip(1. + ssr + sigma_g, 1e-12, 1e12)

            self.sigma_g = sigma_g
            self.sigma_epsilon = invgamma(self.sigma_epsilon_prior[0] + .5*self.N,
                                          loc=0.,
                                          scale=self.sigma_epsilon_prior[1] + .5*self.N*ssr).rvs()


    cpdef sample_global_parameters(self):

        self.sample_pi()
        self.sample_sigma_beta()
        self.sample_sigma_epsilon()

    cpdef get_proportion_causal(self):

        return self.rs_pi.mean()

    cpdef get_heritability(self):

        return self.rs_h2g.mean()

    cpdef fit(self, n_samples=10000, burn_in=2000, continued=False):

        if not continued:
            self.initialize()

        if self.load_ld:
            self.gdl.load_ld()

        self.pip = self.s_gamma.copy()
        self.inf_beta = self.s_beta.copy()

        print("> Sampling from the posterior...")

        for i in tqdm(range(n_samples)):
            self.sample_local_parameters()
            self.sample_global_parameters()

            if i + 1 > burn_in:

                for c in self.shapes:
                    self.rs_gamma[c].push(self.s_gamma[c])
                    self.rs_beta[c].push(self.s_beta[c])

                self.rs_pi.push(self.pi)
                self.rs_sigma_beta.push(self.sigma_beta)
                self.rs_sigma_epsilon.push(self.sigma_epsilon)
                self.rs_h2g.push(self.sigma_g / (self.sigma_g + self.sigma_epsilon))

        self.pip = {c : rs.mean() for c, rs in self.rs_gamma.items()}
        self.inf_beta = {c : rs.mean() for c, rs in self.rs_beta.items()}

        return self
