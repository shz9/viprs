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
from scipy.stats import invgamma
from libc.math cimport exp, log, sqrt
from .PRSModel cimport PRSModel
from .c_utils cimport dot
from .run_stats cimport RunStats, RunStatsVec


cdef class prs_gibbs_sbayes(PRSModel):

    cdef public:
        double pi, sigma_beta, sigma_epsilon  # Global parameters
        double bxxb  # genotypic variance
        bint load_ld
        tuple beta_prior, sigma_beta_prior, sigma_epsilon_prior
        dict s_beta, s_gamma  # Sampled beta and gamma
        dict beta_hat, ld, ld_bounds, yy, sig_e_snp  # Inputs to the algorithm
        dict fix_params  # Helpers
        dict rs_gamma, rs_beta
        RunStats rs_pi, rs_sigma_beta, rs_sigma_epsilon, rs_h2g  # Running stats objects

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

        self.load_ld = load_ld
        self.ld = self.gdl.get_ld_matrices()
        self.ld_bounds = self.gdl.get_ld_boundaries()
        self.yy = {c: y.values for c, y in self.gdl.compute_yy_per_snp().items()}

        self.fix_params = fix_params or {}
        self.fix_params = {k: np.array(v).flatten() for k, v in self.fix_params}

        self.initialize()

    cpdef initialize(self):
        self.beta_hat = {c: b.values for c, b in self.gdl.beta_hats.items()}
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
            self.sigma_beta = np.random.uniform()
        else:
            self.sigma_beta = self.fix_params['sigma_beta'][0]

        if 'sigma_epsilon' not in self.fix_params:
            self.sigma_epsilon = np.random.uniform()
        else:
            self.sigma_epsilon = self.fix_params['sigma_epsilon'][0]

        if 'pi' not in self.fix_params:
            self.pi = np.random.uniform()
        else:
            self.pi = self.fix_params['pi'][0]

    cpdef initialize_local_params(self):

        self.s_beta = {}
        self.s_gamma = {}
        self.sig_e_snp = {}

        for c, c_size in self.shapes.items():

            self.s_gamma[c] = np.random.binomial(1, self.pi, size=c_size).astype(np.float)
            self.s_beta[c] = np.random.normal(scale=np.sqrt(self.sigma_beta), size=c_size)
            self.sig_e_snp[c] = np.repeat(self.sigma_epsilon, c_size)

    cpdef sample_local_parameters(self):
        """

        :return:
        """

        cdef unsigned int j
        cdef double u_j, mu_beta_j
        cdef double[::1] prod, s_beta, s_gamma, beta_hat, s_unif, s_norm, sig_e_snp, Dj
        cdef long[:, ::1] ld_bound

        # The prior variance parameter (can be defined in two ways):
        cdef double prior_var = self.sigma_beta

        cdef double denom, s_var, sqrt_prior_var = sqrt(prior_var)

        # The log(pi) for the gamma updates
        cdef double logodds_pi = log(self.pi / (1. - self.pi))

        for c, c_size in self.shapes.items():

            beta_hat = self.beta_hat[c]
            s_beta = self.s_beta[c]
            s_gamma = self.s_gamma[c]
            ld_bound = self.ld_bounds[c]
            sig_e_snp = self.sig_e_snp[c]
            s_unif = np.random.uniform(size=c_size)
            s_norm = np.random.normal(size=c_size)

            prod = np.multiply(s_gamma, s_beta)

            for j, Dj in enumerate(self.ld[c]):

                denom = (1. + sig_e_snp[j] / (self.N * prior_var))
                s_var = sig_e_snp[j] / (self.N + sig_e_snp[j] / prior_var)

                mu_beta_j = (beta_hat[j] - dot(Dj, prod[ld_bound[0, j]: ld_bound[1, j]]) +
                                  Dj[j - ld_bound[0, j]]*prod[j]) / denom

                u_j = (logodds_pi + .5*log(s_var / prior_var) +
                       (.5/s_var)*mu_beta_j*mu_beta_j)

                s_gamma[j] = 1. if s_unif[j] > 1./(1. + exp(-u_j)) else 0.

                if s_gamma[j] == 1.:
                    s_beta[j] = mu_beta_j + sqrt(s_var)*s_norm[j]
                else:
                    s_beta[j] = sqrt_prior_var*s_norm[j]

                prod[j] = s_gamma[j]*s_beta[j]

    cpdef sample_global_parameters(self):
        """
        In the M-step, we update the global parameters of
        the model.
        :return:
        """

        n_causal = np.sum([
            np.sum(self.s_gamma[c])
            for c in self.shapes
        ])

        # Update pi:
        if 'pi' not in self.fix_params:
            self.pi = np.random.beta(self.beta_prior[0] + n_causal,
                                     self.beta_prior[1] + (self.M - n_causal))

        # Update sigma_beta:
        if 'sigma_beta' not in self.fix_params:
            sum_beta_sq = np.sum([
                np.sum(self.s_gamma[c]*self.s_beta[c]**2)
                for c in self.shapes
            ])

            self.sigma_beta = invgamma(self.sigma_beta_prior[0] + .5*n_causal,
                                       loc=0.,
                                       scale=self.sigma_beta_prior[1] + .5*sum_beta_sq).rvs()

        # Update sigma_epsilon

        cdef double bxxb = 0., ssr = 0., snp_ssr, snp_xxb
        cdef unsigned int i, j, c_size
        cdef double[::1] s_gamma, s_beta, beta_hat, yy, sig_e_snp, Di
        cdef long[:, ::1] ld_bound

        for c, c_size in self.shapes.items():

            beta_hat = self.beta_hat[c]
            s_gamma = self.s_gamma[c]
            s_beta = self.s_beta[c]
            ld_bound = self.ld_bounds[c]
            yy = self.yy[c]
            sig_e_snp = self.sig_e_snp[c]

            for i, Di in enumerate(self.ld[c]):

                snp_ssr = 0.
                snp_xxb = 0.

                if s_gamma[i] == 0:
                    continue

                snp_ssr -= s_beta[i] * beta_hat[i]

                snp_xxb += .5 * s_beta[i] * s_beta[i]

                for j in range(i + 1, ld_bound[1, i]):
                    snp_xxb += Di[j - ld_bound[0, i]] * s_beta[i] * s_gamma[j] * s_beta[j]

                sig_e_snp[i] = yy[i] + 2.*(snp_ssr + snp_xxb)

                bxxb += snp_xxb
                ssr += snp_ssr

            self.sig_e_snp[c] = np.clip(sig_e_snp, 1e-12, 1e12)

        ssr = 1. + 2.*(ssr + bxxb)
        self.bxxb = 2. * bxxb

        if 'sigma_epsilon' not in self.fix_params:

            self.sigma_epsilon = invgamma(self.sigma_epsilon_prior[0] + .5*self.N,
                                          loc=0.,
                                          scale=self.sigma_epsilon_prior[1] + .5*self.N*ssr).rvs()

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

        for i in range(burn_in + n_samples):
            self.sample_local_parameters()
            self.sample_global_parameters()

            if i + 1 > burn_in:

                for c in self.shapes:
                    self.rs_gamma[c].push(self.s_gamma[c])
                    self.rs_beta[c].push(self.s_beta[c])

                self.rs_pi.push(self.pi)
                self.rs_sigma_beta.push(self.sigma_beta)
                self.rs_sigma_epsilon.push(self.sigma_epsilon)
                self.rs_h2g.push(self.bxxb / (self.bxxb + self.sigma_epsilon))

        self.pip = {c : rs.mean() for c, rs in self.rs_gamma.items()}
        self.inf_beta = {c : rs.mean() for c, rs in self.rs_beta.items()}

        if self.load_ld:
            self.gdl.release_ld()

        return self
