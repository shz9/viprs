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
from scipy.stats import invgamma
from libc.math cimport log, sqrt
from .GibbsPRS cimport GibbsPRS
from .c_utils cimport dot, elementwise_add_mult, sigmoid, clip


cdef class GibbsPRSSBayes(GibbsPRS):

    cdef public:
        dict yy, sig_e_snp

    def __init__(self, gdl,
                 beta_prior=(10., 1e4),
                 sigma_beta_prior=(1e-4, 1e-4),
                 sigma_epsilon_prior=(1., 1.),
                 fix_params=None, load_ld=True):

        super().__init__(gdl, beta_prior=beta_prior,
                         sigma_beta_prior=sigma_beta_prior,
                         sigma_epsilon_prior=sigma_epsilon_prior,
                         fix_params=fix_params, load_ld=load_ld)

        self.yy = {c: yy.values for c, yy in self.gdl.compute_yy_per_snp().items()}
        self.sig_e_snp = {}

    cpdef initialize_theta(self):

        if 'sigma_beta' not in self.fix_params:
            self.sigma_beta = np.random.uniform()
        else:
            self.sigma_beta = self.fix_params['sigma_beta'][0]

        if 'sigma_epsilon' not in self.fix_params:
            self.sigma_epsilon = np.random.uniform()
        else:
            self.sigma_epsilon = self.fix_params['sigma_epsilon'][0]

        self.sig_e_snp = {c: np.repeat(self.sigma_epsilon, c_size)
                          for c, c_size in self.shapes.items()}

        if 'pi' not in self.fix_params:
            self.pi = np.random.uniform()
        else:
            self.pi = self.fix_params['pi'][0]

    cpdef sample_local_parameters(self):

        cdef:
            unsigned int j, start, end, j_idx
            double u_j, mu_beta_j, s_var
            double[::1] prod, s_beta, s_gamma, beta_hat, s_unif, s_norm, q, Dj, sig_e
            long[:, ::1] ld_bound
            # The log(pi) for the gamma updates
            double logodds_pi = log(self.pi / (1. - self.pi))
            double sqrt_prior_var = sqrt(self.sigma_beta)

        for c, c_size in self.shapes.items():
            beta_hat = self.beta_hat[c]
            s_beta = self.s_beta[c]
            s_gamma = self.s_gamma[c]
            ld_bound = self.ld_bounds[c]
            sig_e = self.sig_e_snp[c]
            s_unif = np.random.uniform(size=c_size)
            s_norm = np.random.normal(size=c_size)
            q = np.zeros(shape=c_size)

            prod = np.multiply(s_gamma, s_beta)

            for j, Dj in enumerate(self.ld[c]):

                start, end = ld_bound[:, j]
                j_idx = j - start

                s_var = sig_e[j] / (self.N + sig_e[j] / self.sigma_beta)
                mu_beta_j = (beta_hat[j] - dot(Dj, prod[start: end]) +
                                  Dj[j_idx]*prod[j]) / (1. + sig_e[j] / (self.N * self.sigma_beta))

                u_j = (logodds_pi + .5*log(s_var / self.sigma_beta) +
                       (.5/s_var)*mu_beta_j*mu_beta_j)

                if s_unif[j] > sigmoid(u_j):
                    s_gamma[j] = 1.
                    s_beta[j] = mu_beta_j + sqrt(s_var) * s_norm[j]
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

    cpdef sample_sigma_epsilon(self):

        if 'sigma_epsilon' not in self.fix_params:

            sigma_g = 0.
            ssr = 0.

            for c, c_size in self.shapes.items():
                beta_hat = self.beta_hat[c]
                prod = np.multiply(self.s_gamma[c], self.s_beta[c])
                q = self.q[c]

                snp_var = prod*(prod + q)
                sigma_g += snp_var.sum()

                snp_sse = self.yy[c] + -2.*prod*beta_hat + snp_var
                self.sig_e_snp[c] = np.clip(snp_sse, 1e-12, 1e12)

                ssr += -2. * np.dot(prod, beta_hat)

            ssr = clip(1. + ssr + sigma_g, 1e-12, 1e12)

            self.sigma_g = sigma_g
            self.sigma_epsilon = invgamma(self.sigma_epsilon_prior[0] + .5*self.N,
                                          loc=0.,
                                          scale=self.sigma_epsilon_prior[1] + .5*self.N*ssr).rvs()
