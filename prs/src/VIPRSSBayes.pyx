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
from libc.math cimport log

from .VIPRS cimport VIPRS
from .c_utils cimport dot, elementwise_add_mult, sigmoid, clip


cdef class VIPRSSBayes(VIPRS):

    cdef public:
        dict yy, sig_e_snp

    def __init__(self, gdl, fix_params=None, load_ld=True):

        super().__init__(gdl, fix_params=fix_params, load_ld=load_ld)

        self.yy = self.gdl.compute_yy_per_snp()
        self.sig_e_snp = {}

    cpdef initialize_theta(self, theta_0=None):

        super(VIPRSSBayes, self).initialize_theta(theta_0=theta_0)
        self.sig_e_snp = {c: np.repeat(self.sigma_epsilon, c_size)
                          for c, c_size in self.shapes.items()}

    cpdef e_step(self):

        cdef:
            unsigned int j, start, end, j_idx
            double u_j
            double[::1] var_gamma, var_mu_beta, var_sigma_beta, mean_beta, mean_beta_sq, beta_hat, Dj, sig_e
            long[::1] N
            long[:, ::1] ld_bound
            # The log(pi) for the gamma updates
            double logodds_pi = log(self.pi / (1. - self.pi))

        for c, c_size in self.shapes.items():

            beta_hat = self.beta_hat[c]
            sig_e = self.sig_e_snp[c]
            var_gamma = self.var_gamma[c]
            var_mu_beta = self.var_mu_beta[c]
            var_sigma_beta = self.var_sigma_beta[c]
            ld_bound = self.ld_bounds[c]
            mean_beta = self.mean_beta[c]
            mean_beta_sq = self.mean_beta_sq[c]
            N = self.Nj[c]
            q = np.zeros(shape=c_size)

            for j, Dj in enumerate(self.ld[c]):

                start, end = ld_bound[:, j]
                j_idx = j - start

                var_sigma_beta[j] = sig_e[j] / (N[j] + sig_e[j] / self.sigma_beta)

                var_mu_beta[j] = (beta_hat[j] - dot(Dj, mean_beta[start: end], self.threads) +
                                  Dj[j_idx] * mean_beta[j]) / (1. + sig_e[j] / (N[j] * self.sigma_beta))

                u_j = (logodds_pi + .5*log(var_sigma_beta[j] / self.sigma_beta) +
                       (.5/var_sigma_beta[j])*var_mu_beta[j]*var_mu_beta[j])
                var_gamma[j] = clip(sigmoid(u_j), 1e-6, 1. - 1e-6)

                mean_beta[j] = var_gamma[j]*var_mu_beta[j]
                mean_beta_sq[j] = var_gamma[j] * (var_mu_beta[j] * var_mu_beta[j] + var_sigma_beta[j])

                if j_idx > 0:
                    # Update the q factor for snp i by adding the contribution of previous SNPs.
                    q[j] = dot(Dj[:j_idx], mean_beta[start: j], self.threads)
                    # Update the q factors for all previously updated SNPs that are in LD with SNP i
                    q[start: j] = elementwise_add_mult(q[start: j], Dj[:j_idx], mean_beta[j], self.threads)


            self.var_gamma[c] = np.array(var_gamma)
            self.var_mu_beta[c] = np.array(var_mu_beta)
            self.var_sigma_beta[c] = np.array(var_sigma_beta)

            self.mean_beta[c] = np.array(mean_beta)
            self.mean_beta_sq[c] = np.array(mean_beta_sq)
            self.q[c] = np.array(q)

    cpdef update_sigma_epsilon(self):

        if 'sigma_epsilon' not in self.fix_params:

            global_sig_e = 0.

            for c, c_size in self.shapes.items():

                sse_per_snp = (self.mean_beta_sq[c]
                               - 2.*self.mean_beta[c]*self.beta_hat[c]
                               + self.mean_beta[c]*self.q[c])

                global_sig_e += sse_per_snp.sum()
                self.sig_e_snp[c] = np.clip(self.yy[c] + sse_per_snp, 1e-12, 1. - 1e-12)

            self.sigma_epsilon = clip(1. + global_sig_e, 1e-12, 1. - 1e-12)
