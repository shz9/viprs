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
from libc.math cimport log, exp

from .VIPRSMix cimport VIPRSMix
from .c_utils cimport dot, elementwise_add_mult, clip, softmax


cdef class VIPRSMixSBayes(VIPRSMix):

    cdef public:
        dict yy, sig_e_snp

    def __init__(self, gdl, K=1, prior_multipliers=None, fix_params=None, load_ld=True, verbose=True, threads=1):

        super().__init__(gdl, K=K, prior_multipliers=prior_multipliers,
                         fix_params=fix_params, load_ld=load_ld, verbose=verbose, threads=threads)

        self.yy = self.gdl.compute_yy_per_snp()
        self.sig_e_snp = {}

    cpdef initialize_theta(self, theta_0=None):

        super(VIPRSMixSBayes, self).initialize_theta(theta_0=theta_0)
        self.sig_e_snp = {c: np.repeat(self.sigma_epsilon, c_size)
                          for c, c_size in self.gdl.shapes.items()}

    cpdef e_step(self):

        # Initialize memoryviews objects for fast access
        cdef:
            unsigned int j, k, start, end, j_idx
            double mu_beta_j, gamma_denom
            double[::1] u_j, log_null_pi, sig_e
            double[:, ::1] log_pi, sigma_beta  # Per-SNP priors
            double[:, ::1] var_gamma, var_mu_beta, var_sigma_beta  # Variational parameters
            double[::1] std_beta, Dj  # Inputs
            double[::1] mean_beta, mean_beta_sq, q  # Properties of proposed distribution
            double[::1] N  # Sample size per SNP
            long[:, ::1] ld_bound

        for c, c_size in self.shapes.items():

            # Set the numpy vectors into memoryviews for fast access:
            log_pi = np.log(self.pi[c])
            log_null_pi = np.log(1. - self.pi[c].sum(axis=1))
            sig_e = self.sig_e_snp[c]
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
            u_j = np.zeros(shape=self.K + 1)

            for j, Dj in enumerate(self.ld[c]):

                start, end = ld_bound[:, j]
                j_idx = j - start

                # The numerator for all the mu_beta updates:
                mu_beta_j = (std_beta[j] - dot(Dj, mean_beta[start: end], self.threads) + Dj[j_idx]*mean_beta[j])

                for k in range(self.K):

                    var_sigma_beta[j, k] = sig_e[j] / (N[j] + sig_e[j] / sigma_beta[j, k])

                    # Compute the mu beta for component `k`
                    var_mu_beta[j, k] = mu_beta_j / (1. + sig_e[j] / (N[j] * sigma_beta[j, k]))
                    # Compute the unnormalized gamma for component `k`
                    u_j[k] = (log_pi[j, k] + .5 * log(var_sigma_beta[j, k] / sigma_beta[j, k]) +
                              (.5 / var_sigma_beta[j, k]) * var_mu_beta[j, k] * var_mu_beta[j, k])


                u_j[k + 1] = log_null_pi[j]
                u_j = softmax(u_j)

                # Normalize the gammas and update the beta statistics:
                mean_beta[j] = 0.
                mean_beta_sq[j] = 0.

                for k in range(self.K):
                    var_gamma[j, k] = clip(u_j[k], 1e-6, 1. - 1e-6)

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

    cpdef update_sigma_epsilon(self):

        if 'sigma_epsilon' not in self.fix_params:

            global_sig_e = 0.

            for c, c_size in self.shapes.items():

                sse_per_snp = (self.mean_beta_sq[c]
                               - 2.*self.mean_beta[c] * self.std_beta[c]
                               + self.mean_beta[c] * self.q[c])

                global_sig_e += sse_per_snp.sum()
                self.sig_e_snp[c] = np.clip(self.yy[c] + sse_per_snp, 1e-12, 1. - 1e-12)

            self.sigma_epsilon = clip(1. + global_sig_e, 1e-12, 1. - 1e-12)
