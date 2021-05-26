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

    cpdef e_step(self):

        cdef:
            unsigned int j, start, end, j_idx
            double u_j
            double[::1] var_prod, var_mu_beta, var_sigma_beta, var_gamma, beta_hat, Dj, sig_e
            long[:, ::1] ld_bound
            # The log(pi) for the gamma updates
            double logodds_pi = log(self.pi / (1. - self.pi))

        for c, c_size in self.shapes.items():

            # Updates for sigma_beta variational parameters:
            self.var_sigma_beta[c] = np.repeat(
                self.sigma_epsilon / (self.N + self.sigma_epsilon / self.sigma_beta),
                c_size
            )

            beta_hat = self.beta_hat[c]
            sig_e = self.sig_e_snp[c]
            var_mu_beta = self.var_mu_beta[c]
            var_sigma_beta = self.var_sigma_beta[c]
            var_gamma = self.var_gamma[c]
            ld_bound = self.ld_bounds[c]
            q = np.zeros(shape=c_size)

            var_prod = np.multiply(var_gamma, var_mu_beta)

            for j, Dj in enumerate(self.ld[c]):

                start, end = ld_bound[:, j]
                j_idx = j - start

                var_sigma_beta[j] = sig_e[j] / (self.N + sig_e[j] / self.sigma_beta)

                var_mu_beta[j] = (beta_hat[j] - dot(Dj, var_prod[start: end]) +
                                  Dj[j_idx] * var_prod[j]) / (1. + sig_e[j] / (self.N * self.sigma_beta))

                u_j = (logodds_pi + .5*log(var_sigma_beta[j] / self.sigma_beta) +
                       (.5/var_sigma_beta[j])*var_mu_beta[j]*var_mu_beta[j])
                var_gamma[j] = clip(sigmoid(u_j), 1e-6, 1. - 1e-6)

                var_prod[j] = var_gamma[j]*var_mu_beta[j]

                if j_idx > 0:
                    # Update the q factor for snp i by adding the contribution of previous SNPs.
                    q[j] = dot(Dj[:j_idx], var_prod[start: j])
                    # Update the q factors for all previously updated SNPs that are in LD with SNP i
                    q[start: j] = elementwise_add_mult(q[start: j], Dj[:j_idx], var_prod[j])


            self.q[c] = np.array(q)
            self.var_gamma[c] = np.array(var_gamma)
            self.var_mu_beta[c] = np.array(var_mu_beta)

    cpdef update_sigma_epsilon(self):

        if 'sigma_epsilon' not in self.fix_params:

            global_sig_e = 0.

            for c, c_size in self.shapes.items():

                beta_hat = self.beta_hat[c]
                var_gamma = self.var_gamma[c]
                var_mu_beta = self.var_mu_beta[c]
                var_sigma_beta = self.var_sigma_beta[c]
                var_prod = np.multiply(var_gamma, var_mu_beta)
                ld_bound = self.ld_bounds[c]
                sig_e = self.sig_e_snp[c]
                yy = self.yy[c]
                q = self.q[c]


                sse_per_snp = (var_gamma * (var_mu_beta**2 + var_sigma_beta)
                               - 2.*var_prod*beta_hat
                               + var_prod*q)

                global_sig_e += sse_per_snp.sum()
                self.sig_e_snp[c] = np.clip(yy + sse_per_snp, 1e-12, 1e12)

            self.sigma_epsilon = clip(1. + global_sig_e, 1e-12, 1e12)
