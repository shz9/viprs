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
from .utils import dict_elementwise_dot, dict_sum, dict_set


cdef class VIPRSSBayesAlpha(VIPRS):

    cdef public:
        double alpha
        dict alpha_factor, reciprocal_alpha_factor, yy, sig_e_snp

    def __init__(self, gdl, alpha=-.25, fix_params=None, load_ld=True, verbose=True, threads=1):

        super().__init__(gdl, fix_params=fix_params, load_ld=load_ld, verbose=verbose, threads=threads)

        self.alpha = alpha
        self.update_alpha_factor(alpha)
        self.yy = self.gdl.compute_yy_per_snp()
        self.sig_e_snp = {}

    cpdef initialize_theta(self, theta_0=None):

        super(VIPRSSBayesAlpha, self).initialize_theta(theta_0=theta_0)
        self.sig_e_snp = {c: np.repeat(self.sigma_epsilon, c_size)
                          for c, c_size in self.shapes.items()}
        self.sigma_beta = dict_elementwise_dot(self.sigma_beta, self.alpha_factor)

    cpdef update_alpha_factor(self, alpha):
        """
        Alpha factor to be used in updating sigma_beta
        Let pj be the allele frequency of SNP j,
        then the alpha factor is equivalent to pj(1. - pj)^(1. + alpha)
        Since we need the reciprocal of this in the sigma_beta update,
        we raise to the the power of -(1. + alpha)
        """
        self.reciprocal_alpha_factor = {c: (maf * (1. - maf)) ** (-(1. + alpha)) for c, maf in self.gdl.maf.items()}
        self.alpha_factor = {c: (maf * (1. - maf)) ** (1. + alpha) for c, maf in self.gdl.maf.items()}

    cpdef e_step(self):

        cdef:
            unsigned int j, start, end, j_idx
            double u_j
            double[::1] logodds_pi, sigma_beta, sig_e  # Per-SNP priors
            double[::1] var_gamma, var_mu_beta, var_sigma_beta  # Variational parameters
            double[::1] std_beta, Dj  # Inputs
            double[::1] mean_beta, mean_beta_sq, q  # Properties of porposed distribution
            double[::1] N  # Sample size per SNP
            long[:, ::1] ld_bound

        for c, c_size in self.shapes.items():

            # Set the numpy vectors into memoryviews for fast access:
            logodds_pi = np.log(self.pi[c] / (1. - self.pi[c]))
            sigma_beta = self.sigma_beta[c]
            std_beta = self.std_beta[c]
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

                var_sigma_beta[j] = sig_e[j] / (N[j] + sig_e[j] / sigma_beta[j])

                var_mu_beta[j] = (std_beta[j] - dot(Dj, mean_beta[start: end], self.threads) +
                                  Dj[j_idx] * mean_beta[j]) / (1. + sig_e[j] / (N[j] * sigma_beta[j]))

                u_j = (logodds_pi[j] + .5*log(var_sigma_beta[j] / sigma_beta[j]) +
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

    cpdef to_theta_table(self):

        theta_table = super(VIPRSSBayesAlpha, self).to_theta_table()
        return theta_table.append({'Parameter': 'alpha', 'Value': self.alpha}, ignore_index=True)

    cpdef update_sigma_beta(self):
        """
        Update the prior variance on the effect size, sigma_beta
        """

        if 'sigma_beta' not in self.fix_params:

            if 'alpha' in self.fix_params:
                if self.fix_params['alpha'] != self.alpha:
                    self.alpha = self.fix_params['alpha']
                    self.update_alpha_factor(self.alpha)

            # Sigma_beta estimate:
            sigma_beta_estimate = dict_sum(
                dict_elementwise_dot(self.reciprocal_alpha_factor, self.mean_beta_sq)
            ) / dict_sum(self.var_gamma)
            # Clip value:
            sigma_beta_estimate = clip(sigma_beta_estimate, 1e-12, 1. - 1e-12)
            # Update the sigma beta given the new inferred estimate
            self.sigma_beta = dict_set(self.sigma_beta, sigma_beta_estimate)
            self.sigma_beta = dict_elementwise_dot(self.sigma_beta, self.alpha_factor)

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
