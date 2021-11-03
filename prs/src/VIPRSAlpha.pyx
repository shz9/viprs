# cython: linetrace=False
# cython: profile=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True

from .VIPRS cimport VIPRS
from .c_utils cimport clip
from .utils import dict_elementwise_dot, dict_sum, dict_set


cdef class VIPRSAlpha(VIPRS):

    cdef public:
        double alpha
        dict alpha_factor, reciprocal_alpha_factor

    def __init__(self, gdl, alpha=-.25, fix_params=None, load_ld=True, verbose=True, threads=1):

        super().__init__(gdl, fix_params=fix_params, load_ld=load_ld, verbose=verbose, threads=threads)
        self.alpha = alpha
        self.update_alpha_factor(alpha)

    cpdef initialize_theta(self, theta_0=None):
        super(VIPRSAlpha, self).initialize_theta(theta_0=theta_0)
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

    cpdef to_theta_table(self):

        theta_table = super(VIPRSAlpha, self).to_theta_table()
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
            # Set the new sigma_beta per SNP:
            self.sigma_beta = dict_elementwise_dot(self.sigma_beta, self.alpha_factor)
