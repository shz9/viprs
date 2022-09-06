# cython: linetrace=False
# cython: profile=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True

from scipy.optimize import minimize
from .VIPRS cimport VIPRS
from viprs.utils.math_utils cimport clip
from viprs.utils.compute_utils import dict_elementwise_dot, dict_sum, dict_set, dict_repeat


def alpha_objective(a, valpha):

    valpha.update_alpha_factor(a)
    valpha.update_sigma_beta()

    # Regularized objective:
    # Roughly equivalent to assuming that alpha is drawn from a unit Gaussian prior
    # See:
    # Signatures of negative selection in the genetic architecture of human complex traits
    # Zeng et al. 2018

    return -valpha.objective() + a**2


cdef class VIPRSAlpha(VIPRS):

    cdef public:
        double alpha
        dict alpha_factor, reciprocal_alpha_factor

    def __init__(self, gdl, fix_params=None, load_ld='auto', tracked_theta=None, verbose=True, threads=1):

        super().__init__(gdl, fix_params=fix_params, load_ld=load_ld,
                         tracked_theta=tracked_theta, verbose=verbose, threads=threads)

    cpdef initialize_theta(self, theta_0=None):

        if theta_0 is not None and self.fix_params is not None:
            theta_0.update(self.fix_params)
        elif self.fix_params is not None:
            theta_0 = self.fix_params
        elif theta_0 is None:
            theta_0 = {}

        super(VIPRSAlpha, self).initialize_theta(theta_0=theta_0)

        if 'alpha' in theta_0:
            self.alpha = theta_0['alpha']
        else:
            self.alpha = -.25  # Empirical estimate based on analysis of a large number of quantitative traits

        if not isinstance(self.sigma_beta, dict):
            self.sigma_beta = dict_repeat(self.sigma_beta, self.shapes)

        self.update_alpha_factor(self.alpha)
        self.sigma_beta = dict_elementwise_dot(self.sigma_beta, self.alpha_factor)

    cpdef update_alpha_factor(self, alpha):
        """
        Alpha factor to be used in updating sigma_beta
        Let pj be the allele frequency of SNP j,
        then the alpha factor is equivalent to pj(1. - pj)^(1. + alpha)
        Since we need the reciprocal of this in the sigma_beta update,
        we raise to the the power of -(1. + alpha)
        """
        self.reciprocal_alpha_factor = {c: (ss.maf * (1. - ss.maf)) ** (-(1. + alpha))
                                        for c, ss in self.gdl.sumstats_table.items()}
        self.alpha_factor = {c: 1./rca for c, rca in self.self.reciprocal_alpha_factor.items()}

    cpdef to_theta_table(self):

        theta_table = super(VIPRSAlpha, self).to_theta_table()
        return theta_table.append({'Parameter': 'alpha', 'Value': self.alpha}, ignore_index=True)

    cpdef update_alpha(self):
        """
        Update the alpha parameter by optimizing with respect to the ELBO.
        """

        if 'alpha' not in self.fix_params:
            res = minimize(alpha_objective, (self.alpha,), args=(self,))
            self.alpha = res.x
            self.update_alpha_factor(self.alpha)
        else:
            if self.fix_params['alpha'] != self.alpha:
                self.alpha = self.fix_params['alpha']
                self.update_alpha_factor(self.alpha)

    cpdef update_sigma_beta(self):
        """
        Update the prior variance on the effect size, sigma_beta
        """

        if 'sigma_beta' not in self.fix_params:

            # Sigma_beta estimate:
            sigma_beta_estimate = dict_sum(
                dict_elementwise_dot(self.reciprocal_alpha_factor, self.zeta)
            ) / dict_sum(self.var_gamma)
            # Clip value:
            sigma_beta_estimate = clip(sigma_beta_estimate, 1e-12, 1. - 1e-12)
            # Update the sigma beta given the new inferred estimate
            self.sigma_beta = dict_set(self.sigma_beta, sigma_beta_estimate)
            # Set the new sigma_beta per SNP:
            self.sigma_beta = dict_elementwise_dot(self.sigma_beta, self.alpha_factor)

    cpdef m_step(self):
        """
        In the M-step, update the global hyperparameters of the model.
        """

        self.update_pi()
        self.update_alpha()
        self.update_sigma_beta()
        self.update_sigma_epsilon()
