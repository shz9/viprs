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

def m_objective(x, vld):
    vld.h2g, vld.pi = x
    vld.update_sigma_epsilon()
    vld.update_sigma_beta()

    return -vld.objective()

cdef class VIPRSLDPred(VIPRS):
    """
    NOTE: Experimental!
    This model replaces sigma_beta and sigma_epsilon with parametrization
    based on the heritability h2g, similar to the LDPred model.
    """

    cdef public:
        double h2g

    def __init__(self, gdl, fix_params=None, load_ld='auto', tracked_theta=None, verbose=True, threads=1):

        super().__init__(gdl, fix_params=fix_params, load_ld=load_ld,
                         tracked_theta=tracked_theta, verbose=verbose, threads=threads)

    cpdef initialize_theta(self, theta_0=None):

        super(VIPRSLDPred, self).initialize_theta(theta_0=theta_0)
        self.h2g = 1. - self.sigma_epsilon

    cpdef update_sigma_beta(self):
        self.sigma_beta = self.h2g / (self.n_snps*self.get_proportion_causal())

    cpdef update_sigma_epsilon(self):
        self.sigma_epsilon = 1. - self.h2g

    cpdef m_step(self):

        bnds = ((1e-4, 1. - 1e-4), (1./self.n_snps, 1. - 1./self.n_snps))

        res = minimize(m_objective, (self.h2g, self.get_proportion_causal()), bounds=bnds, args=(self,))
        self.h2g, self.pi = res.x
        self.update_sigma_epsilon()
        self.update_sigma_beta()
