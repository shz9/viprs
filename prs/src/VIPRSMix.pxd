
import numpy as np
cimport numpy as np
from .PRSModel cimport PRSModel

cdef class VIPRSMix(PRSModel):

    cdef public:
        int K
        object d  # Multiplier for the prior on the effect size
        double sigma_epsilon  # Residual variance
        dict pi, sigma_beta  # Priors
        bint load_ld, verbose  # Binary flags
        dict var_gamma, var_mu_beta, var_sigma_beta  # Per-SNP Variational parameters
        dict q, mean_beta, mean_beta_sq  # Properties of the variational distribution
        dict std_beta, ld, ld_bounds  # Inputs to the algorithm
        dict history, fix_params  # Helpers
        list tracked_theta
        int threads

    cpdef initialize(self, theta_0=*, param_0=*)
    cpdef init_history(self)
    cpdef initialize_theta(self, theta_0=*)
    cpdef initialize_variational_params(self, param_0=*)
    cpdef e_step(self)
    cpdef update_pi(self)
    cpdef update_sigma_beta(self)
    cpdef update_sigma_epsilon(self)
    cpdef m_step(self)
    cpdef objective(self)
    cpdef to_theta_table(self)
    cpdef update_theta_history(self)
    cpdef write_inferred_theta(self, f_name)
