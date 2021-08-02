
from .PRSModel cimport PRSModel

cdef class VIPRS(PRSModel):

    cdef public:
        double pi, sigma_beta, sigma_epsilon  # Global hyper-parameters
        bint load_ld, verbose  # Binary flags
        dict q, var_mu_beta, var_sigma_beta, var_gamma  # Variational parameters
        dict beta_hat, ld, ld_bounds  # Inputs to the algorithm
        dict history, fix_params  # Helpers
        int threads

    cpdef initialize(self)
    cpdef init_history(self)
    cpdef initialize_theta(self)
    cpdef initialize_variational_params(self)
    cpdef e_step(self)
    cpdef update_pi(self)
    cpdef update_sigma_beta(self)
    cpdef update_sigma_epsilon(self)
    cpdef m_step(self)
    cpdef objective(self)
