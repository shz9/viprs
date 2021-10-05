
from .PRSModel cimport PRSModel

cdef class VIPRS(PRSModel):

    cdef public:
        double pi, sigma_beta, sigma_epsilon  # Global hyper-parameters
        bint load_ld, verbose  # Binary flags
        dict q, var_gamma, var_mu_beta, var_sigma_beta, mean_beta, mean_beta_sq  # Variational parameters
        dict beta_hat, ld, ld_bounds  # Inputs to the algorithm
        dict history, fix_params  # Helpers
        int threads

    cpdef initialize(self, theta_0=*)
    cpdef init_history(self)
    cpdef initialize_theta(self, theta_0=*)
    cpdef initialize_variational_params(self)
    cpdef e_step(self)
    cpdef update_pi(self)
    cpdef update_sigma_beta(self)
    cpdef update_sigma_epsilon(self)
    cpdef m_step(self)
    cpdef objective(self)
    cpdef write_inferred_theta(self, f_name)
