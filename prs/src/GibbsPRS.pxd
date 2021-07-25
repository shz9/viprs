from .PRSModel cimport PRSModel
from .run_stats cimport RunStats

cdef class GibbsPRS(PRSModel):

    cdef public:
        double pi, sigma_beta, sigma_epsilon, sigma_g  # Global parameters
        bint load_ld, verbose  # Binary flags
        tuple beta_prior, sigma_beta_prior, sigma_epsilon_prior
        dict q, s_beta, s_gamma  # q factor, Sampled beta and gamma
        dict beta_hat, ld, ld_bounds  # Inputs to the algorithm
        dict fix_params  # Helpers
        dict rs_gamma, rs_beta
        RunStats rs_pi, rs_sigma_beta, rs_sigma_epsilon, rs_h2g  # Running stats objects


    cpdef initialize(self)
    cpdef initialize_running_stats(self)
    cpdef initialize_theta(self)
    cpdef initialize_local_params(self)
    cpdef sample_local_parameters(self)
    cpdef sample_pi(self)
    cpdef sample_sigma_beta(self)
    cpdef sample_sigma_epsilon(self)
    cpdef sample_global_parameters(self)
