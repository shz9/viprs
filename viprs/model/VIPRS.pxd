
from .PRSModel cimport PRSModel

cdef class VIPRS(PRSModel):

    cdef public:
        double sigma_epsilon  # Residual variance
        pi, sigma_beta  # Priors
        bint load_ld, verbose  # Binary flags
        dict var_gamma, var_mu, var_sigma  # Per-SNP Variational parameters
        dict q, eta, zeta  # Properties of the variational distribution
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
    cpdef get_sigma_epsilon(self)
    cpdef get_sigma_beta(self, chrom=*)
    cpdef get_pi(self, chrom=*)
    cpdef get_average_effect_size_variance(self)
    cpdef to_theta_table(self)
    cpdef update_theta_history(self)
    cpdef compute_pip(self)
    cpdef compute_eta(self)
    cpdef compute_zeta(self)
    cpdef update_posterior_moments(self)
    cpdef write_inferred_theta(self, f_name, sep=*)
