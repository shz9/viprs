from cython cimport floating
cimport numpy as cnp

# --------------------------------------------------
# Define fused data types:

ctypedef fused indptr_type:
    cnp.int32_t
    cnp.int64_t

ctypedef fused noncomplex_numeric:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.float32_t
    cnp.float64_t

# --------------------------------------------------

cpdef void e_step(int[::1] ld_left_bound,
                  indptr_type[::1] ld_indptr,
                  floating[::1] ld_data,
                  floating[::1] std_beta,
                  floating[::1] var_gamma,
                  floating[::1] var_mu,
                  floating[::1] eta,
                  floating[::1] q,
                  floating[::1] eta_diff,
                  floating[::1] u_logs,
                  floating[::1] half_var_tau,
                  floating[::1] mu_mult,
                  int threads,
                  bint use_blas,
                  bint low_memory
                 ) noexcept nogil


cpdef void e_step_mixture(int[::1] ld_left_bound,
                          indptr_type[::1] ld_indptr,
                          floating[::1] ld_data,
                          floating[::1] std_beta,
                          floating[:, ::1] var_gamma,
                          floating[:, ::1] var_mu,
                          floating[::1] eta,
                          floating[::1] q,
                          floating[::1] eta_diff,
                          floating[::1] log_null_pi,
                          floating[:, ::1] u_logs,
                          floating[:, ::1] half_var_tau,
                          floating[:, ::1] mu_mult,
                          int threads,
                          bint use_blas,
                          bint low_memory) noexcept nogil


cpdef void e_step_grid(int[::1] ld_left_bound,
                       indptr_type[::1] ld_indptr,
                       floating[::1] ld_data,
                       floating[::1] std_beta,
                       floating[::1, :] var_gamma,
                       floating[::1, :] var_mu,
                       floating[::1, :] eta,
                       floating[::1, :] q,
                       floating[::1, :] eta_diff,
                       floating[::1, :] u_logs,
                       floating[::1, :] half_var_tau,
                       floating[::1, :] mu_mult,
                       int[:] active_model_idx,
                       int threads,
                       bint use_blas,
                       bint low_memory) noexcept nogil

