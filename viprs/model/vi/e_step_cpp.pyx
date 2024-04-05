# distutils: language = c++
# sources: model/vi/e_step.hpp

from cython cimport floating, integral


cdef extern from "e_step.hpp" nogil:

    bint blas_supported() noexcept nogil
    bint omp_supported() noexcept nogil

    void blas_axpy[T](T* y, T* x, T alpha, int size) noexcept nogil
    T blas_dot[T](T* x, T* y, int size) noexcept nogil

    void e_step[T, U, I](int c_size,
                      int* ld_left_bound,
                      I* ld_indptr,
                      U* ld_data,
                      T* std_beta,
                      T* var_gamma,
                      T* var_mu,
                      T* eta,
                      T* q,
                      T* eta_diff,
                      T* u_logs,
                      T* half_var_tau,
                      T* mu_mult,
                         T dq_scale,
                      int threads,
                      bint use_blas,
                      bint low_memory) noexcept nogil

    void e_step_mixture[T, U, I](int c_size,
                              int K,
                              int* ld_left_bound,
                              I* ld_indptr,
                              U* ld_data,
                              T* std_beta,
                              T* var_gamma,
                              T* var_mu,
                              T* eta,
                              T* q,
                              T* eta_diff,
                              T* log_null_pi,
                              T* u_logs,
                              T* half_var_tau,
                              T* mu_mult,
                                 T dq_scale,
                              int threads,
                              bint use_blas,
                              bint low_memory) noexcept nogil

    void e_step_grid[T, U, I](int c_size,
                           int n_active_models,
                           int* active_model_idx,
                           int* ld_left_bound,
                           I* ld_indptr,
                           U* ld_data,
                           T* std_beta,
                           T* var_gamma,
                           T* var_mu,
                           T* eta,
                           T* q,
                           T* eta_diff,
                           T* u_logs,
                           T* half_var_tau,
                           T* mu_mult,
                              T dq_scale,
                           int threads,
                           bint use_blas,
                           bint low_memory) noexcept nogil


cpdef check_blas_support():
    return blas_supported()


cpdef check_omp_support():
    return omp_supported()


cdef void cpp_blas_axpy(floating[::1] v1, floating[::1] v2, floating alpha) noexcept nogil:
    """v1 := v1 + alpha * v2"""
    cdef int size = v1.shape[0]
    blas_axpy(&v1[0], &v2[0], alpha, size)


cdef floating cpp_blas_dot(floating[::1] v1, floating[::1] v2) noexcept nogil:
    """v1 := v1.Tv2"""
    cdef int size = v1.shape[0]
    return blas_dot(&v1[0], &v2[0], size)


cpdef void cpp_e_step(int[::1] ld_left_bound,
                      integral[::1] ld_indptr,
                      noncomplex_numeric[::1] ld_data,
                      floating[::1] std_beta,
                      floating[::1] var_gamma,
                      floating[::1] var_mu,
                      floating[::1] eta,
                      floating[::1] q,
                      floating[::1] eta_diff,
                      floating[::1] u_logs,
                      floating[::1] half_var_tau,
                      floating[::1] mu_mult,
                      floating dq_scale,
                      int threads,
                      bint use_blas,
                      bint low_memory) noexcept nogil:

    e_step(var_mu.shape[0],
           &ld_left_bound[0],
           &ld_indptr[0],
           &ld_data[0],
           &std_beta[0],
           &var_gamma[0],
           &var_mu[0],
           &eta[0],
           &q[0],
           &eta_diff[0],
           &u_logs[0],
           &half_var_tau[0],
           &mu_mult[0],
           dq_scale,
           threads,
           use_blas,
           low_memory)


cpdef void cpp_e_step_mixture(int[::1] ld_left_bound,
                              integral[::1] ld_indptr,
                              noncomplex_numeric[::1] ld_data,
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
                              floating dq_scale,
                              int threads,
                              bint use_blas,
                              bint low_memory) noexcept nogil:

    e_step_mixture(var_mu.shape[0],
                   var_mu.shape[1],
                   &ld_left_bound[0],
                   &ld_indptr[0],
                   &ld_data[0],
                   &std_beta[0],
                   &var_gamma[0, 0],
                   &var_mu[0, 0],
                   &eta[0],
                   &q[0],
                   &eta_diff[0],
                   &log_null_pi[0],
                   &u_logs[0, 0],
                   &half_var_tau[0, 0],
                   &mu_mult[0, 0],
                   dq_scale,
                   threads,
                   use_blas,
                   low_memory)

cpdef void cpp_e_step_grid(int[::1] ld_left_bound,
                           integral[::1] ld_indptr,
                           noncomplex_numeric[::1] ld_data,
                           floating[::1] std_beta,
                           floating[::1, :] var_gamma,
                           floating[::1, :] var_mu,
                           floating[::1, :] eta,
                           floating[::1, :] q,
                           floating[::1, :] eta_diff,
                           floating[::1, :] u_logs,
                           floating[::1, :] half_var_tau,
                           floating[::1, :] mu_mult,
                           floating dq_scale,
                           int[:] active_model_idx,
                           int threads,
                           bint use_blas,
                           bint low_memory) noexcept nogil:

    e_step_grid(var_mu.shape[0],
                active_model_idx.shape[0],
                &active_model_idx[0],
                &ld_left_bound[0],
                &ld_indptr[0],
                &ld_data[0],
                &std_beta[0],
                &var_gamma[0, 0],
                &var_mu[0, 0],
                &eta[0, 0],
                &q[0, 0],
                &eta_diff[0, 0],
                &u_logs[0, 0],
                &half_var_tau[0, 0],
                &mu_mult[0, 0],
                dq_scale,
                threads,
                use_blas,
                low_memory)
