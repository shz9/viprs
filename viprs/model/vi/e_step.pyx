from cython.parallel import prange, parallel
from ...utils.math_utils cimport (
    sigmoid,
    softmax,
    clip,
    scipy_blas_axpy,
    scipy_blas_dot,
    axpy,
    dot
)
from .e_step_cpp cimport cpp_blas_axpy, cpp_blas_dot
cimport cython
import numpy as np
cimport numpy as np
from cython cimport floating, integral


# A safe way to get the number of the thread currently executing the code:
# This is used to avoid compile-time errors when compiling the code with OpenMP support disabled.
# In earlier iterations, we used:
# cimport openmp
# openmp.omp_get_thread_num()
# But this tends to fail when OpenMP is not enabled.
# The code below is a safer way to get the thread number.
cdef extern from *:
    """
    #ifdef _OPENMP
    #include <omp.h>
    #else
    int omp_get_thread_num() { return 0; }
    #endif
    """
    int omp_get_thread_num() noexcept nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cpdef void update_q_factor_matrix(int[:] active_model_idx,
                                  int[::1] ld_left_bound,
                                  integral[::1] ld_indptr,
                                  floating[::1] ld_data,
                                  floating[::1, :] eta,
                                  floating[::1, :] q,
                                  bint use_blas,
                                  int threads) noexcept nogil:
    """
        Compute or update q factor, defined as the result of a dot product between the
        Linkage-disequilibrium matrix (LD) and the current value of eta (the posterior mean
        for the effect sizes). The definition of the q-factor excludes the diagonal elements.

        This function assumes that we have a matrix of eta values, where each column represents
        eta from a different model (this is relevant for grid-based models).

        The implementation below assumes that the eta and q matrices are column-major matrices.
    """

    cdef:
        int j, m, m_idx, start, end, c_size = eta.shape[0], num_models = active_model_idx.shape[0]
        integral ld_start, ld_end

    for j in prange(c_size, nogil=True, schedule='static', num_threads=threads):

        ld_start = ld_indptr[j]
        ld_end = ld_indptr[j + 1]
        start = ld_left_bound[j]
        end = start + (ld_end - ld_start)

        for m in range(num_models):
            m_idx = active_model_idx[m]
            if use_blas:
                q[j, m_idx] += cpp_blas_dot(eta[start:end, m_idx], ld_data[ld_start:ld_end])
            else:
                q[j, m_idx] += dot(eta[start:end, m_idx], ld_data[ld_start:ld_end])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cpdef void update_q_factor(int[::1] ld_left_bound,
                           integral[::1] ld_indptr,
                           floating[::1] ld_data,
                           floating[::1] eta,
                           floating[::1] q,
                           bint use_blas,
                           int threads) noexcept nogil:

    """
        Compute or update q factor, defined as the result of a dot product between the
        Linkage-disequilibrium matrix (LD) and the current value of eta (the posterior mean
        for the effect sizes). The definition of the q-factor excludes the diagonal elements.

        This function assumes that we have a vector of eta values. If you have a matrix of eta
        values, you should use the `update_q_factor_matrix` function instead.
    """

    cdef:
        int j, start, end, c_size = eta.shape[0]
        integral ld_start, ld_end;

    for j in prange(c_size, nogil=True, schedule='static', num_threads=threads):

        ld_start = ld_indptr[j]
        ld_end = ld_indptr[j + 1]
        start = ld_left_bound[j]
        end = start + (ld_end - ld_start)

        if use_blas:
            q[j] += cpp_blas_dot(ld_data[ld_start:ld_end], eta[start:end])
        else:
            q[j] += dot(ld_data[ld_start:ld_end], eta[start:end])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cpdef void e_step(int[::1] ld_left_bound,
                  integral[::1] ld_indptr,
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
                  bint low_memory) noexcept nogil:

    """
    
    This function performs the E-step of the variational inference algorithm. It updates the variational parameters
    for the latent variables, and the posterior mean of the effect sizes. The function is written in Cython, and 
    compiled to C code. The function is called from the Python code in the vi function in the variational
    inference module (VIPRS.py).
    
    :param ld_left_bound: The index of the leftmost neighboring SNP in the LD matrix (i.e. index of the leftmost that is SNP in LD with SNP j).
    :param ld_indptr: The index pointers for the flattened LD matrix (`ld_data`).
    :param ld_data: The flattened LD matrix.
    :param std_beta: The standardized effect sizes.
    :param var_gamma: The variational gamma parameters.
    :param var_mu: The variational mu parameters.
    :param eta: The posterior mean of the effect sizes.
    :param q: The q factors.
    :param u_logs: The logarithmic factors used to compute the variational gamma parameters.
    :param half_var_tau: The reciprocal sigma factors used to compute the variational gamma parameters.
    :param mu_mult: The multiplicative factor that is used to compute the variational mu parameters.
    :param threads: The number of threads to use for parallelization.
    """

    cdef:
        int j, start, end, c_size = var_mu.shape[0]
        long ld_start, ld_end
        floating u_j

    for j in prange(c_size, nogil=True, schedule='static', num_threads=threads):

        # The start and end coordinates for the flattened LD matrix:
        ld_start = ld_indptr[j]
        ld_end = ld_indptr[j + 1]

        # The start and end coordinates for the neighboring SNPs:
        start = ld_left_bound[j]
        end = start + (ld_end - ld_start)

        # Compute the variational mu beta:
        var_mu[j] = mu_mult[j] * (std_beta[j] - q[j])

        # Compute the variational gamma:
        u_j = u_logs[j] + half_var_tau[j] * var_mu[j] * var_mu[j]
        var_gamma[j] = sigmoid(u_j)

        # Compute the difference between the new and old values for the posterior mean:
        eta_diff[j] = var_gamma[j] * var_mu[j] - eta[j]

        # Update the q factors for all neighboring SNPs that are in LD with SNP j
        if use_blas:
            cpp_blas_axpy(q[start:end], ld_data[ld_start:ld_end], eta_diff[j])
        else:
            axpy(q[start:end], ld_data[ld_start:ld_end], eta_diff[j])

        if start <= j < end:
            # Updating q factors for neighboring SNPs above also updates the q-factor for
            # SNP j, so we correct that here:
            q[j] = q[j] - eta_diff[j]

        # Update the posterior mean:
        eta[j] = eta[j] + eta_diff[j]

    if low_memory:
        # If the LD matrix used in the above operations is upper-triangular, then we would not
        # have updated the q-factors of variants based on the new etas for variants that come after them
        # in the dataset. So, we need to correct for that here:
        update_q_factor(ld_left_bound, ld_indptr, ld_data, eta_diff, q, use_blas, threads)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cpdef void e_step_mixture(int[::1] ld_left_bound,
                          integral[::1] ld_indptr,
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
                          bint low_memory) noexcept nogil:

    """

    This function performs the E-step of the variational inference algorithm. It updates the variational parameters
    for the latent variables, and the posterior mean of the effect sizes. The function is written in Cython, and 
    compiled to C code. The function is called from the Python code in the vi function in the variational
    inference module (VIPRS.py).

    :param ld_left_bound: The index of the leftmost neighboring SNP in the LD matrix 
    (i.e. index of the leftmost that is SNP in LD with SNP j).
    :param ld_indptr: The index pointers for the flattened LD matrix (`ld_data`).
    :param ld_data: The flattened LD matrix.
    :param std_beta: The standardized effect sizes.
    :param var_gamma: The variational gamma parameters.
    :param var_mu: The variational mu parameters.
    :param eta: The posterior mean of the effect sizes.
    :param q: The q factors.
    :param u_logs: The logarithmic factors used to compute the variational gamma parameters.
    :param half_var_tau: The reciprocal sigma factors used to compute the variational gamma parameters.
    :param mu_mult: The multiplicative factor that is used to compute the variational mu parameters.
    :param threads: The number of threads to use for parallelization.
    """

    cdef:
        int j, start, end, thread_offset, c_size = var_mu.shape[0], k, K = var_mu.shape[1]
        int u_size = threads * (K + 1)
        long ld_start, ld_end
        floating mu_beta_j
        floating[::1] u_j

    with gil:
        u_j = np.empty(threads*(K + 1), dtype=[np.float64, np.float32][floating is float])

    for j in prange(c_size, nogil=True, schedule='static', num_threads=threads):

        # Set the thread offset for the u_j array:
        thread_offset = omp_get_thread_num() * (K + 1)

        # The start and end coordinates for the flattened LD matrix:
        ld_start = ld_indptr[j]
        ld_end = ld_indptr[j + 1]

        # The start and end coordinates for the neighboring SNPs:
        start = ld_left_bound[j]
        end = start + (ld_end - ld_start)

        mu_beta_j = std_beta[j] - q[j]

        for k in range(K):
            # Compute the variational mu beta:
            var_mu[j, k] = mu_mult[j, k] * mu_beta_j
            u_j[thread_offset + k] = u_logs[j, k] + half_var_tau[j, k] * var_mu[j, k] * var_mu[j, k]

        # Normalize the variational gammas:
        u_j[thread_offset + K] = log_null_pi[j]
        var_gamma[j, :] = softmax(u_j[thread_offset:thread_offset + K + 1])[:K]

        # Compute the difference between the new and old values for the posterior mean:
        eta_diff[j] = -eta[j]

        for k in range(K):
            eta_diff[j] += var_gamma[j, k] * var_mu[j, k]

        # Update the q factors for all neighboring SNPs that are in LD with SNP j
        if use_blas:
            cpp_blas_axpy(q[start:end], ld_data[ld_start:ld_end], eta_diff[j])
        else:
            axpy(q[start:end], ld_data[ld_start:ld_end], eta_diff[j])

        if start <= j < end:
            # Updating q factors for neighboring SNPs above also updates the q-factor for
            # SNP j, so we correct that here:
            q[j] = q[j] - eta_diff[j]

        # Update the posterior mean:
        eta[j] = eta[j] + eta_diff[j]

    if low_memory:
        # If the LD matrix used in the above operations is upper-triangular, then we would not
        # have updated the q-factors of variants based on the new etas for variants that come after them
        # in the dataset. So, we need to correct for that here:
        update_q_factor(ld_left_bound, ld_indptr, ld_data, eta_diff, q, use_blas, threads)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cpdef void e_step_grid(int[::1] ld_left_bound,
                       integral[::1] ld_indptr,
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
                       bint low_memory) noexcept nogil:

    """

    This function performs the E-step of the variational inference algorithm. It updates the variational parameters
    for the latent variables, and the posterior mean of the effect sizes. The function is written in Cython, and 
    compiled to C code. The function is called from the Python code in the vi function in the variational
    inference module (VIPRS.py).

    :param ld_left_bound: The index of the leftmost neighboring SNP in the LD matrix 
    (i.e. index of the leftmost that is SNP in LD with SNP j).
    :param ld_indptr: The index pointers for the flattened LD matrix (`ld_data`).
    :param ld_data: The flattened LD matrix.
    :param std_beta: The standardized effect sizes.
    :param var_gamma: The variational gamma parameters.
    :param var_mu: The variational mu parameters.
    :param eta: The posterior mean of the effect sizes.
    :param q: The q factors.
    :param u_logs: The logarithmic factors used to compute the variational gamma parameters.
    :param half_var_tau: The reciprocal sigma factors used to compute the variational gamma parameters.
    :param mu_mult: The multiplicative factor that is used to compute the variational mu parameters.
    :param threads: The number of threads to use for parallelization.
    """

    cdef:
        int start, end, j, c_size = var_mu.shape[0], m_idx, m, num_models = active_model_idx.shape[0]
        long ld_start, ld_end
        floating u_j

    for j in prange(c_size, nogil=True, schedule='static', num_threads=threads):

        # The start and end coordinates for the flattened LD matrix:
        ld_start = ld_indptr[j]
        ld_end = ld_indptr[j + 1]

        # The start and end coordinates for the neighboring SNPs:
        start = ld_left_bound[j]
        end = start + (ld_end - ld_start)

        for m_idx in range(num_models):

            # Retrieve the index of the active model:
            m = active_model_idx[m_idx]

            # Compute the variational mu beta:
            var_mu[j, m] = mu_mult[j, m] * (std_beta[j] - q[j, m])

            # Compute the variational gamma:
            u_j = u_logs[j, m] + half_var_tau[j, m] * var_mu[j, m] * var_mu[j, m]
            var_gamma[j, m] = sigmoid(u_j)

            # Compute the difference between the new and old values for the posterior mean:
            eta_diff[j, m] = var_gamma[j, m] * var_mu[j, m] - eta[j, m]

            # Update the q factors for all neighboring SNPs that are in LD with SNP j
            if use_blas:
                cpp_blas_axpy(q[start:end, m], ld_data[ld_start:ld_end], eta_diff[j, m])
            else:
                axpy(q[start:end, m], ld_data[ld_start:ld_end], eta_diff[j, m])

            if start <= j < end:
                # Updating q factors for neighboring SNPs above also updates the q-factor for
                # SNP j, so we correct that here:
                q[j, m] = q[j, m] - eta_diff[j, m]

            # Update the posterior mean:
            eta[j, m] = eta[j, m] + eta_diff[j, m]

    if low_memory:
        # If the LD matrix used in the above operations is upper-triangular, then we would not
        # have updated the q-factors of variants based on the new etas for variants that come after them
        # in the dataset. So, we need to correct for that here:
        update_q_factor_matrix(active_model_idx, ld_left_bound, ld_indptr, ld_data, eta_diff, q, use_blas, threads)
