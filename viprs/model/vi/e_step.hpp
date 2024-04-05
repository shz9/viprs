#ifndef E_STEP_H
#define E_STEP_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <type_traits>

// Check for and include `cblas`:
#ifdef HAVE_CBLAS
    #include <cblas.h>
#endif

// Check for and include `omp`:
#ifdef _OPENMP
    #include <omp.h>
#endif


/* ----------------------------- */
// Helper system-related functions to check for BLAS and OpenMP support

bool omp_supported() {
    /* Check if OpenMP is supported by examining compiler flags. */
    #ifdef _OPENMP
        return true;
    #else
        return false;
    #endif
}

bool blas_supported() {
    /* Check if BLAS is supported by examining compiler flags. */
    #ifdef HAVE_CBLAS
        return true;
    #else
        return false;
    #endif
}

/* ------------------------------------------------------------------------ */
// Linear algebra + math functions

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
clamp(T x, T min, T max)
{
    /* Clamp a scalar value between a minimum `min` and maximum `max` value. */
    if (x < min) x = min;
    if (x > max) x = max;

    return x;
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
c_max(T* x, int size) {
    /* Find the maximum value in a vector x of length `size`. */

    T current_max = x[0];

    for (int i = 1; i < size; ++i) {
        if (current_max < x[i]) {
            current_max = x[i];
        }
    }
    return current_max;
}

/* ------------------------------ */
// Dot product functions

// Define a function pointer for the dot product functions `dot` and `blas_dot`:
template <typename T, typename U>
using dot_func_pt = typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, T>::type (*)(T*, U*, int);

/* * * * * */

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, T>::type
dot(T* x, U* y, int size) {
    /* Perform dot product between two vectors x and y, each of length `size`

    :param x: Pointer to the first element of the first vector
    :param y: Pointer to the first element of the second vector
    :param size: Length of the vectors

    */

    T s = 0.;

    #ifdef _OPENMP
        #ifndef _WIN32
            #pragma omp simd
        #endif
    #endif
    for (int i = 0; i < size; ++i) {
        s += x[i]*static_cast<T>(y[i]);
    }
    return s;
}

/* * * * * */

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, T>::type
blas_dot(T* x, U* y, int size) {
    /*
        Use BLAS (if available) to perform dot product
        between two vectors x and y, each of length `size`.

        :param x: Pointer to the first element of the first vector
        :param y: Pointer to the first element of the second vector
        :param size: Length of the vectors
    */

    #ifdef HAVE_CBLAS
        int incx = 1;
        int incy = 1;

        if constexpr (std::is_same<T, float>::value) {
            if constexpr (std::is_same<U, float>::value) {
                return cblas_sdot(size, x, incx, y, incy);
            } else {
                // Handles the case where y is any data type that is not a float:
                std::vector<float> y_float(size);
                std::transform(y, y + size, y_float.begin(),  [](U val) { return static_cast<float>(val);});
                return cblas_sdot(size, x, incx, y_float.data(), incy);
            }
        }
        else if constexpr (std::is_same<T, double>::value) {
            if constexpr (std::is_same<U, double>::value) {
                return cblas_ddot(size, x, incx, y, incy);
            } else {
                // Handles the case where y is any data type that is not a double:
                std::vector<double> y_double(size);
                std::transform(y, y + size, y_double.begin(),  [](U val) { return static_cast<double>(val);});
                return cblas_ddot(size, x, incx, y_double.data(), incy);
            }
        }
    #else
        return dot(x, y, size);
    #endif
}

/* * * * * */

// Define a function pointer for the axpy functions `axpy` and `blas_axpy`:
template <typename T, typename U>
using axpy_func_pt = typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, void>::type (*)(T*, U*, T, int);

/* * * * * */

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, void>::type
axpy(T* x, U* y, T alpha, int size) {
    /*
        Perform axpy operation on two vectors x and y, each of length `size`.
       axpy is a standard linear algebra operation that performs
       element-wise addition and multiplication:
       x := x + a*y.
    */

    #ifdef _OPENMP
        #ifndef _WIN32
            #pragma omp simd
        #endif
    #endif
    for (int i = 0; i < size; ++i) {
        x[i] += static_cast<T>(y[i]) * alpha;
    }
}

/* * * * * */

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, void>::type
blas_axpy(T *y, U *x, T alpha, int size) {
    /*
        Use BLAS (if available) to perform axpy operation on two vectors x and y,
        each of length `size`.
       axpy is a standard linear algebra operation that performs
       element-wise addition and multiplication:
       x := x + a*y.
    */

    #ifdef HAVE_CBLAS
        int incx = 1;
        int incy = 1;

        if constexpr (std::is_same<T, float>::value) {
            if constexpr (std::is_same<U, float>::value) {
                cblas_saxpy(size, alpha, x, incx, y, incy);
            } else {
                // Handles the case where x is any data type that is not a float:
                std::vector<float> x_float(size);
                std::transform(x, x + size, x_float.begin(),  [](U val) { return static_cast<float>(val);});
                cblas_saxpy(size, alpha, x_float.data(), incx, y, incy);
            }
        }
        else if constexpr (std::is_same<T, double>::value) {
            if constexpr (std::is_same<U, double>::value) {
                cblas_daxpy(size, alpha, x, incx, y, incy);
            } else {
                // Handles the case where x is any data type that is not a float:
                std::vector<double> x_double(size);
                std::transform(x, x + size, x_double.begin(),  [](U val) { return static_cast<double>(val);});
                cblas_daxpy(size, alpha, x_double.data(), incx, y, incy);
            }
        }
    #else
        axpy(y, x, alpha, size);
    #endif
}

/* ------------------------------ */
// Numerically stable softmax and sigmoid functions

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
softmax(T* logits, T* output, int size) {
    /*
        Perform softmax operation on a vector of logits,
        and store the result in `output`. This function implements
        a numerically stable version of softmax to avoid overflow.
    */

    T s = 0., max_val = c_max(logits, size);

    for (int i = 0; i < size; ++i) {
        logits[i] = exp(logits[i] - max_val);
        s += logits[i];
    }

    for (int i = 0; i < size - 1; ++i) {
        output[i] = logits[i] / s;
    }
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
sigmoid(T x) {
    /*
        Perform sigmoid operation on a scalar x.
        This function implements a numerically stable version of sigmoid
        to avoid overflow.
    */

    if (x < 0) {
        T exp_x = exp(x);
        return exp_x / (1. + exp_x);
    }
    else {
        return 1. / (1. + exp(-x));
    }
}

/* ------------------------------------------------------------------------ */
// Helper functions for the E-step

template <typename T, typename U, typename I>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value && std::is_integral<I>::value, void>::type
update_q_factor_matrix(int c_size,
                        int n_active_models,
                        int* active_model_idx,
                        int* ld_left_bound,
                        I* ld_indptr,
                        U* ld_data,
                        T* eta,
                        T* q,
                        T dq_scale,
                        bool use_blas,
                        int threads) {
    /*
        Compute or update q factor, defined as the result of a dot product between the
        Linkage-disequilibrium matrix (LD) and the current value of eta (the posterior mean
        for the effect sizes). The definition of the q-factor excludes the diagonal elements.

        This function assumes that we have a matrix of eta values, where each column represents
        eta from a different model (this is relevant for grid-based models).

        The implementation below assumes that the eta and q matrices are column-major matrices.
    */

    I ld_start, ld_end, mat_idx;

    // Determine the dot function depending on whether we are using BLAS or not:
    dot_func_pt<T, U> dot_func = use_blas ? blas_dot<T, U> : dot<T, U>;

    #ifdef _OPENMP
        #pragma omp parallel for private(ld_start, ld_end, mat_idx) schedule(static) num_threads(threads)
    #endif
    for (int j = 0; j < c_size; ++j) {
        ld_start = ld_indptr[j];
        ld_end = ld_indptr[j + 1];

        for (int m=0; m < n_active_models; ++m){
            mat_idx = active_model_idx[m]*c_size; // Assumes column-major matrices.
            q[mat_idx + j] += dq_scale*dot_func(eta + (mat_idx + ld_left_bound[j]), ld_data + ld_start, ld_end - ld_start);
        }
    }
}

template <typename T, typename U, typename I>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value && std::is_integral<I>::value, void>::type
update_q_factor(int c_size,
                 int* ld_left_bound,
                 I* ld_indptr,
                 U* ld_data,
                 T* eta,
                 T* q,
                 T dq_scale,
                 bool use_blas,
                 int threads) {
    /*
        Compute or update q factor, defined as the result of a dot product between the
        Linkage-disequilibrium matrix (LD) and the current value of eta (the posterior mean
        for the effect sizes). The definition of the q-factor excludes the diagonal elements.

        This function assumes that we have a vector of eta values. If you have a matrix of eta
        values, you should use the `update_q_factor_matrix` function instead.
    */

    I ld_start, ld_end;

    // Determine the dot function depending on whether we are using BLAS or not:
    dot_func_pt<T, U> dot_func = use_blas ? blas_dot<T, U> : dot<T, U>;

    #ifdef _OPENMP
        #pragma omp parallel for private(ld_start, ld_end) schedule(static) num_threads(threads)
    #endif
    for (int j = 0; j < c_size; ++j) {

        ld_start = ld_indptr[j];
        ld_end = ld_indptr[j + 1];

        q[j] += dq_scale*dot_func(eta + ld_left_bound[j], ld_data + ld_start, ld_end - ld_start);
    }
}

/* ------------------------------------------------------------------------ */
// E-step functions

template <typename T, typename U, typename I>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value && std::is_integral<I>::value, void>::type
e_step(int c_size,
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
        bool use_blas,
        bool low_memory) {

    /*
        Perform the E-Step of the coordinate-ascent variational inference (CAVI) algorithm for the
        simple spike-and-slab model.

        The function iterates over the variants in the dataset, and updates the posterior mean (var_mu) and
        posterior probability (var_gamma) according to the update rules in Zabad et al. (2023). The implementation
        assumes that the variational tau (inverse variance) have been updated prior to calling this function.

        In addition to updating the variational posterior parameters var_gamma and var_mu, the function also
        computes certain quantities that are used in the M-step of the CAVI algorithm. These quantities include
        the q-factor for each variant (the dot product between the LD matrix and the posterior mean), and eta_diff,
        defined as the difference between the current value of eta and the updated value of eta.
    */

    int start, end;
    I ld_start, ld_end;
    T u_j;

    // Determine the axpy function depending on whether we are using BLAS or not:
    axpy_func_pt<T, U> axpy_func = use_blas ? blas_axpy<T, U> : axpy<T, U>;

    #ifdef _OPENMP
        #pragma omp parallel for private(start, end, ld_start, ld_end, u_j) schedule(static) num_threads(threads)
    #endif
    for (int j = 0; j < c_size; ++j) {

        ld_start = ld_indptr[j];
        ld_end = ld_indptr[j + 1];
        start = ld_left_bound[j];
        end = start + (ld_end - ld_start);

        /* Update the posterior mean for variant j */
        var_mu[j] = mu_mult[j] * (std_beta[j] - q[j]);

        /* Update the posterior inclusion probability for variant j */
        u_j = u_logs[j] + half_var_tau[j] * var_mu[j] * var_mu[j];
        var_gamma[j] = sigmoid(u_j);

        /* Update eta_diff for variant j */
        eta_diff[j] = var_gamma[j] * var_mu[j] - eta[j];

        /* Update the q-factors for variants that are in LD with variant j */
        axpy_func(q + start, ld_data + ld_start, dq_scale*eta_diff[j], end - start);

        if (!low_memory) {
            /* If the matrix is symmetric, updating q in the previous step would also
            update the q-factor for the focal variant (j). So, we need to correct for
            this here. */
            q[j] = q[j] - eta_diff[j];
        }

        /* Update eta (posterior mean) for variant j */
        eta[j] = eta[j] + eta_diff[j];
    }

    if (low_memory) {
        /* If the LD matrix used in the above operations is upper-triangular, then we would not
           have updated the q-factors of variants based on the new etas for variants that come after them
           in the dataset. So, we need to correct for that here: */
        update_q_factor(c_size, ld_left_bound, ld_indptr, ld_data, eta_diff, q, dq_scale, use_blas, threads);
    }

}


template <typename T, typename U, typename I>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value && std::is_integral<I>::value, void>::type
e_step_mixture(int c_size,
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
                bool use_blas,
                bool low_memory) {

    /*
        Perform the E-Step of the coordinate-ascent variational inference (CAVI) algorithm for the
        sparse scale-mixture of Gaussians model.

        The function iterates over the variants in the dataset, and updates the posterior mean (var_mu) and
        posterior probability (var_gamma) according to the update rules in Zabad et al. (2023). The implementation
        assumes that the variational tau (inverse variance) have been updated prior to calling this function.

        In addition to updating the variational posterior parameters var_gamma and var_mu, the function also
        computes certain quantities that are used in the M-step of the CAVI algorithm. These quantities include
        the q-factor for each variant (the dot product between the LD matrix and the posterior mean), and eta_diff,
        defined as the difference between the current value of eta and the updated value of eta.
    */

    // Determine the axpy function depending on whether we are using BLAS or not:
    axpy_func_pt<T, U> axpy_func = use_blas ? blas_axpy<T, U> : axpy<T, U>;

    /* Delineate the parallel region: */
    #ifdef _OPENMP
        #pragma omp parallel num_threads(threads)
    #endif
    {
        // Declare variables that are private to each thread:
        int start, end, mat_idx;
        I ld_start, ld_end;
        T mu_beta_j;
        T* u_j = new T[K + 1];

        #ifdef _OPENMP
            #pragma omp for schedule(static)
        #endif
        for (int j = 0; j < c_size; ++j) {

            ld_start = ld_indptr[j];
            ld_end = ld_indptr[j + 1];

            start = ld_left_bound[j];
            end = start + (ld_end - ld_start);

            /* Update the posterior mean for each variant j and each of the K mixture components */
            mu_beta_j = std_beta[j] - q[j];

            for (int k = 0; k < K; ++k) {
                mat_idx = j*K + k; // Assumes C-order matrices.
                var_mu[mat_idx] = mu_mult[mat_idx] * mu_beta_j;
                u_j[k] = u_logs[mat_idx] + half_var_tau[mat_idx] * var_mu[mat_idx] * var_mu[mat_idx];
            }

            /* Compute the posterior inclusion probability by applying softmax over the logits (u_j) */
            u_j[K] = log_null_pi[j];
            softmax(u_j, var_gamma + j*K, K + 1);

            /* Update eta_diff for variant j */
            eta_diff[j] = -eta[j];

            for (int k = 0; k < K; ++k) {
                mat_idx = j*K + k; // Assumes C-order matrices.
                eta_diff[j] += var_gamma[mat_idx] * var_mu[mat_idx];
            }

            /* Update the q-factors for variants that are in LD with variant j */
            axpy_func(q + start, ld_data + ld_start, dq_scale*eta_diff[j], end - start);

            if (!low_memory) {
                /* If the matrix is symmetric, updating q in the previous step would also
                update the q-factor for the focal variant (j). So, we need to correct for
                this here. */
                q[j] -= eta_diff[j];
            }

            eta[j] += eta_diff[j];
        }

        delete[] u_j;

    }

    if (low_memory) {
        /*  If the LD matrix used in the above operations is upper-triangular, then we would not
            have updated the q-factors of variants based on the new etas for variants that come after them
            in the dataset. So, we need to correct for that here:
        */
        update_q_factor(c_size, ld_left_bound, ld_indptr, ld_data, eta_diff, q, dq_scale, use_blas, threads);
    }

}

template <typename T, typename U, typename I>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value && std::is_integral<I>::value, void>::type
e_step_grid(int c_size,
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
             bool use_blas,
             bool low_memory) {
     /*
        Perform the E-Step of the coordinate-ascent variational inference (CAVI) algorithm for the
        a grid of spike-and-slab models. The grid is defined over the hyperparameters of the model, so that
        instead of one set of variational parameters for each variant, we have a vector of variational
        parameters consistent with the grid of models. In this implementation, the variational parameters form
        a matrix (rather than a vector) where the dimensions are (number of variants) x (number of models in the grid).

        The function iterates over the variants in the dataset, and updates the posterior mean (var_mu) and
        posterior probability (var_gamma) according to the update rules in Zabad et al. (2023). The implementation
        assumes that the variational tau (inverse variance) have been updated prior to calling this function.

        In addition to updating the variational posterior parameters var_gamma and var_mu, the function also
        computes certain quantities that are used in the M-step of the CAVI algorithm. These quantities include
        the q-factor for each variant (the dot product between the LD matrix and the posterior mean), and eta_diff,
        defined as the difference between the current value of eta and the updated value of eta.
    */

    int start, end, mat_idx, model_idx;
    I ld_start, ld_end;
    T u_j;

    // Determine the axpy function depending on whether we are using BLAS or not:
    axpy_func_pt<T, U> axpy_func = use_blas ? blas_axpy<T, U> : axpy<T, U>;

    #ifdef _OPENMP
        #pragma omp parallel for private(start, end, ld_start, ld_end, mat_idx, model_idx, u_j) schedule(static) num_threads(threads)
    #endif
    for (int j = 0; j < c_size; ++j) {

        ld_start = ld_indptr[j];
        ld_end = ld_indptr[j + 1];
        start = ld_left_bound[j];
        end = start + (ld_end - ld_start);

        /* Loop over the *active* models in the grid (i.e. models that still have not converged) */
        for (int m=0; m < n_active_models; ++m){

            model_idx = active_model_idx[m];
            mat_idx = model_idx*c_size + j; // Assumes column-major matrices.

            /* Update the posterior mean for variant j and model m */
            var_mu[mat_idx] = mu_mult[mat_idx] * (std_beta[j] - q[mat_idx]);

            /* Update the posterior inclusion probability for variant j and model m */
            u_j = u_logs[mat_idx] + half_var_tau[mat_idx] * var_mu[mat_idx] * var_mu[mat_idx];
            var_gamma[mat_idx] = sigmoid(u_j);

            /* Update eta_diff for variant j and model m */
            eta_diff[mat_idx] = var_gamma[mat_idx] * var_mu[mat_idx] - eta[mat_idx];

            /* Update the q-factors for variants that are in LD with variant j */
            axpy_func(q + (model_idx*c_size + start), ld_data + ld_start, dq_scale*eta_diff[mat_idx], end - start);

            if (!low_memory) {
                /* If the matrix is symmetric, updating q in the previous step would also
                update the q-factor for the focal variant (j). So, we need to correct for
                this here. */
                q[mat_idx] -= eta_diff[mat_idx];
            }

            /* Update eta (posterior mean) for variant j and model m */
            eta[mat_idx] += eta_diff[mat_idx];
        }
    }

    if (low_memory) {
        /* If the LD matrix used in the above operations is upper-triangular, then we would not
           have updated the q-factors of variants based on the new etas for variants that come after them
           in the dataset. So, we need to correct for that here:
        */
        update_q_factor_matrix(c_size, n_active_models, active_model_idx,
                               ld_left_bound, ld_indptr, ld_data, eta_diff,
                               q, dq_scale, use_blas, threads);
    }

}


#endif // E_STEP_H
