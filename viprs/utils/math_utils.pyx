# cython: linetrace=False
# cython: profile=False
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True

cimport cython
import numpy as np
from cython cimport floating
from libc.math cimport exp, log
from scipy.linalg.cython_blas cimport saxpy, daxpy, sdot, ddot


# ------------------------------------------------------------
# BLAS implementations of some linear algebra operations

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cdef void scipy_blas_axpy(floating[::1] v1, floating[::1] v2, floating alpha) noexcept nogil:
    """v1 := v1 + alpha * v2"""
    cdef:
        int inc = 1, n=v1.shape[0]

    if floating is float:
        saxpy(&n, &alpha, &v2[0], &inc, &v1[0], &inc)
    else:
        daxpy(&n, &alpha, &v2[0], &inc, &v1[0], &inc)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cdef floating scipy_blas_dot(floating[::1] v1, floating[::1] v2) noexcept nogil:
    """v1 . v2"""
    cdef:
        int inc = 1, n=v1.shape[0]

    if floating is float:
        return sdot(&n, &v1[0], &inc, &v2[0], &inc)
    else:
        return ddot(&n, &v1[0], &inc, &v2[0], &inc)

# ------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cdef floating[::1] softmax(floating[::1] x) noexcept nogil:
    """
    A numerically stable implementation of softmax
    """

    cdef unsigned int i, end = x.shape[0]
    cdef floating s = 0., max_x = c_max(x)

    with nogil:
        for i in range(end):
            x[i] = exp(x[i] - max_x)
            s += x[i]

        for i in range(end):
            x[i] /= s

    return x

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cdef floating sigmoid(floating x) noexcept nogil:
    """
    A numerically stable version of the Sigmoid function.
    """
    if x < 0:
        exp_x = exp(x)
        return exp_x / (1. + exp_x)
    else:
        return 1. / (1. + exp(-x))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cdef floating logit(floating x) noexcept nogil:
    """
    The logit function (inverse of the sigmoid function)
    """
    return log(x / (1. - x))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cdef floating dot(floating[::1] v1, floating[::1] v2) noexcept nogil:
    """
    Dot product between vectors of the same shape
    """

    cdef unsigned int i, end = v1.shape[0]
    cdef floating s = 0.

    with nogil:
        for i in range(end):
            s += v1[i]*v2[i]

    return s

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cdef floating vec_sum(floating[::1] v1) noexcept nogil:
    """
    Vector summation
    """

    cdef unsigned int i, end = v1.shape[0]
    cdef floating s = 0.

    with nogil:
        for i in range(end):
            s += v1[i]

    return s

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cdef void axpy(floating[::1] v1, floating[::1] v2, floating s) noexcept nogil:
    """
    Elementwise addition and multiplication
    """

    cdef unsigned int i, end = v1.shape[0]

    with nogil:
        for i in range(end):
            v1[i] = v1[i] + v2[i] * s

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cdef floating[::1] clip_list(floating[::1] a, floating min_value, floating max_value) noexcept nogil:
    """
    Iterate over a list and clip every element to be between `min_value` and `max_value`
    :param a: A list of floating point numbers
    :param min_value: Minimum values
    :param max_value: Maximum value
    """

    cdef unsigned int i, end = a.shape[0]

    with nogil:
        for i in range(end):
            a[i] = clip(a[i], min_value, max_value)

    return a

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cdef floating c_max(floating[::1] x) noexcept nogil:
    """
    Obtain the maximum value in a vector `x`
    """
    cdef unsigned int i, end = x.shape[0]
    cdef floating current_max = 0.

    with nogil:
        for i in range(end):
            if i == 0 or current_max < x[i]:
                current_max = x[i]

    return current_max

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.exceptval(check=False)
cdef floating clip(floating a, floating min_value, floating max_value) noexcept nogil:
    """
    Clip a scalar value `a` to be between `min_value` and `max_value`
    """

    if a < min_value:
        a = min_value
    if a > max_value:
        a = max_value

    return a

def bernoulli_entropy(p):
    """
    Compute the entropy of a Bernoulli variable given a vector of probabilities.
    :param p: A vector (or scalar) of probabilities between zero and one, 0. < p < 1.
    """
    return -(p*np.log(p) + (1. - p)*np.log(1. - p))
