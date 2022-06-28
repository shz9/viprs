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
from libc.math cimport exp, log

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[::1] softmax(double[::1] x):
    """
    A numerically stable implementation of softmax
    """

    cdef unsigned int i, end = x.shape[0]
    cdef double s = 0., max_x = c_max(x)

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
cdef double sigmoid(double x):
    return 1./(1. + exp(-x))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double logit(double x):
    return log(x / (1. - x))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double dot(double[::1] v1, double[::1] v2):
    """
    Dot product between vectors of the same shape
    """

    cdef unsigned int i, end = v1.shape[0]
    cdef double s = 0.

    for i in range(end):
        s += v1[i]*v2[i]

    return s

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double vec_sum(double[::1] v1):
    """
    Vector summation
    """

    cdef unsigned int i, end = v1.shape[0]
    cdef double s = 0.

    for i in range(end):
        s += v1[i]

    return s

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double[::1] elementwise_add_mult(double[::1] v1, double[::1] v2, double s):
    """
    Elementwise addition and multiplication
    """

    cdef unsigned int i, end = v1.shape[0]

    for i in range(end):
        v1[i] = v1[i] + v2[i] * s

    return v1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[::1] clip_list(double[::1] a, double min_value, double max_value):
    """
    Iterate over a list and clip every element to be between `min_value` and `max_value`
    :param a: A list of doubles
    :param min_value: Minimum values
    :param max_value: Maximum value
    """

    cdef unsigned int i, end = a.shape[0]

    for i in range(end):
        a[i] = clip(a[i], min_value, max_value)

    return a

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double c_max(double[::1] x):
    """
    Obtain the maximum value in a vector `x`
    """
    cdef unsigned int i, end = x.shape[0]
    cdef double current_max = 0.

    for i in range(end):
        if i == 0 or current_max < x[i]:
            current_max = x[i]
    return current_max

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double clip(double a, double min_value, double max_value):
    """
    Clip a scalar value `a` to be between `min_value` and `max_value`
    """
    return min(max(a, min_value), max_value)

