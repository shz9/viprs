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
from libc.math cimport exp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double sigmoid(double x):
    return 1./(1. + exp(-x))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double dot(double[::1] v1, double[::1] v2):
    """
    TODO: Figure out a way to parallelize this with prange
    :param v1: 
    :param v2: 
    :return: 
    """

    cdef unsigned int i, end = v1.shape[0]
    cdef double s = 0.

    with nogil:
        for i in range(end):
            s += v1[i]*v2[i]

    return s

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double[::1] elementwise_add_mult(double[::1] v1, const double[::1] v2, double s):
    """
    TODO: Figure out a way to parallelize this with prange
    :param v1: 
    :param v2: 
    :param s: 
    :return: 
    """

    cdef unsigned int i, end = v1.shape[0]

    for i in range(end):
        v1[i] = v1[i] + v2[i] * s

    return v1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double clip(double a, double min_value, double max_value):
    return min(max(a, min_value), max_value)

