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
cimport numpy as np
import numpy as np


cdef class RunStatsVec:
    """
    Defines a Running statistics object that computes mean and variance in one-pass.
    Implementation is based on https://github.com/grantjenks/python-runstats
    but modified to allow for handling vectors.
    """

    def __init__(self, int size):

        self.size = size
        self.count = np.zeros(size).astype(np.int)
        self.eta = np.zeros(size)
        self.rho = np.zeros(size)

    cpdef void push_element(self, int index, double value):

        cdef:
            double delta = value - self.eta[index]
            double delta_n = delta / (self.count[index] + 1)
            double term = delta * delta_n * self.count[index]

        self.eta[index] += delta_n
        self.count[index] += 1
        self.rho[index] += term

    cpdef void push(self, double[::1] vec_value):

        cdef unsigned int i

        for i in range(self.size):
            self.push_element(i, vec_value[i])

    cpdef mean(self):
        return np.array(self.eta)

    cpdef variance(self, int ddof=1):
        return np.array(self.rho) / (np.array(self.count) - ddof)


cdef class RunStats:

    def __init__(self):

        self.count = self.eta = self.rho = 0

    cpdef void push(self, double value):

        cdef:
            double delta = value - self.eta
            double delta_n = delta / (self.count + 1)
            double term = delta * delta_n * self.count

        self.eta += delta_n
        self.count += 1
        self.rho += term

    cpdef mean(self):
        return self.eta

    cpdef variance(self, int ddof=1):
        return self.rho / (self.count - ddof)