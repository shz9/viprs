cdef class RunStatsVec:

    cdef public:
        unsigned int size
        long[::1] count
        double[::1] eta
        double[::1] rho

    cpdef void push_element(self, int index, double value)
    cpdef void push(self, double[::1] vec_value)
    cpdef mean(self)
    cpdef variance(self, int ddof=*)

cdef class RunStats:

    cdef public:
        unsigned int count
        double eta, rho

    cpdef void push(self, double value)
    cpdef mean(self)
    cpdef variance(self, int ddof=*)
