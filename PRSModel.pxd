from libcpp cimport bool

cdef class PRSModel:

    cdef public int N
    cdef public int M
    cdef public gdl
    cdef public pip
    cdef public inf_beta

    cpdef fit(self)

    cpdef get_heritability(self)
    cpdef get_pip(self)
    cpdef get_inf_beta(self)

    cpdef predict_phenotype(self)

    cpdef write_inferred_params(self)
