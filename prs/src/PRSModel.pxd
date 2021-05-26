
cdef class PRSModel:

    cdef public:
        int N, M
        gdl
        dict pip, inf_beta, shapes

    cpdef fit(self)

    cpdef get_proportion_causal(self)
    cpdef get_heritability(self)
    cpdef get_pip(self)
    cpdef get_inf_beta(self)

    cpdef predict_phenotype(self)

    cpdef read_inferred_params(self, f_name)
    cpdef write_inferred_params(self, f_name)
