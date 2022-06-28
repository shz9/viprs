
cdef class PRSModel:

    cdef public:
        int N, M
        gdl
        dict pip, post_mean_beta, post_var_beta, Nj, shapes

    cpdef fit(self)

    cpdef get_proportion_causal(self)
    cpdef get_heritability(self)
    cpdef get_pip(self)
    cpdef get_posterior_mean_beta(self)
    cpdef get_posterior_variance_beta(self)

    cpdef predict(self, gdl=*)
    cpdef harmonize_data(self, gdl=*, eff_table=*)
    cpdef to_table(self, per_chromosome=*, col_subset=*)

    cpdef read_inferred_params(self, f_name, sep=*)
    cpdef write_inferred_params(self, f_name, per_chromosome=*, sep=*)
