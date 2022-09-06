
cdef class PRSModel:

    cdef public:
        int N
        object gdl
        dict pip, post_mean_beta, post_var_beta, Nj, shapes

    cpdef fit(self)
    cpdef get_proportion_causal(self)
    cpdef get_heritability(self)
    cpdef get_pip(self)
    cpdef get_posterior_mean_beta(self)
    cpdef get_posterior_variance_beta(self)

    cpdef predict(self, gdl=*)
    cpdef harmonize_data(self, gdl=*, parameter_table=*)
    cpdef to_table(self, col_subset=*, per_chromosome=*)
    cpdef set_model_parameters(self, parameter_table)
    cpdef read_inferred_parameters(self, f_name, sep=*)
    cpdef write_inferred_parameters(self, f_name, per_chromosome=*, sep=*)
