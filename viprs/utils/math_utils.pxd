from cython cimport floating

cdef floating[::1] softmax(floating[::1] x) noexcept nogil
cdef floating sigmoid(floating x) noexcept nogil
cdef floating logit(floating x) noexcept nogil
cdef floating dot(floating[::1] v1, floating[::1] v2) noexcept nogil
cdef floating vec_sum(floating[::1] v1) noexcept nogil
cdef void axpy(floating[::1] v1, floating[::1] v2, floating s) noexcept nogil
cdef void scipy_blas_axpy(floating[::1] v1, floating[::1] v2, floating alpha) noexcept nogil
cdef floating scipy_blas_dot(floating[::1] v1, floating[::1] v2) noexcept nogil

cdef floating[::1] clip_list(floating[::1] a, floating min_value, floating max_value) noexcept nogil
cdef floating c_max(floating[::1] x) noexcept nogil
cdef floating clip(floating a, floating min_value, floating max_value) noexcept nogil
