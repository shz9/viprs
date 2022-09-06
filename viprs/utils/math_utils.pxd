cdef double[::1] softmax(double[::1] x) nogil
cdef double sigmoid(double x) nogil
cdef double logit(double x) nogil
cdef double dot(double[::1] v1, double[::1] v2) nogil
cdef double vec_sum(double[::1] v1) nogil
cdef void elementwise_add_mult(double[::1] v1, double[::1] v2, double s) nogil
cdef double[::1] clip_list(double[::1] a, double min_value, double max_value) nogil
cdef double c_max(double[::1] x) nogil
cdef double clip(double a, double min_value, double max_value) nogil
