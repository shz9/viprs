
from .VIPRS cimport VIPRS

cdef class VIPRSGrid(VIPRS):

    cdef public:
        object grid_table
        object validation_result
        list stop_iteration
        int n_models
