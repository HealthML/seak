
#cython: language_level=3

import numpy as np

cimport numpy
cimport cython

ctypedef numpy.float64_t DTYPE_float
ctypedef numpy.int_t DTYPE_int

from libc.math cimport log

@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def optimize_lambda(numpy.ndarray[DTYPE_float, ndim=1] res,
                    numpy.ndarray[DTYPE_int, ndim=1] lambdaind,
                    Py_ssize_t nsim,
                    Py_ssize_t g,
                    int n0,
                    numpy.ndarray[DTYPE_float, ndim=1] ChiSum,
                    numpy.ndarray[DTYPE_float, ndim=1] ChiK,
                    numpy.ndarray[DTYPE_float, ndim=2] Chi1,
                    numpy.ndarray[DTYPE_float, ndim=1] sumlog1plambdaxi,
                    numpy.ndarray[DTYPE_float, ndim=2] fN,
                    numpy.ndarray[DTYPE_float, ndim=2] fD):
    
    cdef Py_ssize_t k = fN.shape[1]
    
    cdef DTYPE_float N
    cdef DTYPE_float D
    cdef DTYPE_float LR
        
    for i_s in range(nsim):
        
        LR = 0.
        
        for i_g in range(g):
            
            N = 0.
            D = 0.
            
            for i_k in range(k):
                N += fN[i_g, i_k] * Chi1[i_s,i_k]
                D += fD[i_g, i_k] * Chi1[i_s,i_k]
            
            D += ChiK[i_s]
            
            LR = n0 * log(N/D+1.) - sumlog1plambdaxi[i_g]
            
            if LR >= res[i_s]:
                res[i_s] = LR
                lambdaind[i_s] = i_g
            else:
                break
