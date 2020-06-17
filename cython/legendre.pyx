'''
Wrapper functions adapted from the wavemoth repository by Dag Sverre Seljebotn.
See https://github.com/wavemoth/wavemoth.
'''

cdef extern from "ylmgen_c.h":
    ctypedef struct Ylmgen_C:
        int *firstl
        double *ylm

    void Ylmgen_init(Ylmgen_C *gen, int l_max, int m_max, int s_max,
                     int spinrec, double epsilon)
    void Ylmgen_set_theta(Ylmgen_C *gen, double *theta, int nth)
    void Ylmgen_destroy (Ylmgen_C *gen)
    void Ylmgen_prepare (Ylmgen_C *gen, int ith, int m)
    void Ylmgen_recalc_Ylm (Ylmgen_C *gen)
    void Ylmgen_recalc_lambda_wx (Ylmgen_C *gen, int spin)
    double *Ylmgen_get_norm (int lmax, int spin, int spinrec)

cimport numpy as np
import numpy as np
from libc.string cimport memcpy

cimport cython

@cython.wraparound(False)
def compute_normalized_associated_legendre(int m, theta, int lmax, double epsilon=1e-300,
                                           out=None):
    '''
    Given a value for m, computes the matrix Ylm(theta,0)
    for all provided theta values and m <= ell <= lmax.

    Parameters
    ----------
    m : int
    theta : (ntheta) array
        Theta values that obey 0 <= theta <= pi.
    lmax : int
    epsilon : float
       Ylm with absolute magnitude < epsilon may be approximated as 0.
    out : (ntheta, nell) array, None
        Optional output array. Values will be overwritten.

    Returns
    -------
    out : (ntheta, nell) array
        Normalized associated Legendre polynomials.

    Raises
    ------
    ValueError
        If lmax < m.
        if m < 0.
        If shape output array does not match input sizes.
    '''
    
    cdef Ylmgen_C ctx
    cdef Py_ssize_t col, row
    cdef np.ndarray[double, mode='c'] theta_ = np.ascontiguousarray(theta, dtype=np.double)
    cdef np.ndarray[double, ndim=2] out_
    cdef int firstl
    
    if lmax < m:
        raise ValueError("lmax < m ({} < {})".format(lmax, m))
    if m < 0:
        raise ValueError("m < 0 (m = {})".format(m))
    
    if out is None:
        out = np.empty((theta_.shape[0], lmax - m + 1), np.double)
    out_ = out
    if out_.shape[0] != theta_.shape[0] or out_.shape[1] != lmax + 1 - m:
        raise ValueError("Invalid shape of out")
    
    Ylmgen_init(&ctx, lmax, lmax, 0, 0, epsilon)
    try:
        Ylmgen_set_theta(&ctx, <double*>theta_.data, theta_.shape[0])
        for row in range(theta_.shape[0]):
            Ylmgen_prepare(&ctx, row, m)
            Ylmgen_recalc_Ylm(&ctx)
            firstl = ctx.firstl[0] # Argument: spin.
            for col in range(m, min(firstl, lmax + 1)):
                out_[row, col - m] = 0
            for col in range(max(m, firstl), lmax + 1):
                out_[row, col - m] = ctx.ylm[col]
    finally:
        Ylmgen_destroy(&ctx)
        
    return out

@cython.wraparound(False)
def normalized_associated_legendre_ms(m, double theta, int lmax, double epsilon=1e-300,
                                      out=None):
    '''
    Given a value for theta, computes the matrix Ylm(theta,0)
    for all provided m values and m <= ell <= lmax.

    Parameters
    ----------
    m : (nm) array
        Non negative m values.
    theta : float
        Theta value that obeys 0 <= theta <= pi.
    lmax : int
    epsilon : float
       Ylm with absolute magnitude < epsilon may be approximated as 0.
    out : (ntheta, nell) array, None
        Optional output array. Values will be overwritten.

    Returns
    -------
    out : (nm, nell) array
        Normalized associated Legendre polynomials.

    Raises
    ------
    ValueError
        if theta does not obey 0 <= theta < np.pi.
        If shape output array does not match input sizes.
    '''
    
    cdef Ylmgen_C ctx
    cdef Py_ssize_t col, row, mval
    cdef np.ndarray[int, mode='c'] m_ = np.ascontiguousarray(m, dtype=np.intc)
    cdef np.ndarray[double, ndim=2] out_
    cdef int firstl

    if not 0 <= theta <= np.pi:
        raise ValueError('Theta : {} outside [0, pi]'.format(theta))
    
    if out is None:
        out = np.empty((m_.shape[0], lmax + 1), np.double)
    out_ = out
    if out_.shape[0] != m_.shape[0] or out_.shape[1] != lmax + 1:
        raise ValueError("Invalid shape of out")
    
    Ylmgen_init(&ctx, lmax, lmax, 0, 0, epsilon)
    try:
        Ylmgen_set_theta(&ctx, &theta, 1)
        for row in range(m_.shape[0]):
            Ylmgen_prepare(&ctx, 0, m_[row])
            Ylmgen_recalc_Ylm(&ctx)
            firstl = ctx.firstl[0] # Argument: spin.
            for col in range(min(firstl, lmax + 1)):
                out[row, col] = 0
            for col in range(firstl, lmax + 1):
                out_[row, col] = ctx.ylm[col]
    finally:
        Ylmgen_destroy(&ctx)
        
    return out
