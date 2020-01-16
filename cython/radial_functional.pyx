from radial_functional cimport compute_radial_func
import numpy as np
cimport numpy as np
import sys

def check_and_return_shape(arr, exp_shape):
    '''
    Return shape of input array if it matches expections.

    Arguments
    ---------
    arr : array
        Array to be checked.
    exp_shape : array_like
        Expected shape of array, use `None` for unknown dimension
        sizes.

    Returns
    -------
    shape : tuple

    Raises
    ------
    ValueError
        Raises ValueError if array shape is incorrect.    

    '''
    if arr.ndim != len(exp_shape):
        raise ValueError(
            "Array dimensions incorrect (expected {}, got {})".format(
                len(exp_shape), arr.ndim))
    
    for idx, (n, n_exp) in enumerate(zip(arr.shape, exp_shape)):
        
        if n_exp is None:
            continue
        
        if n != n_exp:
            raise ValueError(
                "Incorrect size of dim {} (expected {}, got {})".format(
                    idx, n_exp, n))            
    return arr.shape

def radial_func(f_k, tr_ell_k, k, radii, ells):
    '''
    Compute f_ell^X(r) = int k^2 dk f(k) transfer^X_ell(k) j_ell(k r),
    where f(k) is an arbitrary function of wavenumber k.

    Arguments
    ---------
    f_k : (nk, ncomp) array
        Input functions.
    tr_ell_k : (nell, nk, npol) array
        Transfer functions.
    k : (nk) array
        Wavenumbers in 1/Mpc.
    radii : (nr) array
        Radii in Mpc.
    ells : (nell) array
        Multipoles

    Returns
    -------
    f_ell_r : (nr, nell, npol, ncomp) array
        Evaluted integral for arr radii, multipoles, polarizations
        and input function components.
    '''

    # Check input for nans and infs.
    f_k = np.asarray_chkfinite(f_k, dtype=float, order='C')
    tr_ell_k = np.asarray_chkfinite(tr_ell_k, dtype=float, order='C')
    k = np.asarray_chkfinite(k, dtype=float, order='C')
    radii = np.asarray_chkfinite(radii, dtype=float, order='C')
    ells = np.asarray_chkfinite(ells, dtype=int, order='C')

    # Check for input shapes.
    nk, = check_and_return_shape(k, [None])
    nr, = check_and_return_shape(radii, [None])
    nell, = check_and_return_shape(ells, [None])
    nk, ncomp = check_and_return_shape(f_k, [nk, None])
    nell, nk, npol = check_and_return_shape(tr_ell_k, [nell, nk, None])

    # Create output array.
    f_ell_r = np.empty((nr, nell, npol, ncomp), dtype=float)
    
    cdef double [::1] f_k_ = f_k.reshape(-1)
    cdef double [::1] tr_ell_k_ = tr_ell_k.reshape(-1)
    cdef double [::1] k_ = k.reshape(-1)
    cdef double [::1] radii_ = radii.reshape(-1)
    cdef double [::1] f_ell_r_ = f_ell_r.reshape(-1)
    cdef int [::1] ells_ = ells
    
    compute_radial_func(&f_k_[0],
                         &tr_ell_k_[0],
                         &k_[0],
                         &radii_[0],
                         &f_ell_r_[0],
                         &ells_[0],
                         nk,
                         nell,
                         nr,
                         npol,
                         ncomp)
    return f_ell_r
    
