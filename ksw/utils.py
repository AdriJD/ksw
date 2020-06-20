import numpy as np

import healpy as hp

def get_trapz_weights(x):
    '''
    Compute weights dx for trapezoidal integration rule.

    Parameters
    ----------
    x : array
        One dimensional monotonically increasing array.

    Returns
    -------
    dx : array
        Weights for each element in input array.

    Raises
    ------
    ValueError
        If input is not 1D.
        If input is not monotonically increasing.
    '''
    
    if x.ndim != 1:
        raise ValueError('Input dimension {} != 1.'.format(x.ndim))

    if np.any(np.diff(x) < 0):
        raise ValueError('Input array is not monotonically increasing.')
    
    dx = np.empty(x.size, dtype=float)
    dx[1:-1] = x[2:] - x[:-2]
    dx[0] = x[1] - x[0]
    dx[-1] = x[-1] - x[-2]
    dx /= 2.
    
    return dx

def compute_fftlen_fftw(len_min, even=True):
    '''
    Compute optimal array length for FFTW given a minumum length.

    Paramters
    ---------
    len_min : int
        Minumum length.
    even : bool, optional
        Demand even optimal lengths (for real FFTs).

    Returns
    -------
    len_opt : int
        Optimal length

    Notes
    -----
    FFTW likes input sizes that can be factored as 2^a 3^b 5^c 7^d.
    '''

    max_a = int(np.ceil(np.log(len_min) / np.log(2)))
    max_b = int(np.ceil(np.log(len_min) / np.log(3)))
    max_c = int(np.ceil(np.log(len_min) / np.log(5)))
    max_d = int(np.ceil(np.log(len_min) / np.log(7)))       

    len_opt = 2 ** max_a # Reasonable starting point.
    for a in range(max_a):
        for b in range(max_b):
            for c in range(max_c):
                for d in range(max_d):
                    fftlen = 2 ** a * 3 ** b * 5 ** c * 7 ** d
                    if even and fftlen % 2:
                        continue
                    if fftlen < len_min:
                        continue
                    if fftlen == len_min:
                        len_opt = fftlen
                        break
                    if fftlen < len_opt:
                        len_opt = fftlen

    return len_opt

def alm2a_ell_m(alm, out=None, mmax=None):
    '''
    Fill N + 2 dimensional array with N + 1 dimensional alm array.

    Parameters
    ----------
    alm : (..., nelem) array
        Healpix ordered (m-major) alm array.
    out : (..., nell, nm) array, optional
        ell-major alm array to be filled.
    mmax : int, None
        Maxumum m-mode used for alm array,

    Returns
    -------
    a_m_ell : (..., nell, nm) array
        (N + 2) d ell-major alm array.

    Raises
    ------
    ValueError
        If shapes do not match.
    '''

    lmax = hp.Alm.getlmax(alm.shape[-1], mmax=mmax)
    if mmax is None:
        mmax = lmax
        
    if out is not None:
        # First dims must match.
        if alm.shape[:-1] != out.shape[:-2]:
            raise ValueError('Mismatch shapes alm {} and out {}'.
                             format(alm.shape, out.shape))
        # Last dims must match.
        if out.shape[-2:] != (lmax + 1, mmax + 1):
            raise ValueError(
                'Expected out.shape[-2:] (lmax+1, mmax+1) = {}, got {} '.format(
                    (lmax + 1, mmax + 1), out.shape[-2:]))
    else:
        out = np.empty(alm.shape[:-1] + (lmax + 1, mmax + 1), dtype=alm.dtype)
    
    out *= 0
    for m in range(mmax + 1):
        
        start = hp.Alm.getidx(lmax, m, m)
        end = start + lmax + 1 - m
        
        out[...,m:,m] = alm[...,start:end]
        
    return out
    
def a_ell_m2alm(arr, out=None):
    '''
    Fill N + 1 dimensional alm array with N + 2 dimensional array.

    Parameters
    ----------
    arr : (..., nell, nm):
        ell-major alm array.
    out : (..., nelem) array, optional
        Healpix ordered (m-major) alm array to be filled

    Returns
    -------
    alm : (..., nelem) array
        Healpix ordered (m-major) alm array        

    Raises
    ------
    ValueError
        If shapes do not match.
    '''
    
    mmax = arr.shape[-1] - 1
    lmax = arr.shape[-2] - 1

    if out is not None:
        # First dims must match.
        if out.shape[:-1] != arr.shape[:-2]:
            raise ValueError('Mismatch shapes out : {} and arr : {}'.
                             format(out.shape, arr.shape))
        # Last dims must match.
        if out.shape[-1] != hp.Alm.getsize(lmax, mmax=mmax):
            raise ValueError('Expected out.shape[-1] : {}, got : {}'.
                format(hp.Alm.getsize(lmax, mmax=mmax), out.shape[-1]))
    else:
        out = np.empty(arr.shape[:-2] + (hp.Alm.getsize(lmax, mmax=mmax),),
                       dtype=arr.dtype)        
        
    out *= 0
    for m in range(mmax + 1):
        
        start = hp.Alm.getidx(lmax, m, m)
        end = start + lmax + 1 - m
        
        out[...,start:end] = arr[...,m:,m]
        
    return out
    
class FakeMPIComm():
    '''
    Mimic an actual MPI communicator.

    Attributes
    ----------
    size : int    
    rank : int
    '''
    
    def __init__(self):
        pass
    
    def Get_size(self):
        return 1
    def Get_rank(self):        
        return 0
