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
        out = np.zeros(alm.shape[:-1] + (lmax + 1, mmax + 1), dtype=alm.dtype)
    
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
        out = np.zeros(arr.shape[:-2] + (hp.Alm.getsize(lmax, mmax=mmax),),
                       dtype=arr.dtype)        
        
    out *= 0
    for m in range(mmax + 1):
        
        start = hp.Alm.getidx(lmax, m, m)
        end = start + lmax + 1 - m
        
        out[...,start:end] = arr[...,m:,m]
        
    return out

def contract_almxblm(alm, blm):
    '''
    Return sum_lm alm x blm, i.e. the sum of the Hadamard product of two
    sets of spherical harmonic coefficients corresponding to real fields.

    Parameters
    ---------
    alm : (..., nelem) complex array
        Healpix ordered (m-major) alm array.
    blm : (..., nelem) complex array
        Healpix ordered (m-major) alm array.

    Returns
    -------
    had_sum : float
        Sum of Hadamard product (real valued).

    Raises
    ------
    ValueError
        If input arrays have different shapes.
    '''

    if blm.shape != alm.shape:
        raise ValueError('Shape alm ({}) != shape blm ({})'.format(alm.shape, blm.shape))

    lmax = hp.Alm.getlmax(alm.shape[-1])

    csum = complex(np.tensordot(alm, blm, axes=alm.ndim))    
    had_sum = 2 * np.real(csum)

    # We need to subtract the m=0 elements once.
    had_sum -= np.real(np.sum(alm[...,:lmax+1] * blm[...,:lmax+1]))

    return had_sum    

def alm_return_2d(alm, npol, lmax):
    '''
    Check shape of alm array and return 2d view of input.

    Parameters
    ----------
    alm : (nelem) or (n, nelem) array
        Healpix-ordered array to be checked.
    npol : int
        Number of expected polarization dimensions.
    lmax : int
        Expected maximumm multipole.

    Returns
    -------
    alm : array
        2d view of input, so (1, nelem) if input was 1d.

    Raises
    ------
    ValueError
        If input shapes differs from expected.
    '''

    ndim_in = alm.ndim
    if ndim_in == 1:
        alm = alm[np.newaxis,:]
    if alm.ndim != 2:
        raise ValueError('Expected 1d or 2d alm array, got dim = {}'.
                         format(ndim_in))

    npol_in = alm.shape[0]
    if npol_in != npol:
        raise ValueError('Expected alm npol = {}, got {}'.format(npol, npol_in))

    lmax_in = hp.Alm.getlmax(alm.shape[1])
    if lmax_in != lmax:
        raise ValueError('Expected alm lmax = {}, got {}'.format(lmax_in, lmax))

    return alm
 
def reduce_array(arr, comm, op=None, root=0):
    '''
    Reduce numpy array to root.

    Parameters
    ----------
    arr : array
    comm : MPI communicator
    op : mpi4py.MPI.Op object, optional
        Operation during reduce, defaults to SUM.
    root : int, optional
        Reduce array to this rank.
    
    Returns
    -------
    arr_out : array, None
        Reduced array on root, None on other ranks.
    '''

    if isinstance(comm, FakeMPIComm) or comm.Get_size() == 1:
        return arr
    
    if comm.Get_rank() == root:
        arr_out = np.zeros_like(arr)
    else:
        arr_out = None

    if op is not None:
        comm.Reduce(arr, arr_out, op=op, root=root)
    else:
        comm.Reduce(arr, arr_out, root=root)

    return arr_out

def allreduce_array(arr, comm, op=None):
    '''
    Allreduce numpy array.

    Parameters
    ----------
    arr : array
        Array present on all ranks.
    comm : MPI communicator
    op : mpi4py.MPI.Op object, optional
        Operation during reduce, defaults to SUM.
    
    Returns
    -------
    arr_out : array
        Reduced array.
    '''

    if isinstance(comm, FakeMPIComm) or comm.Get_size() == 1:
        return arr
    
    arr_out = np.zeros_like(arr)

    if op is not None:
        comm.Allreduce(arr, arr_out, op=op)
    else:
        comm.Allreduce(arr, arr_out)

    return arr_out

def bcast_array(arr, comm, root=0):
    '''
    Reduce numpy array to root.

    Parameters
    ----------
    arr : array
    comm : MPI communicator
    root : int, optional
        Broadcast array from this rank.
    
    Returns
    -------
    arr_out : array, None
        Broadcasted array on all ranks.
    '''
  
    if isinstance(comm, FakeMPIComm) or comm.Get_size() == 1:
        return arr

    if comm.Get_rank() == root:
        shape = arr.shape
        dtype = arr.dtype
    else:
        shape, dtype = None, None

    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)

    if comm.Get_rank() != root:
        arr = np.zeros(shape, dtype=dtype)

    comm.Bcast(arr, root=root)

    return arr

def reduce(obj, comm, op=None, root=0):
    '''
    Reduce python object to root.

    Parameters
    ----------
    obj : object
    comm : MPI communicator
    op : mpi4py.MPI.Op object, optional
        Operation during reduce, defaults to SUM.
    root : int, optional
        Reduce object to this rank.
    
    Returns
    -------
    obj_out : obj, None
        Reduced object on root, None on other ranks.
    '''
    
    if isinstance(comm, FakeMPIComm) or comm.Get_size() == 1:
        return obj

    if op is not None:
        obj_out = comm.reduce(obj, op=op, root=root)
    else:
        obj_out = comm.reduce(obj, root=root)
    
    return obj_out

def allreduce(obj, comm, op=None):
    '''
    Allreduce python object.

    Parameters
    ----------
    obj : object
    comm : MPI communicator
    op : mpi4py.MPI.Op object, optional
        Operation during reduce, defaults to SUM.
    
    Returns
    -------
    obj_out : obj
        Reduced object (on all ranks).
    '''
    
    if isinstance(comm, FakeMPIComm) or comm.Get_size() == 1:
        return obj

    if op is not None:
        obj_out = comm.allreduce(obj, op=op)
    else:
        obj_out = comm.allreduce(obj)
    
    return obj_out
    
def bcast(obj, comm, root=0):
    '''
    Broadcast python object.

    Parameters
    ----------
    obj : object
        Object on root rank, can be None on other ranks.
    comm : MPI cummunicator
    root : int, optional
        Broadcast object from thi rank.

    Returns
    -------
    obj_out : obj
        Input object on all ranks.
    '''
    
    if isinstance(comm, FakeMPIComm) or comm.Get_size() == 1:
        return obj
    
    return comm.bcast(obj, root=root)

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
