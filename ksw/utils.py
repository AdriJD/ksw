import numpy as np

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
