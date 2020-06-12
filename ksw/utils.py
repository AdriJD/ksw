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
