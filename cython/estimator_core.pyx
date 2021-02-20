cimport cestimator_core
import numpy as np

def step(ct_weights, rule, weights, f_i_ell, a_ell_m, y_m_ell, grad_t, nphi):
    '''
    Calculate the contribution to grad T for a set of rings.

    Arguments
    ---------
    ct_weights : (ntheta) array
        Quadrature weights for cos(theta) on each ring.
    rule : (nrule, 3) array
        Rule to combine unique bispectrum factors 
    weights : (nrule, 3) array
        Amplitude for each element of rule.
    f_i_ell : (nufact, npol, nell) array
        Unique bispectrum factors.
    a_ell_m : (npol, nell, nell) complex array
        SH coefficients in ell-major order.
    y_m_ell : (ntheta, nell, nell) array
        Ylms in m-major order for each ring.
    grad_t : (npol, nell, nell) complex array
        SH coefficients of grad T in ell-major order. Results will be added
	to this array.
    nphi : int
        Number of phi points on each ring.

    Raises
    ------
    ValueError
        If input shapes do not match.
    '''

    ntheta = ct_weights.size
    nrule = rule.shape[0]
    nufact, npol, nell = f_i_ell.shape
    
    if rule.shape != (nrule, 3):
        raise ValueError(f'rule.shape = {rule.shape}, expected {(nrule, 3)}')    

    if weights.shape != (nrule, 3):
        raise ValueError(f'weights.shape = {weights.shape}, expected {(nrule, 3)}')

    if f_i_ell.shape != (nufact, npol, nell):
        raise ValueError(
	f'f_i_ell.shape = {f_i_ell.shape}, expected {(nufact, npol, nell)}')

    if a_ell_m.shape != (npol, nell, nell):
        raise ValueError(
	f'a_ell_m.shape = {a_ell_m.shape}, expected {(npol, nell, nell)}')

    if y_m_ell.shape != (ntheta, nell, nell):
        raise ValueError(
	f'y_m_ell.shape = {y_m_ell.shape}, expected {(ntheta, nell, nell)}')

    if grad_t.shape != (npol, nell, nell):
        raise ValueError(
	f'grad_t.shape = {grad_t.shape}, expected {(npol, nell, nell)}')

    if a_ell_m.dtype == np.complex64:
        _step_sp(ct_weights, rule, weights, f_i_ell, a_ell_m, y_m_ell, grad_t,
             ntheta, nrule, nell, npol, nufact, nphi)
    elif a_ell_m.dtype == np.complex128:
        _step_dp(ct_weights, rule, weights, f_i_ell, a_ell_m, y_m_ell, grad_t,
             ntheta, nrule, nell, npol, nufact, nphi)	
    else:
        raise ValueError(f'dtype : {a_ell_m.dtype} not supported')

def _step_sp(ct_weights, rule, weights, f_i_ell, a_ell_m, y_m_ell, grad_t,
             ntheta, nrule, nell, npol, nufact, nphi):
    ''' Single precision version. '''

    cdef double [::1] ct_weights_ = ct_weights.reshape(-1)
    cdef long long [::1] rule_ = rule.reshape(-1)
    cdef float [::1] weights_ = weights.reshape(-1)
    cdef float [::1] f_i_ell_ = f_i_ell.reshape(-1)
    cdef float complex [::1] a_ell_m_ = a_ell_m.reshape(-1)
    cdef float [::1] y_m_ell_ = y_m_ell.reshape(-1)
    cdef float complex [::1] grad_t_ = grad_t.reshape(-1)

    cestimator_core.step_sp(&ct_weights_[0], &rule_[0], &weights_[0],
                 &f_i_ell_[0], &a_ell_m_[0], &y_m_ell_[0], &grad_t_[0],
		 ntheta, nrule, nell, npol, nufact, nphi)

def _step_dp(ct_weights, rule, weights, f_i_ell, a_ell_m, y_m_ell, grad_t,
             ntheta, nrule, nell, npol, nufact, nphi):
    ''' Double precision version. '''

    cdef double [::1] ct_weights_ = ct_weights.reshape(-1)
    cdef long long [::1] rule_ = rule.reshape(-1)
    cdef double [::1] weights_ = weights.reshape(-1)
    cdef double [::1] f_i_ell_ = f_i_ell.reshape(-1)
    cdef double complex [::1] a_ell_m_ = a_ell_m.reshape(-1)
    cdef double [::1] y_m_ell_ = y_m_ell.reshape(-1)
    cdef double complex [::1] grad_t_ = grad_t.reshape(-1)

    cestimator_core.step_dp(&ct_weights_[0], &rule_[0], &weights_[0],
                 &f_i_ell_[0], &a_ell_m_[0], &y_m_ell_[0], &grad_t_[0],
		 ntheta, nrule, nell, npol, nufact, nphi)

def compute_estimate(ct_weights, rule, weights, f_i_ell, a_ell_m, y_m_ell, nphi):
    '''
    Compute T[a] for a collection of rings.

    Arguments
    ---------
    ct_weights : (ntheta) array
        Quadrature weights for cos(theta) on each ring.
    rule : (nrule, 3) array
        Rule to combine unique bispectrum factors 
    weights : (nrule, 3) array
        Amplitude for each element of rule.
    f_i_ell : (nufact, npol, nell) array
        Unique bispectrum factors.
    a_ell_m : (npol, nell, nell) complex array
        SH coefficients in ell-major order.
    y_m_ell : (ntheta, nell, nell) array
        Ylms in m-major order for each ring.
    nphi : int
        Number of phi points on each ring.

    Returns
    -------
    t_cubic : float
        Contribution to estimate, i.e. T[a], from these rings.

    Raises
    ------
    ValueError
        If input shapes do not match.
    '''

    ntheta = ct_weights.size
    nrule = rule.shape[0]
    nufact, npol, nell = f_i_ell.shape
    
    if rule.shape != (nrule, 3):
        raise ValueError(f'rule.shape = {rule.shape}, expected {(nrule, 3)}')    

    if weights.shape != (nrule, 3):
        raise ValueError(f'weights.shape = {weights.shape}, expected {(nrule, 3)}')

    if f_i_ell.shape != (nufact, npol, nell):
        raise ValueError(
	f'f_i_ell.shape = {f_i_ell.shape}, expected {(nufact, npol, nell)}')

    if a_ell_m.shape != (npol, nell, nell):
        raise ValueError(
	f'a_ell_m.shape = {a_ell_m.shape}, expected {(npol, nell, nell)}')

    if y_m_ell.shape != (ntheta, nell, nell):
        raise ValueError(
	f'y_m_ell.shape = {y_m_ell.shape}, expected {(ntheta, nell, nell)}')

    if a_ell_m.dtype == np.complex64:
        t_cubic = _compute_estimate_sp(ct_weights, rule, weights, f_i_ell, a_ell_m, y_m_ell,
             ntheta, nrule, nell, npol, nufact, nphi)
    elif a_ell_m.dtype == np.complex128:
        t_cubic = _compute_estimate_dp(ct_weights, rule, weights, f_i_ell, a_ell_m, y_m_ell,
             ntheta, nrule, nell, npol, nufact, nphi)	
    else:
        raise ValueError(f'dtype : {a_ell_m.dtype} not supported')

    return t_cubic

def _compute_estimate_sp(ct_weights, rule, weights, f_i_ell, a_ell_m, y_m_ell,
             ntheta, nrule, nell, npol, nufact, nphi):
    ''' Single precision version. '''

    cdef double [::1] ct_weights_ = ct_weights.reshape(-1)
    cdef long long [::1] rule_ = rule.reshape(-1)
    cdef float [::1] weights_ = weights.reshape(-1)
    cdef float [::1] f_i_ell_ = f_i_ell.reshape(-1)
    cdef float complex [::1] a_ell_m_ = a_ell_m.reshape(-1)
    cdef float [::1] y_m_ell_ = y_m_ell.reshape(-1)

    cdef t_cubic = cestimator_core.t_cubic_sp(&ct_weights_[0], &rule_[0], &weights_[0],
                               &f_i_ell_[0], &a_ell_m_[0], &y_m_ell_[0], ntheta, nrule,
		               nell, npol, nufact, nphi)
    return t_cubic

def _compute_estimate_dp(ct_weights, rule, weights, f_i_ell, a_ell_m, y_m_ell,
             ntheta, nrule, nell, npol, nufact, nphi):
    ''' Double precision version. '''

    cdef double [::1] ct_weights_ = ct_weights.reshape(-1)
    cdef long long [::1] rule_ = rule.reshape(-1)
    cdef double [::1] weights_ = weights.reshape(-1)
    cdef double [::1] f_i_ell_ = f_i_ell.reshape(-1)
    cdef double complex [::1] a_ell_m_ = a_ell_m.reshape(-1)
    cdef double [::1] y_m_ell_ = y_m_ell.reshape(-1)

    cdef t_cubic = cestimator_core.t_cubic_dp(&ct_weights_[0], &rule_[0], &weights_[0],
                               &f_i_ell_[0], &a_ell_m_[0], &y_m_ell_[0], ntheta, nrule,
		               nell, npol, nufact, nphi)
    return t_cubic

def _compute_ylm(thetas, lmax, dtype=np.float32):
    '''
    Compute Ylm(theta,0) for a range of thetas.

    Arguments
    ---------
    thetas : (ntheta) array 
        Theta values.
    lmax : int	
        Maximum multipole.
    dtype : type, optional
        dtype used for output, choose between np.float32 and 64.

    Returns
    -------
    y_m_ell : (ntheta, nell, nell) array
        Ylms in m-major order for each ring.            
    '''
    
    nell = lmax + 1
    ntheta = thetas.size
    y_m_ell = np.zeros((ntheta, nell, nell), dtype=dtype)

    if dtype == np.float32:
        _compute_ylm_sp(thetas, y_m_ell, ntheta, lmax)
    elif dtype == np.float64:
        _compute_ylm_dp(thetas, y_m_ell, ntheta, lmax)
    else:
        raise ValueError(f'dtype : {dtype} not supported')

    return y_m_ell

def _compute_ylm_sp(thetas, y_m_ell, ntheta, lmax):
    '''Single precision version.'''

    cdef double [::1] thetas_ = thetas.reshape(-1)
    cdef float [::1] y_m_ell_ = y_m_ell.reshape(-1)

    cestimator_core.compute_ylm_sp(&thetas_[0], &y_m_ell_[0], ntheta, lmax)

def _compute_ylm_dp(thetas, y_m_ell, ntheta, lmax):
    '''Double precision version.'''

    cdef double [::1] thetas_ = thetas.reshape(-1)
    cdef double [::1] y_m_ell_ = y_m_ell.reshape(-1)

    cestimator_core.compute_ylm_dp(&thetas_[0], &y_m_ell_[0], ntheta, lmax)

