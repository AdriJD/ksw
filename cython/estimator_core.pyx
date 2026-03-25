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


def compute_products_Afunc_sst(ct_weights, rule, weights,
		  a_ell_m, y_M_L, 
		  ndeltaL, m_dim, nphi,
		  L_list, deltaL_list, n,
		  w3j_product_scalar, w3j_product_tensor,
		  prefactors_scalar, prefactors_tensor, 
		  A_L_M_scalar, A_L_M_tensor,
		  Lmax, nell,
		  n_L_phi_scalar, n_L_phi_tensor,
		  f_i_phi_scalar, f_i_phi_tensor, 
		  kappa_i_L_scalar, kappa_i_L_tensor){
            
    ntheta = ct_weights.size
    nrule = rule.shape[0]
    nufact, npol, nL = f_i_L.shape
    
    if rule.shape != (nrule, 3):
        raise ValueError(f'rule.shape = {rule.shape}, expected {(nrule, 3)}')    

    if weights.shape != (nrule, 3):
        raise ValueError(f'weights.shape = {weights.shape}, expected {(nrule, 3)}')

    if kappa_i_L_scalar.shape != (nufact, npol, ndeltaL, nL):
        raise ValueError(
	f'kappa_i_L_scalar.shape = {kappa_i_L_scalar.shape}, expected {(nufact, npol, ndeltaL, nL)}')

    if kappa_i_L_tensor.shape != (nufact, npol, ndeltaL, nL):
        raise ValueError(
	f'kappa_i_L_tensor.shape = {kappa_i_L_tensor.shape}, expected {(nufact, npol, ndeltaL, nL)}')   

    if w3j_product_scalar.shape != (ndeltaL, nL, m_dim):
        raise ValueError(
	f'w3j_product_scalar.shape = {w3j_product_scalar.shape}, expected {(ndeltaL, nL, m_dim)}')

    if w3j_product_tensor.shape != (ndeltaL, nL, m_dim):
        raise ValueError(
	f'w3j_product_tensor.shape = {w3j_product_tensor.shape}, expected {(ndeltaL, nL, m_dim)}')

    if prefactors_scalar.shape != (ndeltaL, nL):
        raise ValueError(
    f'prefactors_scalar.shape = {prefactors_scalar.shape}, expected {(ndeltaL, nL)}')   

    if prefactors_tensor.shape != (ndeltaL, nL):
        raise ValueError(
    f'prefactors_tensor.shape = {prefactors_tensor.shape}, expected {(ndeltaL, nL)}')   

    if a_ell_m.shape != (npol, nell, nell):
        raise ValueError(
	f'a_ell_m.shape = {a_ell_m.shape}, expected {(npol, nell, nell)}')

    if y_M_L.shape != (ntheta, nL, nL):
        raise ValueError(
	f'y_M_L.shape = {y_M_L.shape}, expected {(ntheta, nL, nL)}')

    if A_L_M_scalar.shape != (npol, ndeltaL, nL, m_dim):
        raise ValueError(
    f'A_L_M_scalar.shape = {A_L_M_scalar.shape}, expected {(npol, ndeltaL, nL, m_dim)}')

    if A_L_M_tensor.shape != (npol, ndeltaL, nL, m_dim):
        raise ValueError(
    f'A_L_M_tensor.shape = {A_L_M_tensor.shape}, expected {(npol, ndeltaL, nL, m_dim)}')        

    if a_ell_m.dtype == np.complex64:
        t_cubic = _compute_products_Afunc_sst_sp(ct_weights, rule, weights,
		  a_ell_m, y_M_L, 
		  ndeltaL, m_dim, nphi,
		  L_list, deltaL_list, n,
		  w3j_product_scalar, w3j_product_tensor,
		  prefactors_scalar, prefactors_tensor, 
		  A_L_M_scalar, A_L_M_tensor,
		  Lmax, nell,
		  n_L_phi_scalar, n_L_phi_tensor,
		  f_i_phi_scalar, f_i_phi_tensor, 
		  kappa_i_L_scalar, kappa_i_L_tensor)
          
    elif a_ell_m.dtype == np.complex128:
        t_cubic = _compute_products_Afunc_sst_dp(ct_weights, rule, weights,
		  a_ell_m, y_M_L, 
		  ndeltaL, m_dim, nphi,
		  L_list, deltaL_list, n,
		  w3j_product_scalar, w3j_product_tensor,
		  prefactors_scalar, prefactors_tensor, 
		  A_L_M_scalar, A_L_M_tensor,
		  Lmax, nell,
		  n_L_phi_scalar, n_L_phi_tensor,
		  f_i_phi_scalar, f_i_phi_tensor, 
		  kappa_i_L_scalar, kappa_i_L_tensor)
    else:
        raise ValueError(f'dtype : {a_ell_m.dtype} not supported')

    return t_cubic 

    
          }
          
    def _compute_products_Afunc_sst_sp(ct_weights, rule, weights,
		  a_ell_m, y_M_L, 
		  ndeltaL, m_dim, nphi,
		  L_list, deltaL_list, n,
		  w3j_product_scalar, w3j_product_tensor,
		  prefactors_scalar, prefactors_tensor, 
		  A_L_M_scalar, A_L_M_tensor,
		  Lmax, nell,
		  n_L_phi_scalar, n_L_phi_tensor,
		  f_i_phi_scalar, f_i_phi_tensor, 
		  kappa_i_L_scalar, kappa_i_L_tensor):

    cdef float [::1] ct_weights_ = ct_weights.reshape(-1)
    cdef long long [::1] rule_ = rule.reshape(-1)
    cdef float [::1] weights_ = weights.reshape(-1)
    cdef float complex [::1] a_ell_m_ = a_ell_m.reshape(-1)
    cdef float [::1] y_M_L_ = y_M_L.reshape(-1)
    cdef float [::1] w3j_product_scalar_ = w3j_product_scalar.reshape(-1)
    cdef float [::1] w3j_product_tensor_ = w3j_product_tensor.reshape(-1)
    cdef float complex [::1] prefactors_scalar_ = prefactors_scalar.reshape(-1)
    cdef float complex [::1] prefactors_tensor_ = prefactors_tensor.reshape(-1)
    cdef float complex [::1] A_L_M_scalar_ = A_L_M_scalar.reshape(-1)
    cdef float complex [::1] A_L_M_tensor_ = A_L_M_tensor.reshape(-1)
    cdef float complex [::1] n_L_phi_scalar_ = n_L_phi_scalar.reshape(-1)
    cdef float complex [::1] n_L_phi_tensor_ = n_L_phi_tensor.reshape(-1)
    cdef float [::1] f_i_phi_scalar_ = f_i_phi_scalar.reshape(-1)
    cdef float [::1] f_i_phi_tensor_ = f_i_phi_tensor.reshape(-1)
    cdef float [::1] kappa_i_L_scalar_ = kappa_i_L_scalar.reshape(-1)
    cdef float [::1] kappa_i_L_tensor_ = kappa_i_L_tensor.reshape(-1)   

    cdef float [::1] L_list_ = L_list.reshape(-1)
    cdef float [::1] deltaL_list_ = deltaL_list.reshape(-1)

    cdef t_cubic = cestimator_core.t_cubic_sp_sst(&ct_weights_[0], &rule_[0], &weights_[0],
		  &a_ell_m_[0], &y_M_L_[0],
		  ndeltaL, m_dim, nphi,
		  &L_list_[0], &deltaL_list_[0], n,
		  &w3j_product_scalar_[0], &w3j_product_tensor_[0],
		  &prefactors_scalar_[0], &prefactors_tensor_[0],
		  &A_L_M_scalar_[0], &A_L_M_tensor_[0],
		  Lmax, nell,
		  &n_L_phi_scalar_[0], &n_L_phi_tensor_[0],
		  &f_i_phi_scalar_[0], &f_i_phi_tensor_[0], 
		  &kappa_i_L_scalar_[0], &kappa_i_L_tensor_[0])
    return t_cubic

def _compute_products_Afunc_sst_dp(ct_weights, rule, weights,
		  a_ell_m, y_M_L, 
		  ndeltaL, m_dim, nphi,
		  L_list, deltaL_list, n,
		  w3j_product_scalar, w3j_product_tensor,
		  prefactors_scalar, prefactors_tensor, 
		  A_L_M_scalar, A_L_M_tensor,
		  Lmax, nell,
		  n_L_phi_scalar, n_L_phi_tensor,
		  f_i_phi_scalar, f_i_phi_tensor, 
		  kappa_i_L_scalar, kappa_i_L_tensor):

    cdef double [::1] ct_weights_ = ct_weights.reshape(-1)
    cdef long long [::1] rule_ = rule.reshape(-1)
    cdef double [::1] weights_ = weights.reshape(-1)
    cdef double complex [::1] a_ell_m_ = a_ell_m.reshape(-1)
    cdef double [::1] y_M_L_ = y_M_L.reshape(-1)
    cdef double [::1] w3j_product_scalar_ = w3j_product_scalar.reshape(-1)
    cdef double [::1] w3j_product_tensor_ = w3j_product_tensor.reshape(-1)
    cdef double complex [::1] prefactors_scalar_ = prefactors_scalar.reshape(-1)
    cdef double complex [::1] prefactors_tensor_ = prefactors_tensor.reshape(-1)
    cdef double complex [::1] A_L_M_scalar_ = A_L_M_scalar.reshape(-1)
    cdef double complex [::1] A_L_M_tensor_ = A_L_M_tensor.reshape(-1)
    cdef double complex [::1] n_L_phi_scalar_ = n_L_phi_scalar.reshape(-1)
    cdef double complex [::1] n_L_phi_tensor_ = n_L_phi_tensor.reshape(-1)
    cdef double [::1] f_i_phi_scalar_ = f_i_phi_scalar.reshape(-1)
    cdef double [::1] f_i_phi_tensor_ = f_i_phi_tensor.reshape(-1)
    cdef double [::1] kappa_i_L_scalar_ = kappa_i_L_scalar.reshape(-1)
    cdef double [::1] kappa_i_L_tensor_ = kappa_i_L_tensor.reshape(-1)   

    cdef double [::1] L_list_ = L_list.reshape(-1)
    cdef double [::1] deltaL_list_ = deltaL_list.reshape(-1)

    cdef t_cubic = cestimator_core.t_cubic_sp_sst(&ct_weights_[0], &rule_[0], &weights_[0],
		  &a_ell_m_[0], &y_M_L_[0],
		  ndeltaL, m_dim, nphi,
		  &L_list_[0], &deltaL_list_[0], n,
		  &w3j_product_scalar_[0], &w3j_product_tensor_[0],
		  &prefactors_scalar_[0], &prefactors_tensor_[0],
		  &A_L_M_scalar_[0], &A_L_M_tensor_[0],
		  Lmax, nell,
		  &n_L_phi_scalar_[0], &n_L_phi_tensor_[0],
		  &f_i_phi_scalar_[0], &f_i_phi_tensor_[0], 
		  &kappa_i_L_scalar_[0], &kappa_i_L_tensor_[0])
    return t_cubic
    

def compute_ylm(thetas, lmax, dtype=np.float32):
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

