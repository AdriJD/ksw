cimport cfisher_core
import numpy as np

def fisher_nxn(sqrt_icov_ell, f_ell_i, thetas, ct_weights, rule, weights, fisher_nxn):
    '''
    Calculate upper-triangular part of (nrule x nrule) fisher matrix.

    Arguments
    ---------
    sqrt_icov_ell : (nell, npol, npol) array
        Square root of icov per multipole (symmetric in pol).
    f_ell_i : (nell, npol, nufact) array
        Unique factors of reduced bispectrum.
    thetas : (ntheta) array
        Theta value for each ring.
    ct_weights : (ntheta) array
        Quadrature weights for cos(theta) on each ring.
    rule : (nrule, 3) array
        Rule to combine unique bispectrum factors 
    weights : (nrule, 3) array
        Amplitude for each element of rule.    
    fisher_nxn : (nrule * nrule) array
        Fisher matrix for these rings, only upper-tringular part is filled.

    Raises
    ------
    ValueError 
        If input shapes do not match.
    '''

    nell, npol, nufact = f_ell_i.shape
    lmax = nell - 1
    nrule = rule.shape[0]
    ntheta = thetas.size

    if sqrt_icov_ell.shape[-2:] != (npol, npol):
        raise ValueError(f'Pol dimensions of sqrt_icov : {sqrt_icov_ell.shape[-2:]} '
                         f'do not match npol of f_ell_i {npol}')

    if sqrt_icov_ell.shape[0] != nell:
        raise ValueError(f'nell dimension of sqrt_icov : {sqrt_icov_ell.shape[-2:]} '
                         f'do not match nell of f_ell_i {npol}')

    if thetas.size != ct_weights.size:
        raise ValueError(f'Size thetas : {thetas.size} is not equal to size '
	                 f'ct_weights : {ct_weights.size}')

    if rule.shape != weights.shape:
        raise ValueError(f'Shape rule : {rule.shape} is not equal to shape weights : '
                         f'{weights.shape}')

    if f_ell_i.dtype == np.float32:
        _fisher_nxn_sp(sqrt_icov_ell, f_ell_i, thetas, ct_weights, rule, weights,
	               fisher_nxn, nufact, nrule, ntheta, lmax, npol)
    elif f_ell_i.dtype == np.float64:
        _fisher_nxn_dp(sqrt_icov_ell, f_ell_i, thetas, ct_weights, rule, weights,
	               fisher_nxn, nufact, nrule, ntheta, lmax, npol)
    else:
        raise ValueError(f'dtype : {f_ell_i.dtype} not supported')


def _fisher_nxn_sp(sqrt_icov_ell, f_ell_i, thetas, ct_weights, rule, weights,
                   fisher_nxn, nufact, nrule, ntheta, lmax, npol):
    ''' Single precision version.'''

    cdef float [::1] sqrt_icov_ell_ = sqrt_icov_ell.reshape(-1)
    cdef float [::1] f_ell_i_ = f_ell_i.reshape(-1)
    cdef double [::1] thetas_ = thetas.reshape(-1)
    cdef double [::1] ct_weights_ = ct_weights.reshape(-1)
    cdef long long [::1] rule_ = rule.reshape(-1)
    cdef float [::1] weights_ = weights.reshape(-1)
    cdef float [::1] fisher_nxn_ = fisher_nxn.reshape(-1)    
    
    cfisher_core.fisher_nxn_sp(&sqrt_icov_ell_[0], &f_ell_i_[0], &thetas_[0],
    		   &ct_weights_[0], &rule_[0], &weights_[0], &fisher_nxn_[0], 
		   nufact, nrule, ntheta, lmax, npol)

def _fisher_nxn_dp(sqrt_icov_ell, f_ell_i, thetas, ct_weights, rule, weights,
                   fisher_nxn, nufact, nrule, ntheta, lmax, npol):
    ''' Double precision version.'''

    cdef double [::1] sqrt_icov_ell_ = sqrt_icov_ell.reshape(-1)
    cdef double [::1] f_ell_i_ = f_ell_i.reshape(-1)
    cdef double [::1] thetas_ = thetas.reshape(-1)
    cdef double [::1] ct_weights_ = ct_weights.reshape(-1)
    cdef long long [::1] rule_ = rule.reshape(-1)
    cdef double [::1] weights_ = weights.reshape(-1)
    cdef double [::1] fisher_nxn_ = fisher_nxn.reshape(-1)    
    
    cfisher_core.fisher_nxn_dp(&sqrt_icov_ell_[0], &f_ell_i_[0], &thetas_[0],
    		   &ct_weights_[0], &rule_[0], &weights_[0], &fisher_nxn_[0], 
		   nufact, nrule, ntheta, lmax, npol)

    