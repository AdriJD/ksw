import numpy as np
from scipy.special import roots_legendre

import healpy as hp
from optweight import mat_utils
import pyfftw

from ksw import utils, legendre, estimator_core, fisher_core

class KSW():
    '''
    Implementation of the Komatsu Spergel Wandelt estimator using the
    factorization by Smith and Zaldarriaga 2011.

    Parameters
    ----------
    red_bispctra : (list of) ksw.ReducedBispectrum instance(s)
        Estimate fNL for these reduced bispectra.
    icov : callable
        Function takes (npol, nelem) alm-like complex array "a" and returns the 
        inverse-variance-weighted version of that array. Specifically: 
        (B^{-1} N B^{-1} + S)^{-1} B^{-1} a, where a = B s + n, B is the beam
        and N^{-1} and S^{-1} are the inverse noise and signal covariance
        matrices, respectively.
    beam : callable
        Function takes (npol, nelem) alm-like complex array
        and returns beam-convolved version of that array. Defaults to no beam.
    lmax : int
        Max multipole used in estimator. Should match shape of alms.
    pol : str or array-like of strings.
        Data polarization, e.g. "E", or ["T", "E"]. Should match shape of alms.
    precision : str, optional
        Use either "single" precision or "double" precision data types 
        for internal calculations.

    Attributes
    ----------
    red_bispectra : list of ksw.ReducedBispectrum instances
        Reduced bispectra templates.
    icov : callable, None
        The inverse variance weighting operation.
    beam : callable
        The beam convolution operation.
    lmax : int
        Max multipole used in estimator.
    pol : tuple
        Data polarizations.
    mc_idx : int
        Counter for Monte Carlo estimates.
    mc_gt : (npol, nelem) complex array, None
        Current <grad T (C^-1 a)> Monte Carlo estimate (eq 60 Smith Zaldarriaga).
    mc_gt_sq : float, None
        Current <grad T (C^-1 a) C^-1 grad T(C^-1 a)^*> Monte Carlo estimate 
        (eq 61 Smith Zaldarriaga).
    thetas : (ntheta) array
        Coordinates of isolatitude rings.
    theta_weights (ntheta) array
        Weight for each isolatitude ring.
    nphi : int
        Number of phi points.
    dtype : type
        Dtype for real quantities, i.e. np.float32/np.float64 if precision is 
        "single"/"double".
    cdtype : type
        Dtype for complex quantities, i.e. np.complex64/np.complex128 if 
        precision is "single"/"double".
    '''

    def __init__(self, red_bispectra, icov, beam, lmax, pol, precision='single'):

        self.red_bispectra = red_bispectra
        self.icov = icov
        self.beam = beam
        self.mc_idx = 0
        self.mc_gt = None
        self.mc_gt_sq = None

        self.lmax = lmax
        self.pol = pol

        if precision == 'single':
            self.dtype = np.float32
            self.cdtype = np.complex64
        elif precision == 'double':
            self.dtype = np.float64
            self.cdtype = np.complex128
        else:
            raise ValueError('precision {} not understood'.format(precision))

        if len(red_bispectra) > 1:
            raise NotImplementedError('no joint estimation for now.')

        self.thetas, self.theta_weights, self.nphi = self.get_coords()

    @property
    def pol(self):
        return self.__pol
    
    @pol.setter
    def pol(self, pol):
        '''Check input and make sorted tuple.'''
        pol = list(np.atleast_1d(pol))
        sort_order = {"T": 0, "E": 1}

        if pol.count('T') + pol.count('E') != len(pol):
            raise ValueError(f'Pol={pol}, but may only contain T and/or E.')
        elif pol.count('T') != 1 and pol.count('E') != 1:
            raise ValueError(f'Pol={pol}, cannot contain duplicates.')

        pol.sort(key=lambda val: sort_order[val[0]])
        self.__pol = tuple(pol)

    @property
    def npol(self):
        return len(self.pol)

    @property
    def mc_gt(self):
        mc_gt = self.__mc_gt
        try:
            return mc_gt / self.mc_idx
        except TypeError:
            return mc_gt

    @mc_gt.setter
    def mc_gt(self, mc_gt):
        self.__mc_gt = mc_gt

    @property
    def mc_gt_sq(self):
        mc_gt_sq = self.__mc_gt_sq
        try:
            return mc_gt_sq / self.mc_idx
        except TypeError:
            return mc_gt_sq

    @mc_gt_sq.setter
    def mc_gt_sq(self, mc_gt_sq):
        self.__mc_gt_sq = mc_gt_sq

    def get_coords(self):
        '''
        Compute samples on sphere that are sufficient for lmax.

        Returns
        -------
        thetas : (ntheta) array
            Theta coordinates.
        ct_weights : (ntheta) array
            Quadrature weights for cos(theta).
        nphi : int
            Number of phi samples on each isolatitude ring.

        Notes
        -----
        We use Gauss-Legendre weights for isolatitude rings, see astro-ph/0305537.
        '''

        cos_thetas, ct_weights = roots_legendre(int(np.floor(1.5 * self.lmax) + 1))
        thetas = np.arccos(cos_thetas)

        nphi_min = 3 * self.lmax + 1
        nphi = utils.compute_fftlen_fftw(nphi_min, even=True)

        return thetas, ct_weights, nphi

    def _init_reduced_bispectrum(self, red_bisp):
        '''
        Prepare reduced bispectrum for estimation.

        Parameters
        ----------
        red_bisp : ksw.ReducedBispectrum instance
            Assumed to have both T and E.
        
        Returns
        -------
        f_i_ell : (nufact, npol, nell) array
            Unique factors of bispectrum scaled by beam.
        rule : (nfact, 3) array
            Rule to map unique factors to bispectrum.
        weights : (nfact, 3) array
            Amplitude for each element in rule.

        Raises
        ------
        ValueError
            If lmax of reduced bispectrum < lmax.
        '''
        
        if red_bisp.lmax < self.lmax:
            raise ValueError('lmax bispectrum ({}) < lmax ({})'.format(
                red_bisp.lmax, self.lmax))

        nufact = red_bisp.factors.shape[0]
        f_i_ell = np.zeros((nufact, self.npol, self.lmax + 1),
                           dtype=self.dtype)

        # Find index of lmax data in ells of red. bisp.
        try:
            end_ells_full = np.where(red_bisp.ells_full == self.lmax)[0][0] + 1
        except IndexError:
            end_ells_full = None

        # Slice corresponding to data pol. Assume red. bisp. has T and E.
        if self.npol == 1 and 'T' in self.pol:
            pslice = slice(0, 1, None)
        elif self.npol == 1 and 'E' in self.pol:
            pslice = slice(1, 2, None)
        else:
            pslice = slice(0, 2, None)

        f_i_ell[:,:,red_bisp.lmin:red_bisp.lmax+1] = \
            red_bisp.factors[:,pslice,:end_ells_full]
        f_i_ell = f_i_ell.astype(self.dtype)

        rule = red_bisp.rule
        weights = red_bisp.weights.astype(self.dtype)

        return f_i_ell, rule, weights

    def _step(self, alm, theta_batch=25):
        '''
        Calculate grad T (C^-1 a).

        Parameters
        ----------
        alm : (nelem) or (npol, nelem) complex array
            Healpix-ordered unfiltered alm array. Will be overwritten!
        theta_batch : int, optional
            Process loop over theta in batches of this size. Higher values
            take up more memory.
        
        Returns
        -------
        grad_t : (nelem) or (npol, nelem) complex array
            Healpix-ordered alm array.
        '''

        alm = utils.alm_return_2d(alm, self.npol, self.lmax)
        alm = self.icov(alm)
        a_ell_m = utils.alm2a_ell_m(alm)
        a_ell_m = a_ell_m.astype(self.cdtype)
        grad_t = np.zeros_like(a_ell_m)

        red_bisp = self.red_bispectra[0]
        f_i_ell, rule, weights = self._init_reduced_bispectrum(red_bisp)

        for tidx_start in range(0, len(self.thetas), theta_batch):

            thetas_batch = self.thetas[tidx_start:tidx_start+theta_batch]
            ct_weights_batch = self.theta_weights[tidx_start:tidx_start+theta_batch]
            y_m_ell = estimator_core.compute_ylm(thetas_batch, self.lmax,
                                                 dtype=self.dtype)
            estimator_core.step(ct_weights_batch, rule, weights, f_i_ell, a_ell_m, y_m_ell,
                               grad_t, self.nphi)

        # Turn back into healpy shape.
        grad_t = utils.a_ell_m2alm(grad_t).astype(self.cdtype)

        return grad_t

    def step(self, alm, theta_batch=25):
        '''
        Add iteration to <grad T (C^-1 a) C^-1 grad T(C^-1 a)^*> and 
        <grad T (C^-1 a)> Monte Carlo estimates.

        Parameters
        ----------
        alm : (nelem) or (npol, nelem) complex array
            Healpix-ordered unfiltered alm array. Will be overwritten!
        theta_batch : int, optional
            Process loop over theta in batches of this size. Higher values
            take up more memory.

        Raises
        ------
        ValueError
            If shape input alm is not understood.
        '''

        grad_t = self._step(alm, theta_batch=theta_batch)

        # Add to Monte Carlo estimates.
        if self.mc_gt is None:
            self.mc_gt = grad_t
        else:
            self.__mc_gt += grad_t

        mc_gt_sq = utils.contract_almxblm(grad_t, self.icov(self.beam(np.conj(grad_t))))

        if self.mc_gt_sq is None:
            self.mc_gt_sq = mc_gt_sq
        else:
            self.__mc_gt_sq += mc_gt_sq

        self.mc_idx += 1

    def step_batch(self, alm_loader, alm_files, comm=None, verbose=False, **kwargs):
        '''
        Add iterations to <grad T (C^-1 a) C^-1 grad T(C^-1 a)^*> and 
        <grad T (C^-1 a)> Monte Carlo estimates by loading and processing several 
        alms in parallel using MPI.

        Arguments
        ---------
        alm_loader : callable
            Function that returns alms on rank given filename as first argument.
        alm_files : array_like
            List of alm files to load.
        comm : MPI communicator, optional
        verbose : bool, optional
            Print process.
        kwargs : dict, optional
            Optional keyword arguments passed to "_step".        
        '''

        if comm is None:
            comm = utils.FakeMPIComm()

        # Monte carlo quantities local to rank.
        mc_idx_loc = 0
        mc_gt_sq_loc = None
        mc_gt_loc = None

        # Split alm_file loop over ranks.
        for alm_file in alm_files[comm.Get_rank():len(alm_files):comm.Get_size()]:

            if verbose:
                print('rank {:3}: loading {}'.format(comm.Get_rank(), alm_file))
            alm = alm_loader(alm_file)
            if verbose:
                print('rank {:3}: done loading'.format(comm.Get_rank()))
            grad_t = self._step(alm, **kwargs)

            if mc_gt_loc is None:
                mc_gt_loc = grad_t
            else:
                mc_gt_loc += grad_t

            mc_gt_sq = utils.contract_almxblm(grad_t, self.icov(self.beam(np.conj(grad_t))))

            if mc_gt_sq_loc is None:
                mc_gt_sq_loc = mc_gt_sq
            else:
                mc_gt_sq_loc += mc_gt_sq
        
            mc_idx_loc += 1

        print(f'rank : {comm.rank} waiting in step_batch')
        # To allow allreduce when number of ranks > alm files.
        shape, dtype = utils.bcast_array_meta(mc_gt_loc, comm, root=0)
        if mc_gt_loc is None: mc_gt_loc = np.zeros(shape, dtype=dtype)
        if mc_gt_sq_loc is None: mc_gt_sq_loc = 0.
        if mc_idx_loc is None: mc_idx_loc = 0

        mc_gt = utils.allreduce_array(mc_gt_loc, comm)
        mc_gt_sq = utils.allreduce(mc_gt_sq_loc, comm)        
        mc_idx = utils.allreduce(mc_idx_loc, comm)
        print(f'rank : {comm.rank} after reduce in step_batch')

        # All ranks get to update the internal mc variables themselves.
        if self.mc_gt is None:
            self.mc_gt = mc_gt
        else:
            self.__mc_gt += mc_gt

        if self.mc_gt_sq is None:
            self.mc_gt_sq = mc_gt_sq
        else:
            self.__mc_gt_sq += mc_gt_sq
                
        self.mc_idx += mc_idx

    def compute_estimate_batch(self, alm_loader, alm_files, comm=None, 
                               verbose=False, **kwargs):
        '''
        Compute fNL estimates for a collection of maps in parallel using MPI.

        Arguments
        ---------
        alm_loader : callable
            Function that returns alms on rank given filename as first argument.
        alm_files : array_like
            List of alm files to load.
        comm : MPI communicator, optional
        verbose : bool, optional
            Print process.
        kwargs : dict, optional
            Optional keyword arguments passed to "compute_estimate".        

        Returns
        -------
        estimates : (nalm_files) array, None
            Estimates for each input file in same order as "alm_files".            
        '''

        if comm is None:
            comm = utils.FakeMPIComm()

        estimates = np.zeros(len(alm_files))

        # Split alm_file loop over ranks.
        for aidx in range(comm.Get_rank(), len(alm_files), comm.Get_size()):
        
            alm_file = alm_files[aidx]
            if verbose:
                print('rank {:3}: loading {}'.format(comm.Get_rank(), alm_file))
            alm = alm_loader(alm_file)
            
            estimate = self.compute_estimate(alm, **kwargs)
            if verbose:
                print('rank {:3}: estimate : {}'.format(comm.Get_rank(), estimate))

            estimates[aidx] = estimate
            
        return utils.allreduce_array(estimates, comm)
                    
    def compute_estimate(self, alm, theta_batch=25, fisher=None, lin_term=None):
        '''
        Compute fNL estimate for input alm.

        Parameters
        ----------
        alm : (npol, nelem) array
            Healpix-ordered unfiltered alm array. Will be overwritten!
        theta_batch : int, optional
            Process loop over theta in batches of this size. Higher values
            take up more memory.
        fisher : float, optional
            If given, do not compute fisher from internal mc variables.
        lin_term : float, optional
            If given, do not compute linear term from alm and internal mc
            variables.

        Returns
        -------
        estimate : scalar, None
            fNL estimate.

        Raises
        ------
        ValueError
            If shape input alm is not understood.
            If Monte Carlo quantities are not iterated yet.
        '''

        # Similar to step, but only do backward transform, multiply alm with linear term
        # and apply normalization.

        alm = utils.alm_return_2d(alm, self.npol, self.lmax)
        alm = self.icov(alm)

        t_cubic = 0 # The cubic estimate.
        if fisher is None:
            fisher = self.compute_fisher()
        if lin_term is None:
            lin_term = self.compute_linear_term(alm, no_icov=True)
        
        a_ell_m = utils.alm2a_ell_m(alm)
        a_ell_m = a_ell_m.astype(self.cdtype)

        red_bisp = self.red_bispectra[0]
        f_i_ell, rule, weights = self._init_reduced_bispectrum(red_bisp)

        for tidx_start in range(0, len(self.thetas), theta_batch):
            thetas_batch = self.thetas[tidx_start:tidx_start+theta_batch]
            ct_weights_batch = self.theta_weights[tidx_start:tidx_start+theta_batch]
            y_m_ell = estimator_core.compute_ylm(thetas_batch, self.lmax,
                                                 dtype=self.dtype)            
            t_cubic += estimator_core.compute_estimate(ct_weights_batch, rule, weights,
                                                       f_i_ell, a_ell_m, y_m_ell, self.nphi)

        print(f't_cubic : {t_cubic}, lin : {lin_term}, fisher : {fisher}')
        return (t_cubic - lin_term) / fisher

    def compute_fisher(self):
        '''
        Return Fisher information at current iteration.

        Returns
        -------
        fisher : float, None
            Fisher information.
        '''

        if self.mc_gt_sq is None or self.mc_gt is None:
            return None
        
        fisher = self.mc_gt_sq
        fisher -= utils.contract_almxblm(self.mc_gt, self.icov(self.beam(np.conj(self.mc_gt))))
        fisher /= 3.

        return fisher

    def compute_linear_term(self, alm, no_icov=False):
        '''
        Return linear term at current iteration for input data alm.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            Spherical harmonic coeffients of data.
        no_icov : bool, optional
            Do not icov filter input (i.e. input is already filtered).

        Returns
        -------
        lin_term : float, None
            Linear term of the estimator.

        Notes
        -----
        Linear term is defined as sum(a C^-1 grad T[C^-1 a]). See Eq. 57 in
        Smith & Zaldarriaga.
        '''

        if self.mc_gt is None:
            return None

        alm = utils.alm_return_2d(alm, self.npol, self.lmax)
        
        if no_icov:
            return utils.contract_almxblm(alm, np.conj(self.mc_gt))
        else:
            return utils.contract_almxblm(self.icov(alm), np.conj(self.mc_gt))

    def compute_fisher_isotropic(self, icov_ell, return_matrix=False, fsky=1, 
                                     comm=None):
        '''
        Return Fisher information assuming that inverse noise + signal
        covariance is diagonal in harmonic space.
        
        Arguments
        ---------
        icov_ell : (npol, npol, nell) or (npol, nell) array
            Inverse covariance matrix diagonal in ell. Unlike "icov" this 
            should be: 1 / (S_ell + (b^{-1} N b^{-1})_ell), so no beam in 
            the numerator.
        return_matrix : bool, optonal
            If set, also return nfact x nfact Fisher matrix.       
        fsky : int or (npol,) array.
            Fraction of sky observed, allowed to vary between polarizations.
        comm : MPI communicator, optional        

        Returns
        -------
        fisher : float, None
            Fisher information.
        fisher_nxn : (nfact, nfact) array, None
            nfact x nfact Fisher matrix (only if return_matrix is set).
        '''

        if comm is None:
            comm = utils.FakeMPIComm()

        red_bisp = self.red_bispectra[0]
        f_i_ell, rule, weights = self._init_reduced_bispectrum(red_bisp)
        f_ell_i = np.ascontiguousarray(np.transpose(f_i_ell, (2, 1, 0)))
        del f_i_ell
        f_ell_i *= np.atleast_1d(fsky ** (1/6))[np.newaxis,:,np.newaxis]

        sqrt_icov_ell = mat_utils.matpow(icov_ell, 0.5)
        sqrt_icov_ell = np.ascontiguousarray(np.transpose(sqrt_icov_ell, (2, 0, 1)),
                                             dtype=self.dtype)
        
        nrule = rule.shape[0]
        fisher_nxn = np.zeros((nrule, nrule), dtype=self.dtype)

        thetas_per_rank = np.array_split(
            self.thetas, comm.Get_size())[comm.Get_rank()]
        ct_weights_per_rank = np.array_split(
            self.theta_weights, comm.Get_size())[comm.Get_rank()]

        fisher_core.fisher_nxn(sqrt_icov_ell, f_ell_i, thetas_per_rank,
                               ct_weights_per_rank, rule, weights, fisher_nxn)

        fisher_nxn = utils.allreduce_array(fisher_nxn, comm)
        fisher_nxn = np.triu(fisher_nxn, 1).T + np.triu(fisher_nxn)
        fisher = np.sum(fisher_nxn)

        if return_matrix:
            return fisher, fisher_nxn 

        return fisher
