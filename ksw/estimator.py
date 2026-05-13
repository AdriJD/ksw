import os
import numpy as np
from scipy.special import roots_legendre

from optweight import mat_utils
import h5py
from ducc0.misc import wigner3j_int

from ksw import utils, legendre, estimator_core, fisher_core
import ksw.radial_functional as rf
from ksw import Afunctionals as AF



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
        inverse-covariance-weighted version of that array.
    lmax : int
        Max multipole used in estimator. Should match shape of alms.
    pol : str or array-like of strings.
        Data polarization, e.g. "B", or ["T", "E", "B"]. Should match shape of alms.
    precision : str, optional
        Use either "single" precision or "double" precision data types 
        for internal calculations.

    Attributes
    ----------
    red_bispectra : list of ksw.ReducedBispectrum instances
        Reduced bispectra templates.
    icov : callable, None
        The inverse covariance weighting operation.
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

    Notes
    -----
    The inverse-covariance operation should correspond to:

    x^icov = S^{-1} (S^{-1} + P^H N^{-1} P)^{-1} P^H N^{-1} P s,

    where data = P s + n, where s are the spherical harmonic coefficients
    of the signal. P = M Y B, where B is the beam, Y is spherical harmonic
    synthesis (alm2map) and M is the pixel mask and any custom filters. 
    N^{-1} and S^{-1} are the inverse noise and signal covariance matrices,
    respectively. ^H denotes the Hermitian transpose.
    '''

    def __init__(self, red_bispectra, icov, lmax, pol, precision='single'):

        self.red_bispectra = red_bispectra
        self.icov = icov
        self.mc_idx = 0
        self.mc_gt = None
        self.mc_gt_sq = None

        self.gt_local = []
        
        self.lmax = lmax
        self.pol = pol

        if precision == 'single':
            self.dtype = np.float32
            self.cdtype = np.complex64
        elif precision == 'double':
            self.dtype = np.float64
            self.cdtype = np.complex128
        else:
            raise ValueError(f'{precision=} is not supported')


       # if len(red_bispectra) > 1:
        #    raise NotImplementedError('no joint estimation for now.')

        self.thetas, self.theta_weights, self.nphi = self.get_coords()

    @property
    def pol(self):
        return self.__pol
    
    @pol.setter
    def pol(self, pol):
        '''Check input and make sorted tuple.'''
        pol = list(np.atleast_1d(pol))
        sort_order = {"T": 0, "E": 1, "B": 2}

        if pol.count('T') + pol.count('E') + pol.count('B') != len(pol):
            raise ValueError(f'{pol=}, but may only contain T, E, and/or B.')
        elif len(set(pol)) != len(pol):
            raise ValueError(f'{pol=}, cannot contain duplicates.')

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

    def _init_reduced_bispectrum(self, red_bisp, keep_deltaL=False):
        '''
        Prepare reduced bispectrum for estimation.

        Parameters
        ----------
        red_bisp : ksw.ReducedBispectrum instance
            Assumed to have T, E, and/or B in that order.
        
        Returns
        -------
        f_i_ell : array
            Unique factors of bispectrum. Shape is (nufact, npol, nell)
            for standard templates and (nufact, npol, ndeltaL, nell)
            when keep_deltaL=True and the template stores a deltaL axis.
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

        factors = red_bisp.factors
        nufact = factors.shape[0]
        if factors.ndim == 3:
            f_i_ell = np.zeros((nufact, self.npol, self.lmax + 1),
                               dtype=self.dtype)
        elif factors.ndim == 4 and keep_deltaL:
            ndeltaL = factors.shape[2]
            f_i_ell = np.zeros((nufact, self.npol, ndeltaL, self.lmax + 1),
                               dtype=self.dtype)
        elif factors.ndim == 4:
            raise ValueError('4D factors require keep_deltaL=True.')
        else:
            raise ValueError('Unsupported factors ndim: {}'.format(
                             factors.ndim))

        # Find index of lmax data in ells of red. bisp.
        try:
            end_ells_full = np.where(red_bisp.ells_full == self.lmax)[0][0] + 1
        except IndexError:
            end_ells_full = None

        # Slice corresponding to data pol. Assume red. bisp. channels are T, E, B.
        pol_to_idx = {'T': 0, 'E': 1, 'B': 2}
        pslice = [pol_to_idx[p] for p in self.pol]
        if max(pslice) >= factors.shape[1]:
            raise ValueError(
                f'{self.pol=} requires at least {max(pslice) + 1} polarization channels, '
                f'but reduced bispectrum only has shape {factors.shape[1]} along that axis.')

        if factors.ndim == 3:
            f_i_ell[:,:,red_bisp.lmin:red_bisp.lmax+1] = \
                factors[:,pslice,:end_ells_full]
        else:
            f_i_ell[:,:,:,red_bisp.lmin:red_bisp.lmax+1] = \
                factors[:,pslice,:,:end_ells_full]
        f_i_ell = f_i_ell.astype(self.dtype, copy=False)
        
        rule = red_bisp.rule
        weights = red_bisp.weights.astype(self.dtype)

        return f_i_ell, rule, weights

    def _step(self, alm, theta_batch=25):
        '''
        Calculate grad T (C^-1 a).

        Parameters
        ----------
        alm : (nelem) or (npol, nelem) complex array
            HEALPix-ordered inverse-covariance filtered data.
        theta_batch : int, optional
            Process loops over theta in batches of this size. Higher values
            take up more memory.
        
        Returns
        -------
        grad_t : (nelem) or (npol, nelem) complex array
            HEALPix-ordered alm array.
        '''

        alm = utils.alm_return_2d(alm, self.npol, self.lmax)
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

        # Turn back into HEALPix shape.
        grad_t = utils.a_ell_m2alm(grad_t).astype(self.cdtype)

        return grad_t

    def step(self, alm, theta_batch=25):
        '''
        Add iteration to <grad T (C^-1 a) C^-1 grad T(C^-1 a)^*> and 
        <grad T (C^-1 a)> Monte Carlo estimates.

        Parameters
        ----------
        alm : (nelem) or (npol, nelem) complex array
            HEALPix-ordered inverse-covariance filtered data.
        theta_batch : int, optional
            Process loops over theta in batches of this size. Higher values
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

        mc_gt_sq = utils.contract_almxblm(grad_t, np.conj(self.icov(grad_t)))

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

        Parameters
        ----------
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

        # Monte Carlo quantities local to rank.
        mc_idx_loc = 0
        mc_gt_sq_loc = None
        mc_gt_loc = None

        # Split alm_file loop over ranks.
        for alm_file in alm_files[comm.Get_rank():len(alm_files):comm.Get_size()]:

            if verbose:
                print(f'rank {comm.rank:3}: loading {alm_file}')
            alm = alm_loader(alm_file)
            if verbose:
                print(f'rank {comm.rank:3}: done loading')
            grad_t = self._step(alm, **kwargs)

            if mc_gt_loc is None:
                mc_gt_loc = grad_t
            else:
                mc_gt_loc += grad_t

            mc_gt_sq = utils.contract_almxblm(grad_t, np.conj(self.icov(grad_t)))

            if mc_gt_sq_loc is None:
                mc_gt_sq_loc = mc_gt_sq
            else:
                mc_gt_sq_loc += mc_gt_sq
        
            mc_idx_loc += 1

        print(f'rank : {comm.rank:3} waiting in step_batch')
        # To allow allreduce when number of ranks > alm files.
        shape, dtype = utils.bcast_array_meta(mc_gt_loc, comm, root=0)
        if mc_gt_loc is None: mc_gt_loc = np.zeros(shape, dtype=dtype)
        if mc_gt_sq_loc is None: mc_gt_sq_loc = 0.
        if mc_idx_loc is None: mc_idx_loc = 0

        mc_gt = utils.allreduce_array(mc_gt_loc, comm)
        mc_gt_sq = utils.allreduce(mc_gt_sq_loc, comm)        
        mc_idx = utils.allreduce(mc_idx_loc, comm)
        print(f'rank : {comm.rank:3} after reduce in step_batch')

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

    def step_batch_2pass(self, alm_loader, alm_files, comm=None, verbose=False, **kwargs):
        '''
        Compute grad T its mean for a set of simulations, by loading and processing
        several alms in parallel using MPI.

        Parameters
        ----------
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

        # Monte Carlo quantities local to rank.
        mc_idx_loc = 0
        mc_gt_loc = None

        # Split alm_file loop over ranks.
        for alm_file in alm_files[comm.Get_rank():len(alm_files):comm.Get_size()]:

            if verbose:
                print(f'rank {comm.rank:3}: loading {alm_file}')
            alm = alm_loader(alm_file)
            if verbose:
                print(f'rank {comm.rank:3}: done loading')
            grad_t = self._step(alm, **kwargs)

            if mc_gt_loc is None:
               mc_gt_loc = grad_t
            else:
               mc_gt_loc += grad_t

            # # NOTE
            # if mc_gt_loc is None:
            #    mc_gt_loc = grad_t.copy()
            # else:
            #    mc_gt_loc += grad_t.copy()

            # The copy here is important, otherwise each iteration of the
            # loop adds grad to the first element of the list.
            self.gt_local.append(grad_t.copy())
            
            mc_idx_loc += 1

        print(f'rank : {comm.rank:3} waiting in step_batch')
        # To allow allreduce when number of ranks > alm files.
        shape, dtype = utils.bcast_array_meta(mc_gt_loc, comm, root=0)
        if mc_gt_loc is None: mc_gt_loc = np.zeros(shape, dtype=dtype)
        if mc_idx_loc is None: mc_idx_loc = 0

        mc_gt = utils.allreduce_array(mc_gt_loc, comm)
        mc_idx = utils.allreduce(mc_idx_loc, comm)
        print(f'rank : {comm.rank:3} after reduce in step_batch')

        # All ranks get to update the internal mc variable themselves.
        if self.mc_gt is None:
            self.mc_gt = mc_gt
        else:
            self.__mc_gt += mc_gt

        self.mc_idx += mc_idx

    def compute_fisher_2pass(self, comm):
        '''
        Compute the Fisher information from the grad T maps computed
        on all ranks.

        Returns
        -------
        fisher : float, None
            Fisher information.
        '''

        if self.mc_gt is None:
            return None

        fisher_local = 0.
        for gidx, gt in enumerate(self.gt_local):

            diff = gt - self.mc_gt
            diff_icov = self.icov(diff)
            dot = utils.contract_almxblm(diff, np.conj(diff_icov))
            print(f'{comm.rank=}, {gidx=}, fisher estimate={dot / 3}')
            fisher_local += dot

        fisher = utils.allreduce(fisher_local, comm)
        fisher /= (3 * self.mc_idx)

        return fisher
        
    def compute_estimate_batch(self, alm_loader, alm_files, comm=None, 
                               verbose=False, **kwargs):
        '''
        Compute fNL estimates for a collection of maps in parallel using MPI.

        Parameters
        ----------
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
        estimates : (nalm_files) array
            Estimates for each input file in same order as "alm_files".
        cubic_terms : (nalm_files) array
            Cubic term for each input file in same order as "alm_files".
        lin_terms : (nalm_files) array
            Linear terms for each input file in same order as "alm_files".
        fishers : (nalm_files) array
            Fisher information for each input file in same order as "alm_files".
        '''

        if comm is None:
            comm = utils.FakeMPIComm()

        estimates = np.zeros(len(alm_files))
        cubic_terms = np.zeros(len(alm_files))
        lin_terms = np.zeros(len(alm_files))
        fishers = np.zeros(len(alm_files))        

        # Split alm_file loop over ranks.
        for aidx in range(comm.Get_rank(), len(alm_files), comm.Get_size()):
        
            alm_file = alm_files[aidx]
            if verbose:
                print(f'rank {comm.rank:3}: loading {alm_file}')
            alm = alm_loader(alm_file)
            
            estimate, cubic, lin_term, fisher = self.compute_estimate(alm, **kwargs)
            if verbose:
                print(f'rank {comm.rank:3}: {estimate=}')

            estimates[aidx] = estimate
            cubic_terms[aidx] = cubic
            lin_terms[aidx] = lin_term
            fishers[aidx] = fisher          
            
        estimates = utils.allreduce_array(estimates, comm)
        cubic_terms = utils.allreduce_array(cubic_terms, comm)
        lin_terms = utils.allreduce_array(lin_terms, comm)
        fishers = utils.allreduce_array(fishers, comm)

        return estimates, cubic_terms, lin_terms, fishers
                    
    def compute_estimate(self, alm, theta_batch=25, fisher=None, lin_term=None):
        '''
        Compute fNL estimate for input alm.

        Parameters
        ----------
        alm : (npol, nelem) array
            HEALPix-ordered inverse-covariance filtered data.        
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
        estimate : float
            fNL estimate.
        cubic : float
            Cubic term.
        lin_term : float
            Linear term.
        fisher : float
            Fisher information.
        
        Raises
        ------
        ValueError
            If shape input alm is not understood.
            If Monte Carlo quantities are not iterated yet.

        Notes
        -----
        Similar to step, but only do backward transform, multiply alm
        with linear term and apply normalization.        
        '''

        alm = utils.alm_return_2d(alm, self.npol, self.lmax)

        t_cubic = 0 # The cubic estimate.
        if fisher is None:
            fisher = self.compute_fisher()
        if lin_term is None:
            lin_term = self.compute_linear_term(alm)
        
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

        fnl = (t_cubic - lin_term) / fisher
        print(f'{fnl=}, {t_cubic=}, {lin_term=}, {fisher=}')
        return fnl, t_cubic, lin_term, fisher

    def compute_estimate_sst(self, alm, L_list, Lmax, theta_batch=25, fisher=None, lin_term=None):
        '''
        Compute fNL estimate for input alm for sst spectra.

        Parameters
        ----------
        alm : (npol, nelem) array
            HEALPix-ordered inverse-covariance filtered data.
        L_list : array
            List of L values.
        Lmax : int
            Maximum L value.
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
        estimate : float
            fNL estimate.
        cubic : float
            Cubic term.
        lin_term : float
            Linear term.
        fisher : float
            Fisher information.

        Raises
        ------
        ValueError
            If shape input alm is not understood.
            If Monte Carlo quantities are not iterated yet.
        Notes
        -----
        Similar to compute_estimate, but for sst spectra and apply the functions written in C for this. 
        '''
        alm = utils.alm_return_2d(alm, self.npol, self.lmax)

        t_cubic = 0 # The cubic estimate.
        Afunc_product = 0 # The product of A functionals that goes into the cubic term.

        if fisher is None:
            fisher = 1
        if lin_term is None:
            lin_term = 0

        a_ell_m = utils.alm2a_ell_m(alm)
        a_ell_m = a_ell_m.astype(self.cdtype)

        
        red_bisp_scalar = self.red_bispectra[0]
        kappa_i_L_scalar1, rule, weights = self._init_reduced_bispectrum(
            red_bisp_scalar, keep_deltaL=True)
        print(kappa_i_L_scalar1.shape)
        kappa_i_L_scalar2 = kappa_i_L_scalar1.copy()
        print(kappa_i_L_scalar1[0, 0, 0, 0])

        red_bisp_tensor = self.red_bispectra[-1]
        kappa_i_L_tensor, _, _ = self._init_reduced_bispectrum(
            red_bisp_tensor, keep_deltaL=True)
        #print(kappa_i_L_tensor.shape)
        
        # (nscalar1, nscalar2, ntensor): {(1, 1, -2); (1, 0, -1); (0, 1, -1);
        #                                 (1, -1, 0); (0, 0, 0)}. The other 4 combinations are not included yet.

        combins = [(1,1,-2), (1,0,-1), (0,1,-1), (1,-1,0), (0,0,0)]
        L_list = np.asarray(L_list, dtype=np.int64)
        nL = L_list.size
        nell = self.lmax + 1
        m_dim = self.nphi
        nufact = kappa_i_L_scalar1.shape[0]
        ndeltaL_scalar = 2
        ndeltaL_tensor = 5

        deltaL_list_scalar = np.array([-1, 1])
        deltaL_list_tensor = np.array([-2, -1, 0, 1, 2])

        for com_idx in range(len(combins)):
            n_scalar1, n_scalar2, n_tensor = combins[com_idx]
            w3j_product_scalar1 = AF.products_3j_array(S=1, n=n_scalar1, L_list=L_list, deltaL_list=deltaL_list_scalar, Jindex=(0, 0, 0))
            w3j_product_scalar2 = AF.products_3j_array(S=1, n=n_scalar2, L_list=L_list, deltaL_list=deltaL_list_scalar, Jindex=(0, 0, 0))
            w3j_product_tensor = AF.products_3j_array(S=2, n=n_tensor, L_list=L_list, deltaL_list=deltaL_list_tensor, Jindex=(-2, 0, 2))

            prefactors_scalar1 = AF.prefactor_product(deltaL_list_scalar, L_list, self.pol, "zeta", cdtype=self.cdtype)
            prefactors_scalar2 = AF.prefactor_product(deltaL_list_scalar, L_list, self.pol, "zeta", cdtype=self.cdtype)
            prefactors_tensor = AF.prefactor_product(deltaL_list_tensor, L_list, self.pol, "h", cdtype=self.cdtype)

            n_L_phi_scalar = np.zeros((self.npol, ndeltaL_scalar, nL, self.nphi), dtype=self.cdtype)
            n_L_phi_tensor = np.zeros((self.npol, ndeltaL_tensor, nL, self.nphi), dtype=self.cdtype)
            
            #kappa_i_L_scalar1 = rf.radial_func_dL()
            #kappa_i_L_scalar2 = np.zeros(())
            #kappa_i_L_tensor = np.zeros(())

            for tidx_start in range(0, len(self.thetas), theta_batch):
                thetas_batch = self.thetas[tidx_start:tidx_start+theta_batch]
                ct_weights_batch = self.theta_weights[tidx_start:tidx_start+theta_batch].astype(self.dtype, copy=False)
                y_M_L = estimator_core.compute_ylm(thetas_batch, nL - 1, dtype=self.dtype)

                A_L_M_scalar1 = estimator_core.compute_A_LM(L_list, deltaL_list_scalar, n_scalar1, a_ell_m, y_M_L, w3j_product_scalar1, prefactors_scalar1, Lmax)
                A_L_M_scalar2 = estimator_core.compute_A_LM(L_list, deltaL_list_scalar, n_scalar2, a_ell_m, y_M_L, w3j_product_scalar2, prefactors_scalar2, Lmax)
                A_L_M_tensor = estimator_core.compute_A_LM(L_list, deltaL_list_tensor, n_tensor, a_ell_m, y_M_L, w3j_product_tensor, prefactors_tensor, Lmax)

                Afunc_product += estimator_core.compute_products_Afunc_sst(ct_weights_batch, rule, weights,
                                                                    a_ell_m, y_M_L,
                                                                    self.nphi,
                                                                    L_list,
                                                                    n_scalar1, n_scalar2, n_tensor,
                                                                    w3j_product_scalar1, w3j_product_scalar2, w3j_product_tensor,
                                                                    prefactors_scalar1, prefactors_scalar2, prefactors_tensor,
                                                                    A_L_M_scalar1, A_L_M_scalar2, A_L_M_tensor,
                                                                    self.lmax, nell,
                                                                    n_L_phi_scalar, n_L_phi_tensor,
                                                                    kappa_i_L_scalar1, kappa_i_L_scalar2, kappa_i_L_tensor)
            l1_min, vals = wigner3j_int(1, 2, n_scalar2, n_tensor)# l1_min=1, vals is all w3j values in the order of increasing l1
            t_cubic += vals[0] * Afunc_product

        fnl = (t_cubic - lin_term) / fisher
        print(f'{fnl=}, {t_cubic=}, {lin_term=}, {fisher=}')
        return fnl, t_cubic, lin_term, fisher


    def compute_fisher(self, return_icov_mc_gt=False):
        '''
        Return Fisher information at current iteration.

        Returns
        -------
        fisher : float, None
            Fisher information.
        '''

        if self.mc_gt_sq is None or self.mc_gt is None:
            return None
        
        icov_mc_gt = self.icov(self.mc_gt)

        mc_gt_icov_mc_gt = utils.contract_almxblm(self.mc_gt, np.conj(icov_mc_gt))
        fisher = (self.mc_gt_sq - mc_gt_icov_mc_gt) / 3.

        print(f'{fisher=}, {self.mc_gt_sq=}, {mc_gt_icov_mc_gt=}')

        if return_icov_mc_gt:
            return fisher, icov_mc_gt
        else:
            return fisher

    def compute_linear_term(self, alm):
        '''
        Return linear term at current iteration for input data alm.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            HEALPix-ordered inverse-covariance filtered data.
        
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
            
        return utils.contract_almxblm(alm, np.conj(self.mc_gt))

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
        #with np.printoptions(threshold=np.inf):
        #    print(fisher_nxn)
        print(f'{fisher=}')
        
        if return_matrix:
            return fisher, fisher_nxn 

        return fisher

    def compute_ng_sim(self, alm, theta_batch=25):
        '''
        Compute non-Gaussian perturbation to input simulation. 

        Parameters
        ----------
        alm : (nelem) or (npol, nelem) complex array
            HEALPix-ordered inverse-signal-covariance filtered alms of
            input Gaussian field.
        theta_batch : int, optional
            Process loops over theta in batches of this size. Higher values
            take up more memory.

        Returns
        -------
        alm_ng : (nelem) or (npol, nelem) complex array
            Non-Gaussian perturbation.
        
        Raises
        ------
        ValueError
            If shape input alm is not understood.

        Notes
        -----
        Follwing Eq. 83 in Smith and Zaldarriaga. This function only
        computes the non-Gaussian component: 1 / 3 grad T [Cl^{-1} a].
        '''

        alm_ng = self._step(alm, theta_batch=theta_batch)
        alm_ng /= 3
        
        return alm_ng

    def compute_ng_sim_batch(self, alm_loader, alm_files, alm_writer,
                             oalm_files, comm=None, verbose=False, **kwargs):
        '''
        Compute non-Gaussian perturbation to input simulation by loading
        and processing several alms in parallel using MPI.

        Parameters
        ----------
        alm_loader : callable
            Function that returns alms on rank given filename as first argument.
        alm_files : array_like
            List of alm files to load.
        alm_writer : callable
            Function that writes alms to disk given filename as first argument
            and alm array as second.
        oalm_files : array_like
            List of output filepaths for eatch input alm file.
        comm : MPI communicator, optional
        verbose : bool, optional
            Print process.
        kwargs : dict, optional
            Optional keyword arguments passed to "compute_ng_sim".

        Raises
        ------
        ValueError
            If alm_files and oalm_files do not match.
        '''

        if len(alm_files) != len(oalm_files):
            raise ValueError(f'{len(alm_files)=} != {oalm_files=}')
        
        if comm is None:
            comm = utils.FakeMPIComm()

        idx_on_rank = np.array_split(
            np.arange(len(alm_files)), comm.size)[comm.rank]
        for idx in idx_on_rank:

            alm_file = alm_files[idx]
            oalm_file = oalm_files[idx]

            if verbose:
                print(f'rank {comm.rank:3}: loading {alm_file}')
            alm = alm_loader(alm_file)
            if verbose:
                print(f'rank {comm.rank:3}: done loading {alm_file}')
                
            alm_ng = self.compute_ng_sim(alm, **kwargs)

            if verbose:
                print(f'rank {comm.rank:3}: writing {oalm_file}')
                
            alm_writer(oalm_file, alm_ng)

            if verbose:
                print(f'rank {comm.rank:3}: done writing {oalm_file}')

    def write_state_2pass(self, filename, fisher, comm=None):
        '''
        Write internal state, i.e. mc_gt, fisher and mc_idx, for the 2pass-mode
        of the code.

        Parameters
        ----------
        filename : str
            Absolute path to output file.
        fisher : float
            Output from compute_fisher_2pass.
        comm : MPI communicator, optional
            If provided, rank 0 is assumed to do the writing, so must be present.
        '''

        if comm is None:
            comm = utils.FakeMPIComm()

        if comm.Get_rank() == 0:
            # Remove file extension to be consistent.
            filename, _ = os.path.splitext(filename)

            mc_idx_to_save = np.asarray([self.mc_idx], dtype=np.int64)

            fisher_to_save = np.asarray([fisher], dtype=np.float64)            

            if self.__mc_gt is None:
                mc_gt_to_save = np.asarray([np.nan], dtype=self.cdtype)
            else:
                mc_gt_to_save = self.__mc_gt

            with h5py.File(filename + '.hdf5', 'w') as f:
                f.create_dataset('mc_idx', data=mc_idx_to_save)
                f.create_dataset('fisher', data=fisher_to_save)
                f.create_dataset('mc_gt', data=mc_gt_to_save)

    def _read_state_2pass(self, filename, comm=None):
        '''
        Read internal state, i.e. mc_gt, fisher and mc_idx, from hdf5 file.

        Parameters
        ----------
        filename : str
            Absolute path to output file.
        comm : MPI communicator, optional
            If provided, rank 0 is assumed to do the reading, result will be 
            broadcasted to all ranks.

        Returns
        -------
        mc_idx : int
            Counter for Monte Carlo estimates.
        fisher : float, None
            Fisher information.
        mc_gt : (npol, nelem) complex array, None
            <grad T (C^-1 a)> Monte Carlo estimate.
        '''

        if comm is None:
            comm = utils.FakeMPIComm()

        if comm.Get_rank() == 0:
            # Remove file extension to be consistent.
            filename, _ = os.path.splitext(filename)

            with h5py.File(filename + '.hdf5', 'r') as f:
                mc_idx_read = f['mc_idx'][()]
                fisher_read = f['fisher'][()]
                mc_gt_read = f['mc_gt'][()]

            assert mc_idx_read.size == 1, (f'mc_idx has to be single int, got '
                                      f'{mc_idx.size}-sized array')
            mc_idx_read = int(mc_idx_read[0])
                        
        else:
            mc_idx_read = None
            fisher_read = None
            mc_gt_read = None

        mc_idx_read = utils.bcast(mc_idx_read, comm, root=0)
        fisher_read = utils.bcast_array(fisher_read, comm, root=0)
        mc_gt_read = utils.bcast_array(mc_gt_read, comm, root=0)

        if fisher_read.size == 1 and np.isnan(fisher_read)[0]:
            fisher_read = None
        else:
            fisher_read = float(fisher_read[0])

        if mc_gt_read.size == 1 and np.isnan(mc_gt_read)[0]:
            mc_gt_read = None
        
        return mc_idx_read, fisher_read, mc_gt_read
    
    def start_from_read_state_2pass(self, filename, comm=None):
        '''
        Return Fisher information and update estimator state with
        mc_gt and mc_idx read from .hdf5 file.

        Parameters
        ----------
        filename : str
            Absolute path to output file.
        comm : MPI communicator, optional
            If provided, rank 0 is assumed to do the reading, result will be 
            broadcasted to all ranks.

        Returns
        -------
        fisher : float
            Fisher information.
        '''
        
        mc_idx_read, fisher_read, mc_gt_read = self._read_state_2pass(
            filename, comm=comm)
        
        self.mc_idx = mc_idx_read
        self.__mc_gt = mc_gt_read

        return fisher_read
                
    def write_state(self, filename, comm=None):
        '''
        Write internal state, i.e. mc_gt, mc_gt_sq and mc_idx, to hdf5 file.

        Parameters
        ----------
        filename : str
            Absolute path to output file.
        comm : MPI communicator, optional
            If provided, rank 0 is assumed to do the writing, so must be present.
        '''

        if comm is None:
            comm = utils.FakeMPIComm()

        if comm.Get_rank() == 0:
            # Remove file extension to be consistent.
            filename, _ = os.path.splitext(filename)

            mc_idx_to_save = np.asarray([self.mc_idx], dtype=np.int64)

            if self.__mc_gt_sq is None:
                mc_gt_sq_to_save = np.asarray([np.nan], dtype=np.float64)
            else:
                mc_gt_sq_to_save = np.asarray([self.__mc_gt_sq], dtype=np.float64)            

            if self.__mc_gt is None:
                mc_gt_to_save = np.asarray([np.nan], dtype=self.cdtype)
            else:
                mc_gt_to_save = self.__mc_gt

            with h5py.File(filename + '.hdf5', 'w') as f:
                f.create_dataset('mc_idx', data=mc_idx_to_save)
                f.create_dataset('mc_gt_sq', data=mc_gt_sq_to_save)
                f.create_dataset('mc_gt', data=mc_gt_to_save)
        
    def _read_state(self, filename, comm=None):
        '''
        Read internal state, i.e. mc_gt, mc_gt_sq and mc_idx, from hdf5 file.

        Parameters
        ----------
        filename : str
            Absolute path to output file.
        comm : MPI communicator, optional
            If provided, rank 0 is assumed to do the reading, result will be 
            broadcasted to all ranks.

        Returns
        -------
        mc_idx : int
            Counter for Monte Carlo estimates.
        mc_gt_sq : float, None
            <grad T (C^-1 a) C^-1 grad T(C^-1 a)^*>
        mc_gt : (npol, nelem) complex array, None
            <grad T (C^-1 a)> Monte Carlo estimate.
        '''

        if comm is None:
            comm = utils.FakeMPIComm()

        if comm.Get_rank() == 0:
            # Remove file extension to be consistent.
            filename, _ = os.path.splitext(filename)

            with h5py.File(filename + '.hdf5', 'r') as f:
                mc_idx_read = f['mc_idx'][()]
                mc_gt_sq_read = f['mc_gt_sq'][()]
                mc_gt_read = f['mc_gt'][()]

            assert mc_idx_read.size == 1, (f'mc_idx has to be single int, got '
                                      f'{mc_idx.size}-sized array')
            mc_idx_read = int(mc_idx_read[0])
                        
        else:
            mc_idx_read = None
            mc_gt_sq_read = None
            mc_gt_read = None

        mc_idx_read = utils.bcast(mc_idx_read, comm, root=0)
        mc_gt_sq_read = utils.bcast_array(mc_gt_sq_read, comm, root=0)
        mc_gt_read = utils.bcast_array(mc_gt_read, comm, root=0)

        if mc_gt_sq_read.size == 1 and np.isnan(mc_gt_sq_read)[0]:
            mc_gt_sq_read = None
        else:
            mc_gt_sq_read = float(mc_gt_sq_read[0])

        if mc_gt_read.size == 1 and np.isnan(mc_gt_read)[0]:
            mc_gt_read = None
        
        return mc_idx_read, mc_gt_sq_read, mc_gt_read
        
    def start_from_read_state(self, filename, comm=None):
        '''
        Update estimator state with mc_gt, mc_gt_sq and mc_idx read from .hdf5 file.

        Parameters
        ----------
        filename : str
            Absolute path to output file.
        comm : MPI communicator, optional
            If provided, rank 0 is assumed to do the reading, result will be 
            broadcasted to all ranks.
        '''
        
        mc_idx_read, mc_gt_sq_read, mc_gt_read = self._read_state(
            filename, comm=comm)
        
        self.mc_idx = mc_idx_read
        self.__mc_gt_sq = mc_gt_sq_read
        self.__mc_gt = mc_gt_read
