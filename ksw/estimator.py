import os
import numpy as np
from scipy.special import roots_legendre

from optweight import mat_utils
import h5py

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
        inverse-covariance-weighted version of that array.
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
            raise ValueError(f'{pol=}, but may only contain T and/or E.')
        elif pol.count('T') != 1 and pol.count('E') != 1:
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
            Unique factors of bispectrum.
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
                print(f'rank {comm.rank:3}: loading {alm_file}')
            alm = alm_loader(alm_file)
            
            estimate = self.compute_estimate(alm, **kwargs)
            if verbose:
                print(f'rank {comm.rank:3}: {estimate=}')

            estimates[aidx] = estimate
            
        return utils.allreduce_array(estimates, comm)
                    
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
        estimate : scalar, None
            fNL estimate.

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
        return fnl

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

        Raises
        ------
        ValueError
            If shape input alm is not understood.

        Notes
        -----
        Follwing Eq. 83 in Smith and Zaldarriaga. This function only
        computes the non-Gaussian component: 1 / 3 grad T [Cl^{-1} a].
        '''

        grad_t = self._step(alm, theta_batch=theta_batch)
        grad_t /= 3
        
        return grad_t

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
            raise ValueError(f'{len(alm_files=} != {oalm_files=}')
        
        if comm is None:
            comm = utils.FakeMPIComm()

        idx_on_rank = np.arange(comm.rank, len(alm_files), comm.size)
        for idx in idx_on_rank:

            alm_file = alm_files[idx]
            oalm_file = oalm_files[idx]

            if verbose:
                print(f'rank {comm.rank:3}: loading {alm_file}')
            alm = alm_loader(alm_file)
            if verbose:
                print(f'rank {comm.rank:3}: done loading')
                
            alm_ng = self.compute_ng_sim(alm, **kwargs)

            if verbose:
                print(f'rank {comm.rank:3}: writing {oalm_file}')
                
            alm_writer(oalm_file, alm_ng)

            if verbose:
                print(f'rank {comm.rank:3}: done writing')
            
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
            mc_idx_read = int(mc_idx_read)
                        
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
            mc_gt_sq_read = float(mc_gt_sq_read)

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
