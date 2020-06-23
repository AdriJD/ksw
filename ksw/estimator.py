import numpy as np
from scipy.special import roots_legendre

import healpy as hp
import pyfftw

from ksw import utils, legendre

class KSW():
    '''
    Implementation of the Komatsu Spergel Wandelt estimator using the 
    factorization by Smith and Zaldarriaga 2011.
    
    Parameters
    ----------
    data : ksw.Data instance
        Data instance containing a cosmology instance, beams and covariance.
    icov : callable, None
        Function takes (npol, nelem) alm-like complex array
        and returns the inverse-variance-weighted version of
        that array. Defaults to diagonal weighting.

    Attributes
    ----------
    data : ksw.Data instance
        Data instance.
    icov : callable, None
        Function takes (npol, nelem) alm-like complex array
        and returns the inverse-variance-weighted version of
        that array. 
    mc_idx : int
        Counter for Monte Carlo estimates.
    mc_gt : (npol, nelem) complex array, None
        Current <grad T (C^-1 a)> Monte Carlo estimate (eq 60 Smith Zaldarriaga).
    mc_gt_sq : float, None
        Current <grad T (C^-1 a) C^-1 grad T(C^-1 a)> Monte Carlo
        estimate (eq 61 Smith Zaldarriaga).
    thetas : (ntheta) array
        Coordinates of isolatitude rings.
    theta_weights (ntheta) array
        Weight for each isolatitude ring.
    nphi : int
        Number of phi points.
    m_ell_m : (npol, nell, nm) complex array            
        M_lm in Eq. 73 in Smith Zaldarriaga 2011.
    n_ell_phi : (npol, nell, nphi) array
        N_lphi in Eq. 73 in Smith Zaldarriaga 2011.
    fft_backward : pyfftw.pyfftw.FFTW instance
        The FFT from m_ell_m to n_ell_phi.
    fft_forward : pyfftw.pyfftw.FFTW instance
        The FFT from n_ell_phi to m_ell_m.
    '''
    
    def __init__(self, data, icov=None):

        self.data = data
        self.cosmology = data.cosmology
        if icov is None:
            icov = data.icov_diag_nonlensed
        self.icov = icov
        self.mc_idx = 0    
        self.mc_gt = None
        self.mc_gt_sq = None

        if len(self.cosmology.red_bispectra) > 1:
            raise NotImplementedError('no joint estimation for now.')

        self.thetas, self.theta_weights, self.nphi = self.get_coords()
        self.m_ell_m, self.n_ell_phi, self.fft_backward, self.fft_forward = \
            self.init_fft()

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
        
        lmax = self.data.lmax
        
        cos_thetas, ct_weights = roots_legendre(int(np.floor(1.5 * lmax) + 1))
        thetas = np.arccos(cos_thetas)
        
        nphi_min = 3 * lmax + 1 
        nphi = utils.compute_fftlen_fftw(nphi_min, even=True)
        
        return thetas, ct_weights, nphi
        
    def init_fft(self):
        '''
        Initialize m <-> phi FFTs.

        Returns
        -------
        m_ell_m : (npol, nell, nm) complex array            
            M_lm in Eq. 73 in Smith Zaldarriaga 2011.
        n_ell_phi : (npol, nell, nphi) array
            N_lphi in Eq. 73 in Smith Zaldarriaga 2011.
        fft_backward : pyfftw.pyfftw.FFTW instance
            The FFT from m_ell_m to n_ell_phi.
        fft_forward : pyfftw.pyfftw.FFTW instance
            The FFT from n_ell_phi to m_ell_m.

        Notes
        -----
        Forward: real -> complex.
        Backward: complex -> real.

        If phi runs over [0, 2 pi) and m runs over [0, lmax]:
        FFTW backward adds factor 1/N (= 1/Y.size) compared to Y_phi = sum_m X_m e^i m phi.
        FFTW forward matches X_m = sum_phi Y_phi e^-i m phi.
        '''

        n_ell_phi = pyfftw.empty_aligned((self.data.npol, self.data.lmax + 1, self.nphi),
                                       dtype=np.float64) 
        m_ell_m = pyfftw.empty_aligned(n_ell_phi.shape[:2] + (n_ell_phi.shape[-1] // 2 + 1,),
                                      dtype=np.complex128)
        n_ell_phi[:] = 0 
        m_ell_m[:] = 0 
        
        fft_forward = pyfftw.FFTW(n_ell_phi, m_ell_m, direction='FFTW_FORWARD',
                                   flags=('FFTW_MEASURE',))         
        fft_backward = pyfftw.FFTW(m_ell_m, n_ell_phi, direction='FFTW_BACKWARD',
                                   flags=('FFTW_MEASURE',)) 
                
        return m_ell_m, n_ell_phi, fft_backward, fft_forward

    def _init_reduced_bispectrum(self, red_bisp):
        '''
        Combine reduced bispectrum factors, rule and weights with
        the data beam into X_{i ell}, Y_{i ell}, Z_{i ell} arrays.  

        Parameters
        ----------
        red_bisp : ksw.ReducedBispectrum instance

        Returns
        -------
        x_i_ell : (nfact, npol, nell) array
            Reduced bispectrum factors for ell_1.
        y_i_ell : (nfact, npol, nell) array
            Reduced bispectrum factors for ell_2.
        z_i_ell : (nfact, npol, nell) array
            Reduced bispectrum factors for ell_3.

        Raises
        ------
        ValueError
            If lmax of reduced bispectrum < lmax of data.
        '''
        
        # We use lmax data as reference.
        if red_bisp.lmax < self.data.lmax:
            raise ValueError('lmax bispectrum ({}) < lmax data ({})'.format(
                red_bisp.lmax, self.data.lmax))

        x_i_ell = np.zeros((red_bisp.nfact, self.data.npol, self.data.lmax + 1))
        y_i_ell = np.zeros_like(x_i_ell)
        z_i_ell = np.zeros_like(x_i_ell)

        # Find index of lmax data in ells of red. bisp.
        try:
            end_ells_full = np.where(red_bisp.ells_full == self.data.lmax)[0][0] + 1
        except IndexError:
            end_ells_full = None

        # Slice corresponding to data pol. Assume red. bisp. has T and E.
        if self.data.npol == 1 and 'T' in self.data.pol:
            pslice = slice(0, 1, None)
        elif self.data.npol == 1 and 'E' in self.data.pol:            
            pslice = slice(1, 2, None)
        else:
            pslice = slice(0, 2, None)
            
        x_i_ell[:,:,red_bisp.lmin:red_bisp.lmax+1] = \
            red_bisp.factors[red_bisp.rule[:,0],pslice,:end_ells_full]
        x_i_ell *= (red_bisp.weights[:,0,:])[:,pslice,np.newaxis]
        x_i_ell *= self.data.b_ell        

        y_i_ell[:,:,red_bisp.lmin:red_bisp.lmax+1] = \
            red_bisp.factors[red_bisp.rule[:,1],pslice,:end_ells_full]
        y_i_ell *= (red_bisp.weights[:,1,:])[:,pslice,np.newaxis]
        y_i_ell *= self.data.b_ell                

        z_i_ell[:,:,red_bisp.lmin:red_bisp.lmax+1] = \
            red_bisp.factors[red_bisp.rule[:,2],pslice,:end_ells_full]
        z_i_ell *= (red_bisp.weights[:,2,:])[:,pslice,np.newaxis]
        z_i_ell *= self.data.b_ell                

        return x_i_ell, y_i_ell, z_i_ell
        
    def _init_rings(self, nfact):
        '''
        Allocate the X_{i phi}, Y_{i phi}, Z_{i phi} ring arrays.

        Parameters
        ----------
        nfact : int
            Number of factorized elements for a reduced bispectrum.

        Returns
        -------
        x_i_phi : (nfact, nphi) array
            Factors on isolatitude ring corresponding to ell_1.
        y_i_phi : (nfact, nphi) array
            Factors on isolatitude ring corresponding to ell_2.
        z_i_phi : (nfact, nphi) array
            Factors on isolatitude ring corresponding to ell_3.
        '''

        x_i_phi = np.zeros((nfact, self.nphi))
        y_i_phi = np.zeros_like(x_i_phi)
        z_i_phi = np.zeros_like(x_i_phi)

        return x_i_phi, y_i_phi, z_i_phi
    
    def step(self, alm, comm=None):
        '''
        Add iteration to <grad T (C^-1 a) C^-1 grad T(C^-1 a)> 
        and <grad T (C^-1 a)> Monte Carlo estimates.

        Parameters
        ----------
        alm : (npol, nelem) array
            Healpix-ordered unfiltered alm array. Will be overwritten!
        comm : MPI communicator, optional

        Raises
        ------
        ValueError
            If shape input alm is not understood.
        '''

        if comm is None:
            comm = utils.FakeMPIComm

        if alm.shape != (self.data.npol, hp.Alm.getsize(self.data.lmax)):
            raise ValueError('alm shape not understood, expected {}, got {}'.
                format(alm.shape, (self.data.npol, hp.Alm.getsize(self.data.lmax))))

        alm = self.icov(alm)
        alm = utils.alm2a_ell_m(alm)
        grad_t = np.zeros_like(alm)
        
        red_bisp = self.cosmology.red_bispectra[0]
        x_i_ell, y_i_ell, z_i_ell = self._init_reduced_bispectrum(red_bisp)
        x_i_phi, y_i_phi, z_i_phi = self._init_rings(red_bisp.nfact)

        ms = np.arange(self.data.lmax + 1)
        m_phase = (-1) ** ms.astype(float)
        
        # Distribute rings over ranks.
        for tidx in range(comm.Get_rank(), len(self.thetas), comm.Get_size()):

            theta = self.thetas[tidx]
            ct_weight = self.theta_weights[tidx]
            ylm = np.ascontiguousarray(np.transpose(
                legendre.normalized_associated_legendre_ms(ms, theta, self.data.lmax)))

            self.backward(alm, x_i_ell, y_i_ell, z_i_ell,
                          x_i_phi, y_i_phi, z_i_phi, ylm)

            self.forward(grad_t, x_i_ell, y_i_ell, z_i_ell,
                         x_i_phi, y_i_phi, z_i_phi, ylm, ct_weight)
        
        grad_t = utils.reduce_array(grad_t, comm) 

        if comm.Get_rank() == 0:        
            # Correct for -m here. See notes `forward()`.
            np.conj(grad_t, out=grad_t)
            grad_t *= m_phase[np.newaxis,np.newaxis,:]

            # Turn back into healpy shape.
            grad_t = utils.a_ell_m2alm(grad_t)

            # Add to Monte Carlo estimates.
            if self.mc_gt is None:
                self.mc_gt = grad_t                
            else:
                self.mc_gt += grad_t

            mc_gt_sq = np.einsum('ij, ij', grad_t.copy(), self.icov(grad_t))
            
            if self.mc_gt_sq is None:
                self.mc_gt_sq = mc_gt_sq
            else:
                self.mc_gt_sq += mc_gt_sq
            
        self.mc_idx += 1

    def compute_estimate(self, alm, comm=None):
        '''
        Compute fNL estimate for input alm.

        Parameters
        ----------
        alm : (npol, nelem) array
            Healpix-ordered unfiltered alm array. Will be overwritten!
        comm : MPI communicator, optional

        Returns
        -------
        estimate : scalar, None
            fNL estimate on root.

        Raises
        ------
        ValueError
            If shape input alm is not understood.
            If Monte Carlo quantities are not iterated yet.
        '''

        # Similar to step, but only do backward transform, multiply alm with linear term
        # and apply normalization.
        if comm is None:
            comm = utils.FakeMPIComm

        if alm.shape != (self.data.npol, hp.Alm.getsize(self.data.lmax)):
            raise ValueError('alm shape not understood, expected {}, got {}'.
                format(alm.shape, (self.data.npol, hp.Alm.getsize(self.data.lmax))))

        t_cubic = 0 # The cubic estimate.
        fisher = self.compute_fisher()
        lin_term = self.compute_linear_term(alm)
        
        alm = self.icov(alm)
        alm = utils.alm2a_ell_m(alm)
        grad_t = np.zeros_like(alm)
        
        red_bisp = self.cosmology.red_bispectra[0]
        x_i_ell, y_i_ell, z_i_ell = self._init_reduced_bispectrum(red_bisp)
        x_i_phi, y_i_phi, z_i_phi = self._init_rings(red_bisp.nfact)

        ms = np.arange(self.data.lmax + 1)
        
        # Distribute rings over ranks.
        for tidx in range(comm.Get_rank(), len(self.thetas), comm.Get_size()):

            theta = self.thetas[tidx]
            ct_weight = self.theta_weights[tidx]
            ylm = np.ascontiguousarray(np.transpose(
                legendre.normalized_associated_legendre_ms(ms, theta, self.data.lmax)))

            self.backward(alm, x_i_ell, y_i_ell, z_i_ell,
                          x_i_phi, y_i_phi, z_i_phi, ylm)

            t_a = np.einsum('ij, ij, ij', x_i_phi, y_i_phi, z_i_phi, optimize=True)
            t_a *= np.pi * ct_weight / 3 / self.nphi
            t_cubic += t_a

        t_cubic = utils.reduce(t_cubic, comm)

        if comm.Get_rank() == 0:
            return (t_cubic - lin_term) / fisher
        else:
            return None
    
    def backward(self, a_ell_m, x_i_ell, y_i_ell, z_i_ell,
                 x_i_phi, y_i_phi, z_i_phi, y_ell_m):
        '''
        a_ell_m -> X_{i phi} Y_{i phi} Z_{i phi}.

        Parameters
        ----------
        a_ell_m : (npol, nell, nm) complex array
            Input ell-major alm.
        x_i_ell : (nfact, npol, nell) array
            Reduced bispectrum factors for ell_1.
        y_i_ell : (nfact, npol, nell) array
            Reduced bispectrum factors for ell_2.
        z_i_ell : (nfact, npol, nell) array
            Reduced bispectrum factors for ell_3.
        x_i_phi : (nfact, nphi) array
            Output factors on isolatitude ring corresponding to ell_1.
        y_i_phi : (nfact, nphi) array
            Output factors on isolatitude ring corresponding to ell_2.
        z_i_phi : (nfact, nphi) array
            Output factors on isolatitude ring corresponding to ell_3.
        y_ell_m : (nm, nell)
            Spherical harmonics stored in ell-major order.
        '''

        self.m_ell_m *= 0
        self.m_ell_m[:,:,:self.data.lmax+1] = a_ell_m
        self.m_ell_m[:,:,:self.data.lmax+1] *= y_ell_m
        self.fft_backward()
        self.n_ell_phi *= self.nphi # Correct for FFT normalization.

        np.einsum('ijk, jkm -> im', x_i_ell, self.n_ell_phi,
                  optimize='optimal', out=x_i_phi) 
        np.einsum('ijk, jkm -> im', y_i_ell, self.n_ell_phi,
                  optimize='optimal', out=y_i_phi) 
        np.einsum('ijk, jkm -> im', z_i_ell, self.n_ell_phi,
                  optimize='optimal', out=z_i_phi) 

    def forward(self, a_ell_m, x_i_ell, y_i_ell, z_i_ell,
                 x_i_phi, y_i_phi, z_i_phi, y_ell_m, ct_weight):        
        '''
        X_{i phi} Y_{i phi} Z_{i phi} -> a_ell_m.

        Parameters
        ----------
        a_ell_m : (npol, nell, nm) complex array
            Output is added to this ell-major alm array.
        x_i_ell : (nfact, npol, nell) array
            Reduced bispectrum factors for ell_1.
        y_i_ell : (nfact, npol, nell) array
            Reduced bispectrum factors for ell_2.
        z_i_ell : (nfact, npol, nell) array
            Reduced bispectrum factors for ell_3.
        x_i_phi : (nfact, nphi) array
            Input factors on isolatitude ring corresponding to ell_1.
        y_i_phi : (nfact, nphi) array
            Input factors on isolatitude ring corresponding to ell_2.
        z_i_phi : (nfact, nphi) array
            Input factors on isolatitude ring corresponding to ell_3.
        y_ell_m : (nm, nell)
            Spherical harmonics stored in ell-major order.
        ct_weight : float
            Quadrature weight for isolatitude ring.

        Notes
        -----
        Output is the complex conjugate * (-1)^m of Eq. 79 of 
        Smith & Zaldarriaga. This is because the FFTW forward 
        transform does e^{-i m phi). So we calculate the answer
        for negative m compared to Eq. 79.
        '''
        
        # Construct dT/dX, dT/dY, dT/dZ (Eq. 76).
        x_i_phi_tmp = x_i_phi.copy()
        y_i_phi_tmp = y_i_phi.copy()        
        
        x_i_phi[:] = y_i_phi
        x_i_phi *= z_i_phi

        y_i_phi[:] = x_i_phi_tmp
        y_i_phi *= z_i_phi

        z_i_phi[:] = x_i_phi_tmp
        z_i_phi *= y_i_phi_tmp

        # Fill dT/dN (Eq. 77).        
        n_ell_phi_tmp = np.zeros_like(self.n_ell_phi)                

        np.einsum('ijk, il -> jkl', x_i_ell, x_i_phi, out=n_ell_phi_tmp,
                  optimize='optimal')
        self.n_ell_phi[:] = n_ell_phi_tmp

        np.einsum('ijk, il -> jkl', y_i_ell, y_i_phi, out=n_ell_phi_tmp,
                  optimize='optimal')
        self.n_ell_phi[:] += n_ell_phi_tmp

        np.einsum('ijk, il -> jkl', z_i_ell, z_i_phi, out=n_ell_phi_tmp,
                  optimize='optimal')
        self.n_ell_phi[:] += n_ell_phi_tmp

        # Multiply dT/dN with weight.
        weight = np.pi * ct_weight / 3 / self.nphi
        self.n_ell_phi *= weight
                
        # dt_dm: fft forward. Forward normalization is already correct.
        self.fft_forward()
        self.m_ell_m[:,:,:self.data.lmax+1] *= y_ell_m

        # Result must be added to a_ell_m.
        a_ell_m += self.m_ell_m[:,:,:self.data.lmax+1]
    
    def compute_fisher(self):
        '''
        Return Fisher information at current iteration. 

        Returns
        -------
        fisher : float, None
        '''

        if self.mc_gt_sq is None or self.mc_gt is None:
            return None
        
        fisher = self.mc_gt_sq
        fisher -= np.einsum('ij, ij', self.mc_gt, self.icov(self.mc_gt.copy()))
        fisher /= 3.

        return fisher
        
    def compute_linear_term(self, alm):
        '''
        Return linear term at current iteration for input data alm.

        Parameters
        ----------
        alm : (npol, nelem) complex array
            Spherical harmonic coeffients of data.

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
        
        return np.einsum('ij, ij', alm, self.icov(self.mc_gt.copy()))        
        

