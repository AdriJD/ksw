import numpy as np

import healpy as hp

class Data():
    '''
    A Data instance contains data alms, beam and S+N covariance.
    Can generate Gaussian sims of data.

    Parameters
    ----------
    alm_data : (nelem) or (npol, nelem) complex array
        Spherical harmonic coefficients of data in uK and
        in Healpix ordering. Order pol: T, E.
    n_ell : (nell) or (nspec, nell) array
        Noise covariance matrix (without beam) in uK^2.
        Order: TT, EE, TE.
    b_ell : (nell) or (npol, nell) array
        Beam window function of alms.
    pol : str, array-like of str, optional
        Data polarization, e.g. "E", or ["T", "E"]. Order
        should always be T, E.

    Raises
    ------
    ValueError
        If input shapes are not consistent.
        If elements of pol are invalid.

    Attributes
    ----------
    pol : tuple
        Included polarizations.
    npol : int
        Number of included polarization.
    lmax : int
        Maximum multipole of data.
    alm_data : (npol, nelem) complex array
        Spherical harmonic coefficients of data in uK and
        Healpix ordering.
    b_ell : (npol, nell) array
        Beam window function for each polarization.
    n_ell : (nspec, nell) array
        Noise covariance matrix (without beam) in uK^2.
        Order=TT, (EE, TE).
    alm_sim : (npol, nelem) complex array, None
        Simulated spherical harmonic coefficients with
        same shape, covariance and beam as data alms.
    totcov_diag : dict of (nspec, nell) arrays, None
        Lensed and unlensed covariance matrix diagonal in multipole.
    inv_totcov_diag : dict of (nspec, nell) arrays, None
        Lensed and unlensed inverse covariance matrix diagonal in multipole.
    '''

    def __init__(self, alm_data, n_ell, b_ell, pol):

        self.pol = pol
        self.alm_data = alm_data
        self.b_ell = b_ell
        self.n_ell = n_ell
        self.alm_sim = None
        self.totcov_diag = None
        self.inv_totcov_diag = None

    @property
    def pol(self):
        return self.__pol
    @pol.setter
    def pol(self, pol):
        '''Make tuple and check contents.'''
        pol = tuple(pol)
        if pol.count('T') + pol.count('E') != len(pol):
            raise ValueError('Pol={}, but may only contain T and/or E.'.
                             format(pol))
        elif pol.count('T') != 1 and pol.count('E') != 1:
            raise ValueError('Pol={}, cannot contain duplicates.'.
                             format(pol))
        elif pol[0] == 'E' and len(pol) > 1:
            raise ValueError('Incorrect order of pol: {}.'.format(pol))
        self.__pol = pol

    @property
    def npol(self):
        return len(self.pol)

    @property
    def alm_data(self):
        return self.__alm_data
    @alm_data.setter
    def alm_data(self, alm):
        '''Check if shape is allowed, make 2d. Store copy.'''
        alm = np.ascontiguousarray(np.atleast_2d(alm.copy()))
        if alm.shape[0] != self.npol:
            raise ValueError(
                'Shape alm: {} does not match with length of pol: {}.'
                .format(alm.shape, self.npol))

        nelem = alm.shape[1]
        if hp.Alm.getlmax(nelem) == -1:
            raise ValueError(
                'Invalid size of alm : nelem = {}'.format(nelem))
        self.__alm_data = alm

    @property
    def lmax(self):
        nelem = self.alm_data.shape[1]
        return hp.Alm.getlmax(nelem)

    @property
    def b_ell(self):
        return self.__b_ell
    @b_ell.setter
    def b_ell(self, b_ell):
        '''Make 2d. Check shape. Store copy.'''
        b_ell = np.ascontiguousarray(np.atleast_2d(b_ell.copy()))
        if b_ell.shape != (self.npol, self.lmax + 1):
            raise ValueError(
                'Invalid shape b_ell. Expected {}, got {}.'
                .format((self.npol, self.lmax + 1),
                        b_ell.shape))
        self.__b_ell = b_ell

    @property
    def n_ell(self):
        return self.__n_ell
    @n_ell.setter
    def n_ell(self, n_ell):
        '''Make 2d. Check shape. Store copy.'''
        n_ell = np.ascontiguousarray(np.atleast_2d(n_ell.copy()))

        if self.npol == 1:
            nspec = 1
        elif self.npol == 2:
            nspec = 3

        if n_ell.shape != (nspec, self.lmax + 1):
            raise ValueError(
                'Invalid shape n_ell. Expected {}, got {}.'
                .format((nspec, self.lmax + 1), n_ell.shape))

        self.__n_ell = n_ell

    def compute_alm_sim(self, lens_power=False):
        '''
        Draw isotropic Gaussian realisation from (S+N) covariance.

        Parameters
        ----------
        lens_power : bool, optional
            Include lensing power in covariance.

        Raises
        ------
        AttributeError
            If compute_totcov_diag() has not been called.
        '''

        if self.totcov_diag is None:
            raise AttributeError('No covariance matrix found. '
                'Call compute_totcov_diag() first.')

        if lens_power:
            totcov_diag = self.totcov_diag['lensed']
        else:
            totcov_diag = self.totcov_diag['unlensed']
            
        # Synalm expects 1D TT array or (TT, EE, BB, TE) array.
        if 'E' in self.pol:
            nspec, nell = totcov_diag.shape
            cls_in = np.zeros((4, nell))
            if 'T' in self.pol:
                cls_in[0,:] = totcov_diag[0]
                cls_in[1,:] = totcov_diag[1]
                cls_in[3,:] = totcov_diag[2]
            else:
                cls_in[1,:] = totcov_diag[0]
        else:
            cls_in = totcov_diag

        alm = hp.synalm(cls_in, lmax=self.lmax, new=True)

        if self.pol == ('T', 'E'):
            # Only return I and E.
            alm = alm[:2,:]
        elif self.pol == ('E',):
            alm = (alm[1,:])[np.newaxis,:]
        else:
            alm = alm

        self.alm_sim = alm

    def compute_totcov_diag(self, cosmo):
        '''
        Compute data covariance: (Nl + Cl * bl^2) and its inverse.
        Diagonal in multipole but can include correlations between
        polarizations.

        Parameters
        ----------
        cosmo : ksw.Cosmology instance
            Cosmology instance to get Cls.

        Raises
        ------
        ValueError
            If data lmax > transfer function lmax.

        Notes
        -----
        We multipy Cls by b_ell**2. We leave Nls unmodified. This 
        assumes that the data alms are beam convolved. (C^-1 a) 
        will thus divide out b_ell^2, which is problematic when Nl
        is zero. If Nl is nonzero at high ell this is fine again.
        The factors of the reduced bispectrum need a factor of b_ell.
        We "forward propagate" the beam.

        The alternative: alm has b_ell divided out, is then mutiplied
        with (Nell / b_ell^2 + Cl)^-1. In this case, you do not
        multiply factors of reduced bispectrum by b_ell.
        '''

        if not hasattr(cosmo, 'cls'):
            cosmo.compute_cls()

        self.totcov_diag = {}
        self.inv_totcov_diag = {}        
        
        for cls_type in ['lensed', 'unlensed']:            
            
            cls_ells = cosmo.cls[cls_type+'_scalar']['ells']
            cls = cosmo.cls[cls_type+'_scalar']['cls']

            cls_lmax = cls_ells[-1]

            if cls_lmax < self.lmax:
                raise ValueError('lmax Cls : {} < lmax data : {}'
                                 .format(cls_lmax, self.lmax))

            # CAMB Cls are (nell, 4), convert to (4, nell).
            totcov = cls.transpose()[:,:self.lmax+2].copy()

            # Turn into correct shape and multiply with beam.
            if self.pol == ('T',):
                totcov = totcov[0,:]
                totcov = totcov[np.newaxis,:]
                totcov *= self.b_ell ** 2
            elif self.pol == ('E',):
                totcov = totcov[1,:]
                totcov = totcov[np.newaxis,:]
                totcov *= self.b_ell ** 2
            elif self.pol == ('T', 'E'):
                polmask = np.ones(4, dtype=bool)
                polmask[2] = False # Mask B.
                totcov = totcov[polmask,:]
                totcov[0] *= self.b_ell[0] ** 2
                totcov[1] *= self.b_ell[1] ** 2
                totcov[2] *= self.b_ell[0] * self.b_ell[1]

            totcov += self.n_ell
            totcov = np.ascontiguousarray(totcov)

            self.totcov_diag[cls_type] = totcov
            self.inv_totcov_diag[cls_type] = self._invert_totcov_diag(totcov)
            
    def _invert_totcov_diag(self, totcov_diag):
        '''
        Return inverse covariance matrix.
        
        Parameters
        ----------
        totcov_diag : (nspec, nell) array
            Covariance matrix diagonal with multipoles.

        Returns
        -------
        inv_totcov_diag : (nspec, nell) array
            Inverse covariance matrix.
        '''

        inv_totcov_diag = np.zeros_like(totcov_diag)
        nell = totcov_diag.shape[1]
        inv_totcov = np.zeros((self.npol, self.npol, nell))

        inv_totcov[0,0] = totcov_diag[0]
        if self.pol == ('T', 'E'):
            inv_totcov[0,1] = totcov_diag[2]
            inv_totcov[1,0] = totcov_diag[2]
            inv_totcov[1,1] = totcov_diag[1]

        # Temporarily put pol dimensions last for faster inverse.
        inv_totcov = np.transpose(inv_totcov, (2, 0, 1))
        inv_totcov = np.linalg.inv(np.ascontiguousarray(inv_totcov))
        inv_totcov = np.transpose(inv_totcov, (1, 2, 0))

        inv_totcov_diag[0] = inv_totcov[0,0]
        if self.pol == ('T', 'E'):        
            inv_totcov_diag[1] = inv_totcov[1,1]
            inv_totcov_diag[2] = inv_totcov[1,0]        
        
        return inv_totcov_diag
            
    def get_c_inv_a_diag(self, sim=False, lens_power=False):
        '''
        Return inverse covariance weighted copy of data.
        Assuming that covariance is diagonal in multipole.

        Parameters
        ----------
        sim : bool, optional
            Return weighted copy of simulated data.
        lens_power : bool, optional
            Include lensing power in invserse covariance.

        Returns
        -------
        c_inv_a : (npol, nelem) complex array
            Inverse covariance weighted copy of data.

        Raises
        ------
        AttributeError
            If compute_alm_sim() has not been called but sim is set.
            If compute_totcov_diag() has not been called.
        '''

        if sim:
            alm = self.alm_sim
            if alm is None:
                raise AttributeError('Call compute_alm_sim() first.')
        else:
            alm = self.alm_data

        if self.inv_totcov_diag is None:
            raise AttributeError('No inverse covariance found. '
                'Call compute_totcov_diag() first.')

        if lens_power:
            inv_totcov = self.inv_totcov_diag['lensed']
        else:
            inv_totcov = self.inv_totcov_diag['unlensed']
            
        c_inv_a = np.zeros_like(alm)        
        c_inv_a[0] = hp.almxfl(alm[0], inv_totcov[0])
        
        if self.pol == ('T', 'E'):
            c_inv_a[0] += hp.almxfl(alm[1], inv_totcov[2])
            c_inv_a[1] = hp.almxfl(alm[0], inv_totcov[2])
            c_inv_a[1] += hp.almxfl(alm[1], inv_totcov[1])

        return c_inv_a

    # And, you can make a data method
    # def c_inv(self, alm) that filters a set of alms.
    # In estimator class you would then call: estimator.set_c_inv(data.c_inv)
