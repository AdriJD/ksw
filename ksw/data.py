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
        in Healpix ordering.
    n_ell : (nell) or (nspec, nell) array
        Noise covariance matrix (without beam) in uK^2.
        Order=TT, (EE, TE).
    b_ell : (nell) or (npol, nell) array
        Beam window function of alms.
    pol : str, array-like of str
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
    totcov_diag : (nspec, nell) array, None
        Covariance matrix diagonal in multipole.
    '''

    def __init__(self, alm_data, n_ell, b_ell, pol):

        self.pol = pol
        self.alm_data = alm_data
        self.b_ell = b_ell
        self.n_ell = n_ell
        self.alm_sim = None
        self.totcov_diag = None

    @property
    def pol(self):
        return self.__pol

    @pol.setter
    def pol(self, pol):
        '''Make tuple and check contents'''
        pol = tuple(pol)
        if pol.count('T') + pol.count('E') != len(pol):
            raise ValueError('Pol={}, but may only contain T and/or E.'.
                             format(pol))
        elif pol.count('T') != 1:
            raise ValueError('Pol={}, cannot contain duplicates.'.
                             format(pol))
        if pol[0] == 'E' and len(pol) > 1:
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

    def compute_alm_sim(self):
        '''
        Draw isotropic Gaussian realisation from (S+N) covariance.
        '''

        totcov_diag = self.totcov_diag
        if totcov_diag is None:
            raise AttributeError(
                'Call compute_totcov_diag() first.')
        
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

        if self.npol == ('T', 'E'):
            # Only return I and E.
            alms = alm[:2,:]
        if self.pol == ('E',):
            alm = alm[1,:]
        else:
            alm = alm

        self.alm_sim = alm

    def compute_totcov_diag(self, cosmo, add_lens_power=False):
        '''
        Return data covariance: (Nl + Cl * bl^2). Diagonal in
        multipole but may include correlations between polarizations.

        Parameters
        ----------
        cosmo : ksw.Cosmology instance
            Cosmology instance to get Cls.
        add_lens_power : bool, optional
            Include lensing contribution to Cls.

        Raises
        ------
        ValueError
            If data lmax > transfer function lmax.

        Notes
        -----
        We multipy Cls by b_ell**2. We leave Nls unmodified.
        This assumes that the data alms are beam convolved.
        (C^-1 a) will thus divide out b_ell^2, which is
        problematic when Nl is zero. If Nl is nonzero at high ell
        this is fine again. The factors of the reduced bispectrum
        need a factor of b_ell. We "forward propagate" the beam.

        The alternative: alm has b_ell divided out, is then mutiplied
        with (Nell / b_ell^2 + Cl)^-1. In this case, you do not
        multiply factors of reduced bispectrum by b_ell.
        '''

        if not hasattr(cosmo, 'cls'):
            cosmo.compute_cls()

        if add_lens_power is True:
            cls_type = 'lensed_scalar'
        else:
            cls_type = 'unlensed_scalar'

        cls_ells = cosmo.cls[cls_type]['ells']
        cls = cosmo.cls[cls_type]['cls']

        cls_lmax = cls_ells[-1]

        if cls_lmax < self.lmax:
            raise ValueError('lmax Cls : {} < lmax data : {}'
                             .format(cls_lmax, self.lmax))

        # CAMB Cls are (nell, 4), convert to (4, nell).
        totcov = cls.transpose()[:,:self.lmax+2].copy()

        # Turn into correct shape and multiply with beam.
        if self.pol == ('T',):
            totcov = totcov[0,:]
            totcov *= self.b_ell ** 2
            totcov = totcov[np.newaxis,:]
        elif self.pol == ('E',):
            totcov = totcov[1,:]
            totcov *= self.b_ell ** 2
            totcov = totcov[np.newaxis,:]
        elif self.pol == ('T', 'E'):
            polmask = np.ones(4, dtype=bool)
            polmask[2] = False # Mask B.
            totcov = totcov[polmask,:]
            totcov[0] *= self.b_ell[0] ** 2
            totcov[1] *= self.b_ell[1] ** 2
            totcov[2] *= self.b_ell[0] * self.b_ell[1]

        totcov += self.n_ell

        self.totcov_diag = totcov

    def get_c_inv_a_diag(self, sim=False):
        '''
        Return inverse covariance weighted copy of data.
        Assuming that covariance is diagonal in multipole.

        Parameters
        ----------
        sim : bool, optional
            Return weighted copy of simulated data.

        Returns
        -------
        c_inv_a : (npol, nelem) complex array
            Inverse covariance weighted copy of data.

        Raises
        ------
        AttributeError
            If compute_alm_sim() has not been called.
            If compute_totcov_diag() has not been called.
        '''

        if sim:
            alm = self.alm_sim
            if alm is None:
                raise AttributeError(
                    'Call compute_alm_sim() first.')
        else:
            alm = self.alm_data

        totcov = self.totcov_diag # (nspec, nell).
        if totcov is None:
            raise AttributeError(
                'Call compute_totcov_diag() first.')

        c_inv_a = np.zeros_like(alm)
        nell = totcov.shape[1]
        c_inv = np.zeros((self.npol, self.npol, nell))

        c_inv[0,0] = totcov[0]
        if npol == 2:            
            c_inv[0,1] = totcov[2]
            c_inv[1,0] = totcov[2]
            c_inv[1,1] = totcov[1]

        c_inv = np.ascontiguousarray(np.transpose(c_inv, (2, 0, 1)))
        c_inv = np.linalg.inv(c_inv)
        c_inv = np.ascontiguousarray(np.transpose(c_inv, (1, 2, 0)))        
        
        if self.pol == ('T', 'E'):
            c_inv_a[0] = hp.almxfl(a[0], c_inv[1])
            c_inv_a[0] += hp.almxfl(a[1], c_inv[2])
            c_inv_a[1] = hp.almxfl(a[0], c_inv[2])
            c_inv_a[1] += hp.almxfl(a[1], c_inv[1])

        elif self.npol == 1:
            c_inv_a[0] = hp.almxfl(a[0], c_inv[0])

        return c_inv_a
