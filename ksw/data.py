import numpy as np

import healpy as hp

class Data():
    '''
    A Data instance contains data alms, beam and S+N covariance.
    Can generate Gaussian sims of data.

    Parameters
    ----------
    lmax : int
        Maximum multipole assumed for alms and related quantities.
    n_ell : (nell) or (nspec, nell) array
        Noise covariance matrix (without beam) in uK^2. Order: TT, EE, TE.
    b_ell : (nell) or (npol, nell) array
        Beam window function of alms.
    pol : str, array-like of str
        Data polarization, e.g. "E", or ["T", "E"]. Order should be T, E.
    cosmo : ksw.Cosmology instance
        Cosmology instance to get Cls.

    Raises
    ------
    ValueError
        If input shapes are inconsistent.
        If elements of pol are invalid.

    Attributes
    ----------
    pol : tuple
        Included polarizations.
    npol : int
        Number of included polarization.
    lmax : int
        Maximum multipole of data.
    b_ell : (npol, nell) array
        Beam window function for each polarization.
    n_ell : (nspec, nell) array
        Noise covariance matrix (without beam) in uK^2. Order=TT(, EE, TE).
    cov_ell_lensed : (nspec, nell) array
        Lensed covariance matrix diagonal in multipole.
    cov_ell_nonlensed : (nspec, nell) array
        Non-lensed covariance matrix diagonal in multipole.
    icov_ell_lensed : (nspec, nell) array
        Inverse lensed covariance matrix diagonal in multipole.
    icov_ell_nonlensed : (nspec, nell) array
        Inverse non-lensed covariance matrix diagonal in multipole.

    Notes
    -----
    Lowering lmax after initiation will truncate all quantities correctly.
    '''

    # Would be nice to add option to interpret n_ell as signal + noise.
    # You run into issues with lensed and non-lensed then, perhaps fine.
    # You could make both lensed and nonlensed return the same thing then?
    
    def __init__(self, lmax, n_ell, b_ell, pol, cosmo):

        self.pol = pol
        self.lmax = lmax
        self.b_ell = b_ell
        self.n_ell = n_ell
        self.cosmology = cosmo

        covs = self._compute_totcov_diag()
        self.cov_ell_lensed = covs[0]
        self.icov_ell_lensed = covs[1]
        self.cov_ell_nonlensed = covs[2]         
        self.icov_ell_nonlensed = covs[3]

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

    def _trunc_x_ell(self, x_ell, name):
        '''If needed, truncate last dim to lmax or raise IndexError.'''
        if x_ell.shape[-1] < self.lmax + 1:
            raise IndexError('lmax {} < lmax ({} < {})'
            .format(name, x_ell.shape[-1], self.lmax))
        return np.ascontiguousarray(x_ell[...,:self.lmax+1])
    
    @property
    def b_ell(self):
        return self._trunc_x_ell(self.__b_ell, 'b_ell')
    
    @b_ell.setter
    def b_ell(self, b_ell):
        '''Make 2d. Check shape. Store copy.'''
        b_ell = np.ascontiguousarray(np.atleast_2d(b_ell.copy()))
        if b_ell.shape[0] != self.npol:
            raise ValueError(
                'Invalid shape[0] of b_ell. Expected {}, got {}.'
                .format(self.npol, b_ell.shape[0]))
        self.__b_ell = b_ell

    @property
    def n_ell(self):
        return self._trunc_x_ell(self.__n_ell, 'n_ell')        
    
    @n_ell.setter
    def n_ell(self, n_ell):
        '''Make 2d. Check shape. Store copy.'''
        n_ell = np.ascontiguousarray(np.atleast_2d(n_ell.copy()))

        if self.npol == 1:
            nspec = 1
        elif self.npol == 2:
            nspec = 3

        if n_ell.shape[0] != nspec:
            raise ValueError(
                'Invalid shape[0] n_ell. Expected {}, got {}.'
                .format(nspec, n_ell.shape[0]))

        self.__n_ell = n_ell

    @property
    def cov_ell_lensed(self):
        return self._trunc_x_ell(
            self.__cov_ell_lensed, 'cov_ell_lensed')

    @cov_ell_lensed.setter
    def cov_ell_lensed(self, cov):
        self.__cov_ell_lensed = cov

    @property
    def cov_ell_nonlensed(self):
        return self._trunc_x_ell(
            self.__cov_ell_nonlensed, 'cov_ell_nonlensed')        
        
    @cov_ell_nonlensed.setter
    def cov_ell_nonlensed(self, cov):
        self.__cov_ell_nonlensed = cov

    @property
    def icov_ell_lensed(self):
        return self._trunc_x_ell(
            self.__icov_ell_lensed, 'icov_ell_lensed')        

    @icov_ell_lensed.setter
    def icov_ell_lensed(self, icov):
        self.__icov_ell_lensed = icov

    @property
    def icov_ell_nonlensed(self):
        return self._trunc_x_ell(
            self.__icov_ell_nonlensed, 'icov_ell_nonlensed')        
        
    @icov_ell_nonlensed.setter
    def icov_ell_nonlensed(self, icov):
        self.__icov_ell_nonlensed = icov
        
    def _compute_totcov_diag(self):
        '''
        Compute data covariance: (Nl + Cl * bl^2) and its inverse.

        Raises
        ------
        ValueError
            If data lmax > transfer function lmax.

        Returns
        -------
        cov_ell_lensed : (nspec, nell) array
            Lensed covariance matrix diagonal in multipole.
        cov_ell_nonlensed : (nspec, nell) array
            Non-lensed covariance matrix diagonal in multipole.
        icov_ell_lensed : (nspec, nell) array
            Inverse lensed covariance matrix diagonal in multipole.
        icov_ell_nonlensed : (nspec, nell) array
            Inverse non-lensed covariance matrix diagonal in multipole.

        Notes
        -----
        Diagonal in multipole but can include correlations between
        polarizations.

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
        
        if not hasattr(self.cosmology, 'c_ell'):
            self.cosmology.compute_c_ell()

        ret = []
        for c_ell_type in ['lensed', 'unlensed']:            
            
            cls_ells = self.cosmology.c_ell[c_ell_type+'_scalar']['ells']
            c_ell = self.cosmology.c_ell[c_ell_type+'_scalar']['c_ell']
                  
            cls_lmax = cls_ells[-1]

            if cls_lmax < self.lmax:
                raise ValueError('lmax Cls : {} < lmax data : {}'
                                 .format(cls_lmax, self.lmax))

            # CAMB Cls are (nell, 4), convert to (4, nell).
            totcov = c_ell.transpose()[:,:self.lmax+1].copy()

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
            inv_totcov = self._invert_cov_diag(totcov)

            ret.append(totcov)
            ret.append(inv_totcov)

        return ret
            
    def _invert_cov_diag(self, cov_diag):
        '''
        Return inverse covariance matrix.
        
        Parameters
        ----------
        cov_diag : (nspec, nell) array
            Covariance matrix diagonal with multipoles.

        Returns
        -------
        icov_diag : (nspec, nell) array
            Inverse covariance matrix.
        '''

        icov_diag = np.zeros_like(cov_diag)
        nell = cov_diag.shape[1]
        icov = np.zeros((self.npol, self.npol, nell))

        icov[0,0] = cov_diag[0]
        if self.pol == ('T', 'E'):
            icov[0,1] = cov_diag[2]
            icov[1,0] = cov_diag[2]
            icov[1,1] = cov_diag[1]

        # Temporarily put pol dimensions last for faster inverse.
        icov = np.transpose(icov, (2, 0, 1))
        icov = np.linalg.inv(np.ascontiguousarray(icov))
        icov = np.transpose(icov, (1, 2, 0))

        icov_diag[0] = icov[0,0]
        if self.pol == ('T', 'E'):   
            icov_diag[1] = icov[1,1]
            icov_diag[2] = icov[1,0]        
        
        return icov_diag
    
    def compute_alm_sim(self, lens_power=False):
        '''
        Draw isotropic Gaussian realisation from (S+N) covariance.

        Parameters
        ----------
        lens_power : bool, optional
            Include lensing power in covariance.

        Returns
        -------
        alm_sim : (npol, nelem) complex array
            Simulated alm with ells up to lmax.
        '''

        if lens_power:
            cov_ell = self.cov_ell_lensed
        else:
            cov_ell = self.cov_ell_nonlensed
            
        # Synalm expects 1D TT array or (TT, EE, BB, TE) array.
        if 'E' in self.pol:
            nspec, nell = cov_ell.shape
            c_ell_in = np.zeros((4, nell))
            if 'T' in self.pol:
                c_ell_in[0,:] = cov_ell[0]
                c_ell_in[1,:] = cov_ell[1]
                c_ell_in[3,:] = cov_ell[2]
            else:
                c_ell_in[1,:] = cov_ell[0]
        else:
            c_ell_in = cov_ell

        alm = hp.synalm(c_ell_in, lmax=self.lmax, new=True)

        if self.pol == ('T', 'E'):
            # Only return I and E.
            alm = alm[:2,:]
        elif self.pol == ('E',):
            alm = (alm[1,:])[np.newaxis,:]
        else:
            alm = alm

        return alm
                    
    def _icov_diag(self, alm, lens_power=False):    
        '''
        Return (in-place) inverse covariance weighted version if input.
        Assuming that covariance is diagonal in multipole.

        Parameters
        ----------
        alm : (npol, nelem) array
            Healpix-ordered alm array to be inverse weighted. Order: T, E.
        lens_power : bool, optional
            Include lensing power in invserse covariance.

        Returns
        -------
        c_inv_a : (npol, nelem) complex array
            Inverse covariance weighted input array.

        Raises
        ------
        ValueError
            If shape input alm does not match npol or exceeds lmax.
        '''

        if alm.ndim == 1:
            input_1d = True
            alm = np.atleast_2d(alm)
        else:
            input_1d = False

        if alm.shape[0] != self.npol:
            raise ValueError(
                'Shape alm: {} does not match with length of pol: {}.'
                .format(alm.shape, self.npol))
                    
        lmax_alm = hp.Alm.getlmax(alm.shape[-1])
        if lmax_alm > self.lmax:
            raise ValueError('lmax alm exceeds lmax icov ({} > {})'.
                             format(lmax_alm, self.lmax))
               
        if lens_power:
            icov = self.icov_ell_lensed
        else:
            icov = self.icov_ell_nonlensed

        if self.pol == ('T', 'E'):
            # Would be nice get rid of this copy with C or cython.
            tmp = alm.copy()
            hp.almxfl(tmp[0], icov[2,:lmax_alm+1], inplace=True)
            hp.almxfl(tmp[1], icov[2,:lmax_alm+1], inplace=True)
            hp.almxfl(alm[0], icov[0,:lmax_alm+1], inplace=True)            
            hp.almxfl(alm[1], icov[1,:lmax_alm+1], inplace=True)
            alm[1] += tmp[0]
            alm[0] += tmp[1]
        else:
            hp.almxfl(alm[0], icov[0,:lmax_alm+1], inplace=True)            
            
        if input_1d:
            return alm[0]
        else:
            return alm

    def icov_diag_lensed(self, alm):
        '''
        Return (in-place) inverse lensed covariance weighted version if input.
        Assuming that covariance is diagonal in multipole.

        Parameters
        ----------
        alm : (npol, nelem) array
            Healpix-ordered alm array to be inverse weighted. Order: T, E.

        Returns
        -------
        c_inv_a : (npol, nelem) complex array
            Inverse covariance weighted input array.

        Raises
        ------
        ValueError
            If shape input alm does not match npol or exceeds lmax.
        '''
        
        return self._icov_diag(alm, lens_power=True)
    
    def icov_diag_nonlensed(self, alm):
        '''
        Return (in-place) inverse non-lensed covariance weighted version if input.
        Assuming that covariance is diagonal in multipole.

        Parameters
        ----------
        alm : (npol, nelem) array
            Healpix-ordered alm array to be inverse weighted. Order: T, E.

        Returns
        -------
        c_inv_a : (npol, nelem) complex array
            Inverse covariance weighted input array.

        Raises
        ------
        ValueError
            If shape input alm does not match npol or exceeds lmax.
        '''
        
        return self._icov_diag(alm, lens_power=False)
        
