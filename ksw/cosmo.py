import numpy as np
from scipy.interpolate import CubicSpline

import camb

from ksw import radial_functional as rf

class Cosmology:
    '''
    A cosmology instance represent a specific cosmology and
    can be used to calculate its observables.

    Parameters
    ---------
    camb_params : camb.model.CAMBparams instance
        CAMB input parameter object.
    verbose : bool, optional
        Report progress.

    Raises
    ------
    ValueError
        If input CAMB parameters are invalid.

    Attributes
    ----------
    camb_params : camb.model.CAMBparams instance
        Possibly modified copy of input CAMB parameters.
    transfer : dict of arrays
    cls : dict of arrays
    b_ell_r : (nr, nell, npol, ncomp) array
    '''

    def __init__(self, camb_params, verbose=True):

        if camb_params.validate() is False:
            raise ValueError('Input CAMB params file invalid')

        self.camb_params = camb_params.copy()

        # Do not change any cosmological or primordial setting,
        # but check the accuracy and calculation settings.
        self._setattr_camb('WantScalars', True, verbose=verbose)
        self._setattr_camb('WantTensors', False, verbose=verbose)
        self._setattr_camb('WantCls', True, verbose=verbose)
        self._setattr_camb('WantTransfer', False,
                          verbose=verbose) # This is the matter transfer.
        self._setattr_camb('DoLateRadTruncation', False,
                          verbose=verbose)
        self._setattr_camb('DoLensing', True, verbose=verbose)
        self._setattr_camb('AccuracyBoost', 2, subclass='Accuracy',
                          verbose=verbose)
        self._setattr_camb('BessIntBoost', 30, subclass='Accuracy',
                          verbose=verbose)
        self._setattr_camb('KmaxBoost', 3, subclass='Accuracy',
                          verbose=verbose)
        self._setattr_camb('IntTolBoost', 4, subclass='Accuracy',
                          verbose=verbose)
        self._setattr_camb('TimeStepBoost', 4, subclass='Accuracy',
                          verbose=verbose)
        self._setattr_camb('SourcekAccuracyBoost', 5, subclass='Accuracy',
                          verbose=verbose)
        self._setattr_camb('BesselBoost', 5, subclass='Accuracy',
                          verbose=verbose)
        self._setattr_camb('IntkAccuracyBoost', 5, subclass='Accuracy',
                          verbose=verbose)
        self._setattr_camb('lSampleBoost', 2, subclass='Accuracy',
                          verbose=verbose)
        self._setattr_camb('lAccuracyBoost', 2, subclass='Accuracy',
                          verbose=verbose)
        self._setattr_camb('AccurateBB', True, subclass='Accuracy',
                          verbose=verbose)
        self._setattr_camb('AccurateReionization', True, subclass='Accuracy',
                          verbose=verbose)
        self._setattr_camb('AccuratePolarization', True, subclass='Accuracy',
                          verbose=verbose)

        if self.camb_params.validate() is False:
            raise ValueError('Invalid CAMB input')

    def _setattr_camb(self, name, value, subclass=None, verbose=True):
        '''
        Set a CAMB parameter and print message if the parameter
        changed its value.

        Parameters
        ---------
        name : str
            Parameter.
        value : obj
            New value.
        subclass : str, optional
            Name of subclass in parameter file.
        verbose : bool, optional
            Print if parameter values change.

        Notes
        -----
        subclass can be: "Accuracy", "NonLinearModel", "DarkEnergy",
            "SourceTerms", "CustomSources", "Reion", "Recomb",
            "InitPower" or "Transfer".

        Raises
        ------
        AttributeError
            If parameter does not exist.
        ValueError
            If new parameter value invalidates parameter object.
        '''

        if subclass is not None:
            params = getattr(self.camb_params, subclass)
        else:
            params = self.camb_params

        # CAMB parameter file initiates all its attributes,
        # so we can trust getattr to work here.
        old_value = getattr(params, name)

        if value != old_value:
            setattr(params, name, value)
            if verbose:
                print('Updated CAMB param: {} from {} to {}.'.
                      format(name, old_value, value))

        if self.camb_params.validate() is False:
            raise ValueError(
                'new value {} for param {} makes params invalid'.format(
                    value, name))

    def calc_transfer(self, lmax, verbose=True):
        '''
        Call CAMB to calculate radiation transfer functions.

        Parameters
        ---------
        lmax : int
            Maximum multipole.
        verbose : bool, optional
            Print if CAMB parameter values change.

        Raises
        ------
        AttributeError
            If CAMB parameters have not been initialized.
        ValueError
            If lmax is too low (lmax < 300).
        '''

        self.transfer = {}

        if lmax < 300:
            # CAMB crashes for too low lmax.
            raise ValueError('Pick lmax >= 300.')

        k_eta_fac = 2.5 # Default used by CAMB.
        max_eta_k = k_eta_fac * lmax
        max_eta_k = max(max_eta_k, 1000)

        self._setattr_camb('max_l', lmax, verbose=verbose)
        self._setattr_camb('max_eta_k', max_eta_k, verbose=verbose)

        if self.camb_params.validate() is False:
            raise ValueError('Invalid CAMB input')

        # Make CAMB do the actual calculations (slow).
        data = camb.get_transfer_functions(self.camb_params)
        self._camb_data = data

        # Copy in resulting transfer functions (fast).
        tr = data.get_cmb_transfer_data('scalar')

        # Modify scalar E-mode and tensor I transfer functions, see
        # Zaldarriaga 1997 eq. 18 and 39. (CAMB applies these factors
        # at a later stage).
        ells = tr.L
        # CAMB ells are in int32, so convert.
        ells = ells.astype(int)
        prefactor = np.sqrt((ells + 2) * (ells + 1) * ells * (ells - 1))

        tr.delta_p_l_k[1,...] *= prefactor[:,np.newaxis]

        # Scale with CMB temperature in uK.
        tr.delta_p_l_k *= (self.camb_params.TCMB * 1e6)

        # Transfer function is now of shape (NumSources, nell, nk),
        # where NumSources = 3 (T, E, lensing potential).
        # We want the shape to be (nell, nk, npol=2).
        nk = tr.q.size
        nell = ells.size
        npol = 2
        tr_ell_k = np.empty((nell, nk, npol), dtype=float)

        tr_view = tr.delta_p_l_k[:2,...]
        tr_view = np.swapaxes(tr_view, 0, 2) # (nk, nell, npol)
        tr_view = np.swapaxes(tr_view, 0, 1) # (nell, nk, npol)
        tr_ell_k[:] = np.ascontiguousarray(tr_view)

        self.transfer['tr_ell_k'] = tr_ell_k
        self.transfer['k'] = tr.q
        self.transfer['ells'] = ells # Probably sparse.

    def calc_cls(self):
        '''
        Calculate angular power spectra using precomputed
        transfer functions.
        '''

        self.cls = {}

        self._camb_data.power_spectra_from_transfer()

        cls_unlensed_scalar = self._camb_data.get_unlensed_scalar_cls(
            lmax=None, CMB_unit='muK', raw_cl=True)

        cls_lensed_scalar = self._camb_data.get_lensed_scalar_cls(
            lmax=None, CMB_unit='muK', raw_cl=True)

        # CAMB cls are column-major (nell, npol)
        # Pol order is TT, EE, BB, TE.
        # nell is different for lensed and unlensed.
        # Monopole and dipole are included.

        ells_unlensed = np.arange(cls_unlensed_scalar.shape[0])
        ells_lensed = np.arange(cls_lensed_scalar.shape[0])

        self.cls['unlensed_scalar'] = {}
        self.cls['unlensed_scalar']['ells'] = ells_unlensed
        self.cls['unlensed_scalar']['cls'] = cls_unlensed_scalar

        self.cls['lensed_scalar'] = {}
        self.cls['lensed_scalar']['ells'] = ells_lensed
        self.cls['lensed_scalar']['cls'] = cls_lensed_scalar

    def calc_reduced_bispectrum(self, shape, radii):
        '''
        Compute the factors of the reduced bispectrum of
        a given primordial shape.

        Parameters
        ----------
        shape : ksw.Shape instance
            Primordial shape function
        radii : array-like
            Radii in Mpc.
        '''

        tr_ell_k = self.transfer['tr_ell_k']
        k = self.transfer['k']
        ells_sparse = self.transfer['ells']

        f_k = shape.get_f_k(k)

        # Call cython code.
        b_ell_r = rf.radial_func(f_k, tr_ell_k, k, radii, ells_sparse)

        lmin = ells_sparse[0]
        lmax = ells_sparse[-1]
        ells_full = np.arange(lmin, lmax+1, dtype=int)
        
        # Interpolate over ells.
        self.b_ell_r = self._interp_reduced_bispec_over_ell(
            b_ell_r, ells_sparse, ells_full)

    @staticmethod
    def _interp_reduced_bispec_over_ell(b_ell_r, ells_sparse,
                                        ells_full):
        '''
        Interpolate factors of reduced bispectrum over all 
        multipoles.

        Parameters
        ----------
        b_ell_r : (nr, nell, npol, ncomp) array
            Factors of reduced bispectrum that are sparsly sampled
            over multipoles.
        ells_sparse : (nell) array
            Sparsly sampled multipoles.
        ells_full : (nell_full) array
            Fully sampled multipoles.

        Returns
        -------
        b_ell_r_full : (nr, nell_full, npol, ncomp) array
            Fully sampled factors of reduced bispectrum.
        '''
        
        # Scipy does internal transpose of input array,
        # transposing beforehand is less efficient.
        cs = CubicSpline(ells_sparse, b_ell_r, axis=1)        

        b_ell_r_full = cs(ells_full)

        return b_ell_r_full
        

    # some functions to read and write transfer functions and b_ell_rs
    # When reading, should automatically get correct state of class instance
    # so also read/write camb params with transfer (and poss also with b_ell_r)
