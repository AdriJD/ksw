import numpy as np
from scipy.interpolate import CubicSpline
import inspect
import json

import camb
import h5py

from ksw import radial_functional as rf

class Cosmology:
    '''
    A Cosmology instance represent a specific cosmology and
    can calculate power spectra and bispectra.

    Parameters
    ---------
    camb_params : camb.model.CAMBparams instance
        CAMB input parameter object.
    verbose : bool, optional
        Report progress.

    Raises
    ------
    ValueError
        If CAMB parameters are invalid.
        If Omega_K parameter is nonzero.

    Attributes
    ----------
    camb_params : camb.model.CAMBparams instance
        Possibly modified copy of input CAMB parameters.
    transfer : dict of arrays
        Radiation transfer functions and metadata.
    cls : dict of arrays
        Angular power spectra and metadata.
    red_bisp : dict of arrays
        Factors of the reduced bispectrum and metadata.
    '''

    def __init__(self, camb_params, verbose=True):

        # In future, allow for several bispectra, i.e.
        # Several primordial, ISW-lensing, etc.

        if camb_params.validate() is False:
            raise ValueError('Input CAMB params file invalid.')

        if camb_params.omk != 0:
            raise ValueError('Nonzero Omega_K not supported.')

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
                'New value {} for param {} makes params invalid.'.format(
                    value, name))

    def compute_transfer(self, lmax, verbose=True):
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

        self.camb_params.set_for_lmax(lmax, lens_margin=0,
                                      k_eta_fac=k_eta_fac)

        if self.camb_params.validate() is False:
            raise ValueError(
                'Value {} for lmax makes params invalid'.format(
                    lmax))

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
        tr_view = np.swapaxes(tr_view, 0, 2) # (nk, nell, npol).
        tr_view = np.swapaxes(tr_view, 0, 1) # (nell, nk, npol).
        tr_ell_k[:] = np.ascontiguousarray(tr_view)

        self.transfer['tr_ell_k'] = tr_ell_k
        self.transfer['k'] = tr.q
        self.transfer['ells'] = ells # Probably sparse.

    def compute_cls(self):
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

    def compute_prim_reduced_bispectrum(self, prim_shape, radii):
        '''
        Compute the factors of the reduced bispectrum of
        a given primordial shape.

        Parameters
        ----------
        prim_shape : ksw.Shape instance
            Primordial shape function.
        radii : array-like
            Radii in Mpc.
        '''

        tr_ell_k = self.transfer['tr_ell_k']
        k = self.transfer['k']
        ells_sparse = self.transfer['ells']

        f_k = prim_shape.get_f_k(k)

        # Call cython code.
        red_bisp = rf.radial_func(f_k, tr_ell_k, k, radii, ells_sparse)

        lmin = ells_sparse[0]
        lmax = ells_sparse[-1]
        ells_full = np.arange(lmin, lmax+1, dtype=int)

        self.red_bisp = {}
        self.red_bisp['ells'] = ells_full
        self.red_bisp['radii'] = radii

        # Interpolate over ells.
        self.red_bisp['red_bisp'] = self._interp_reduced_bispec_over_ell(
            red_bisp, ells_sparse, ells_full)

    @staticmethod
    def _interp_reduced_bispec_over_ell(red_bisp, ells_sparse,
                                        ells_full):
        '''
        Interpolate factors of reduced bispectrum over all
        multipoles.

        Parameters
        ----------
        red_bisp : (nr, nell, npol, ncomp) array
            Factors of reduced bispectrum that are sparsly sampled
            over multipoles.
        ells_sparse : (nell) array
            Sparsly sampled multipoles.
        ells_full : (nell_full) array
            Fully sampled multipoles.

        Returns
        -------
        red_bisp_full : (nr, nell_full, npol, ncomp) array
            Fully sampled factors of reduced bispectrum.
        '''

        # Scipy does internal transpose of input array,
        # transposing beforehand is less efficient.
        cs = CubicSpline(ells_sparse, red_bisp, axis=1)

        red_bisp_full = cs(ells_full)

        return red_bisp_full

    def write_transfer(self, filename):
        '''
        Write the transfer function to disk.

        Parameters
        ----------
        filename : str
            Filename
        '''

        with h5py.File(filename + '.hdf5', 'w') as f:
            f.create_dataset('tr_ell_k', data=self.transfer['tr_ell_k'])
            f.create_dataset('k', data=self.transfer['k'])
            f.create_dataset('ells', data=self.transfer['ells'])

    def write_camb_params(self, filename):
        '''
        Dump the CAMB parameters to human readable file.

        Parameters
        ----------
        filename : str
            Filename
        '''

        # Put parameters in dict of dicts.
        params = {}

        for idx in inspect.getmembers(self.camb_params):
            if idx[0].startswith('_'):
                continue
            elif inspect.ismethod(idx[1]):
                continue
            elif isinstance(idx[1], (int, float, bool, list)):
                params[idx[0]] = idx[1]
            elif isinstance(idx[1], camb.model.CAMB_Structure):
                params[idx[0]] = {}
                
                for jdx in inspect.getmembers(idx[1]):

                    if jdx[0].startswith('_'):
                        continue
                    elif inspect.ismethod(jdx[1]):
                        continue
                    elif isinstance(jdx[1], (int, float, bool, list)):
                        params[idx[0]][jdx[0]] = jdx[1]
                                                       
        # Store dict.
        with open(filename + '.json', 'w') as f:
            json.dump(params, f, sort_keys=True, indent=4)

    def write_red_bisp(self, filename):
        '''
        Write the reduced bispectrum to disk.

        Parameters
        ----------
        filename : str
            Filename
        '''

        with h5py.File(filename + '.hdf5', 'w') as f:
            f.create_dataset('red_bisp', data=self.red_bisp['red_bisp'])
            f.create_dataset('radii', data=self.red_bisp['radii'])
            f.create_dataset('ells', data=self.red_bisp['ells'])
    
    def write_cls(self, filename):
        '''
        Write the angular power spectra to disk.

        Parameters
        ----------
        filename : str
            Filename
        '''
        with h5py.File(filename + '.hdf5', 'w') as f:
            lens_scal = f.create_group('lensed_scalar')
            lens_scal.create_dataset('ells',
                    data=self.cls['lensed_scalar']['ells'])
            lens_scal.create_dataset('cls',
                    data=self.cls['lensed_scalar']['cls'])

            unlens_scal = f.create_group('unlensed_scalar')
            unlens_scal.create_dataset('ells',
                    data=self.cls['unlensed_scalar']['ells'])
            unlens_scal.create_dataset('cls',
                    data=self.cls['unlensed_scalar']['cls'])
            
    def read_transfer(self, filename):
        '''
        Read in tranfer file and populate transfer attribute.

        Parameters
        ----------
        filename : str
            Filename
        '''
        
        self.transfer = {}

        with h5py.File(filename + '.hdf5', 'r') as f:
            self.transfer['tr_ell_k'] = f['tr_ell_k'][()]
            self.transfer['k'] = f['k'][()]
            self.transfer['ells'] = f['ells'][()]

    def read_red_bisp(self, filename):
        '''
        Read in reduced bispectrum file and populate red_bisp attribute.

        Parameters
        ----------
        filename : str
            Filename
        '''

        self.red_bisp = {}

        with h5py.File(filename + '.hdf5', 'r') as f:
            self.red_bisp['red_bisp'] = f['red_bisp'][()]
            self.red_bisp['radii'] = f['radii'][()]
            self.red_bisp['ells'] = f['ells'][()]
    
    def read_cls(self, filename):
        '''
        Read in cls file and populate cls attribute.

        Parameters
        ----------
        filename : str
            Filename
        '''

        self.cls = {}

        with h5py.File(filename + '.hdf5', 'r') as f:
            ells = f['lensed_scalar/ells'][()]
            cls = f['lensed_scalar/cls'][()]
            self.cls['lensed_scalar'] = {}
            self.cls['lensed_scalar']['ells'] = ells
            self.cls['lensed_scalar']['cls'] = cls

            ells = f['unlensed_scalar/ells'][()]
            cls = f['unlensed_scalar/cls'][()]
            self.cls['unlensed_scalar'] = {}
            self.cls['unlensed_scalar']['ells'] = ells
            self.cls['unlensed_scalar']['cls'] = cls            
            
