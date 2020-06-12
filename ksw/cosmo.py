import numpy as np
from scipy.interpolate import CubicSpline
import inspect
import json

import camb
import h5py

from ksw import utils, radial_functional as rf

# NOTE, using cls for power spectra is confusing in python,
# perhaps c_ell. Matches n_ell and b_ell in data too.

class Cosmology:
    '''
    A Cosmology instance represents a specific cosmology. It is
    used to calculate power spectra and reduced bispectra.

    Parameters
    ----------
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
    transfer : dict
        Radiation transfer functions and metadata.
    cls : dict
        Angular power spectra and metadata.
    red_bispectra : list
        Collection of ReducedBispectrum instances.
    '''
    
    def __init__(self, camb_params, verbose=True):

        if camb_params.validate() is False:
            raise ValueError('Input CAMB params file invalid.')

        if camb_params.omk != 0:
            raise ValueError('Nonzero Omega_K not supported.')

        self.camb_params = camb_params.copy()

        # Check accuracy and calculation settings, but do not change
        # cosmological or primordial settings.
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

        self.transfer = {}
        self.cls = {}
        self.red_bispectra = []
        
    def _setattr_camb(self, name, value, subclass=None, verbose=True):
        '''
        Set a CAMB parameter and print message if the parameter
        changed its value.

        Parameters
        ----------
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
        ----------
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

        if lmax < 300:
            # CAMB crashes for too low lmax.
            raise ValueError('Pick lmax >= 300.')

        k_eta_fac = 2.5 # Default used by CAMB.
        self.camb_params.set_for_lmax(lmax, lens_margin=0,
                                      k_eta_fac=k_eta_fac)

        if self.camb_params.validate() is False:
            raise ValueError(
            'Value {} for lmax makes params invalid'.format(lmax))

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
        Calculate angular power spectra (Cls) using precomputed
        transfer functions.

        Notes
        -----
        Spectra are CAMB output and are therefore column-major
        (nell, npol). The pol order is TT, EE, BB, TE. nell can 
        be different for lensed and unlensed spectra. The monopole
        and dipole are included. Units are muK^2.
        '''

        self._camb_data.power_spectra_from_transfer()

        cls_unlensed_scalar = self._camb_data.get_unlensed_scalar_cls(
            lmax=None, CMB_unit='muK', raw_cl=True)

        cls_lensed_scalar = self._camb_data.get_lensed_scalar_cls(
            lmax=None, CMB_unit='muK', raw_cl=True)

        ells_unlensed = np.arange(cls_unlensed_scalar.shape[0])
        ells_lensed = np.arange(cls_lensed_scalar.shape[0])

        self.cls['unlensed_scalar'] = {}
        self.cls['unlensed_scalar']['ells'] = ells_unlensed
        self.cls['unlensed_scalar']['cls'] = cls_unlensed_scalar

        self.cls['lensed_scalar'] = {}
        self.cls['lensed_scalar']['ells'] = ells_lensed
        self.cls['lensed_scalar']['cls'] = cls_lensed_scalar

    def add_prim_reduced_bispectrum(self, prim_shape, radii, name=None):
        '''
        Compute the factors of the reduced bispectrum for a given 
        primordial shape and add to internal list of reduced bispectra.

        Parameters
        ----------
        prim_shape : ksw.Shape instance
            Primordial shape function.
        radii : array-like
            Radii in Mpc.
        name : str, optional
            Name to identify reduced bispectrum, defaults to name  
            attribute of primordial shape.
        '''

        tr_ell_k = self.transfer['tr_ell_k']
        k = self.transfer['k']
        ells_sparse = self.transfer['ells']

        f_k = prim_shape.get_f_k(k)
        f_k *= 2 * self.camb_params.InitPower.As ** 2
        
        # Call C code.
        red_bisp = rf.radial_func(f_k, tr_ell_k, k, radii, ells_sparse)

        factors, rule, weights = self._parse_prim_reduced_bispec(
            red_bisp, radii, prim_shape.rule, prim_shape.amps)
        
        if name is None:
            name = prim_shape.name
            
        self.red_bispectra.append(
            ReducedBispectrum(factors, rule, weights, ells_sparse, name))
        
    def _parse_prim_reduced_bispec(self, red_bisp, radii, prim_rule, amps):
        '''
        Rearrange factors of reduced bispectrum for a primordial model
        into the form required by the ReducedBispectrum class.

        Parameters
        ----------
        red_bisp : (nr, nell, npol, ncomp) array
            Factors of reduced bispectrum.
        radii : (nr) array-like
            Radii in Mpc.
        prim_rule : (nprim) sequence of array-like
            Rule to combine factors into primordial shape, see ksw.Shape.
        amps : (nprim) array-like
            Amplitude for each element in rule, see ksw.Shape.

        Returns
        -------
        factors : (ncomp * nr, npol, nell) array
            Rearranged factors of reduced bispectrum.
        rule : (nfact, 3) int array
            Array of indices to first dimension of factors that form the
            reduced bispectrum.
        weights : (nfact, 3, npol) float array
            (nprim * nr) weights (amp * r^2 dr) for each element in rule.
        '''

        nr, nell, npol, ncomp = red_bisp.shape
        nfact = nr * len(prim_rule)        

        # Unique factors, just reshaped version of input reduced bispectrum.
        factors = np.ascontiguousarray(np.transpose(red_bisp, (3, 0, 2, 1)))
        factors = factors.reshape((ncomp * nr, npol, nell), order='C')
        
        # For primordial bispectra the last two dims of weights are constant.
        weights = np.ones((nfact, 3, npol))
        rule = np.ones((nfact, 3), dtype=int)        
        
        dr = utils.get_trapz_weights(radii) * radii ** 2
        ridxs = np.arange(nr) # Indices to radii.
        start = 0
        for amp, ru in zip(amps, prim_rule):
            weights[start:start+nr] = (amp * dr)[:,np.newaxis,np.newaxis]
            # Indices into first dim of factors (which is (ncomp, nr)).
            rule[start:start+nr,:] = ridxs[:,np.newaxis] + (np.asarray(ru) * nr)
            start += nr
                        
        return factors, rule, weights

    def add_reduced_bispectrum_from_file():
        pass
    
    def write_transfer(self, filename):
        '''
        Write the transfer functions to disk.

        Parameters
        ----------
        filename : str
            Absolute path to transfer file.
        '''

        with h5py.File(filename + '.hdf5', 'w') as f:
            f.create_dataset('tr_ell_k', data=self.transfer['tr_ell_k'])
            f.create_dataset('k', data=self.transfer['k'])
            f.create_dataset('ells', data=self.transfer['ells'])

    def read_transfer(self, filename):
        '''
        Read in transfer file and populate transfer attribute.

        Parameters
        ----------
        filename : str
            Absolute path to transfer file.
        '''

        self.transfer = {}

        with h5py.File(filename + '.hdf5', 'r') as f:
            self.transfer['tr_ell_k'] = f['tr_ell_k'][()]
            self.transfer['k'] = f['k'][()]
            self.transfer['ells'] = f['ells'][()]

    def write_cls(self, filename):
        '''
        Write the angular power spectra to disk.

        Parameters
        ----------
        filename : str
            Absolute path to output file.
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

    def read_cls(self, filename):
        '''
        Read in cls file and populate cls attribute.

        Parameters
        ----------
        filename : str
            Absolute path to spectra file.
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
                        
    def write_camb_params(self, filename):
        '''
        Write the CAMB parameters to human readable file.

        Parameters
        ----------
        filename : str
            Absolute path to parameter file.
        '''

        # Put CAMB parameters in dict: {param : value}.
        params = {}
        # Loop over attr ibutes of CAMB param file.
        for idx in inspect.getmembers(self.camb_params):
            # idx : (name, value).
            if idx[0].startswith('_'):
                continue
            elif inspect.ismethod(idx[1]):
                continue
            elif isinstance(idx[1], (int, float, bool, list)):
                params[idx[0]] = idx[1]
            elif isinstance(idx[1], camb.model.CAMB_Structure):
                # Subclass, do separate loop over its attributes.
                params[idx[0]] = {}

                for jdx in inspect.getmembers(idx[1]):
                    if jdx[0].startswith('_'):
                        continue
                    elif inspect.ismethod(jdx[1]):
                        continue
                    elif isinstance(jdx[1], (int, float, bool, list)):
                        params[idx[0]][jdx[0]] = jdx[1]

        with open(filename + '.json', 'w') as f:
            json.dump(params, f, sort_keys=True, indent=4)

class ReducedBispectrum:
    '''
    A ReducedBispectrum instance represents a reduced bispectrum
    that consists of a sum of terms that are each seperable in
    the l1, l2, l3 multipoles:

    b_l1_l2_l3 = sum_i^nfact X(i)_l1 Y(i)_l1 Z(i)_l1.

    Parameters
    ----------
    factors = (n, npol, nell_sparse)
        Unique f_ells that make up the bispectrum.
    rule : (nfact, 3) int array
        Indices to first dimension of unique factors array that
        create the (nfact, 3, npol, nell) reduced bispectrum.
    weights : (nfact, 3, npol) float array
        Weights for each element in rule.
    ells : (nell_sparse) array
        Possibily sparse array of monotonicially increasing
        multipoles.
    name : str
        A name to identify the reduced bispectrum.

    Attributes
    ----------
    factors : (n, npol, nell)
        Unique f_ells that make up the bispectrum.
    rule : (nfact, 3) int array
        Indices to first dimension of unique factors array that
        create the (nfact, 3, npol, nell) reduced bispectrum.
    weights : (nfact, 3, npol) float array
        Weights for each element in rule.
    ells_sparse : (nell_sparse) int array
        Possibily sparse array of monotonicially increasing
        multipoles.        
    ells_full : (nell) int array
        Fully sampled array of multipoles.
    npol : int
        Number of polarizations.

    Raises
    ------
    ValueError
        If input shapes do not match.
        If rule refers to nonexisting factors.
        If name is not an identifiable string.
    TypeError
        If rule is not an array of integers.
    '''

    def __init__(self, factors, rule, weights, ells, name):

        self.ells_sparse = ells
        self.ells_full = np.arange(ells[0], ells[-1] + 1)
        self.factors = factors
        _, self.npol, _ = self.factors.shape
        self.weights = weights
        self.rule = rule
        self.name = name
        
    @property
    def factors(self):
        return self.__factors
    
    @factors.setter
    def factors(self, factors):
        '''Check shape. Interpolate if needed.'''

        if self.ells_sparse.size != factors.shape[2]:
            raise ValueError('Shape of factors {}, does not '
            'match with shape ells {}.'.format(factors.shape,
                                        self.ells_sparse.size))
            
        if self.ells_full.size != self.ells_sparse.size:
            factors = self._interp_factors(factors)
        else:
            factors = factors.copy()

        self.__factors = factors
        
    @property
    def rule(self):
        return self.__rule
    
    @rule.setter
    def rule(self, rule):
        '''Check shape, type and values.'''

        if rule.shape != (self.weights.shape[0], 3):
            raise ValueError('Shape of rule is {}, expected {}.'
                    .format(rule.shape, (self.weights.shape[0], 3)))
        if rule.dtype != int:
            raise TypeError('dtype rule is {} instead of int'.
                            format(rule.dtype))
        if rule.min() < 0:
            raise ValueError('Rule does not support negative indices.')
        if rule.max() >= self.factors.shape[0]:
            raise ValueError('Rule points to at least one index that is '
                             'out of bounds.')
        self.__rule = rule

    @property
    def weights(self):
        return self.__weights
    
    @weights.setter
    def weights(self, weights):
        '''Check shape.'''
        
        if weights.shape[1:] != (3, self.npol):
            raise ValueError('Shape[1:] of weights is {}, expected {}.'
                             .format(weights.shape[1:], (3, self.npol)))
        self.__weights = weights

    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self, name):
        '''Test for empty string.'''
        
        errmsg = ('Shape name "{}" is not an identifiable string'.
                  format(name))
        try:
            if not name.strip():
                raise ValueError(errmsg)
        except AttributeError as e:
            raise ValueError(errmsg) from e
        
        self.__name = name
                
    def _interp_factors(self, factors):
        '''
        Return factors of reduced bispectrum interpolated 
        over all multipoles.

        Parameters
        ----------
        factors : (n, npol, nell_sparse) array
            Factors of reduced bispectrum to be interpolated.

        Returns
        -------
        factors_full : (n, npol, nell) array
            Input interpolated over multipole.
        '''

        cs = CubicSpline(self.ells_sparse, factors, axis=2)
        return cs(self.ells_full)

    # Use name as prefix..
    def write_red_bisp(self, filename):
        '''
        Write the reduced bispectrum to disk.

        Parameters
        ----------
        filename : str
            Absolute path to output file.
        '''

        with h5py.File(filename + '.hdf5', 'w') as f:
            f.create_dataset('red_bisp', data=self.red_bisp['red_bisp'])
            f.create_dataset('radii', data=self.red_bisp['radii'])
            f.create_dataset('ells', data=self.red_bisp['ells'])
            
    # NOTE, how to do this..
    # Put this I/O stuff in ReducedBispectrum
    # save and load the processed bispectrum.
    # filename could use name attribute.

    # Like a static method that return an instance of the class?
    # like this
    # @classmethod
    # def init_from_file(cls, files)
    #     read and parse files
    #     return cls(parsed_files)
    #
    # you can then just do red_bisp = ReducedBispectrum.init_from_file(...)        
    def read_red_bisp(self, filename):
        '''
        Read in reduced bispectrum file and populate red_bisp attribute.

        Parameters
        ----------
        filename : str
            Absolute path to reduced bispectrum file.
        '''

        # now needs to save the attributes if ReducedBispectrum class.
        
        self.red_bisp = {}

        with h5py.File(filename + '.hdf5', 'r') as f:
            self.red_bisp['red_bisp'] = f['red_bisp'][()]
            self.red_bisp['radii'] = f['radii'][()]
            self.red_bisp['ells'] = f['ells'][()]

    
    # in estimator you can also concatenate X, Y and Z
    # and do unique on that one. Because eq. 74 is the same for
    # X, Y, Z. Otherwise for eg orthogonal rule: (1,1,0), (2,2,2), (1,2,3)
    # you would do 1+2+1, 1+2+2, 0+2+3. I.e you repeat 1 twice, 2 three times.
    # so if you concatenate you get back at alpha, beta, gamma, delta.
    # then you create an (nfact, 3) int array that tells you which i index in
    # X_(i,phi) you pick for X, Y and Z in eq. 75.

# reduced bispectrum class?
# so that cosmo class can have list of bispectrum instances.
# name
# radii
# ells sparse
# ells
# method for interp
# method for reading/writing
# Store in shape that makes sense for prim and secondary.
