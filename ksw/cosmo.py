import numpy as np
from scipy.interpolate import CubicSpline
import inspect
import json

import camb
import h5py

from ksw import utils, radial_functional as rf

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
    c_ell : dict
        Angular power spectra and metadata.
    red_bispectra : list
        Collection of ReducedBispectrum instances.
    '''

    def __init__(self, camb_params, verbose=False):

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
        self.c_ell = {}
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
        # Modify scalar E-mode, see Zaldarriaga 1997 Eqs. 18 and 39.
        # (CAMB applies these factors at a later stage).
        try:
            # CAMB changed this at least one time.
            # See Nov 19 CAMB commit: Formatting; py 3.8 test.
            ells = tr.L
        except AttributeError:
            ells = tr.l
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

    def compute_c_ell(self, lmax=None):
        '''
        Calculate angular power spectra (Cls) using precomputed
        transfer functions.

        Notes
        -----
        Spectra are CAMB output and are therefore column-major
        (nell, npol). The pol order is TT, EE, BB, TE for the CMB
        spectra and phiphi, phiT, phiE for the lenspotential. Nell can
        be different for each type of spectra. The monopole
        and dipole are included. Units are muK^2.
        '''

        self._camb_data.power_spectra_from_transfer()

        c_ell_unlensed_scalar = self._camb_data.get_unlensed_scalar_cls(
            lmax=lmax, CMB_unit='muK', raw_cl=True)

        c_ell_lensed_scalar = self._camb_data.get_lensed_scalar_cls(
            lmax=lmax, CMB_unit='muK', raw_cl=True)

        c_ell_lenspotential = self._camb_data.get_lens_potential_cls(
            lmax=lmax, CMB_unit='muK', raw_cl=True)
        
        ells_unlensed = np.arange(c_ell_unlensed_scalar.shape[0])
        ells_lensed = np.arange(c_ell_lensed_scalar.shape[0])
        ells_lenspotential = np.arange(c_ell_lenspotential.shape[0])        

        self.c_ell['unlensed_scalar'] = {}
        self.c_ell['unlensed_scalar']['ells'] = ells_unlensed
        self.c_ell['unlensed_scalar']['c_ell'] = c_ell_unlensed_scalar

        self.c_ell['lensed_scalar'] = {}
        self.c_ell['lensed_scalar']['ells'] = ells_lensed
        self.c_ell['lensed_scalar']['c_ell'] = c_ell_lensed_scalar

        self.c_ell['lenspotential'] = {}
        self.c_ell['lenspotential']['ells'] = ells_lenspotential
        self.c_ell['lenspotential']['c_ell'] = c_ell_lenspotential
        
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

        Notes
        -----
        We need to correct of the fact that the radiation transfer functions we
        extract from CAMB are for the curvature perturbation zeta instead of the
        Bardeen potential phi (see shape.py, we use phi to match the Planck 
        conventions). We also have to take into account that the scalar amplitude
        "As" extracted from CAMB is different than the value "A" in the Planck convention.

        During matter domination on superhorizon scales we have for adiabatic 
        perturbations: zeta = (5 / 3) phi. 

        Planck defines A as <phi_k1 phi_k2> = (2pi)^3 delta(k12) A / k^3.
        CAMB defines As as <zeta_k2 zeta_k2> = (2pi)^3 delta(k12) 2 * pi^2 As / k^3.
        So A = (3/5)^2 * 2 * pi^2 As.

        The radiation transfer functions have to be multiplied by (5/3) to interpret
        them as phi transfer functions. So the 3 factors of (5/3) from the 3 transfer
        functions in b_l1l2l3 partly cancel with the 4 factors of (3/5) from A^2.

        So finally, to get the 2 A_phi^2 amplitude specified in shape.py and convert
        zeta to phi we need to multiply our templates by:
        
        2 * (2 * pi^2)^2 * A_s^2 * (3/5)        
        '''

        tr_ell_k = self.transfer['tr_ell_k']
        k = self.transfer['k']
        ells_sparse = self.transfer['ells']

        f_k = prim_shape.get_f_k(k) 
        amps = np.asarray(prim_shape.amps)
        amps *= 2 * (2 * np.pi ** 2 * self.camb_params.InitPower.As) ** 2 * (3 / 5)

        # Call C code.
        red_bisp = rf.radial_func(f_k, tr_ell_k, k, radii, ells_sparse)

        factors, rule, weights = self._parse_prim_reduced_bispec(
            red_bisp, radii, prim_shape.rule, amps)

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
            Amplitude for each element in rule.

        Returns
        -------
        factors : (ncomp * nr, npol, nell) array
            Rearranged factors of reduced bispectrum.
        rule : (nfact, 3) int array
            Array of indices to first dimension of factors that form the
            reduced bispectrum.
        weights : (nfact, 3) float array
            (nprim * nr) weights (amp * r^2 dr) for each element in rule.
        '''

        nr, nell, npol, ncomp = red_bisp.shape
        nfact = nr * len(prim_rule)

        # Unique factors, just reshaped version of input reduced bispectrum.
        factors = np.ascontiguousarray(np.transpose(red_bisp, (3, 0, 2, 1)))
        factors = factors.reshape((ncomp * nr, npol, nell), order='C')

        # For primordial bispectra the last dim of weights is constant.
        weights = np.ones((nfact, 3))
        rule = np.ones((nfact, 3), dtype=int)

        dr = utils.get_trapz_weights(radii) * radii ** 2
        ridxs = np.arange(nr) # Indices to radii.
        start = 0
        for amp, ru in zip(amps, prim_rule):

            # Note, each term needs to be multiplied by number of distinct 
            # permutations in rule, so e.g. 3 for local. 
            nperm = self.num_permutations(ru)
            amp_per_r = (amp * dr * nperm)

            # Note each factor gets 1/3 power of overall amplitude such that 
            # fl1 * fl2 * fl3 has correct amplitude.

            # If amp * dr is negative you get complex answers to the 3rd root.
            # To extract the real value of the root (always exists for n=3), we 
            # take the root of the absolute value and apply the sign afterwards.

            signs = np.sign(amp_per_r)
            amp_per_r = signs * np.abs(amp_per_r) ** (1 / 3)
            weights[start:start+nr] = amp_per_r[:,np.newaxis]
            # Indices into first dim of factors (which is (ncomp, nr)).
            rule[start:start+nr,:] = ridxs[:,np.newaxis] + (np.asarray(ru) * nr)
            start += nr

        return factors, rule, weights

    def add_ttt_lensing_bispectrum(self):
        '''
        Compute the factors of the reduced bispectrum due to the correlation
        between the lensing potential phi and the late-time ISW effect and add
        to internal list of reduced bispectra.

        Notes
        -----
        This is not the full ISW-lensing bispectrum, because pol is ignored.

        Implements: B_l1l2l3^TTT = Cl2^Tphi * tilde(C)l3^TT fl1l2l3^T + 5 perm.
        see first line of Eq. 38 in Planck NG 2018.        
        '''

        ells = self.c_ell['lensed_scalar']['ells']
        assert np.allclose(ells, self.c_ell['lenspotential']['ells'])
        
        c_ell_tt = self.c_ell['lensed_scalar']['c_ell'][:,0]
        c_ell_tphi = self.c_ell['lenspotential']['c_ell'][:,1]

        factors = np.zeros((6, 2, ells.size))
        # We include the E part of the factors, but it is always zero.
        factors[0,0,:] = np.ones(ells.size)
        factors[1,0,:] = c_ell_tphi * ells * (ells + 1)
        factors[2,0,:] = c_ell_tt
        factors[3,0,:] = c_ell_tt * ells * (ells + 1)
        factors[4,0,:] = ells * (ells + 1)
        factors[5,0,:] = c_ell_tphi

        # Set all monopoles and dipoles to zero.
        factors[:,:,:2] = 0
        
        rule = np.asarray([[0, 1, 2], [0, 5, 3], [4, 5, 2]])        
        weights = np.asarray([[1., 1., 1.], [1., 1., 1.], [-1., -1., -1.]])
        # This is due to the 1/6 in the def of b_l1l2l3 and the factor 1/2 in f_l1l2l3        
        weights *= 3 ** (1 / 3) 
        name = 'ttt_lensing'

        self.red_bispectra.append(
            ReducedBispectrum(factors, rule, weights, ells, name))

    def add_lensing_bispectrum(self):
        '''
        Compute the factors of the reduced bispectrum due to the correlation
        between the lensing potential phi and the late-time ISW effect and
        reionization and add to internal list of reduced bispectra.

        Notes
        -----
        This implements the full lensing-ISW/reion bispectrum, i.e. both lines
        from Eq. 38 in Planck NG 2018.        
        '''

        ells = self.c_ell['lensed_scalar']['ells']
        assert np.allclose(ells, self.c_ell['lenspotential']['ells'])

        # Common terms that will appear in many of the factors.
        denom_fact = np.zeros(ells.size)
        denom_fact[2:] = 1 / np.sqrt((ells[2:] - 1) * (ells[2:] + 2) * ells[2:] * (ells[2:] + 1))
        num_fact = np.zeros_like(denom_fact)
        num_fact[2:] = np.sqrt(ells[2:] * (ells[2:] + 1) / ((ells[2:] - 1) * (ells[2:] + 1)))
        
        c_ell_tt = self.c_ell['lensed_scalar']['c_ell'][:,0]
        c_ell_ee = self.c_ell['lensed_scalar']['c_ell'][:,1]
        c_ell_te = self.c_ell['lensed_scalar']['c_ell'][:,3]
        
        c_ell_tphi = self.c_ell['lenspotential']['c_ell'][:,1]
        c_ell_ephi = self.c_ell['lenspotential']['c_ell'][:,2]        

        factors = np.zeros((20, 2, ells.size))
        # Start with the X1 = T case.
        # For the factors that do not contain C_ells the E index is always zero.
        factors[0,0,:] = np.ones(ells.size)
        
        factors[1,0,:] = c_ell_tphi * ells * (ells + 1)
        factors[1,1,:] = c_ell_ephi * ells * (ells + 1)        
        
        factors[2,0,:] = c_ell_tt
        factors[2,1,:] = c_ell_te
        
        factors[3,0,:] = c_ell_tt * ells * (ells + 1)
        factors[3,1,:] = c_ell_te * ells * (ells + 1)
        
        factors[4,0,:] = ells * (ells + 1)
        
        factors[5,0,:] = c_ell_tphi
        factors[5,1,:] = c_ell_ephi

        # Now the additional factors needed for the X1 = E case.
        # Here, factors without C_ells have zero for the T index.
        factors[6,1,:] = ells ** 2 * (ells + 1) ** 2 * denom_fact

        factors[7,0,:] = c_ell_te * denom_fact
        factors[7,1,:] = c_ell_ee * denom_fact
        
        factors[8,0,:] = ells * (ells + 1) * c_ell_te * denom_fact
        factors[8,1,:] = ells * (ells + 1) * c_ell_ee * denom_fact

        factors[9,1,:] = ells ** 3 * (ells + 1) ** 3 * denom_fact

        factors[10,1,:] = 1 * denom_fact
        
        factors[11,0,:] = ells ** 2 * (ells + 1) ** 2 * c_ell_te * denom_fact
        factors[11,1,:] = ells ** 2 * (ells + 1) ** 2 * c_ell_ee * denom_fact

        factors[12,0,:] = ells ** 3 * (ells + 1) ** 3 * c_ell_te * denom_fact
        factors[12,1,:] = ells ** 3 * (ells + 1) ** 3 * c_ell_ee * denom_fact

        factors[13,1,:] = ells * (ells + 1) * denom_fact

        factors[14,0,:] = ells ** 3 * (ells + 1) ** 3 * c_ell_tphi
        factors[14,1,:] = ells ** 3 * (ells + 1) ** 3 * c_ell_ephi

        factors[15,0,:] = ells ** 2 * (ells + 1) ** 2 * c_ell_tphi
        factors[15,1,:] = ells ** 2 * (ells + 1) ** 2 * c_ell_ephi

        factors[16,1,:] = num_fact
        
        factors[17,0,:] = num_fact * c_ell_te
        factors[17,1,:] = num_fact * c_ell_ee

        factors[18,0,:] = ells * (ells + 1) * num_fact * c_ell_te
        factors[18,1,:] = ells * (ells + 1) * num_fact * c_ell_ee

        factors[19,1,:] = ells * (ells + 1) * num_fact
        
        # Set all monopoles and dipoles to zero.
        factors[:,:,:2] = 0

        # Note, the fact that we have the same number of unique factors
        # and bispectrum termsn (nrule) is a concidence I think.
        rule = np.zeros((20, 3), dtype=np.int64)
        weights = np.ones((20, 3))

        rule[0] =  [0,  1,  2]
        rule[1] =  [0,  5,  3]
        rule[2] =  [4,  5,  2]
        rule[3] =  [6,  1,  7] #  1D.
        rule[4] =  [6,  5,  8] #  1E.
        rule[5] =  [9,  5,  7] #  1F.
        rule[6] =  [10, 1,  11] # 2D.
        rule[7] =  [10, 5,  12] # 2E.
        rule[8] =  [13, 5,  11] # 2F.
        rule[9] =  [10, 14, 7] #  3D.
        rule[10] = [10, 15, 8] #  3E.
        rule[11] = [13, 15, 7] #  3F.
        rule[12] = [13, 1,  8] #  4D.
        #rule[13] = [13, 5,  11] # 4E.
        #rule[14] = [6,  5,  8] #  4F.
        #rule[15] = [13, 15, 7] #  5D.
        #rule[16] = [13, 1,  8] #  5E.
        #rule[17] = [6,  1,  7] #  5F.
        #rule[18] = [10, 15, 8] #  6D.
        #rule[19] = [10, 1,  11] # 6E.
        #rule[20] = [13, 1,  8] #  6F.
        rule[13] = [13, 1,  7] #  7D.
        #rule[22] = [13, 5,  8] #  7E.
        rule[14] = [6,  5,  7] #  7F.
        #rule[24] = [10, 1,  8] #  8D.
        rule[15] = [10, 5,  11] # 8E
        #rule[26] = [13, 5,  8] #  8F.
        rule[16] = [10, 15, 7] #  9D.
        #rule[28] = [10, 1,  8] #  9E.
        #rule[29] = [13, 1,  7] #  9F.
        rule[17] = [16, 1,  17] # 10D.
        rule[18] = [16, 5,  18] # 10E.
        rule[19] = [19, 5,  17] # 10F.

        weights[0]  *= 3    ** (1 / 3)
        weights[1]  *= 3    ** (1 / 3)
        weights[2]  *= -3   ** (1 / 3)
        weights[3]  *= 4.5  ** (1 / 3) # 1D.
        weights[4]  *= -1.5 ** (1 / 3) # 1E.
        weights[5]  *= -1.5 ** (1 / 3) # 1F.
        weights[6]  *= -1.5 ** (1 / 3) # 2D.
        weights[7]  *= 1.5  ** (1 / 3) # 2E.
        weights[8]  *= 1.5  ** (1 / 3) # 2F.
        weights[9]  *= 1.5  ** (1 / 3) # 3D.
        weights[10] *= -1.5 ** (1 / 3) # 3E.
        weights[11] *= -4.5 ** (1 / 3) # 3F.
        weights[12] *= 3    ** (1 / 3) # 4D.
        #weights[13] *= 3    ** (1 / 3) # 4E.
        #weights[14] *= -3   ** (1 / 3) # 4F.
        #weights[15] *= -3   ** (1 / 3) # 5D.
        #weights[16] *= -3   ** (1 / 3) # 5E.
        #weights[17] *= 3    ** (1 / 3) # 5F.
        #weights[18] *= -3   ** (1 / 3) # 6D.
        #weights[19] *= -3   ** (1 / 3) # 6E.
        #weights[20] *= 3    ** (1 / 3) # 6F.
        weights[13] *= -6  ** (1 / 3) # 7D.
        #weights[22] *= -3   ** (1 / 3) # 7E.
        weights[14] *= 3   ** (1 / 3) # 7F.
        #weights[24] *= -3   ** (1 / 3) # 8D.
        weights[15] *= -3  ** (1 / 3) # 8E.
        #weights[26] *= 3    ** (1 / 3) # 8F.
        weights[16] *= 3   ** (1 / 3) # 9D.
        #weights[28] *= 3    ** (1 / 3) # 9E.
        #weights[29] *= -3   ** (1 / 3) # 9F.
        weights[17] *= -3  ** (1 / 3) # 10D.
        weights[18] *= -3  ** (1 / 3) # 10E.
        weights[19] *= 3   ** (1 / 3) # 10F.
                
        name = 'lensing'

        self.red_bispectra.append(
            ReducedBispectrum(factors, rule, weights, ells, name))
        
    @staticmethod
    def num_permutations(rule):
        '''
        Return number of distinct permutations of rule.

        Parameters
        ----------
        rule : (3,) array-like
            
        Returns
        -------
        n_perm : int
            1 if XXX, 3 if e.g. YXX, 6 if e.g. XYZ.

        Raises
        ------
        ValueError
            If rule has wrong length.
        '''

        if len(rule) != 3:
            raise ValueError('Length input rule must be 3, got {}'.format(len(rule)))

        n_uni = np.unique(rule).size

        return (n_uni ** 2 + n_uni) // 2

    def add_reduced_bispectrum_from_file(self, filename, comm=None):
        '''
        Load reduced bispectrum and add to internal list
        of reduced bispectra.

        Parameters
        ----------
        filename : str
            Absolute path to file.
        comm : MPI communicator, optional
            If provided, broadcast after load.
        '''

        if comm is None:
            comm = utils.FakeMPIComm()

        if comm.Get_rank() == 0:
            rb = ReducedBispectrum.init_from_file(filename)
        else:
            rb = None

        rb = utils.bcast(rb, comm)
        self.red_bispectra.append(rb)
        
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

    def write_c_ell(self, filename):
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
                    data=self.c_ell['lensed_scalar']['ells'])
            lens_scal.create_dataset('c_ell',
                    data=self.c_ell['lensed_scalar']['c_ell'])

            unlens_scal = f.create_group('unlensed_scalar')
            unlens_scal.create_dataset('ells',
                    data=self.c_ell['unlensed_scalar']['ells'])
            unlens_scal.create_dataset('c_ell',
                    data=self.c_ell['unlensed_scalar']['c_ell'])
            
            lenspotential = f.create_group('lenspotential')
            lenspotential.create_dataset('ells',
                    data=self.c_ell['lenspotential']['ells'])
            lenspotential.create_dataset('c_ell',
                    data=self.c_ell['lenspotential']['c_ell'])

    def read_c_ell(self, filename, comm=None):
        '''
        Read in c_ell file and populate c_ell attribute.

        Parameters
        ----------
        filename : str
            Absolute path to spectra file.
        comm : MPI communicator, optional
            If provided, broadcast after load.
        '''
        
        if comm is None:
            comm = utils.FakeMPIComm()

        if comm.Get_rank() == 0:
            self.c_ell = {}

            with h5py.File(filename + '.hdf5', 'r') as f:
                ells = f['lensed_scalar/ells'][()]
                c_ell = f['lensed_scalar/c_ell'][()]
                self.c_ell['lensed_scalar'] = {}
                self.c_ell['lensed_scalar']['ells'] = ells
                self.c_ell['lensed_scalar']['c_ell'] = c_ell

                ells = f['unlensed_scalar/ells'][()]
                c_ell = f['unlensed_scalar/c_ell'][()]
                self.c_ell['unlensed_scalar'] = {}
                self.c_ell['unlensed_scalar']['ells'] = ells
                self.c_ell['unlensed_scalar']['c_ell'] = c_ell

                ells = f['lenspotential/ells'][()]
                c_ell = f['lenspotential/c_ell'][()]
                self.c_ell['lenspotential'] = {}
                self.c_ell['lenspotential']['ells'] = ells
                self.c_ell['lenspotential']['c_ell'] = c_ell                
        else:
            self.c_ell = None
    
        self.c_ell = utils.bcast(self.c_ell, comm)
                            
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
    that consists of a sum of terms that are each separable in
    the l1, l2, l3 multipoles:

    b_l1_l2_l3 = (1/6) sum_i^nfact X(i)_l1 Y(i)_l2 Z(i)_l3 + 5 perm.

    Parameters
    ----------
    factors = (n, npol, nell_sparse)
        Unique f_ells (e.g. X(i)_ell, Y(i)_ell) that make up the reduced 
        bispectrum.
    rule : (nfact, 3) int array
        Indices to first dimension of unique factors array that
        create the (nfact, 3, npol, nell) reduced bispectrum.
    weights : (nfact, 3) float array
        Weights for each element in rule.
    ells : (nell_sparse) array
        Potentially sparse array of monotonicially increasing
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
    weights : (nfact, 3) float array
        Weights for each element in rule.
    ells_sparse : (nell_sparse) int array
        Potentially sparse array of monotonicially increasing
        multipoles.
    ells_full : (nell) int array
        Fully sampled array of multipoles.
    npol : int
        Number of polarizations.
    nfact : int
        Number of factored sums making up this reduced bispectrum.
    lmax : int
        Maximum multipole describing this reduced bispectrum.
    lmin : int
        Minimum multipole describing this reduced bispectrum.

    Raises
    ------
    ValueError
        If input shapes do not match.
        If rule refers to nonexisting factors.
        If name is not an identifiable string.
    TypeError
        If rule is not an array of integers.
        If rule contains repeated triplets.
    '''

    def __init__(self, factors, rule, weights, ells, name):

        self.ells_sparse = ells
        self.ells_full = np.arange(ells[0], ells[-1] + 1)
        self.factors = factors
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
        if np.unique(np.sort(rule, axis=1), axis=0).size != rule.size:
            raise ValueError('Rule cannot contain multiple equivalent triplets.')
        self.__rule = rule

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights):
        '''Check shape.'''

        if weights.shape[1:] != (3,):
            raise ValueError('Shape[1:] of weights is {}, expected (3,).'
                             .format(weights.shape[1:]))
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

    @property
    def npol(self):
        return self.factors.shape[1]

    @property
    def nfact(self):
        return self.rule.shape[0]

    @property
    def lmax(self):
        return self.ells_full[-1]

    @property
    def lmin(self):
        return self.ells_full[0]    
        
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

    def write(self, filename):
        '''
        Write the reduced bispectrum to disk.

        Parameters
        ----------
        filename : str
            Absolute path to output file.
        '''

        with h5py.File(filename + '.hdf5', 'w') as f:
            f.create_dataset('factors', data=self.factors)
            f.create_dataset('rule', data=self.rule)
            f.create_dataset('weights', data=self.weights)
            f.create_dataset('ells_full', data=self.ells_full)
            f.create_dataset('name', data=np.string_(self.name))

    @classmethod
    def init_from_file(cls, filename):
        '''
        Read reduced bispectrum and return class instance.

        Parameters
        ----------
        filename : str
            Absolute path to output file.
        '''

        with h5py.File(filename + '.hdf5', 'r') as f:
            factors = f['factors'][()]
            rule = f['rule'][()]
            weights = f['weights'][()]
            ells = f['ells_full'][()]
            name = f['name'][()].decode("utf-8") 

        return cls(factors, rule, weights, ells, name)
            
