import unittest
import numpy as np
from scipy.special import spherical_jn
import os
import tempfile
import pathlib
import json

import camb

from ksw import Cosmology, Shape, ReducedBispectrum

class TestCosmo(unittest.TestCase):

    def setUp(self):
        # Is called before each test.

        self.cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
    def tearDown(self):
        # Is called after each test.
        pass

    def test_cosmology_init(self):

        pars = camb.CAMBparams(**self.cosmo_opts)

        cosmo = Cosmology(pars)

        self.assertIs(cosmo.camb_params.WantTensors, False)
        self.assertIs(cosmo.camb_params.DoLateRadTruncation, False)
        self.assertIs(cosmo.camb_params.WantCls, True)
        self.assertIs(cosmo.camb_params.WantTransfer, False)
        self.assertIs(cosmo.camb_params.DoLensing, True)
        self.assertEqual(cosmo.camb_params.Accuracy.AccuracyBoost, 2)
        self.assertEqual(cosmo.camb_params.Accuracy.lSampleBoost, 2)
        self.assertEqual(cosmo.camb_params.Accuracy.lAccuracyBoost, 2)
        self.assertIs(cosmo.camb_params.Accuracy.AccurateBB, True)
        self.assertIs(cosmo.camb_params.Accuracy.AccurateReionization, True)
        self.assertEqual(cosmo.camb_params.Accuracy.BessIntBoost, 30)
        self.assertEqual(cosmo.camb_params.Accuracy.KmaxBoost, 3)
        self.assertEqual(cosmo.camb_params.Accuracy.IntTolBoost, 4)
        self.assertEqual(cosmo.camb_params.Accuracy.TimeStepBoost, 4)
        self.assertEqual(cosmo.camb_params.Accuracy.SourcekAccuracyBoost, 5)
        self.assertEqual(cosmo.camb_params.Accuracy.BesselBoost, 5)
        self.assertEqual(cosmo.camb_params.Accuracy.IntkAccuracyBoost, 5)
        self.assertEqual(cosmo.camb_params.Accuracy.lSampleBoost, 2)
        self.assertEqual(cosmo.camb_params.Accuracy.IntkAccuracyBoost, 5)

        self.assertEqual(cosmo.transfer, {})
        self.assertEqual(cosmo.c_ell, {})
        self.assertEqual(cosmo.red_bispectra, [])

    def test_cosmology_init_omk(self):

        self.cosmo_opts['omk'] = 1.
        pars = camb.CAMBparams(**self.cosmo_opts)

        self.assertRaises(ValueError, Cosmology, pars)

    def test_cosmology_setattr_camb(self):

        pars = camb.CAMBparams(**self.cosmo_opts)
        self.assertTrue(pars.validate())
        self.assertEqual(pars.WantTensors, False)

        cosmo = Cosmology(pars)

        self.assertEqual(cosmo.camb_params.WantTensors, False)
        cosmo._setattr_camb('WantTensors', True)
        self.assertEqual(cosmo.camb_params.WantTensors, True)

    def test_cosmology_setattr_camb_acc(self):

        pars = camb.CAMBparams(**self.cosmo_opts)
        self.assertTrue(pars.validate())

        cosmo = Cosmology(pars)

        self.assertEqual(cosmo.camb_params.Accuracy.lSampleBoost, 2)
        cosmo._setattr_camb('lSampleBoost', 3, subclass='Accuracy')
        self.assertEqual(cosmo.camb_params.Accuracy.lSampleBoost, 3)

    def test_cosmology_setattr_camb_attr_err(self):

        pars = camb.CAMBparams(**self.cosmo_opts)
        self.assertTrue(pars.validate())

        cosmo = Cosmology(pars)

        self.assertRaises(AttributeError, cosmo._setattr_camb,
                          'NonExistingParam', 100)

    def test_cosmology_compute_transfer(self):

        lmax = 300
        pars = camb.CAMBparams(**self.cosmo_opts)

        cosmo = Cosmology(pars)

        cosmo.compute_transfer(lmax)

        ells = cosmo.transfer['ells']
        k = cosmo.transfer['k']
        tr_ell_k = cosmo.transfer['tr_ell_k']

        npol = 2
        nell = ells.size
        nk = k.size

        self.assertEqual(tr_ell_k.shape, (nell, nk, npol))
        self.assertTrue(tr_ell_k.flags['C_CONTIGUOUS'])
        self.assertTrue(tr_ell_k.flags['OWNDATA'])
        self.assertEqual(ells[0], 2)
        self.assertEqual(ells[-1], lmax)

    def test_cosmology_compute_transfer_err_value(self):

        lmax = 299 # Too low value.
        pars = camb.CAMBparams(**self.cosmo_opts)

        cosmo = Cosmology(pars)

        self.assertRaises(ValueError, cosmo.compute_transfer, lmax)

    def test_cosmology_compute_c_ell(self):

        lmax = 450
        pars = camb.CAMBparams(**self.cosmo_opts)

        cosmo = Cosmology(pars)

        cosmo.compute_transfer(lmax)
        cosmo.compute_c_ell()

        ells_unlensed = cosmo.c_ell['unlensed_scalar']['ells']
        np.testing.assert_equal(ells_unlensed, np.arange(lmax+1,dtype=int))
        ells_lensed = cosmo.c_ell['lensed_scalar']['ells']
        np.testing.assert_equal(ells_lensed, np.arange(ells_lensed.size,
                                                       dtype=int))
        c_ell_unlensed = cosmo.c_ell['unlensed_scalar']['c_ell']
        self.assertEqual(c_ell_unlensed.shape, (lmax+1, 4))
        self.assertEqual(c_ell_unlensed.dtype, float)

        c_ell_lensed = cosmo.c_ell['lensed_scalar']['c_ell']
        self.assertEqual(c_ell_unlensed.shape, (lmax+1, 4))
        self.assertEqual(c_ell_lensed.dtype, float)

        # EE amplitude at ell=300 is approximately 50e-5 uK^2.
        self.assertTrue(40e-5 < c_ell_unlensed[300,1] < 60e-5)
        self.assertTrue(40e-5 < c_ell_lensed[300,1] < 60e-5)

        # BB amplitude should be zero for unlensed and
        # approx 5 uk-arcmin rms for large-scale lensed BB.
        np.testing.assert_almost_equal(c_ell_unlensed[:,2],
                                       np.zeros(lmax+1))
        # Cls are in uK^2 radians^2, so convert arcmin to rad.
        exp_BB_amp = np.radians(5 / 60.) ** 2
        exp_BB_cl = np.ones(ells_lensed.size) * exp_BB_amp
        exp_BB_cl[:2] = 0. # Monopole and dipole.
        np.testing.assert_almost_equal(c_ell_lensed[:,2],
                                       exp_BB_cl, decimal=1)

    def test_cosmology_compute_transfer_power(self):

        # Check if transfer functions obey Eq. C5 in the BTT paper.

        lmax = 300
        pars = camb.CAMBparams(**self.cosmo_opts) # add ns=1.

        cosmo = Cosmology(pars)
        cosmo._setattr_camb('ns', 1., subclass='InitPower')
        cosmo.compute_transfer(lmax)
        cosmo.compute_c_ell()

        ells = cosmo.transfer['ells']
        k = cosmo.transfer['k']
        tr = cosmo.transfer['tr_ell_k']

        p_k = 2 * np.pi ** 2 * cosmo.camb_params.InitPower.As * k ** (-3)

        c_ell = np.zeros((ells.size, 4))
        
        for lidx in range(ells.size):
            
            # TT.
            c_ell[lidx,0] = np.trapz(k ** 2 * p_k * tr[lidx,:,0] ** 2, k)

            # EE.
            c_ell[lidx,1]  = np.trapz(k ** 2 * p_k * tr[lidx,:,1] ** 2, k)

            # TE.
            c_ell[lidx,3] = np.trapz(k ** 2 * p_k * tr[lidx,:,0] * tr[lidx,:,1], k)

        c_ell *= (2 / np.pi)

        ells_unlensed = cosmo.c_ell['unlensed_scalar']['ells']
        c_ell_nonlensed = cosmo.c_ell['unlensed_scalar']['c_ell']
        lmin = ells_unlensed[0]
        c_ell_exp = c_ell_nonlensed[ells-lmin,:]

        np.testing.assert_array_almost_equal(c_ell, c_ell_exp)

    def test_cosmology_compute_transfer_power_ns(self):

        # Check if transfer functions obey Eq. C5 in the BTT paper.
        # Add ns. 

        ns = 0.6
        pivot = 0.05

        lmax = 300
        pars = camb.CAMBparams(**self.cosmo_opts)

        cosmo = Cosmology(pars)
        cosmo._setattr_camb('ns', ns, subclass='InitPower')
        cosmo._setattr_camb('pivot_scalar', pivot, subclass='InitPower')
        cosmo.compute_transfer(lmax)
        cosmo.compute_c_ell()

        ells = cosmo.transfer['ells']
        k = cosmo.transfer['k']
        tr = cosmo.transfer['tr_ell_k']

        p_k = 2 * np.pi ** 2 * cosmo.camb_params.InitPower.As \
              * k ** (-3) * (k / pivot) ** (ns - 1)

        c_ell = np.zeros((ells.size, 4))
        
        for lidx in range(ells.size):
            
            # TT.
            c_ell[lidx,0] = np.trapz(k ** 2 * p_k * tr[lidx,:,0] ** 2, k)

            # EE.
            c_ell[lidx,1]  = np.trapz(k ** 2 * p_k * tr[lidx,:,1] ** 2, k)

            # TE.
            c_ell[lidx,3] = np.trapz(k ** 2 * p_k * tr[lidx,:,0] * tr[lidx,:,1], k)

        c_ell *= (2 / np.pi)

        ells_unlensed = cosmo.c_ell['unlensed_scalar']['ells']
        c_ell_nonlensed = cosmo.c_ell['unlensed_scalar']['c_ell']
        lmin = ells_unlensed[0]
        c_ell_exp = c_ell_nonlensed[ells-lmin,:]

        np.testing.assert_array_almost_equal(c_ell, c_ell_exp)

    def test_cosmology_add_prim_reduced_bispectrum(self):

        lmax = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        pars = camb.CAMBparams(**self.cosmo_opts)

        cosmo = Cosmology(pars)
        cosmo.compute_transfer(lmax)

        prim_shape = Shape.prim_equilateral(ns=1)

        self.assertTrue(len(cosmo.red_bispectra) == 0)
        cosmo.add_prim_reduced_bispectrum(prim_shape, radii)
        self.assertTrue(len(cosmo.red_bispectra) == 1)

        red_bisp = cosmo.red_bispectra[0]

        nell = lmax - 1 # Reduced Bispectrum starts from ell=2.
        nr = len(radii)
        ncomp = len(prim_shape.funcs)
        npol = 2
        nfact = nr * len(prim_shape.rule)

        self.assertEqual(red_bisp.factors.shape,
                         (ncomp * nr, npol, nell))
        self.assertEqual(red_bisp.factors.dtype, float)

        self.assertEqual(red_bisp.rule.shape, (nfact, 3))
        self.assertEqual(red_bisp.rule.dtype, int)

        self.assertEqual(red_bisp.weights.shape, (nfact, 3, npol))
        self.assertEqual(red_bisp.weights.dtype, float)

        ells = np.arange(2, lmax+1)
        np.testing.assert_equal(red_bisp.ells_full, ells)

        # Equilateral rule = [(1,1,0), (3,3,3), (1,2,3)].
        rule_exp = np.asarray([[2, 2, 0], # r[0].
                               [3, 3, 1], # r[1].
                               [6, 6, 6],
                               [7, 7, 7],
                               [2, 4, 6],
                               [3, 5, 7]], dtype=int)
        np.testing.assert_array_equal(red_bisp.rule, rule_exp)

        weights_exp = np.ones((nfact, 3, npol))
        # nfact is ordered like flattened (nprim, radii) array.
        # I assume equilateral shape here, so nprim = 3.
        weights_exp[0,...] = (dr * radii[0] ** 2) ** (1 / 3)
        weights_exp[1,...] = (dr * radii[1] ** 2) ** (1 / 3)
        weights_exp[2,...] = (dr * radii[0] ** 2) ** (1 / 3)
        weights_exp[3,...] = (dr * radii[1] ** 2) ** (1 / 3)
        weights_exp[4,...] = (dr * radii[0] ** 2) ** (1 / 3)
        weights_exp[5,...] = (dr * radii[1] ** 2) ** (1 / 3)

        weights_exp *= (2 * (2 * np.pi ** 2 * cosmo.camb_params.InitPower.As) ** 2 \
                        * (3 / 5)) ** (1 / 3)

        signs = np.sign(prim_shape.amps)
        weights_exp[0:2] *= signs[0] * np.abs(prim_shape.amps[0]) ** (1 / 3)
        weights_exp[2:4] *= signs[1] * np.abs(prim_shape.amps[1]) ** (1 / 3)
        weights_exp[4:6] *= signs[2] * np.abs(prim_shape.amps[2]) ** (1 / 3)

        np.testing.assert_array_almost_equal(red_bisp.weights, weights_exp)

        # Manually compute reduced bispec factors for given r, ell.
        k = cosmo.transfer['k']
        tr_ell_k = cosmo.transfer['tr_ell_k'] # (nell, nk, npol).
        ells_sparse = cosmo.transfer['ells']

        lidx = 40 # Randomly picked.
        pidx = 1 # Look at E-mode.
        ridx = 1
        ell = ells_sparse[lidx]
        radius = radii[ridx]
        func = k ** -3 # Look at second shape function.
        cidx = 1 # Second shape function.

        integrand = k ** 2 * spherical_jn(ell, radius * k)
        integrand *= tr_ell_k[lidx,:,pidx] * func
        ans_expec = (2 / np.pi) * np.trapz(integrand, k)

        lidx_full = ell - 2
        ans = red_bisp.factors[cidx*nr+ridx,pidx,lidx_full]
        self.assertAlmostEqual(ans, ans_expec, places=6)

class TestCosmoIO(unittest.TestCase):

    def setUp(self):

        # Get location of this script.
        self.path = pathlib.Path(__file__).parent.absolute()

        # Cosmo instance with transfer, c_ell and camb params.
        self.cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)

        pars = camb.CAMBparams(**self.cosmo_opts)
        cosmo = Cosmology(pars)

        nell = 4
        nk = 3
        npol = 2
        ncomp = 2
        nr = 3
        tr_ell_k = np.ones((nell, nk, npol), dtype=float)
        k = np.arange(nk, dtype=float)
        ells = np.arange(nell, dtype=int)
        radii = np.arange(nr, dtype=float)
        red_bisp =  np.ones((nr, nell, npol, ncomp), dtype=float)

        cosmo.transfer = {'tr_ell_k' : tr_ell_k,
                          'k' : k, 'ells' : ells}

        c_ell_lensed = np.ones((nell, 4), dtype=float)
        c_ell_unlensed = np.ones((nell, 4), dtype=float) * 2.
        cosmo.c_ell = {'lensed_scalar' : {'ells' : ells,
                                        'c_ell' : c_ell_lensed},
                     'unlensed_scalar' : {'ells' : ells,
                                          'c_ell' : c_ell_unlensed}}
        cosmo.red_bisp = {'red_bisp' : red_bisp,
                          'radii' : radii,
                          'ells' : ells}

        self.cosmo = cosmo

    def tearDown(self):
        # Is called after each test.
        pass

    def test_cosmo_io_read_write_transfer(self):

        pars = camb.CAMBparams(**self.cosmo_opts)
        cosmo_new = Cosmology(pars)

        with tempfile.TemporaryDirectory(dir=self.path) as tmpdirname:

            filename = os.path.join(tmpdirname, 'transfer')
            self.cosmo.write_transfer(filename)

            cosmo_new.read_transfer(filename)

            np.testing.assert_almost_equal(
                cosmo_new.transfer['tr_ell_k'],
                self.cosmo.transfer['tr_ell_k'])

            np.testing.assert_almost_equal(
                cosmo_new.transfer['k'],
                self.cosmo.transfer['k'])

            np.testing.assert_equal(
                cosmo_new.transfer['ells'],
                self.cosmo.transfer['ells'])

    def test_cosmo_io_read_write_c_ell(self):

        pars = camb.CAMBparams(**self.cosmo_opts)
        cosmo_new = Cosmology(pars)

        with tempfile.TemporaryDirectory(dir=self.path) as tmpdirname:

            filename = os.path.join(tmpdirname, 'c_ell')
            self.cosmo.write_c_ell(filename)

            cosmo_new.read_c_ell(filename)

            np.testing.assert_almost_equal(
                cosmo_new.c_ell['lensed_scalar']['c_ell'],
                self.cosmo.c_ell['lensed_scalar']['c_ell'])

            np.testing.assert_equal(
                cosmo_new.c_ell['lensed_scalar']['ells'],
                self.cosmo.c_ell['lensed_scalar']['ells'])

            np.testing.assert_almost_equal(
                cosmo_new.c_ell['unlensed_scalar']['c_ell'],
                self.cosmo.c_ell['unlensed_scalar']['c_ell'])

            np.testing.assert_equal(
                cosmo_new.c_ell['unlensed_scalar']['ells'],
                self.cosmo.c_ell['unlensed_scalar']['ells'])

    def test_cosmo_io_write_camb_params(self):

        with tempfile.TemporaryDirectory(dir=self.path) as tmpdirname:

            filename = os.path.join(tmpdirname, 'camb_params')
            self.cosmo.write_camb_params(filename)

            with open(filename + '.json') as f:
                params = json.load(f)

        params_new = camb.CAMBparams()

        # Perhaps put this in method of cosmo, and let cosmo.__init__
        # accept a filename as param as well.

        for key in params:
            if type(params[key]) is dict: # i.e. camb.model.CAMB_Structure.
                subclass = getattr(params_new, key)
                for subkey in params[key]:
                    try:
                        setattr(subclass, subkey, params[key][subkey])
                    except AttributeError as e:
                        # Test if property without setter.
                        if not hasattr(subclass, subkey):
                            raise e

            else:
                try:
                    setattr(params_new, key, params[key])
                except AttributeError as e:
                    # Test if property without setter.
                    if not hasattr(subclass, subkey):
                        raise e

        # Check if new CAMB parameter object is equal to original.
        self.assertIs(self.cosmo.camb_params.diff(params_new), None)

    def test_cosmology_add_reduced_bispectrum_from_file(self):

        pars = camb.CAMBparams(**self.cosmo_opts)
        cosmo = Cosmology(pars)

        n_unique = 2
        nfact = 4
        npol = 2
        ells_sparse = np.asarray([3, 5, 7, 10, 13, 15])
        ells_full = np.arange(3, 16)

        weights = np.ones((nfact, 3, npol))
        rule = np.ones((nfact, 3), dtype=int)
        factors = np.ones((n_unique, npol, ells_sparse.size))
        name = 'test_bispec'

        rb = ReducedBispectrum(factors, rule, weights, ells_sparse, name)

        with tempfile.TemporaryDirectory(dir=self.path) as tmpdirname:

            filename = os.path.join(tmpdirname, 'red_bisp')
            rb.write(filename)

            cosmo.add_reduced_bispectrum_from_file(filename)

        rb2 = cosmo.red_bispectra[0]
        np.testing.assert_array_almost_equal(rb2.factors, rb.factors)
        np.testing.assert_array_equal(rb2.rule, rb.rule)
        np.testing.assert_array_almost_equal(rb2.weights, rb.weights)
        np.testing.assert_array_equal(rb2.ells_full, ells_full)
        np.testing.assert_array_equal(rb2.ells_sparse, ells_full)
        self.assertEqual(rb2.name, rb.name)
