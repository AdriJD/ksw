import unittest
import numpy as np
from scipy.special import spherical_jn
import os
import tempfile
import pathlib
import json

import camb

from ksw import Cosmology, Shape

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

    def test_cosmology_compute_cls(self):

        lmax = 450
        pars = camb.CAMBparams(**self.cosmo_opts)

        cosmo = Cosmology(pars)

        cosmo.compute_transfer(lmax)
        cosmo.compute_cls()

        ells_unlensed = cosmo.cls['unlensed_scalar']['ells']
        np.testing.assert_equal(ells_unlensed, np.arange(lmax+1,dtype=int))
        ells_lensed = cosmo.cls['lensed_scalar']['ells']
        np.testing.assert_equal(ells_lensed, np.arange(ells_lensed.size,
                                                       dtype=int))
        cls_unlensed = cosmo.cls['unlensed_scalar']['cls']
        self.assertEqual(cls_unlensed.shape, (lmax+1, 4))
        self.assertEqual(cls_unlensed.dtype, float)

        cls_lensed = cosmo.cls['lensed_scalar']['cls']
        self.assertEqual(cls_unlensed.shape, (lmax+1, 4))
        self.assertEqual(cls_lensed.dtype, float)

        # EE amplitude at ell=300 is approximately 50e-5 uK^2.
        self.assertTrue(40e-5 < cls_unlensed[300,1] < 60e-5)
        self.assertTrue(40e-5 < cls_lensed[300,1] < 60e-5)

        # BB amplitude should be zero for unlensed and
        # approx 5 uk-arcmin rms for large-scale lensed BB.
        np.testing.assert_almost_equal(cls_unlensed[:,2],
                                       np.zeros(lmax+1))
        # Cls are in uK^2 radians^2, so convert arcmin to rad.
        exp_BB_amp = np.radians(5 / 60.) ** 2
        exp_BB_cl = np.ones(ells_lensed.size) * exp_BB_amp
        exp_BB_cl[:2] = 0. # Monopole and dipole.
        np.testing.assert_almost_equal(cls_lensed[:,2],
                                       exp_BB_cl, decimal=1)

    def test_cosmology_interp_reduced_bispec_over_ell(self):

        ells_sparse = np.asarray([0, 5, 10, 15, 20, 25, 30])
        ells_full = np.arange(0, 31)

        nr = 3
        nell = ells_sparse.size
        npol = 2
        ncomp = 1

        red_bisp = np.ones((nr, nell, npol, ncomp), dtype=float)
        red_bisp *= (
            np.sin(0.1 * ells_sparse)[np.newaxis,:,np.newaxis,np.newaxis])

        red_bisp_full = Cosmology._interp_reduced_bispec_over_ell(
            red_bisp, ells_sparse, ells_full)

        nell_full = ells_full.size

        self.assertEqual(red_bisp_full.shape, (nr, nell_full, npol, ncomp))

        red_bisp_full_expec = np.ones((nr, nell_full, npol, ncomp))
        red_bisp_full_expec *= (
            np.sin(0.1 * ells_full)[np.newaxis,:,np.newaxis,np.newaxis])

        np.testing.assert_almost_equal(red_bisp_full, red_bisp_full_expec,
                                       decimal=2)

    def test_cosmology_compute_prim_reduced_bispectrum(self):

        lmax = 300
        radii = np.asarray([11000., 14000.])
        pars = camb.CAMBparams(**self.cosmo_opts)

        cosmo = Cosmology(pars)
        cosmo.compute_transfer(lmax)

        local = Shape.prim_local(ns=1)

        cosmo.compute_prim_reduced_bispectrum(local, radii)

        # Check shape of attribute.
        # Check dtype of attribute.
        nell = lmax - 1 # red_bisp starts from ell=2.
        nr = len(radii)
        ncomp = len(local.funcs)
        npol = 2
        self.assertEqual(cosmo.red_bisp['red_bisp'].shape,
                         (nr, nell, npol, ncomp))
        self.assertEqual(cosmo.red_bisp['red_bisp'].dtype, float)

        ells = np.arange(2, lmax+1)
        np.testing.assert_equal(cosmo.red_bisp['ells'], ells)
        np.testing.assert_almost_equal(cosmo.red_bisp['radii'], radii)

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
        cidx = 1

        integrand = k ** 2 * spherical_jn(ell, radius * k)
        integrand *= tr_ell_k[lidx,:,pidx] * func
        ans_expec = (2 / np.pi) * np.trapz(integrand, k)

        lidx_full = ell - 2

        ans = cosmo.red_bisp['red_bisp'][ridx,lidx_full,pidx, cidx]
        self.assertAlmostEqual(ans, ans_expec, places=6)

class TestCosmoIO(unittest.TestCase):

    def setUp(self):

        # Get location of this script.
        self.path = pathlib.Path(__file__).parent.absolute()

        # Cosmo instance with transfer, cls and camb params.
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

        cls_lensed = np.ones((nell, 4), dtype=float)
        cls_unlensed = np.ones((nell, 4), dtype=float) * 2.
        cosmo.cls = {'lensed_scalar' : {'ells' : ells,
                                        'cls' : cls_lensed},
                     'unlensed_scalar' : {'ells' : ells,
                                          'cls' : cls_unlensed}}
        cosmo.red_bisp = {'red_bisp' : red_bisp,
                          'radii' : radii,
                          'ells' : ells}

        self.cosmo = cosmo

    def tearDown(self):
        # Is called after each test.
        pass

    def test_read_write_transfer(self):

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

    def test_read_write_cls(self):

        pars = camb.CAMBparams(**self.cosmo_opts)
        cosmo_new = Cosmology(pars)

        with tempfile.TemporaryDirectory(dir=self.path) as tmpdirname:

            filename = os.path.join(tmpdirname, 'cls')
            self.cosmo.write_cls(filename)

            cosmo_new.read_cls(filename)

            np.testing.assert_almost_equal(
                cosmo_new.cls['lensed_scalar']['cls'],
                self.cosmo.cls['lensed_scalar']['cls'])

            np.testing.assert_equal(
                cosmo_new.cls['lensed_scalar']['ells'],
                self.cosmo.cls['lensed_scalar']['ells'])

            np.testing.assert_almost_equal(
                cosmo_new.cls['unlensed_scalar']['cls'],
                self.cosmo.cls['unlensed_scalar']['cls'])

            np.testing.assert_equal(
                cosmo_new.cls['unlensed_scalar']['ells'],
                self.cosmo.cls['unlensed_scalar']['ells'])

    def test_read_write_red_bisp(self):

        pars = camb.CAMBparams(**self.cosmo_opts)
        cosmo_new = Cosmology(pars)

        with tempfile.TemporaryDirectory(dir=self.path) as tmpdirname:

            filename = os.path.join(tmpdirname, 'red_bisp')
            self.cosmo.write_red_bisp(filename)

            cosmo_new.read_red_bisp(filename)

            np.testing.assert_almost_equal(
                cosmo_new.red_bisp['red_bisp'],
                self.cosmo.red_bisp['red_bisp'])

            np.testing.assert_almost_equal(
                cosmo_new.red_bisp['radii'],
                self.cosmo.red_bisp['radii'])

            np.testing.assert_equal(
                cosmo_new.red_bisp['ells'],
                self.cosmo.red_bisp['ells'])

    def test_write_camb_params(self):

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
