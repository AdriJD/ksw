import unittest
import numpy as np
from scipy.special import spherical_jn
import os

import camb

from ksw import Cosmology, Shape

class TestTools(unittest.TestCase):

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
        self.assertEqual(cosmo.camb_params.Accuracy.AccuracyBoost, 3)
        self.assertEqual(cosmo.camb_params.Accuracy.lSampleBoost, 2)
        self.assertEqual(cosmo.camb_params.Accuracy.lAccuracyBoost, 2)
        self.assertIs(cosmo.camb_params.Accuracy.AccurateBB, True)
        self.assertIs(cosmo.camb_params.Accuracy.AccurateReionization, True)
        self.assertIs(cosmo.camb_params.Accuracy.AccuratePolarization, True)
    
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

    def test_cosmology_calc_transfer(self):

        lmax = 300
        pars = camb.CAMBparams(**self.cosmo_opts)

        cosmo = Cosmology(pars)

        cosmo.calc_transfer(lmax)

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

    def test_cosmology_calc_transfer_err_value(self):

        lmax = 299 # Too low value.
        pars = camb.CAMBparams(**self.cosmo_opts)

        cosmo = Cosmology(pars)

        self.assertRaises(ValueError, cosmo.calc_transfer, lmax)

    def test_cosmology_calc_cls(self):

        lmax = 450
        pars = camb.CAMBparams(**self.cosmo_opts)

        cosmo = Cosmology(pars)

        cosmo.calc_transfer(lmax)
        cosmo.calc_cls()

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
        
        b_ell_r = np.ones((nr, nell, npol, ncomp), dtype=float)
        b_ell_r *= (
            np.sin(0.1 * ells_sparse)[np.newaxis,:,np.newaxis,np.newaxis])
        
        b_ell_r_full = Cosmology._interp_reduced_bispec_over_ell(
            b_ell_r, ells_sparse, ells_full)

        nell_full = ells_full.size

        self.assertEqual(b_ell_r_full.shape, (nr, nell_full, npol, ncomp))
        
        b_ell_r_full_expec = np.ones((nr, nell_full, npol, ncomp))
        b_ell_r_full_expec *= (
            np.sin(0.1 * ells_full)[np.newaxis,:,np.newaxis,np.newaxis])

        np.testing.assert_almost_equal(b_ell_r_full, b_ell_r_full_expec,
                                       decimal=2)
                        
    def test_cosmology_calc_reduced_bispectrum(self):
        
        lmax = 450
        radii = np.asarray([11000., 14000.])
        pars = camb.CAMBparams(**self.cosmo_opts)

        cosmo = Cosmology(pars)
        cosmo.calc_transfer(lmax)
        
        funcs, rule, amps = Shape.prim_local(ns=1)        
        local = Shape(funcs, rule, amps)

        cosmo.calc_reduced_bispectrum(local, radii)

        # check cshape of attribute
        # check dtype of attribute
        nell = lmax - 1 # b_ell_r starts from ell=2.
        nr = len(radii)
        ncomp = len(funcs)
        npol = 2
        self.assertEqual(cosmo.b_ell_r.shape, (nr, nell, npol, ncomp))
        self.assertEqual(cosmo.b_ell_r.dtype, float)
        
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

        ans = cosmo.b_ell_r[ridx,lidx_full,pidx, cidx]
        self.assertAlmostEqual(ans, ans_expec, places=6)

