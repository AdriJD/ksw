import unittest
import tempfile
import pathlib
import os
import numpy as np

import healpy as hp
import camb

from ksw import KSW
from ksw import Cosmology
from ksw import Shape

class TestKSW_64(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.precision = 'double'
        cls.decimal = 7

    def setUp(self):
        # Is called before each test.

        # Get location of this script.
        self.path = pathlib.Path(__file__).parent.absolute()

        class FakeReducedBispectrum():
            def __init__(self):

                self.npol = 2
                self.nfact = 4

                self.ells_sparse = np.arange(2, 10)
                self.ells_full = np.arange(2, 10)
                self.factors = np.ones((3, self.npol, len(self.ells_full)))
                self.rule = np.ones((self.nfact, 3), dtype=int)
                self.weights = np.ones((self.nfact, 3))
                self.name = 'test'

                self.lmax = self.ells_full[-1]
                self.lmin = self.ells_full[0]

        class FakeCosmology():
            def __init__(self):
                self.red_bispectra = [FakeReducedBispectrum()]

        class FakeData():
            def __init__(self):
                self.pol = ('T', 'E')
                self.npol = len(self.pol)
                self.lmax = 300
                self.cosmology = FakeCosmology()

            @property
            def b_ell(self):
                b_ell = np.ones((self.npol, self.lmax+1)) * 0.1
                return b_ell

            def icov_diag_nonlensed(self, alm):
                return alm

        def y00(theta, phi):
            return 1 / np.sqrt(4 * np.pi)

        def y10(theta, phi):
            return np.sqrt(3 / 4 / np.pi) * np.cos(theta)

        def y11(theta, phi):
            return -np.sqrt(3 / 8 / np.pi) * np.sin(theta) * np.exp(1j * phi)

        def y20(theta, phi):
            return np.sqrt(5 / 16 / np.pi) * (3 * np.cos(theta) ** 2 - 1)

        def y21(theta, phi):
            return -np.sqrt(15 / 8 / np.pi) * np.sin(theta) * np.cos(theta) * np.exp(1j * phi)

        def y22(theta, phi):
            return np.sqrt(15 / 32 / np.pi) * np.sin(theta) ** 2 * np.exp(2j * phi)

        def cubic_term_direct(alm1, alm2, alm3, red_bisp):
            '''
            Calculate cubic term directly.

            Parameters
            ----------
            alm1, alm2, alm3 : (nelem) array
                1D Healpix-ordered alm arrays
            red_bisp : callable
                Function with ell1, ell2, ell3 arguments returning reduced bispectrum.

            Returns
            -------
            cubic_term : float
            '''

            lmax = hp.Alm.getlmax(alm1.size)

            estimate_exp = 0
            for ell3 in range(lmax + 1):
                for m3 in range(-ell3, ell3 + 1):
                    for ell2 in range(lmax + 1):
                        for m2 in range(-ell2, ell2 + 1):
                            m1 = -(m2 + m3)
                            ells1 = np.arange(abs(ell3 - ell2), ell2 + ell3 + 1)

                            # Get wigner prefactors
                            wig_000 = camb.mathutils.threej(ell2, ell3, 0, 0)
                            wig_m = np.zeros(ells1.size)
                            tmp = camb.mathutils.threej(ell2, ell3, m2, m3)
                            wig_m[wig_m.size-tmp.size:] = tmp

                            for lidx1, ell1 in enumerate(ells1):

                                if ell1 > lmax or abs(m1) > lmax:
                                    continue
                                idx1 = hp.Alm.getidx(lmax, ell1, abs(m1))
                                idx2 = hp.Alm.getidx(lmax, ell2, abs(m2))
                                idx3 = hp.Alm.getidx(lmax, ell3, abs(m3))

                                a1 = np.conj(alm1[idx1]) * (-1) ** m1 if m1 < 0 else alm1[idx1]
                                a2 = np.conj(alm2[idx2]) * (-1) ** m2 if m2 < 0 else alm2[idx2]
                                a3 = np.conj(alm3[idx3]) * (-1) ** m3 if m3 < 0 else alm3[idx3]

                                nl = (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) / 4 / np.pi
                                nl = np.sqrt(nl)

                                est = red_bisp(ell1, ell2, ell3) * \
                                      nl * wig_000[lidx1] * wig_m[lidx1] * a1 * a2 * a3
                                estimate_exp += est

            return estimate_exp / 6

        def grad_direct(alm2, alm3, red_bisp):
            '''
            Calculate gradient term directly.

            Parameters
            ----------
            alm2, alm3 : (nelem) array
                1D Healpix-ordered alm arrays
            red_bisp : callable
                Function with ell1, ell2, ell3 arguments returning reduced bispectrum.

            Returns
            -------
            grad : (nelem) array
            '''

            lmax = hp.Alm.getlmax(alm2.size)
            grad = np.zeros_like(alm2)

            for ell3 in range(lmax + 1):
                for m3 in range(-ell3, ell3 + 1):
                    for ell2 in range(lmax + 1):
                        for m2 in range(-ell2, ell2 + 1):
                            m1 = -(m2 + m3)
                            if m1 < 0:
                                continue
                            ells1 = np.arange(abs(ell3 - ell2), ell2 + ell3 + 1)

                            # Get wigner prefactors
                            wig_000 = camb.mathutils.threej(ell2, ell3, 0, 0)
                            wig_m = np.zeros(ells1.size)
                            tmp = camb.mathutils.threej(ell2, ell3, m2, m3)
                            wig_m[wig_m.size-tmp.size:] = tmp

                            for lidx1, ell1 in enumerate(ells1):

                                if ell1 > lmax or abs(m1) > lmax:
                                    continue
                                idx1 = hp.Alm.getidx(lmax, ell1, abs(m1))
                                idx2 = hp.Alm.getidx(lmax, ell2, abs(m2))
                                idx3 = hp.Alm.getidx(lmax, ell3, abs(m3))

                                a2 = np.conj(alm2[idx2]) * (-1) ** m2 if m2 < 0 else alm2[idx2]
                                a3 = np.conj(alm3[idx3]) * (-1) ** m3 if m3 < 0 else alm3[idx3]

                                nl = (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) / 4 / np.pi
                                nl = np.sqrt(nl)
                                
                                grad[idx1] += red_bisp(ell1, ell2, ell3) * \
                                      nl * wig_000[lidx1] * wig_m[lidx1] * np.conj(a2) * np.conj(a3)

            return grad / 2

        def fisher_direct(lmax, npol, red_bisp, icov):
            '''
            Calculate fisher information directly.

            Parameters
            ----------
            lmax : int
            npol : int
            red_bisp : callable
                Function with ell1, ell2, ell3 arguments returning reduced bispectrum.
            icov : callable
                Function with ell, pix1, pidx2 arguments returning inverse covariance.

            Returns
            -------
            fisher : float
            '''

            pols = range(npol)

            pol_indices = []
            # Get pol combinations.
            for pidx1 in pols:
                for pidx2 in pols:
                    for pidx3 in pols:
                        for pidx4 in pols:
                            for pidx5 in pols:
                                for pidx6 in pols:
                                    pol_indices.append([pidx1, pidx2, pidx3, 
                                                        pidx4, pidx5, pidx6])
                                
            pol_indices = np.asarray(pol_indices)
            fisher = 0
            
            for ell2 in range(lmax + 1):
                for ell3 in range(lmax + 1):

                    ells1 = np.arange(abs(ell3 - ell2), ell2 + ell3 + 1)

                    # Get wigner prefactor
                    wig_000 = camb.mathutils.threej(ell2, ell3, 0, 0)

                    for lidx1, ell1 in enumerate(ells1):
                        
                        if ell1 > lmax:
                            continue

                        prefactor = (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1)
                        prefactor /= 4 * np.pi * 6

                        for pidx in range(pol_indices.shape[0]):
                            pidx1, pidx2, pidx3, pidx4, pidx5, pidx6 = pol_indices[pidx]

                            fisher += wig_000[lidx1] ** 2 * prefactor * \
                                      red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3) * \
                                      icov(ell1, pidx1, pidx4) * \
                                      icov(ell2, pidx2, pidx5) * \
                                      icov(ell3, pidx3, pidx6) * \
                                      red_bisp(ell1, ell2, ell3, pidx4, pidx5, pidx6)
            return fisher
            
        self.FakeReducedBispectrum = FakeReducedBispectrum
        self.FakeData = FakeData

        self.y00 = y00
        self.y10 = y10
        self.y11 = y11
        self.y20 = y20
        self.y21 = y21
        self.y22 = y22

        self.cubic_term_direct = cubic_term_direct
        self.grad_direct = grad_direct
        self.fisher_direct = fisher_direct

    def tearDown(self):
        # Is called after each test.
        pass

    def test_ksw_init(self):
        
        pol = ('T', 'E')
        lmax = 300
        red_bispectra = [self.FakeReducedBispectrum()]
        icov = self.FakeData().icov_diag_nonlensed
        beam = lambda alm : alm

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)

        self.assertIs(estimator.red_bispectra[0], red_bispectra[0])

        self.assertTrue(callable(estimator.icov))
        self.assertTrue(callable(estimator.beam))
        self.assertEqual(estimator.icov, icov)
        self.assertEqual(estimator.beam, beam)

        self.assertEqual(estimator.mc_idx, 0)
        self.assertIs(estimator.mc_gt, None)
        self.assertIs(estimator.mc_gt_sq, None)

        self.assertEqual(estimator.thetas.size, estimator.theta_weights.size)

        self.assertEqual(estimator.pol, ('T', 'E'))
        self.assertEqual(estimator.npol, 2)

        self.assertEqual(estimator.lmax, lmax)

        if self.precision == 'double':
            self.assertEqual(estimator.dtype, np.float64)
            self.assertEqual(estimator.cdtype, np.complex128)
        elif self.precision == 'single':
            self.assertEqual(estimator.dtype, np.float32)
            self.assertEqual(estimator.cdtype, np.complex64)

    def test_ksw_init_pol(self):
        
        lmax = 300
        red_bispectra = [self.FakeReducedBispectrum()]
        icov = self.FakeData().icov_diag_nonlensed
        beam = lambda alm : alm

        pol = ('T', 'E')
        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        self.assertEqual(estimator.pol, ('T', 'E'))
        self.assertEqual(estimator.npol, 2)

        pol = ['T', 'E']
        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        self.assertEqual(estimator.pol, ('T', 'E'))
        self.assertEqual(estimator.npol, 2)

        pol = ('E', 'T')
        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        self.assertEqual(estimator.pol, ('T', 'E'))
        self.assertEqual(estimator.npol, 2)

        pol = ('T',)
        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        self.assertEqual(estimator.pol, ('T',))
        self.assertEqual(estimator.npol, 1)

        pol = 'T'
        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        self.assertEqual(estimator.pol, ('T',))
        self.assertEqual(estimator.npol, 1)

        pol = ('E',)
        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        self.assertEqual(estimator.pol, ('E',))
        self.assertEqual(estimator.npol, 1)

        pol = ('B',)
        self.assertRaises(ValueError, KSW, red_bispectra, icov, beam, lmax, pol,
                          precision=self.precision)

    def test_ksw_mc_gt_sq(self):

        lmax = 300
        red_bispectra = [self.FakeReducedBispectrum()]
        icov = self.FakeData().icov_diag_nonlensed
        beam = lambda alm : alm
        pol = ('T', 'E')

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)

        self.assertIs(estimator.mc_gt_sq, None)

        estimator.mc_gt_sq = 10
        estimator.mc_idx = 1

        self.assertEqual(estimator.mc_gt_sq, 10)

        estimator.mc_idx = 3

        self.assertEqual(estimator.mc_gt_sq, 10 / 3.)

    def test_ksw_mc_gt(self):

        lmax = 300
        red_bispectra = [self.FakeReducedBispectrum()]
        icov = self.FakeData().icov_diag_nonlensed
        beam = lambda alm : alm
        pol = ('T', 'E')

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)

        self.assertIs(estimator.mc_gt, None)

        mc_gt = np.ones((len(pol), hp.Alm.getsize(lmax)),
                                      dtype=complex)
        estimator.mc_gt = mc_gt.copy() * 10
        estimator.mc_idx = 1

        np.testing.assert_almost_equal(estimator.mc_gt, mc_gt * 10, 
                                       decimal=self.decimal)

        estimator.mc_idx = 3

        np.testing.assert_almost_equal(estimator.mc_gt, 10 * mc_gt / 3.,
                                       decimal=self.decimal)

    def test_ksw_get_coords(self):

        lmax = 300
        red_bispectra = [self.FakeReducedBispectrum()]
        icov = self.FakeData().icov_diag_nonlensed
        beam = lambda alm : alm
        pol = ('T', 'E')

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)

        self.assertEqual(estimator.thetas.size, (3 * lmax) // 2 + 1)
        self.assertTrue(np.all(0 <= estimator.thetas))
        self.assertTrue(np.all(estimator.thetas <= np.pi))
        self.assertTrue(estimator.nphi >= 3 * lmax + 1)

        # Updating lmax will result in different coords.
        estimator.lmax = 10
        thetas, tw, nphi = estimator.get_coords()
        self.assertEqual(thetas.size, (3 * 10) // 2 + 1)
        self.assertTrue(np.all(0 <= thetas))
        self.assertTrue(np.all(thetas <= np.pi))
        self.assertTrue(nphi >= 3 * 10 + 1)

    def test_ksw_init_reduced_bispectrum_err(self):

        lmax = 300
        red_bispectra = [self.FakeReducedBispectrum()]
        icov = self.FakeData().icov_diag_nonlensed
        beam = lambda alm : alm
        pol = ('T', 'E')

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)

        # Bisp lmax is smaller than estimator lmax.
        self.assertRaises(ValueError, estimator._init_reduced_bispectrum,
            estimator.red_bispectra[0])

    def test_ksw_init_reduced_bispectrum(self):

        lmax = 6
        red_bispectra = [self.FakeReducedBispectrum()]
        icov = self.FakeData().icov_diag_nonlensed
        beam = lambda alm : alm
        pol = ('T', 'E')

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)

        rb = estimator.red_bispectra[0]
        nell = rb.factors.shape[-1]
        # Shape = (3, npol, red_bisp.nell).
        rb.factors[:,:,:] = np.arange(nell, dtype=float)
        rb.factors[0,0,:] *= 2
        rb.factors[0,1,:] *= 3
        rb.factors[1,0,:] *= 4
        rb.factors[1,1,:] *= 5
        rb.factors[2,0,:] *= 6
        rb.factors[2,1,:] *= 7

        # Shape = (nfact, 3).
        rb.rule[0] = [0, 0, 0]
        rb.rule[1] = [1, 1, 0]
        rb.rule[2] = [1, 2, 2]
        rb.rule[3] = [0, 1, 2]

        # Shape = (nfact, 3, npol).
        rb.weights[:,0] = 1
        rb.weights[:,1] = 2
        rb.weights[:,2] = 3

        f_i_ell, rule, weights = estimator._init_reduced_bispectrum(
            estimator.red_bispectra[0])

        np.testing.assert_allclose(rule, rb.rule)
        self.assertTrue(np.shares_memory(rule, rb.rule))

        np.testing.assert_allclose(weights, rb.weights)
        self.assertEqual(weights.dtype, estimator.dtype)
        
        nufact = rb.factors.shape[0]
        npol = len(pol)
        
        self.assertEqual(f_i_ell.shape, (nufact, npol, lmax + 1))
        self.assertEqual(f_i_ell.dtype, estimator.dtype)
        f_i_ell_exp = np.zeros((nufact, npol, lmax + 1), dtype=estimator.dtype)
        f_i_ell_exp[:,:,2:] = rb.factors[:,:,:lmax+1-2]
        np.testing.assert_allclose(f_i_ell, f_i_ell_exp)

    def test_ksw_init_reduced_bispectrum_I(self):

        lmax = 6
        red_bispectra = [self.FakeReducedBispectrum()]
        icov = self.FakeData().icov_diag_nonlensed
        beam = lambda alm : alm
        pol = ('T',)

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)

        rb = estimator.red_bispectra[0]
        nell = rb.factors.shape[-1]
        # Shape = (3, npol, red_bisp.nell).
        rb.factors[:,:,:] = np.arange(nell, dtype=float)
        rb.factors[0,0,:] *= 2
        rb.factors[0,1,:] *= 3
        rb.factors[1,0,:] *= 4
        rb.factors[1,1,:] *= 5
        rb.factors[2,0,:] *= 6
        rb.factors[2,1,:] *= 7

        # Shape = (nfact, 3).
        rb.rule[0] = [0, 0, 0]
        rb.rule[1] = [1, 1, 0]
        rb.rule[2] = [1, 2, 2]
        rb.rule[3] = [0, 1, 2]

        # Shape = (nfact, 3, npol).
        rb.weights[:,0] = 1
        rb.weights[:,1] = 2
        rb.weights[:,2] = 3

        f_i_ell, rule, weights = estimator._init_reduced_bispectrum(
            estimator.red_bispectra[0])

        np.testing.assert_allclose(rule, rb.rule)
        self.assertTrue(np.shares_memory(rule, rb.rule))

        np.testing.assert_allclose(weights, rb.weights)
        self.assertEqual(weights.dtype, estimator.dtype)
        
        nufact = rb.factors.shape[0]
        npol = len(pol)
        
        self.assertEqual(f_i_ell.shape, (nufact, npol, lmax + 1))
        self.assertEqual(f_i_ell.dtype, estimator.dtype)
        f_i_ell_exp = np.zeros((nufact, npol, lmax + 1), dtype=estimator.dtype)
        #f_i_ell_exp[:,:,2:] = rb.factors[:,0:1,:lmax+1-2] * 0.1 # 0.1 is beam factor.
        f_i_ell_exp[:,:,2:] = rb.factors[:,0:1,:lmax+1-2]
        np.testing.assert_allclose(f_i_ell, f_i_ell_exp)

    def test_ksw_init_reduced_bispectrum_E(self):

        lmax = 6
        red_bispectra = [self.FakeReducedBispectrum()]
        icov = self.FakeData().icov_diag_nonlensed
        beam = lambda alm : alm
        pol = ('E',)

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)

        rb = estimator.red_bispectra[0]
        nell = rb.factors.shape[-1]
        # Shape = (3, npol, red_bisp.nell).
        rb.factors[:,:,:] = np.arange(nell, dtype=float)
        rb.factors[0,0,:] *= 2
        rb.factors[0,1,:] *= 3
        rb.factors[1,0,:] *= 4
        rb.factors[1,1,:] *= 5
        rb.factors[2,0,:] *= 6
        rb.factors[2,1,:] *= 7

        # Shape = (nfact, 3).
        rb.rule[0] = [0, 0, 0]
        rb.rule[1] = [1, 1, 0]
        rb.rule[2] = [1, 2, 2]
        rb.rule[3] = [0, 1, 2]

        # Shape = (nfact, 3, npol).
        rb.weights[:,0] = 1
        rb.weights[:,1] = 2
        rb.weights[:,2] = 3

        f_i_ell, rule, weights = estimator._init_reduced_bispectrum(
            estimator.red_bispectra[0])

        np.testing.assert_allclose(rule, rb.rule)
        self.assertTrue(np.shares_memory(rule, rb.rule))

        np.testing.assert_allclose(weights, rb.weights)
        self.assertEqual(weights.dtype, estimator.dtype)
        
        nufact = rb.factors.shape[0]
        npol = len(pol)
        
        self.assertEqual(f_i_ell.shape, (nufact, npol, lmax + 1))
        self.assertEqual(f_i_ell.dtype, estimator.dtype)
        f_i_ell_exp = np.zeros((nufact, npol, lmax + 1), dtype=estimator.dtype)
        f_i_ell_exp[:,:,2:] = rb.factors[:,1:2,:lmax+1-2]
        np.testing.assert_allclose(f_i_ell, f_i_ell_exp)

    def test_ksw_compute_fisher(self):

        lmax = 300
        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T', 'E')
        def icov(alm):
            alm *= 2
            return alm

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)

        self.assertIs(estimator.compute_fisher(), None)

        mc_gt = np.ones((len(pol), hp.Alm.getsize(lmax)), dtype=complex)
        mc_gt *= 10
        mc_gt_sq = 5
        mc_idx = 4
        estimator.mc_gt = mc_gt
        estimator.mc_gt_sq = mc_gt_sq
        estimator.mc_idx = mc_idx

        mc_gt_copy = estimator.mc_gt.copy()
        mc_gt_sq_copy = estimator.mc_gt_sq

        # Using Eq. 58 in Smith & Zaldarriaga.
        fisher_exp = mc_gt_sq / mc_idx
        # Following two lines are the sum over the contraction, second line Eq. 58.
        # See utils.contract_almxblm
        fisher_exp -= 2 * np.real(np.sum(mc_gt / mc_idx * icov(mc_gt.copy() / mc_idx)))
        fisher_exp -= -(np.real(np.sum((mc_gt / mc_idx * icov(mc_gt.copy() / mc_idx))
                                       [...,:lmax+1])))
        fisher_exp /= 3

        fisher = estimator.compute_fisher()
        self.assertAlmostEqual(fisher, fisher_exp, places=self.decimal)

        # I want internal quantities unchanged.
        self.assertEqual(estimator.mc_gt_sq, mc_gt_sq_copy)
        np.testing.assert_array_almost_equal(estimator.mc_gt, mc_gt_copy, 
                                             decimal=self.decimal)

    def test_ksw_compute_linear_term(self):

        lmax = 300
        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T', 'E')
        def icov(alm):
            alm *= 2
            return alm

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)

        alm = np.zeros((len(pol), hp.Alm.getsize(lmax)),
                      dtype=complex)
        alm[:] += np.arange(alm.size, dtype=alm.dtype).reshape(alm.shape)

        self.assertIs(estimator.compute_linear_term(alm), None)

        mc_gt = np.ones_like(alm)
        mc_gt *= 10
        mc_gt_sq = 5
        mc_idx = 4
        estimator.mc_gt = mc_gt
        estimator.mc_gt_sq = mc_gt_sq
        estimator.mc_idx = mc_idx

        mc_gt_copy = estimator.mc_gt.copy()
        mc_gt_sq_copy = estimator.mc_gt_sq

        # Using second line of Eq. 57.
        lin_term_exp = 2 * np.real(np.sum(alm * icov(mc_gt.copy() / mc_idx)))
        lin_term_exp -= np.real(np.sum((alm * icov(mc_gt.copy() / mc_idx))
                                       [...,:lmax+1]))
        lin_term = estimator.compute_linear_term(alm)

        self.assertAlmostEqual(lin_term, lin_term_exp, places=self.decimal)

        # I want internal quantities unchanged.
        self.assertEqual(estimator.mc_gt_sq, mc_gt_sq_copy)
        np.testing.assert_array_almost_equal(estimator.mc_gt, mc_gt_copy, 
                                             decimal=self.decimal)

    def test_ksw_compute_linear_term_1d(self):

        # Check if still works with (nelem) input array.
        lmax = 300
        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T',)
        def icov(alm):
            alm *= 2
            return alm

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)

        alm = np.zeros(hp.Alm.getsize(lmax),
                      dtype=complex)
        alm[:] += np.arange(alm.size, dtype=alm.dtype).reshape(alm.shape)

        self.assertIs(estimator.compute_linear_term(alm), None)

        mc_gt = np.ones((1, hp.Alm.getsize(lmax)), dtype=alm.dtype)
        mc_gt *= 10
        mc_gt_sq = 5
        mc_idx = 4
        estimator.mc_gt = mc_gt
        estimator.mc_gt_sq = mc_gt_sq
        estimator.mc_idx = mc_idx

        mc_gt_copy = estimator.mc_gt.copy()
        mc_gt_sq_copy = estimator.mc_gt_sq

        # Using second line of Eq. 57.
        lin_term_exp = 2 * np.real(np.sum(alm * icov(mc_gt.copy() / mc_idx)))
        lin_term_exp -= np.real(np.sum((alm * icov(mc_gt.copy() / mc_idx))
                                       [...,:lmax+1]))
        lin_term = estimator.compute_linear_term(alm)

        self.assertAlmostEqual(lin_term, lin_term_exp, places=self.decimal)

        # I want internal quantities unchanged.
        self.assertEqual(estimator.mc_gt_sq, mc_gt_sq_copy)
        np.testing.assert_array_almost_equal(estimator.mc_gt, mc_gt_copy,
                                             decimal=self.decimal)

    def test_ksw_compute_fisher_isotropic_I_simple(self):

        lmax = 6
        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T',)
        npol = 1
        icov = lambda alm: alm
        icov_ell = np.ones((1, lmax + 1))
                
        fsky = 0.4

        # Create a reduced bispectrum that is just b_l1l2l3 = 1.
        rb = red_bispectra[0]
        rb.npol = 1
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        fisher = estimator.compute_fisher_isotropic(icov_ell, fsky=fsky)

        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            return 1. 
            
        def icov(ell, pidx1, pidx2):
            return 1.

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov) * fsky

        self.assertAlmostEqual(fisher, fisher_exp, places=self.decimal)

    def test_ksw_compute_fisher_isotropic_pol_simple(self):

        lmax = 5
        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T', 'E')
        npol = 2
        icov = lambda alm: alm
        icov_ell = np.ones((npol, npol, lmax + 1))
                
        # Create a reduced bispectrum that is just b_l1l2l3 = 1.
        rb = red_bispectra[0]
        rb.npol = npol
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        fisher = estimator.compute_fisher_isotropic(icov_ell)

        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            return 1.
            
        def icov(ell, pidx1, pidx2):
            return 1.

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov)

        self.assertAlmostEqual(fisher / fisher_exp, 1, places=self.decimal)

    def test_ksw_compute_fisher_isotropic_I_simple_2d(self):

        # Compare to direct 4 dimensional sum over (l,m).
        lmax = 5
        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T',)
        npol = 1
        icov = lambda alm: alm
        icov_ell = np.ones((1, lmax + 1))
                
        # Create a reduced bispectrum that is b_l1l2l3 = 1 * 1 * 1 + 2 * 2 * 2.
        rb = red_bispectra[0]
        rb.npol = 1
        rb.nfact = 2
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0

        rb.factors = np.ones((2, npol, lmax + 1))
        rb.factors[1] *= 2
        rb.rule = np.zeros((2, 3), dtype=int)
        rb.rule[1] = 1
        rb.weights = np.ones((2, 3))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        fisher = estimator.compute_fisher_isotropic(icov_ell)
        
        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            return (1 * 1 * 1 + 2 * 2 * 2)

        def icov(ell, pidx1, pidx2):
            return 1.

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov)

        self.assertAlmostEqual(fisher / fisher_exp, 1, places=self.decimal)

    def test_ksw_compute_fisher_isotropic_pol_simple_2d(self):

        # Compare to direct 4 dimensional sum over (l,m).
        lmax = 5
        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T', 'E')
        npol = 2
        icov = lambda alm: alm
        icov_ell = np.ones((2, 2, lmax + 1))
                                
        # Create a reduced bispectrum that is sum of 1 for I and 2 for E
        # and 3 for I and 6 for E.
        rb = red_bispectra[0]
        rb.npol = npol
        rb.nfact = 2
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0

        rb.factors = np.ones((2, npol, lmax + 1))
        rb.factors[0,1,:] = 2
        rb.factors[1,0,:] = 3
        rb.factors[1,1,:] = 6
        rb.rule = np.zeros((2, 3), dtype=int)
        rb.rule[1] = 1
        rb.weights = np.ones((2, 3))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        fisher = estimator.compute_fisher_isotropic(icov_ell)

        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            b1_l1 = 2 if pidx1 else 1
            b1_l2 = 2 if pidx2 else 1
            b1_l3 = 2 if pidx3 else 1

            b2_l1 = 6 if pidx1 else 3
            b2_l2 = 6 if pidx2 else 3
            b2_l3 = 6 if pidx3 else 3

            return b1_l1 * b1_l2 * b1_l3 + b2_l1 * b2_l2 * b2_l3
            
        def icov(ell, pidx1, pidx2):
            return 1.

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov)
        
        self.assertAlmostEqual(fisher / fisher_exp, 1, places=self.decimal)

    def test_ksw_compute_fisher_isotropic_I_local(self):

        lmax_transfer = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
        pars = camb.CAMBparams()
        pars.set_cosmology(**cosmo_opts)


        cosmo = Cosmology(pars)
        cosmo.compute_transfer(lmax_transfer)

        prim_shape = Shape.prim_local(ns=1)

        self.assertTrue(len(cosmo.red_bispectra) == 0)
        cosmo.add_prim_reduced_bispectrum(prim_shape, radii)
        self.assertTrue(len(cosmo.red_bispectra) == 1)

        rb = cosmo.red_bispectra[0]

        # Lmax and pol of data should overrule those of bispectrum.
        lmax = 5
        red_bispectra = [rb]
        beam = lambda alm : alm
        pol = ('T',)
        npol = 1
        icov = lambda alm: alm
        icov_ell = np.ones((1, lmax + 1))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        fisher = estimator.compute_fisher_isotropic(icov_ell)

        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            '''
            b_l1l2l3 = int dr r^2 1/6 (alpha_l1(r) beta_l2(r) beta_l3(r) + 5 symm.)
            '''
            # Factors are direct output from radial functional, 
            # so they do not have the 2 * As^2 factor.

            if ell1 < 2 or ell2 < 2 or ell3 < 2:
                return 0

            # Determine indices into ell dimension of factors.
            lidx1 = np.where(rb.ells_full == ell1)[0][0]
            lidx2 = np.where(rb.ells_full == ell2)[0][0]
            lidx3 = np.where(rb.ells_full == ell3)[0][0]

            amp = 2 * (2 * np.pi ** 2 * cosmo.camb_params.InitPower.As) ** 2 * (3 / 5)
            # First alpha_l1(r1) beta_l2(r1) beta_l3(r1).
            ret = dr * radii[0] ** 2 * \
                  rb.factors[0,0,lidx1] * rb.factors[2,0,lidx2] * rb.factors[2,0,lidx3]
            # beta_l1(r1) alpha_l2(r1) beta_l3(r1).
            ret += dr * radii[0] ** 2 * \
                   rb.factors[2,0,lidx1] * rb.factors[0,0,lidx2] * rb.factors[2,0,lidx3]
            # beta_l1(r1) beta_l2(r1) alpha_l3(r1).
            ret += dr * radii[0] ** 2 * \
                   rb.factors[2,0,lidx1] * rb.factors[2,0,lidx2] * rb.factors[0,0,lidx3]
            
            # Then alpha_l1(r2) beta_l2(r2) beta_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[1,0,lidx1] * rb.factors[3,0,lidx2] * rb.factors[3,0,lidx3]
            # beta_l1(r2) alpha_l2(r2) beta_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[3,0,lidx1] * rb.factors[1,0,lidx2] * rb.factors[3,0,lidx3]
            # beta_l1(r2) beta_l2(r2) alpha_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[3,0,lidx1] * rb.factors[3,0,lidx2] * rb.factors[1,0,lidx3]

            ret *= amp

            return ret

        def icov(ell, pidx1, pidx2):
            return 1.

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov)

        self.assertAlmostEqual(fisher / fisher_exp, 1, places=self.decimal)

    def test_ksw_compute_fisher_isotropic_pol_local(self):

        lmax_transfer = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
        #pars = camb.CAMBparams(**cosmo_opts)
        pars = camb.CAMBparams()
        pars.set_cosmology(**cosmo_opts)

        cosmo = Cosmology(pars)
        cosmo.compute_transfer(lmax_transfer)

        prim_shape = Shape.prim_local(ns=1)

        self.assertTrue(len(cosmo.red_bispectra) == 0)
        cosmo.add_prim_reduced_bispectrum(prim_shape, radii)
        self.assertTrue(len(cosmo.red_bispectra) == 1)

        rb = cosmo.red_bispectra[0]

        # Lmax and pol of data should overrule those of bispectrum.
        lmax = 5
        red_bispectra = [rb]
        beam = lambda alm : alm
        pol = ('T', 'E')
        npol = 2
        icov = lambda alm: alm
        icov_ell = np.ones((2, 2, lmax + 1))
        icov_ell[0,1] = 0.4
        icov_ell[1,0] = 0.4

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        fisher = estimator.compute_fisher_isotropic(icov_ell)

        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            '''
            b^X1X2X3_l1l2l3 = int dr r^2 1/6 (alpha^X1_l1(r) beta^X2_l2(r) beta^X3_l3(r) + 5 symm.)
            '''
            # Factors are direct output from radial functional, 
            # so they do not have the 2 * As^2 factor.

            if ell1 < 2 or ell2 < 2 or ell3 < 2:
                return 0

            # Determine indices into ell dimension of factors.
            lidx1 = np.where(rb.ells_full == ell1)[0][0]
            lidx2 = np.where(rb.ells_full == ell2)[0][0]
            lidx3 = np.where(rb.ells_full == ell3)[0][0]

            # Note that the symmetrisation below is not needed for the cubic term, but still
            # formally correct.
            
            amp = 2 * (2 * np.pi ** 2 * cosmo.camb_params.InitPower.As) ** 2 * (3 / 5)
            # First alpha_l1(r1) beta_l2(r1) beta_l3(r1).
            ret = dr * radii[0] ** 2 * \
                  rb.factors[0,pidx1,lidx1] * rb.factors[2,pidx2,lidx2] * rb.factors[2,pidx3,lidx3]
            # beta_l1(r1) alpha_l2(r1) beta_l3(r1).
            ret += dr * radii[0] ** 2 * \
                   rb.factors[2,pidx1,lidx1] * rb.factors[0,pidx2,lidx2] * rb.factors[2,pidx3,lidx3]
            # beta_l1(r1) beta_l2(r1) alpha_l3(r1).
            ret += dr * radii[0] ** 2 * \
                   rb.factors[2,pidx1,lidx1] * rb.factors[2,pidx2,lidx2] * rb.factors[0,pidx3,lidx3]
            
            # Then alpha_l1(r2) beta_l2(r2) beta_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[1,pidx1,lidx1] * rb.factors[3,pidx2,lidx2] * rb.factors[3,pidx3,lidx3]
            # beta_l1(r2) alpha_l2(r2) beta_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[3,pidx1,lidx1] * rb.factors[1,pidx2,lidx2] * rb.factors[3,pidx3,lidx3]
            # beta_l1(r2) beta_l2(r2) alpha_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[3,pidx1,lidx1] * rb.factors[3,pidx2,lidx2] * rb.factors[1,pidx3,lidx3]

            ret *= amp

            return ret

        def icov(ell, pidx1, pidx2):
            if pidx1 == pidx2:
                return 1. 
            else:
                return 0.4

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov)

        self.assertAlmostEqual(fisher / fisher_exp, 1, places=self.decimal)

    def test_ksw_compute_fisher_isotropic_matrix(self):
        
        lmax = 5
        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T',)
        npol = 1
        icov = lambda alm: alm
        icov_ell = np.ones((1, lmax + 1))

        # Create a reduced bispectrum that is just b_l1l2l3 = 1.
        rb = red_bispectra[0]
        rb.npol = npol
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        fisher = estimator.compute_fisher_isotropic(icov_ell)

        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            return 1.

        def icov(ell, pidx1, pidx2):
            return 1.

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov)

        fisher, fisher_nxn = estimator.compute_fisher_isotropic(
            icov_ell, return_matrix=True)

        self.assertAlmostEqual(fisher, fisher_exp, places=self.decimal)
        self.assertAlmostEqual(fisher, np.sum(fisher_nxn), places=self.decimal)

    def test_ksw_compute_estimate_cubic_I_simple(self):

        # Compare to direct 5 dimensional sum over (l,m).

        lmax = 5
        alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        alm += np.random.randn(alm.size)
        alm += np.random.randn(alm.size) * 1j
        alm[:lmax+1] = alm[:lmax+1].real # Make sure m=0 is real.
        alm = alm.reshape((1, alm.size))

        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T',)
        npol = 1
        icov = lambda alm: alm
        icov_ell = np.ones((1, lmax + 1))

        # Create a reduced bispectrum that is just b_l1l2l3 = 1.
        rb = red_bispectra[0]
        rb.npol = 1
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        estimator.mc_idx = 1

        # Make sure Fisher is 1 and linear term is 0.
        estimator.mc_gt_sq = 3.
        estimator.mc_gt = np.zeros_like(alm)

        self.assertEqual(estimator.compute_fisher(), 1)
        self.assertEqual(estimator.compute_linear_term(alm), 0.)

        estimate = estimator.compute_estimate(alm.copy())

        def red_bisp(ell1, ell2, ell3):
            return 1.

        alm = alm[0]
        estimate_exp = self.cubic_term_direct(alm, alm, alm, red_bisp)
        self.assertAlmostEqual(estimate / estimate_exp, 1, places=self.decimal)

    def test_ksw_compute_estimate_cubic_I_fisher_lin_term(self):

        # Compare to direct 5 dimensional sum over (l,m).
        
        fisher = 10
        lin_term = 5

        lmax = 5
        alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        alm += np.random.randn(alm.size)
        alm += np.random.randn(alm.size) * 1j
        alm[:lmax+1] = alm[:lmax+1].real # Make sure m=0 is real.
        alm = alm.reshape((1, alm.size))

        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T',)
        npol = 1
        icov = lambda alm: alm
        icov_ell = np.ones((1, lmax + 1))

        # Create a reduced bispectrum that is just b_l1l2l3 = 1.
        rb = red_bispectra[0]
        rb.npol = 1
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        estimator.mc_idx = 1

        # Make sure Fisher is 1 and linear term is 0.
        estimator.mc_gt_sq = 3.
        estimator.mc_gt = np.zeros_like(alm)

        self.assertEqual(estimator.compute_fisher(), 1)
        self.assertEqual(estimator.compute_linear_term(alm), 0.)

        # Test if giving fisher and linear terms manually works.
        estimate = estimator.compute_estimate(alm.copy(), fisher=fisher, lin_term=lin_term)

        def red_bisp(ell1, ell2, ell3):
            return 1.

        alm = alm[0]
        estimate_exp = (self.cubic_term_direct(alm, alm, alm, red_bisp) - lin_term) / fisher

        self.assertAlmostEqual(estimate / estimate_exp, 1, places=self.decimal)

    def test_ksw_compute_estimate_cubic_pol_simple(self):

        # Compare to direct 5 dimensional sum over (l,m).

        lmax = 5
        alm = np.zeros((2, hp.Alm.getsize(lmax)), dtype=np.complex128)
        alm += np.random.randn(alm.size).reshape(alm.shape)
        alm += np.random.randn(alm.size).reshape(alm.shape) * 1j
        alm[:,:lmax+1] = alm[:,:lmax+1].real # Make sure m=0 is real.

        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T', 'E')
        npol = 2
        icov = lambda alm: alm
        icov_ell = np.ones((2, 2, lmax + 1))

        # Create a reduced bispectrum with factors that are 1 for I and 2 for E.
        rb = red_bispectra[0]
        rb.npol = 2
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.factors[:,1,:] = 2
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3))

        #estimator = KSW(data, precision=self.precision)
        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        estimator.mc_idx = 1

        # Make sure Fisher is 1 and linear term is 0.
        estimator.mc_gt_sq = 3.
        estimator.mc_gt = np.zeros_like(alm)

        self.assertEqual(estimator.compute_fisher(), 1)
        self.assertEqual(estimator.compute_linear_term(alm), 0.)

        estimate = estimator.compute_estimate(alm.copy())

        def red_bisp_III(ell1, ell2, ell3):
            return 1 * 1 * 1
        def red_bisp_IIE(ell1, ell2, ell3):
            return 1 * 1 * 2
        def red_bisp_IEI(ell1, ell2, ell3):
            return 1 * 2 * 1
        def red_bisp_EII(ell1, ell2, ell3):
            return 2 * 1 * 1
        def red_bisp_IEE(ell1, ell2, ell3):
            return 1 * 2 * 2
        def red_bisp_EIE(ell1, ell2, ell3):
            return 2 * 1 * 2
        def red_bisp_EEI(ell1, ell2, ell3):
            return 2 * 2 * 1
        def red_bisp_EEE(ell1, ell2, ell3):
            return 2 * 2 * 2

        alm_I = alm[0]
        alm_E = alm[1]

        estimate_exp = self.cubic_term_direct(alm_I, alm_I, alm_I, red_bisp_III)
        estimate_exp += self.cubic_term_direct(alm_I, alm_I, alm_E, red_bisp_IIE)
        estimate_exp += self.cubic_term_direct(alm_I, alm_E, alm_I, red_bisp_IEI)
        estimate_exp += self.cubic_term_direct(alm_E, alm_I, alm_I, red_bisp_EII)
        estimate_exp += self.cubic_term_direct(alm_I, alm_E, alm_E, red_bisp_IEE)
        estimate_exp += self.cubic_term_direct(alm_E, alm_I, alm_E, red_bisp_EIE)
        estimate_exp += self.cubic_term_direct(alm_E, alm_E, alm_I, red_bisp_EEI)
        estimate_exp += self.cubic_term_direct(alm_E, alm_E, alm_E, red_bisp_EEE)

        self.assertAlmostEqual(estimate / estimate_exp, 1, places=self.decimal)

    def test_ksw_compute_estimate_cubic_I_2d(self):

        # Compare to direct 5 dimensional sum over (l,m).
        # For 2 term reduced bispectrum.

        lmax = 5
        alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        alm += np.random.randn(alm.size)
        alm += np.random.randn(alm.size) * 1j
        alm[:lmax+1] = alm[:lmax+1].real # Make sure m=0 is real.
        alm = alm.reshape((1, alm.size))

        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T',)
        npol = 1
        icov = lambda alm: alm
        icov_ell = np.ones((1, lmax + 1))

        # Create a reduced bispectrum that is b_l1l2l3 = 1 * 1 * 1 + 2 * 2 * 2.
        rb = red_bispectra[0]
        rb.npol = 1
        rb.nfact = 2
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0

        rb.factors = np.ones((2, npol, lmax + 1))
        rb.factors[1] *= 2
        rb.rule = np.zeros((2, 3), dtype=int)
        rb.rule[1] = 1
        rb.weights = np.ones((2, 3))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        estimator.mc_idx = 1

        # Make sure Fisher is 1 and linear term is 0.
        estimator.mc_gt_sq = 3.
        estimator.mc_gt = np.zeros_like(alm)

        self.assertEqual(estimator.compute_fisher(), 1)
        self.assertEqual(estimator.compute_linear_term(alm), 0.)

        estimate = estimator.compute_estimate(alm.copy())

        def red_bisp(ell1, ell2, ell3):
            return 1 * 1 * 1 + 2 * 2 * 2

        alm = alm[0]
        estimate_exp = self.cubic_term_direct(alm, alm, alm, red_bisp)

        self.assertAlmostEqual(estimate / estimate_exp, 1, places=self.decimal)

    def test_ksw_compute_estimate_cubic_pol_2d(self):

        # Compare to direct 5 dimensional sum over (l,m).
        # For 2 term reduced bispectrum.

        lmax = 5
        alm = np.zeros((2, hp.Alm.getsize(lmax)), dtype=np.complex128)
        alm += np.random.randn(alm.size).reshape(alm.shape)
        alm += np.random.randn(alm.size).reshape(alm.shape) * 1j
        alm[:,:lmax+1] = alm[:,:lmax+1].real # Make sure m=0 is real.
        
        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T', 'E')
        npol = 2
        icov = lambda alm: alm
        icov_ell = np.ones((2, 2, lmax + 1))

        # Create a reduced bispectrum that is sum of 1 for I and 2 for E
        # and 3 for I and 6 for E.
        rb = red_bispectra[0]
        rb.npol = 1
        rb.nfact = 2
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0

        rb.factors = np.ones((2, npol, lmax + 1))
        rb.factors[0,1,:] = 2
        rb.factors[1,0,:] = 3
        rb.factors[1,1,:] = 6
        rb.rule = np.zeros((2, 3), dtype=int)
        rb.rule[1] = 1
        rb.weights = np.ones((2, 3))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        estimator.mc_idx = 1

        # Make sure Fisher is 1 and linear term is 0.
        estimator.mc_gt_sq = 3.
        estimator.mc_gt = np.zeros_like(alm)

        self.assertEqual(estimator.compute_fisher(), 1)
        self.assertEqual(estimator.compute_linear_term(alm), 0.)

        estimate = estimator.compute_estimate(alm.copy())

        def red_bisp_III(ell1, ell2, ell3):
            return 1 * 1 * 1 + 3 * 3 * 3
        def red_bisp_IIE(ell1, ell2, ell3):
            return 1 * 1 * 2 + 3 * 3 * 6
        def red_bisp_IEI(ell1, ell2, ell3):
            return 1 * 2 * 1 + 3 * 6 * 3
        def red_bisp_EII(ell1, ell2, ell3):
            return 2 * 1 * 1 + 6 * 3 * 3
        def red_bisp_IEE(ell1, ell2, ell3):
            return 1 * 2 * 2 + 3 * 6 * 6
        def red_bisp_EIE(ell1, ell2, ell3):
            return 2 * 1 * 2 + 6 * 3 * 6
        def red_bisp_EEI(ell1, ell2, ell3):
            return 2 * 2 * 1 + 6 * 6 * 3
        def red_bisp_EEE(ell1, ell2, ell3):
            return 2 * 2 * 2 + 6 * 6 * 6

        alm_I = alm[0]
        alm_E = alm[1]

        estimate_exp = self.cubic_term_direct(alm_I, alm_I, alm_I, red_bisp_III)
        estimate_exp += self.cubic_term_direct(alm_I, alm_I, alm_E, red_bisp_IIE)
        estimate_exp += self.cubic_term_direct(alm_I, alm_E, alm_I, red_bisp_IEI)
        estimate_exp += self.cubic_term_direct(alm_E, alm_I, alm_I, red_bisp_EII)
        estimate_exp += self.cubic_term_direct(alm_I, alm_E, alm_E, red_bisp_IEE)
        estimate_exp += self.cubic_term_direct(alm_E, alm_I, alm_E, red_bisp_EIE)
        estimate_exp += self.cubic_term_direct(alm_E, alm_E, alm_I, red_bisp_EEI)
        estimate_exp += self.cubic_term_direct(alm_E, alm_E, alm_E, red_bisp_EEE)

        self.assertAlmostEqual(estimate / estimate_exp, 1, places=self.decimal)

    def test_ksw_compute_estimate_cubic_local_I(self):

        # Compare to direct 5 dimensional sum over (l,m).
        # For local reduced bispectrum.

        lmax_transfer = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
#        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
#                               omk=0, TCMB=2.7255)

#        pars = camb.CAMBparams(**cosmo_opts)
        pars = camb.CAMBparams()
        pars.set_cosmology(**cosmo_opts)


        cosmo = Cosmology(pars)
        cosmo.compute_transfer(lmax_transfer)

        prim_shape = Shape.prim_local(ns=1)

        self.assertTrue(len(cosmo.red_bispectra) == 0)
        cosmo.add_prim_reduced_bispectrum(prim_shape, radii)
        self.assertTrue(len(cosmo.red_bispectra) == 1)

        rb = cosmo.red_bispectra[0]

        # Lmax and pol of data should overrule those of bispectrum.
        lmax = 5
        alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        alm += np.random.randn(alm.size)
        alm += np.random.randn(alm.size) * 1j
        alm[:lmax+1] = alm[:lmax+1].real # Make sure m=0 is real.
        alm = alm.reshape((1, alm.size))

        red_bispectra = [rb]
        beam = lambda alm : alm
        pol = ('T',)
        npol = 1
        icov = lambda alm: alm
        icov_ell = np.ones((1, lmax + 1))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        estimator.mc_idx = 1

        # Make sure Fisher is 1 and linear term is 0.
        estimator.mc_gt_sq = 3.
        estimator.mc_gt = np.zeros_like(alm)

        self.assertEqual(estimator.compute_fisher(), 1)
        self.assertEqual(estimator.compute_linear_term(alm), 0.)

        estimate = estimator.compute_estimate(alm.copy())

        def red_bisp(ell1, ell2, ell3):
            '''
            b_l1l2l3 = int dr r^2 1/6 (alpha_l1(r) beta_l2(r) beta_l3(r) + 5 symm.)
            '''
            # Factors are direct output from radial functional, 
            # so they do not have the 2 * As^2 factor.

            if ell1 < 2 or ell2 < 2 or ell3 < 2:
                return 0

            # Determine indices into ell dimension of factors.
            lidx1 = np.where(rb.ells_full == ell1)[0][0]
            lidx2 = np.where(rb.ells_full == ell2)[0][0]
            lidx3 = np.where(rb.ells_full == ell3)[0][0]

            # Note that the symmetrisation below is not needed for the cubic term, but still
            # formally correct.
            
            amp = 2 * (2 * np.pi ** 2 * cosmo.camb_params.InitPower.As) ** 2 * (3 / 5)
            # First alpha_l1(r1) beta_l2(r1) beta_l3(r1).
            ret = dr * radii[0] ** 2 * \
                  rb.factors[0,0,lidx1] * rb.factors[2,0,lidx2] * rb.factors[2,0,lidx3]
            # beta_l1(r1) alpha_l2(r1) beta_l3(r1).
            ret += dr * radii[0] ** 2 * \
                   rb.factors[2,0,lidx1] * rb.factors[0,0,lidx2] * rb.factors[2,0,lidx3]
            # beta_l1(r1) beta_l2(r1) alpha_l3(r1).
            ret += dr * radii[0] ** 2 * \
                   rb.factors[2,0,lidx1] * rb.factors[2,0,lidx2] * rb.factors[0,0,lidx3]
            
            # Then alpha_l1(r2) beta_l2(r2) beta_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[1,0,lidx1] * rb.factors[3,0,lidx2] * rb.factors[3,0,lidx3]
            # beta_l1(r2) alpha_l2(r2) beta_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[3,0,lidx1] * rb.factors[1,0,lidx2] * rb.factors[3,0,lidx3]
            # beta_l1(r2) beta_l2(r2) alpha_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[3,0,lidx1] * rb.factors[3,0,lidx2] * rb.factors[1,0,lidx3]

            ret *= amp

            return ret

        alm = alm[0]
        estimate_exp = self.cubic_term_direct(alm, alm, alm, red_bisp)

        self.assertAlmostEqual(estimate / estimate_exp, 1, places=self.decimal)

    def test_ksw_compute_estimate_cubic_local_pol(self):

        # Compare to direct 5 dimensional sum over (l,m).
        # For local reduced bispectrum.

        lmax_transfer = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
        pars = camb.CAMBparams()
        pars.set_cosmology(**cosmo_opts)

        cosmo = Cosmology(pars)
        cosmo.compute_transfer(lmax_transfer)

        prim_shape = Shape.prim_local(ns=1)

        self.assertTrue(len(cosmo.red_bispectra) == 0)
        cosmo.add_prim_reduced_bispectrum(prim_shape, radii)
        self.assertTrue(len(cosmo.red_bispectra) == 1)

        rb = cosmo.red_bispectra[0]

        # Lmax and pol of data should overrule those of bispectrum.
        lmax = 5
        alm = np.zeros((2, hp.Alm.getsize(lmax)), dtype=np.complex128)
        alm += np.random.randn(alm.size).reshape(alm.shape)
        alm += np.random.randn(alm.size).reshape(alm.shape) * 1j
        alm[:,:lmax+1] = alm[:,:lmax+1].real # Make sure m=0 is real.

        red_bispectra = [rb]
        beam = lambda alm : alm
        pol = ('T', 'E')
        npol = 2
        icov = lambda alm: alm

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        estimator.mc_idx = 1

        # Make sure Fisher is 1 and linear term is 0.
        estimator.mc_gt_sq = 3.
        estimator.mc_gt = np.zeros_like(alm)

        self.assertEqual(estimator.compute_fisher(), 1)
        self.assertEqual(estimator.compute_linear_term(alm), 0.)

        estimate = estimator.compute_estimate(alm.copy())

        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            '''
            b^X1X2X3_l1l2l3 = int dr r^2 1/6 (alpha^X1_l1(r) beta^X2_l2(r) beta^X3_l3(r) + 5 symm.)
            '''
            # Factors are direct output from radial functional, 
            # so they do not have the 2 * As^2 factor.

            if ell1 < 2 or ell2 < 2 or ell3 < 2:
                return 0

            # Determine indices into ell dimension of factors.
            lidx1 = np.where(rb.ells_full == ell1)[0][0]
            lidx2 = np.where(rb.ells_full == ell2)[0][0]
            lidx3 = np.where(rb.ells_full == ell3)[0][0]

            # Note that the symmetrisation below is not needed for the cubic term, but still
            # formally correct.
            
            amp = 2 * (2 * np.pi ** 2 * cosmo.camb_params.InitPower.As) ** 2 * (3 / 5)
            # First alpha_l1(r1) beta_l2(r1) beta_l3(r1).
            ret = dr * radii[0] ** 2 * \
                  rb.factors[0,pidx1,lidx1] * rb.factors[2,pidx2,lidx2] * rb.factors[2,pidx3,lidx3]
            # beta_l1(r1) alpha_l2(r1) beta_l3(r1).
            ret += dr * radii[0] ** 2 * \
                   rb.factors[2,pidx1,lidx1] * rb.factors[0,pidx2,lidx2] * rb.factors[2,pidx3,lidx3]
            # beta_l1(r1) beta_l2(r1) alpha_l3(r1).
            ret += dr * radii[0] ** 2 * \
                   rb.factors[2,pidx1,lidx1] * rb.factors[2,pidx2,lidx2] * rb.factors[0,pidx3,lidx3]
            
            # Then alpha_l1(r2) beta_l2(r2) beta_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[1,pidx1,lidx1] * rb.factors[3,pidx2,lidx2] * rb.factors[3,pidx3,lidx3]
            # beta_l1(r2) alpha_l2(r2) beta_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[3,pidx1,lidx1] * rb.factors[1,pidx2,lidx2] * rb.factors[3,pidx3,lidx3]
            # beta_l1(r2) beta_l2(r2) alpha_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[3,pidx1,lidx1] * rb.factors[3,pidx2,lidx2] * rb.factors[1,pidx3,lidx3]

            ret *= amp

            return ret

        # Create separate function for each pol combo.
        red_bisp_III = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 0, 0, 0)
        red_bisp_IIE = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 0, 0, 1)
        red_bisp_IEI = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 0, 1, 0)
        red_bisp_EII = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 1, 0, 0)
        red_bisp_IEE = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 0, 1, 1)
        red_bisp_EIE = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 1, 0, 1)
        red_bisp_EEI = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 1, 1, 0)
        red_bisp_EEE = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 1, 1, 1)
        
        alm_I = alm[0]
        alm_E = alm[1]

        estimate_exp = self.cubic_term_direct(alm_I, alm_I, alm_I, red_bisp_III)
        estimate_exp += self.cubic_term_direct(alm_I, alm_I, alm_E, red_bisp_IIE)
        estimate_exp += self.cubic_term_direct(alm_I, alm_E, alm_I, red_bisp_IEI)
        estimate_exp += self.cubic_term_direct(alm_E, alm_I, alm_I, red_bisp_EII)
        estimate_exp += self.cubic_term_direct(alm_I, alm_E, alm_E, red_bisp_IEE)
        estimate_exp += self.cubic_term_direct(alm_E, alm_I, alm_E, red_bisp_EIE)
        estimate_exp += self.cubic_term_direct(alm_E, alm_E, alm_I, red_bisp_EEI)
        estimate_exp += self.cubic_term_direct(alm_E, alm_E, alm_E, red_bisp_EEE)

        # Added -1. Was needed on Rusty when compiled with no optimization...
        self.assertAlmostEqual(estimate / estimate_exp, 1, places=self.decimal-1)

    def test_ksw_compute_estimate_cubic_equilateral_I(self):

        # Compare to direct 5 dimensional sum over (l,m).
        # For equilateral reduced bispectrum.

        lmax_transfer = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
        pars = camb.CAMBparams()
        pars.set_cosmology(**cosmo_opts)

        cosmo = Cosmology(pars)
        cosmo.compute_transfer(lmax_transfer)

        prim_shape = Shape.prim_equilateral(ns=1)

        self.assertTrue(len(cosmo.red_bispectra) == 0)
        cosmo.add_prim_reduced_bispectrum(prim_shape, radii)
        self.assertTrue(len(cosmo.red_bispectra) == 1)

        rb = cosmo.red_bispectra[0]

        # Lmax and pol of data should overrule those of bispectrum.
        lmax = 5
        alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        alm += np.random.randn(alm.size)
        alm += np.random.randn(alm.size) * 1j
        alm[:lmax+1] = alm[:lmax+1].real # Make sure m=0 is real.
        alm = alm.reshape((1, alm.size))

        red_bispectra = [rb]
        beam = lambda alm : alm
        pol = ('T',)
        npol = 1
        icov = lambda alm: alm

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        estimator.mc_idx = 1

        # Make sure Fisher is 1 and linear term is 0.
        estimator.mc_gt_sq = 3.
        estimator.mc_gt = np.zeros_like(alm)

        self.assertEqual(estimator.compute_fisher(), 1)
        self.assertEqual(estimator.compute_linear_term(alm), 0.)

        estimate = estimator.compute_estimate(alm.copy())

        def red_bisp(ell1, ell2, ell3):
            '''
            b_l1l2l3 = int dr r^2 1/6 (alpha_l1(r) beta_l2(r) beta_l3(r) + 5 symm.)
            '''
            # Factors are direct output from radial functional, 
            # so they do not have the 2 * As^2 factor.

            if ell1 < 2 or ell2 < 2 or ell3 < 2:
                return 0

            # Determine indices into ell dimension of factors.
            lidx1 = np.where(rb.ells_full == ell1)[0][0]
            lidx2 = np.where(rb.ells_full == ell2)[0][0]
            lidx3 = np.where(rb.ells_full == ell3)[0][0]

            amp = 2 * (2 * np.pi ** 2 * cosmo.camb_params.InitPower.As) ** 2 * (3 / 5)

            # First -3 alpha_l1(r1) beta_l2(r1) beta_l3(r1).
            ret = -3 * dr * radii[0] ** 2 * \
                  rb.factors[0,0,lidx1] * rb.factors[2,0,lidx2] * rb.factors[2,0,lidx3]
            # -3 beta_l1(r1) alpha_l2(r1) beta_l3(r1).
            ret -= 3 * dr * radii[0] ** 2 * \
                  rb.factors[2,0,lidx1] * rb.factors[0,0,lidx2] * rb.factors[2,0,lidx3]
            # -3 beta_l1(r1) beta_l2(r1) alpha_l3(r1).
            ret -= 3 * dr * radii[0] ** 2 * \
                  rb.factors[2,0,lidx1] * rb.factors[2,0,lidx2] * rb.factors[0,0,lidx3]
            
            # Then -6 delta_l1(r1) delta_l2(r1) delta_l3(r1).
            ret -= 6 * dr * radii[0] ** 2 * \
                   rb.factors[6,0,lidx1] * rb.factors[6,0,lidx2] * rb.factors[6,0,lidx3]

            # Then +3 beta_l1(r1) gamma_l2(r1) delta_l3(r1).
            ret += 3 * dr * radii[0] ** 2 * \
                   rb.factors[2,0,lidx1] * rb.factors[4,0,lidx2] * rb.factors[6,0,lidx3]
            # +3 delta_l1(r1) beta_l2(r1) gamma_l3(r1).
            ret += 3 * dr * radii[0] ** 2 * \
                   rb.factors[6,0,lidx1] * rb.factors[2,0,lidx2] * rb.factors[4,0,lidx3]
            # +3 gamma_l1(r1) delta_l2(r1) beta_l3(r1).
            ret += 3 * dr * radii[0] ** 2 * \
                   rb.factors[4,0,lidx1] * rb.factors[6,0,lidx2] * rb.factors[2,0,lidx3]
            # +3 beta_l1(r1) delta_l2(r1) gamma_l3(r1).
            ret += 3 * dr * radii[0] ** 2 * \
                   rb.factors[2,0,lidx1] * rb.factors[6,0,lidx2] * rb.factors[4,0,lidx3]
            # +3 gamma_l1(r1) beta_l2(r1) delta_l3(r1).
            ret += 3 * dr * radii[0] ** 2 * \
                   rb.factors[4,0,lidx1] * rb.factors[2,0,lidx2] * rb.factors[6,0,lidx3]
            # +3 delta_l1(r1) gamma_l2(r1) beta_l3(r1).
            ret += 3 * dr * radii[0] ** 2 * \
                   rb.factors[6,0,lidx1] * rb.factors[4,0,lidx2] * rb.factors[2,0,lidx3]

            # Then, the same for r2: -3 alpha_l1(r2) beta_l2(r2) beta_l3(r2).
            ret -= 3 * dr * radii[1] ** 2 * \
                   rb.factors[1,0,lidx1] * rb.factors[3,0,lidx2] * rb.factors[3,0,lidx3]
            # -3 beta_l1(r2) alpha_l2(r2) beta_l3(r2).
            ret -= 3 * dr * radii[1] ** 2 * \
                   rb.factors[3,0,lidx1] * rb.factors[1,0,lidx2] * rb.factors[3,0,lidx3]
            # -3 beta_l1(r2) beta_l2(r2) alpha_l3(r2).
            ret -= 3 * dr * radii[1] ** 2 * \
                   rb.factors[3,0,lidx1] * rb.factors[3,0,lidx2] * rb.factors[1,0,lidx3]
            
            # Then -6 delta_l1(r2) delta_l2(r2) delta_l3(r2).
            ret -= 6 * dr * radii[1] ** 2 * \
                   rb.factors[7,0,lidx1] * rb.factors[7,0,lidx2] * rb.factors[7,0,lidx3]

            # Then +3 beta_l1(r2) gamma_l2(r2) delta_l3(r2).
            ret += 3 * dr * radii[1] ** 2 * \
                   rb.factors[3,0,lidx1] * rb.factors[5,0,lidx2] * rb.factors[7,0,lidx3]
            # +3 delta_l1(r2) beta_l2(r2) gamma_l3(r2).
            ret += 3 * dr * radii[1] ** 2 * \
                   rb.factors[7,0,lidx1] * rb.factors[3,0,lidx2] * rb.factors[5,0,lidx3]
            # +3 gamma_l1(r2) delta_l2(r2) beta_l3(r2).
            ret += 3 * dr * radii[1] ** 2 * \
                   rb.factors[5,0,lidx1] * rb.factors[7,0,lidx2] * rb.factors[3,0,lidx3]
            # +3 beta_l1(r2) delta_l2(r2) gamma_l3(r2).
            ret += 3 * dr * radii[1] ** 2 * \
                   rb.factors[3,0,lidx1] * rb.factors[7,0,lidx2] * rb.factors[5,0,lidx3]
            # +3 gamma_l1(r2) beta_l2(r2) delta_l3(r2).
            ret += 3 * dr * radii[1] ** 2 * \
                   rb.factors[5,0,lidx1] * rb.factors[3,0,lidx2] * rb.factors[7,0,lidx3]
            # +3 delta_l1(r2) gamma_l2(r2) beta_l3(r2).
            ret += 3 * dr * radii[1] ** 2 * \
                   rb.factors[7,0,lidx1] * rb.factors[5,0,lidx2] * rb.factors[3,0,lidx3]

            ret *= amp

            return ret

        alm = alm[0]
        estimate_exp = self.cubic_term_direct(alm, alm, alm, red_bisp)

        self.assertAlmostEqual(estimate / estimate_exp, 1, places=self.decimal)

    def test_ksw_step_I_simple(self):

        np.random.seed(1)

        # Compare to direct 4 dimensional sum over (l,m).
        lmax = 5
        alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        alm += np.random.randn(alm.size)
        alm += np.random.randn(alm.size) * 1j
        alm[:lmax+1] = alm[:lmax+1].real # Make sure m=0 is real.
        alm = alm.reshape((1, alm.size))

        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T',)
        npol = 1
        icov = lambda alm: alm

        # Create a reduced bispectrum that is just b_l1l2l3 = 1.
        rb = red_bispectra[0]
        rb.npol = 1
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        estimator.step(alm.copy())

        def red_bisp(ell1, ell2, ell3):
            return 1.

        alm = alm[0]
        grad_exp = self.grad_direct(alm, alm, red_bisp)

        np.testing.assert_array_almost_equal(estimator.mc_gt[0], grad_exp, 
                                             decimal=self.decimal)

        mc_gt_sq_exp = np.sum(2 * np.real(grad_exp * np.conj(grad_exp)))
        mc_gt_sq_exp -= np.sum(grad_exp[:lmax+1].real ** 2 )
        
        np.testing.assert_array_almost_equal(estimator.mc_gt_sq / mc_gt_sq_exp,
                                             np.ones_like(mc_gt_sq_exp), 
                                             decimal=self.decimal)
    def test_ksw_step_pol_simple(self):

        np.random.seed(1)

        lmax = 5
        alm = np.zeros((2, hp.Alm.getsize(lmax)), dtype=np.complex128)
        alm += np.random.randn(alm.size).reshape(alm.shape)
        alm += np.random.randn(alm.size).reshape(alm.shape) * 1j
        alm[:,:lmax+1] = alm[:,:lmax+1].real # Make sure m=0 is real.

        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T', 'E')
        npol = 2
        icov = lambda alm: alm

        # Create a reduced bispectrum with factors that are 1 for I and 2 for E.
        rb = red_bispectra[0]
        rb.npol = 2
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.factors[:,1,:] = 2
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        estimator.step(alm.copy())

        def red_bisp_III(ell1, ell2, ell3):
            return 1 * 1 * 1
        def red_bisp_IIE(ell1, ell2, ell3):
            return 1 * 1 * 2
        def red_bisp_IEI(ell1, ell2, ell3):
            return 1 * 2 * 1
        def red_bisp_EII(ell1, ell2, ell3):
            return 2 * 1 * 1
        def red_bisp_IEE(ell1, ell2, ell3):
            return 1 * 2 * 2
        def red_bisp_EIE(ell1, ell2, ell3):
            return 2 * 1 * 2
        def red_bisp_EEI(ell1, ell2, ell3):
            return 2 * 2 * 1
        def red_bisp_EEE(ell1, ell2, ell3):
            return 2 * 2 * 2

        alm_I = alm[0]
        alm_E = alm[1]
        
        # Grad is now 2d and is sum of all pol combinations.
        grad_exp_I = self.grad_direct(alm_I, alm_I, red_bisp_III)
        grad_exp_I += self.grad_direct(alm_I, alm_E, red_bisp_IIE)
        grad_exp_I += self.grad_direct(alm_E, alm_I, red_bisp_IEI)
        grad_exp_I += self.grad_direct(alm_E, alm_E, red_bisp_IEE)

        grad_exp_E = self.grad_direct(alm_I, alm_I, red_bisp_EII)
        grad_exp_E += self.grad_direct(alm_I, alm_E, red_bisp_EIE)
        grad_exp_E += self.grad_direct(alm_E, alm_I, red_bisp_EEI)
        grad_exp_E += self.grad_direct(alm_E, alm_E, red_bisp_EEE)

        np.testing.assert_array_almost_equal(estimator.mc_gt[0], grad_exp_I, 
                                             decimal=self.decimal)
        np.testing.assert_array_almost_equal(estimator.mc_gt[1], grad_exp_E, 
                                             decimal=self.decimal)

        # We use diagonal cov in ell, m and pol in this example, so just sum of the two.
        mc_gt_sq_exp = np.sum(2 * np.real(grad_exp_I * np.conj(grad_exp_I)))
        mc_gt_sq_exp -= np.sum(grad_exp_I[:lmax+1].real ** 2 )

        mc_gt_sq_exp += np.sum(2 * np.real(grad_exp_E * np.conj(grad_exp_E)))
        mc_gt_sq_exp -= np.sum(grad_exp_E[:lmax+1].real ** 2 )
        
        np.testing.assert_array_almost_equal(estimator.mc_gt_sq / mc_gt_sq_exp, 
                                             np.ones_like(mc_gt_sq_exp),
                                             decimal=self.decimal)
    def test_ksw_step_I_simple_2d(self):

        np.random.seed(1)

        # Compare to direct 4 dimensional sum over (l,m).
        lmax = 5
        alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        alm += np.random.randn(alm.size)
        alm += np.random.randn(alm.size) * 1j
        alm[:lmax+1] = alm[:lmax+1].real # Make sure m=0 is real.
        alm = alm.reshape((1, alm.size))

        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T',)
        npol = 1
        icov = lambda alm: alm

        # Create a reduced bispectrum that is b_l1l2l3 = 1 * 1 * 1 + 2 * 2 * 2.
        rb = red_bispectra[0]
        rb.npol = 1
        rb.nfact = 2
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0

        rb.factors = np.ones((2, npol, lmax + 1))
        rb.factors[1] *= 2
        rb.rule = np.zeros((2, 3), dtype=int)
        rb.rule[1] = 1
        rb.weights = np.ones((2, 3))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        estimator.step(alm.copy())

        def red_bisp(ell1, ell2, ell3):
            return 1 * 1 * 1 + 2 * 2 * 2

        alm = alm[0]
        grad_exp = self.grad_direct(alm, alm, red_bisp)

        np.testing.assert_array_almost_equal(estimator.mc_gt[0], grad_exp, 
                                             decimal=self.decimal)

        mc_gt_sq_exp = np.sum(2 * np.real(grad_exp * np.conj(grad_exp)))
        mc_gt_sq_exp -= np.sum(grad_exp[:lmax+1].real ** 2 )
        
        np.testing.assert_array_almost_equal(estimator.mc_gt_sq / mc_gt_sq_exp, 
                                             np.ones_like(mc_gt_sq_exp),
                                             decimal=self.decimal)

    def test_ksw_step_pol_simple_2d(self):

        np.random.seed(1)

        lmax = 5
        alm = np.zeros((2, hp.Alm.getsize(lmax)), dtype=np.complex128)
        alm += np.random.randn(alm.size).reshape(alm.shape)
        alm += np.random.randn(alm.size).reshape(alm.shape) * 1j
        alm[:,:lmax+1] = alm[:,:lmax+1].real # Make sure m=0 is real.

        red_bispectra = [self.FakeReducedBispectrum()]
        beam = lambda alm : alm
        pol = ('T', 'E')
        npol = 2
        icov = lambda alm: alm

        # Create a reduced bispectrum that is sum of 1 for I and 2 for E
        # and 3 for I and 6 for E.
        rb = red_bispectra[0]
        rb.npol = 2
        rb.nfact = 2
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0

        rb.factors = np.ones((2, npol, lmax + 1))
        rb.factors[0,1,:] = 2
        rb.factors[1,0,:] = 3
        rb.factors[1,1,:] = 6
        rb.rule = np.zeros((2, 3), dtype=int)
        rb.rule[1] = 1
        rb.weights = np.ones((2, 3))

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        estimator.step(alm.copy())

        def red_bisp_III(ell1, ell2, ell3):
            return 1 * 1 * 1 + 3 * 3 * 3
        def red_bisp_IIE(ell1, ell2, ell3):
            return 1 * 1 * 2 + 3 * 3 * 6
        def red_bisp_IEI(ell1, ell2, ell3):
            return 1 * 2 * 1 + 3 * 6 * 3
        def red_bisp_EII(ell1, ell2, ell3):
            return 2 * 1 * 1 + 6 * 3 * 3
        def red_bisp_IEE(ell1, ell2, ell3):
            return 1 * 2 * 2 + 3 * 6 * 6
        def red_bisp_EIE(ell1, ell2, ell3):
            return 2 * 1 * 2 + 6 * 3 * 6
        def red_bisp_EEI(ell1, ell2, ell3):
            return 2 * 2 * 1 + 6 * 6 * 3
        def red_bisp_EEE(ell1, ell2, ell3):
            return 2 * 2 * 2 + 6 * 6 * 6

        alm_I = alm[0]
        alm_E = alm[1]
        
        # Grad is now 2d and is sum of all pol combinations.
        grad_exp_I = self.grad_direct(alm_I, alm_I, red_bisp_III)
        grad_exp_I += self.grad_direct(alm_I, alm_E, red_bisp_IIE)
        grad_exp_I += self.grad_direct(alm_E, alm_I, red_bisp_IEI)
        grad_exp_I += self.grad_direct(alm_E, alm_E, red_bisp_IEE)

        grad_exp_E = self.grad_direct(alm_I, alm_I, red_bisp_EII)
        grad_exp_E += self.grad_direct(alm_I, alm_E, red_bisp_EIE)
        grad_exp_E += self.grad_direct(alm_E, alm_I, red_bisp_EEI)
        grad_exp_E += self.grad_direct(alm_E, alm_E, red_bisp_EEE)

        np.testing.assert_array_almost_equal(estimator.mc_gt[0] / grad_exp_I,
                                             np.ones_like(grad_exp_I),
                                             decimal=self.decimal)
        np.testing.assert_array_almost_equal(estimator.mc_gt[1] / grad_exp_E, 
                                             np.ones_like(grad_exp_E),
                                             decimal=self.decimal)

        # We use diagonal cov in ell, m and pol in this example, so just sum of the two.
        mc_gt_sq_exp = np.sum(2 * np.real(grad_exp_I * np.conj(grad_exp_I)))
        mc_gt_sq_exp -= np.sum(grad_exp_I[:lmax+1].real ** 2 )

        mc_gt_sq_exp += np.sum(2 * np.real(grad_exp_E * np.conj(grad_exp_E)))
        mc_gt_sq_exp -= np.sum(grad_exp_E[:lmax+1].real ** 2 )
        
        np.testing.assert_array_almost_equal(estimator.mc_gt_sq / mc_gt_sq_exp, 
                                             np.ones_like(mc_gt_sq_exp),
                                             decimal=self.decimal)

    def test_ksw_step_local_I(self):

        # Compare to direct 4 dimensional sum over (l,m).
        # For local reduced bispectrum.

        lmax_transfer = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
        pars = camb.CAMBparams()
        pars.set_cosmology(**cosmo_opts)

        cosmo = Cosmology(pars)
        cosmo.compute_transfer(lmax_transfer)

        prim_shape = Shape.prim_local(ns=1)

        self.assertTrue(len(cosmo.red_bispectra) == 0)
        cosmo.add_prim_reduced_bispectrum(prim_shape, radii)
        self.assertTrue(len(cosmo.red_bispectra) == 1)

        rb = cosmo.red_bispectra[0]

        # Lmax and pol of data should overrule those of bispectrum.
        lmax = 5
        alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        alm += np.random.randn(alm.size)
        alm += np.random.randn(alm.size) * 1j
        alm[:lmax+1] = alm[:lmax+1].real # Make sure m=0 is real.
        alm = alm.reshape((1, alm.size))

        red_bispectra = [rb]
        beam = lambda alm : alm
        pol = ('T',)
        npol = 1
        icov = lambda alm: alm

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        estimator.step(alm.copy())

        def red_bisp(ell1, ell2, ell3):
            '''
            b_l1l2l3 = int dr r^2 1/6 (alpha_l1(r) beta_l2(r) beta_l3(r) + 5 symm.)
            '''
            # Factors are direct output from radial functional, 
            # so they do not have the 2 * As^2 factor.

            if ell1 < 2 or ell2 < 2 or ell3 < 2:
                return 0

            # Determine indices into ell dimension of factors.
            lidx1 = np.where(rb.ells_full == ell1)[0][0]
            lidx2 = np.where(rb.ells_full == ell2)[0][0]
            lidx3 = np.where(rb.ells_full == ell3)[0][0]

            amp = 2 * (2 * np.pi ** 2 * cosmo.camb_params.InitPower.As) ** 2 * (3 / 5)
            # First alpha_l1(r1) beta_l2(r1) beta_l3(r1).
            ret = dr * radii[0] ** 2 * \
                  rb.factors[0,0,lidx1] * rb.factors[2,0,lidx2] * rb.factors[2,0,lidx3]
            # beta_l1(r1) alpha_l2(r1) beta_l3(r1).
            ret += dr * radii[0] ** 2 * \
                  rb.factors[2,0,lidx1] * rb.factors[0,0,lidx2] * rb.factors[2,0,lidx3]
            # beta_l1(r1) beta_l2(r1) alpha_l3(r1).
            ret += dr * radii[0] ** 2 * \
                  rb.factors[2,0,lidx1] * rb.factors[2,0,lidx2] * rb.factors[0,0,lidx3]
            
            # Then alpha_l1(r2) beta_l2(r2) beta_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[1,0,lidx1] * rb.factors[3,0,lidx2] * rb.factors[3,0,lidx3]
            # beta_l1(r2) alpha_l2(r2) beta_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[3,0,lidx1] * rb.factors[1,0,lidx2] * rb.factors[3,0,lidx3]
            # beta_l1(r2) beta_l2(r2) alpha_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[3,0,lidx1] * rb.factors[3,0,lidx2] * rb.factors[1,0,lidx3]

            ret *= amp

            return ret

        alm = alm[0]
        grad_exp = self.grad_direct(alm, alm, red_bisp)

        np.testing.assert_allclose(estimator.mc_gt[0] , grad_exp, 
                                   rtol=10 ** -self.decimal)

        mc_gt_sq_exp = np.sum(2 * np.real(grad_exp * np.conj(grad_exp)))
        mc_gt_sq_exp -= np.sum(grad_exp[:lmax+1].real ** 2 )
        
        np.testing.assert_array_almost_equal(estimator.mc_gt_sq /  mc_gt_sq_exp, 
                                             np.ones_like(mc_gt_sq_exp),
                                             decimal=self.decimal)

    def test_ksw_step_local_pol(self):

        # Compare to direct 4 dimensional sum over (l,m).
        # For local reduced bispectrum.

        np.random.seed(1)

        lmax_transfer = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
        pars = camb.CAMBparams()
        pars.set_cosmology(**cosmo_opts)

        cosmo = Cosmology(pars)
        cosmo.compute_transfer(lmax_transfer)

        prim_shape = Shape.prim_local(ns=1)

        self.assertTrue(len(cosmo.red_bispectra) == 0)
        cosmo.add_prim_reduced_bispectrum(prim_shape, radii)
        self.assertTrue(len(cosmo.red_bispectra) == 1)

        rb = cosmo.red_bispectra[0]

        # Lmax and pol of data should overrule those of bispectrum.
        lmax = 5
        alm = np.zeros((2, hp.Alm.getsize(lmax)), dtype=np.complex128)
        alm += np.random.randn(alm.size).reshape(alm.shape)
        alm += np.random.randn(alm.size).reshape(alm.shape) * 1j
        alm[:,:lmax+1] = alm[:,:lmax+1].real # Make sure m=0 is real.

        red_bispectra = [rb]
        beam = lambda alm : alm
        pol = ('T', 'E')
        npol = 2
        icov = lambda alm: alm

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)
        estimator.step(alm.copy())

        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            '''
            b^X1X2X3_l1l2l3 = int dr r^2 1/6 (alpha^X1_l1(r) beta^X2_l2(r) beta^X3_l3(r) + 5 symm.)
            '''
            # Factors are direct output from radial functional, 
            # so they do not have the 2 * As^2 factor.

            if ell1 < 2 or ell2 < 2 or ell3 < 2:
                return 0

            # Determine indices into ell dimension of factors.
            lidx1 = np.where(rb.ells_full == ell1)[0][0]
            lidx2 = np.where(rb.ells_full == ell2)[0][0]
            lidx3 = np.where(rb.ells_full == ell3)[0][0]

            # Note that the symmetrisation below is not needed for the cubic term, but still
            # formally correct.
            
            amp = 2 * (2 * np.pi ** 2 * cosmo.camb_params.InitPower.As) ** 2 * (3 / 5)
            # First alpha_l1(r1) beta_l2(r1) beta_l3(r1).
            ret = dr * radii[0] ** 2 * \
                  rb.factors[0,pidx1,lidx1] * rb.factors[2,pidx2,lidx2] * rb.factors[2,pidx3,lidx3]
            # beta_l1(r1) alpha_l2(r1) beta_l3(r1).
            ret += dr * radii[0] ** 2 * \
                   rb.factors[2,pidx1,lidx1] * rb.factors[0,pidx2,lidx2] * rb.factors[2,pidx3,lidx3]
            # beta_l1(r1) beta_l2(r1) alpha_l3(r1).
            ret += dr * radii[0] ** 2 * \
                   rb.factors[2,pidx1,lidx1] * rb.factors[2,pidx2,lidx2] * rb.factors[0,pidx3,lidx3]
            
            # Then alpha_l1(r2) beta_l2(r2) beta_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[1,pidx1,lidx1] * rb.factors[3,pidx2,lidx2] * rb.factors[3,pidx3,lidx3]
            # beta_l1(r2) alpha_l2(r2) beta_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[3,pidx1,lidx1] * rb.factors[1,pidx2,lidx2] * rb.factors[3,pidx3,lidx3]
            # beta_l1(r2) beta_l2(r2) alpha_l3(r2).
            ret += dr * radii[1] ** 2 * \
                   rb.factors[3,pidx1,lidx1] * rb.factors[3,pidx2,lidx2] * rb.factors[1,pidx3,lidx3]

            ret *= amp

            return ret

        # Create separate function for each pol combo.
        red_bisp_III = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 0, 0, 0)
        red_bisp_IIE = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 0, 0, 1)
        red_bisp_IEI = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 0, 1, 0)
        red_bisp_EII = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 1, 0, 0)
        red_bisp_IEE = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 0, 1, 1)
        red_bisp_EIE = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 1, 0, 1)
        red_bisp_EEI = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 1, 1, 0)
        red_bisp_EEE = lambda ell1, ell2, ell3: red_bisp(ell1, ell2, ell3, 1, 1, 1)

        alm_I = alm[0]
        alm_E = alm[1]
        
        # Grad is now 2d and is sum of all pol combinations.
        grad_exp_I = self.grad_direct(alm_I, alm_I, red_bisp_III)
        grad_exp_I += self.grad_direct(alm_I, alm_E, red_bisp_IIE)
        grad_exp_I += self.grad_direct(alm_E, alm_I, red_bisp_IEI)
        grad_exp_I += self.grad_direct(alm_E, alm_E, red_bisp_IEE)

        grad_exp_E = self.grad_direct(alm_I, alm_I, red_bisp_EII)
        grad_exp_E += self.grad_direct(alm_I, alm_E, red_bisp_EIE)
        grad_exp_E += self.grad_direct(alm_E, alm_I, red_bisp_EEI)
        grad_exp_E += self.grad_direct(alm_E, alm_E, red_bisp_EEE)

        # The 4th element sometimes gets error that slightly larger than 1e-5.
        np.testing.assert_allclose(estimator.mc_gt[0], grad_exp_I, 
                                   rtol=10 ** -(self.decimal - 1))
        np.testing.assert_allclose(estimator.mc_gt[1], grad_exp_E, 
                                   rtol=10 ** -self.decimal)

        # We use diagonal cov in ell, m and pol in this example, so just sum of the two.
        mc_gt_sq_exp = np.sum(2 * np.real(grad_exp_I * np.conj(grad_exp_I)))
        mc_gt_sq_exp -= np.sum(grad_exp_I[:lmax+1].real ** 2 )

        mc_gt_sq_exp += np.sum(2 * np.real(grad_exp_E * np.conj(grad_exp_E)))
        mc_gt_sq_exp -= np.sum(grad_exp_E[:lmax+1].real ** 2 )
        
        np.testing.assert_allclose(estimator.mc_gt_sq, mc_gt_sq_exp, 
                                   rtol=10 ** -self.decimal)

    def test_ksw_read_write_state(self):

        lmax = 300
        red_bispectra = [self.FakeReducedBispectrum()]
        icov = self.FakeData().icov_diag_nonlensed
        beam = lambda alm : alm
        pol = ('T', 'E')

        estimator = KSW(red_bispectra, icov, beam, lmax, pol, precision=self.precision)

        with tempfile.TemporaryDirectory(dir=self.path) as tmpdirname:
            
            filename = os.path.join(tmpdirname, 'state')
            estimator.write_state(filename)

            estimator.start_from_read_state(filename)

        self.assertEqual(estimator.mc_idx, 0)
        self.assertIs(estimator.mc_gt_sq, None)
        self.assertIs(estimator.mc_gt, None)
        
        # Now update state and repeat.
        mc_gt = np.ones((len(pol), hp.Alm.getsize(lmax)),
                        dtype=complex) * 10
        estimator.mc_gt = mc_gt.copy()
        estimator.mc_gt_sq = 100
        estimator.mc_idx = 5

        with tempfile.TemporaryDirectory(dir=self.path) as tmpdirname:
            
            filename = os.path.join(tmpdirname, 'state')
            estimator.write_state(filename)

            estimator.start_from_read_state(filename)

        self.assertEqual(estimator.mc_idx, 5)
        self.assertAlmostEqual(estimator.mc_gt_sq, 100 / 5)
        np.testing.assert_allclose(estimator.mc_gt, mc_gt / 5)

class TestKSW_32(TestKSW_64):

    @classmethod
    def setUpClass(cls):
        cls.precision = 'single'
        cls.decimal = 5
