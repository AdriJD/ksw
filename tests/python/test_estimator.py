import unittest
import numpy as np

import healpy as hp
import camb

from ksw import KSW
from ksw import Cosmology
from ksw import Shape
from ksw import legendre
from ksw import utils

class TestKSW(unittest.TestCase):

    def setUp(self):
        # Is called before each test.

        class FakeReducedBispectrum():
            def __init__(self):

                self.npol = 2
                self.nfact = 4

                self.ells_sparse = np.arange(2, 10)
                self.ells_full = np.arange(2, 10)
                self.factors = np.ones((3, self.npol, len(self.ells_full)))
                self.rule = np.ones((self.nfact, 3), dtype=int)
                self.weights = np.ones((self.nfact, 3, self.npol))
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

        data = self.FakeData()
        estimator = KSW(data)

        self.assertIs(estimator.data, data)
        self.assertIs(estimator.cosmology, data.cosmology)
        self.assertTrue(callable(estimator.icov))
        self.assertEqual(estimator.icov, data.icov_diag_nonlensed)

        self.assertEqual(estimator.mc_idx, 0)
        self.assertIs(estimator.mc_gt, None)
        self.assertIs(estimator.mc_gt_sq, None)

        self.assertEqual(estimator.thetas.size, estimator.theta_weights.size)
        self.assertEqual(estimator.n_ell_phi.shape,
                (estimator.data.npol, estimator.data.lmax + 1, estimator.nphi))
        self.assertEqual(estimator.m_ell_m.shape,
                (estimator.data.npol, estimator.data.lmax + 1, estimator.nphi // 2 + 1))
        self.assertTrue(callable(estimator.fft_forward))
        self.assertTrue(callable(estimator.fft_backward))

    def test_ksw_mc_gt_sq(self):

        data = self.FakeData()
        estimator = KSW(data)

        self.assertIs(estimator.mc_gt_sq, None)

        estimator.mc_gt_sq = 10
        estimator.mc_idx = 1

        self.assertEqual(estimator.mc_gt_sq, 10)

        estimator.mc_idx = 3

        self.assertEqual(estimator.mc_gt_sq, 10 / 3.)

    def test_ksw_mc_gt(self):

        data = self.FakeData()
        estimator = KSW(data)

        self.assertIs(estimator.mc_gt, None)

        mc_gt = np.ones((data.npol, hp.Alm.getsize(data.lmax)),
                                      dtype=complex)
        estimator.mc_gt = mc_gt.copy() * 10
        estimator.mc_idx = 1

        np.testing.assert_almost_equal(estimator.mc_gt, mc_gt * 10)

        estimator.mc_idx = 3

        np.testing.assert_almost_equal(estimator.mc_gt, 10 * mc_gt / 3.)

    def test_ksw_get_coords(self):

        data = self.FakeData()
        estimator = KSW(data)

        self.assertEqual(estimator.thetas.size, (3 * data.lmax) // 2 + 1)
        self.assertTrue(np.all(0 <= estimator.thetas))
        self.assertTrue(np.all(estimator.thetas <= np.pi))
        self.assertTrue(estimator.nphi >= 3 * data.lmax + 1)

        # Updating lmax will result in different coords.
        data.lmax = 10
        thetas, tw, nphi = estimator.get_coords()
        self.assertEqual(thetas.size, (3 * 10) // 2 + 1)
        self.assertTrue(np.all(0 <= thetas))
        self.assertTrue(np.all(thetas <= np.pi))
        self.assertTrue(nphi >= 3 * 10 + 1)

    def test_ksw_init_fft(self):

        # Verify that I understand the fft normalization.

        data = self.FakeData()
        estimator = KSW(data)

        # Forward.
        phis = np.linspace(0, 2 * np.pi, estimator.nphi, endpoint=False)
        m = 10
        estimator.n_ell_phi[:,:,:] = 1.
        estimator.n_ell_phi[:,:,:] *= np.cos(m * phis)
        estimator.fft_forward()

        amp_exp = phis.size / 2.
        m_ell_m_exp = np.zeros_like(estimator.m_ell_m)
        m_ell_m_exp[:,:,m] = amp_exp

        np.testing.assert_array_almost_equal(estimator.m_ell_m, m_ell_m_exp)

        # Now the backward transform.
        estimator.m_ell_m *= 0
        estimator.m_ell_m[:,:,m] = 1.
        estimator.fft_backward()

        n_ell_phi_exp = np.ones((data.npol, data.lmax + 1, estimator.nphi),
                                   dtype=np.float64)
        n_ell_phi_exp[:,:,:] *= np.cos(m * phis) * 2 / estimator.nphi

        np.testing.assert_array_almost_equal(estimator.n_ell_phi, n_ell_phi_exp)

    def test_ksw_sht_monopole(self):

        # Test round-trip of SHT transforms for monopole alm input.
        data = self.FakeData()
        data.lmax = 50
        estimator = KSW(data)

        # SHT backward: alm2map.
        ms = np.arange(data.lmax + 1)
        theta = estimator.thetas[2]
        weight = estimator.theta_weights[2]

        estimator.m_ell_m *= 0
        estimator.m_ell_m[:,0,0] = 1
        # Correcting for the fft normalization.
        estimator.m_ell_m[:,0,0] *= estimator.nphi
        estimator.m_ell_m[:,:,:data.lmax+1] *= np.transpose(
            legendre.normalized_associated_legendre_ms(ms, theta, data.lmax))

        estimator.fft_backward()
        ring = np.sum(estimator.n_ell_phi[0], axis=0) # Sum over ells.

        ring_exp = np.ones(estimator.nphi) * 1 / np.sqrt(4 * np.pi)
        np.testing.assert_array_almost_equal(ring, ring_exp)

        # Forwards: map2alm.
        alm_exp = np.zeros_like(estimator.m_ell_m)
        alm_exp[:,0,0] = 1.
        estimator.fft_forward() # Note, only ell=0 row is nonzero.
        ylm = np.transpose(legendre.normalized_associated_legendre_ms(ms, theta, data.lmax))
        alm = np.zeros_like(estimator.m_ell_m)
        for weight in estimator.theta_weights:
            alm[:,:data.lmax+1,:data.lmax+1] += weight * \
                estimator.m_ell_m[:,:data.lmax+1,:data.lmax+1] * \
                (2 * np.pi / estimator.nphi) * ylm

        np.testing.assert_array_almost_equal(alm, alm_exp)

    def test_ksw_sht(self):

        data = self.FakeData()
        data.lmax = 150
        estimator = KSW(data)

        # SHT backward: alm2map.
        ms = np.arange(data.lmax + 1)

        rings = np.empty((estimator.thetas.size, estimator.nphi))

        for tidx, theta in enumerate(estimator.thetas):

            estimator.m_ell_m *= 0
            estimator.m_ell_m[:,0,0] = 1 # ell = m = 0.
            estimator.m_ell_m[:,1,1] = 1j # ell = m = 1.
            estimator.m_ell_m[:,2,1] = 1 # ell = m = 1.
            # Correcting for the fft normalization.
            estimator.m_ell_m[:,:,:] *= estimator.nphi # You dont need factor 2.
            estimator.m_ell_m[:,:,:data.lmax+1] *= np.transpose(
                legendre.normalized_associated_legendre_ms(ms, theta, data.lmax))

            estimator.fft_backward()
            rings[tidx,:] = np.sum(estimator.n_ell_phi[0], axis=0) # Sum over ells.

        # Compare to healpy.
        nside = 256
        alm = np.zeros(hp.Alm.getsize(data.lmax), dtype=complex)
        alm[hp.Alm.getidx(data.lmax, 0, 0)] = 1
        alm[hp.Alm.getidx(data.lmax, 1, 1)] = 1j
        alm[hp.Alm.getidx(data.lmax, 2, 1)] = 1
        map_hp = hp.alm2map(alm, nside)

        phis = np.linspace(0, 2 * np.pi, estimator.nphi, endpoint=False)

        for tidx, theta in enumerate(estimator.thetas):
            np.testing.assert_array_almost_equal(
                rings[tidx], map_hp[hp.ang2pix(nside, theta, phis)],
                decimal=2) # Bit worrisome, but I think only a few elements are off.

            # Compare to analytical result.
            ring_exp = 2 * np.sqrt(3 / 8 / np.pi) * np.sin(theta) * np.sin(phis) \
                + 1 / np.sqrt(4 * np.pi) + \
                -2 * np.sqrt(15 / 8 / np.pi) * np.sin(theta) * np.cos(theta) * np.cos(phis)

            np.testing.assert_array_almost_equal(rings[tidx], ring_exp)

        # Forward: map2alm.
        alm_exp = np.zeros_like(estimator.m_ell_m)
        alm_exp[:,0,0] = 1.
        alm_exp[:,1,1] = 1j
        alm_exp[:,2,1] = 1.

        alm = np.zeros_like(estimator.m_ell_m)
        for tidx, (theta, weight) in enumerate(zip(estimator.thetas, estimator.theta_weights)):
            estimator.n_ell_phi *= 0
            estimator.n_ell_phi[:,:,:] = rings[tidx]
            estimator.fft_forward()
            ylm = np.transpose(
                legendre.normalized_associated_legendre_ms(ms, theta, data.lmax))

            alm[:,:data.lmax+1,:data.lmax+1] += weight * \
                estimator.m_ell_m[:,:data.lmax+1,:data.lmax+1] * \
                (2 * np.pi / estimator.nphi) * ylm

        np.testing.assert_array_almost_equal(alm, alm_exp)

    def test_ksw_init_reduced_bispectrum_err(self):

        data = self.FakeData()
        estimator = KSW(data)

        # Data bisp lmax is smaller than data lmax.
        self.assertRaises(ValueError, estimator._init_reduced_bispectrum,
            estimator.cosmology.red_bispectra[0])

    def test_ksw_init_reduced_bispectrum(self):

        data = self.FakeData()
        data.lmax = 6
        estimator = KSW(data)

        rb = estimator.cosmology.red_bispectra[0]
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
        rb.weights[:,0,0] = 1
        rb.weights[:,0,1] = 0.1
        rb.weights[:,0,0] = 2
        rb.weights[:,0,1] = 0.2
        rb.weights[:,0,0] = 3
        rb.weights[:,0,1] = 0.3

        x_i_ell, y_i_ell, z_i_ell = estimator._init_reduced_bispectrum(
            estimator.cosmology.red_bispectra[0])

        # Shape (nfact, npol, data.nell).
        self.assertEqual(x_i_ell.shape, (rb.nfact, data.npol, data.lmax + 1))
        self.assertEqual(y_i_ell.shape, (rb.nfact, data.npol, data.lmax + 1))
        self.assertEqual(z_i_ell.shape, (rb.nfact, data.npol, data.lmax + 1))

        # Do some spot checks.
        # Shape (nfact, npol, data.nell).
        self.assertEqual(x_i_ell[0,0,0], 0) # Because lmin bisp = 2.
        # Factor 0.1 is for the beam.
        self.assertEqual(x_i_ell[0,0,2], rb.factors[0,0,0] * rb.weights[0,0,0] * 0.1)

        self.assertEqual(x_i_ell[3,1,6], rb.factors[0,1,4] * rb.weights[3,0,1] * 0.1)

        self.assertEqual(y_i_ell[2,1,4], rb.factors[2,1,2] * rb.weights[2,1,1] * 0.1)

        self.assertEqual(z_i_ell[2,0,5], rb.factors[2,0,3] * rb.weights[2,2,0] * 0.1)

    def test_ksw_init_reduced_bispectrum_I(self):

        data = self.FakeData()
        data.pol = ['T']
        data.npol = 1
        data.lmax = 6
        estimator = KSW(data)

        rb = estimator.cosmology.red_bispectra[0]
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
        rb.weights[:,0,0] = 1
        rb.weights[:,0,1] = 0.1
        rb.weights[:,0,0] = 2
        rb.weights[:,0,1] = 0.2
        rb.weights[:,0,0] = 3
        rb.weights[:,0,1] = 0.3

        x_i_ell, y_i_ell, z_i_ell = estimator._init_reduced_bispectrum(
            estimator.cosmology.red_bispectra[0])

        # Shape (nfact, npol, data.nell).
        self.assertEqual(x_i_ell.shape, (rb.nfact, data.npol, data.lmax + 1))
        self.assertEqual(y_i_ell.shape, (rb.nfact, data.npol, data.lmax + 1))
        self.assertEqual(z_i_ell.shape, (rb.nfact, data.npol, data.lmax + 1))

        # Do some spot checks.
        # Shape (nfact, npol, data.nell).
        self.assertEqual(x_i_ell[0,0,0], 0) # Because lmin bisp = 2.
        # Factor 0.1 is for the beam.
        self.assertEqual(x_i_ell[0,0,2], rb.factors[0,0,0] * rb.weights[0,0,0] * 0.1)

        self.assertEqual(x_i_ell[3,0,6], rb.factors[0,0,4] * rb.weights[3,0,0] * 0.1)

        self.assertEqual(y_i_ell[2,0,4], rb.factors[2,0,2] * rb.weights[2,1,0] * 0.1)

        self.assertEqual(z_i_ell[2,0,5], rb.factors[2,0,3] * rb.weights[2,2,0] * 0.1)

    def test_ksw_init_reduced_bispectrum_E(self):

        data = self.FakeData()
        data.pol = ['E']
        data.npol = 1
        data.lmax = 6
        estimator = KSW(data)

        rb = estimator.cosmology.red_bispectra[0]
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
        rb.weights[:,0,0] = 1
        rb.weights[:,0,1] = 0.1
        rb.weights[:,0,0] = 2
        rb.weights[:,0,1] = 0.2
        rb.weights[:,0,0] = 3
        rb.weights[:,0,1] = 0.3

        x_i_ell, y_i_ell, z_i_ell = estimator._init_reduced_bispectrum(
            estimator.cosmology.red_bispectra[0])

        # Shape (nfact, npol, data.nell).
        self.assertEqual(x_i_ell.shape, (rb.nfact, data.npol, data.lmax + 1))
        self.assertEqual(y_i_ell.shape, (rb.nfact, data.npol, data.lmax + 1))
        self.assertEqual(z_i_ell.shape, (rb.nfact, data.npol, data.lmax + 1))

        # Do some spot checks.
        # Shape (nfact, npol, data.nell).
        self.assertEqual(x_i_ell[0,0,0], 0) # Because lmin bisp = 2.
        # Factor 0.1 is for the beam.
        self.assertEqual(x_i_ell[0,0,2], rb.factors[0,1,0] * rb.weights[0,0,1] * 0.1)

        self.assertEqual(x_i_ell[3,0,6], rb.factors[0,1,4] * rb.weights[3,0,1] * 0.1)

        self.assertEqual(y_i_ell[2,0,4], rb.factors[2,1,2] * rb.weights[2,1,1] * 0.1)

        self.assertEqual(z_i_ell[2,0,5], rb.factors[2,1,3] * rb.weights[2,2,1] * 0.1)

    def test_ksw_init_rings(self):

        data = self.FakeData()
        estimator = KSW(data)

        nfact = 12

        x_i_phi, y_i_phi, z_i_phi = estimator._init_rings(nfact)

        arr_exp = np.zeros((nfact, estimator.nphi))
        np.testing.assert_array_equal(x_i_phi, arr_exp)
        np.testing.assert_array_equal(y_i_phi, arr_exp)
        np.testing.assert_array_equal(z_i_phi, arr_exp)

    def test_ksw_backward(self):

        data = self.FakeData()
        data.lmax = 350
        estimator = KSW(data)
        rb = estimator.cosmology.red_bispectra[0]
        theta = 0.1
        phis = np.linspace(0, 2 * np.pi, estimator.nphi, endpoint=False)

        a_ell_m = np.zeros((data.npol, data.lmax + 1, data.lmax + 1), dtype=complex)
        a_ell_m[0,1,1] = 1
        a_ell_m[1,1,1] = 0.5

        y_ell_m = np.zeros((data.lmax + 1, data.lmax + 1))
        y_ell_m[1,1] = -np.sqrt(3 / 8 / np.pi) * np.sin(theta)

        x_i_ell = np.ones((rb.nfact, data.npol, data.lmax + 1))
        y_i_ell = np.ones((rb.nfact, data.npol, data.lmax + 1)) * 2
        z_i_ell = np.ones((rb.nfact, data.npol, data.lmax + 1)) * 3

        x_i_phi = np.zeros((rb.nfact, estimator.nphi))
        y_i_phi = np.zeros((rb.nfact, estimator.nphi))
        z_i_phi = np.zeros((rb.nfact, estimator.nphi))

        x_i_phi_exp = np.zeros((rb.nfact, estimator.nphi))
        x_i_phi_exp += -(1 + 0.5) * 2 * np.sqrt(3 / 8 / np.pi) * np.sin(theta) * np.cos(phis)
        y_i_phi_exp = x_i_phi_exp.copy() * 2
        z_i_phi_exp = x_i_phi_exp.copy() * 3

        estimator.backward(a_ell_m, x_i_ell, y_i_ell, z_i_ell,
                           x_i_phi, y_i_phi, z_i_phi, y_ell_m)

        decimal = 10
        np.testing.assert_almost_equal(x_i_phi, x_i_phi_exp, decimal=decimal)
        np.testing.assert_almost_equal(y_i_phi, y_i_phi_exp, decimal=decimal)
        np.testing.assert_almost_equal(z_i_phi, z_i_phi_exp, decimal=decimal)

    def test_ksw_forward(self):

        data = self.FakeData()
        data.lmax = 350
        estimator = KSW(data)
        rb = estimator.cosmology.red_bispectra[0]
        theta = 0.1
        ct_weight = 0.1
        phis = np.linspace(0, 2 * np.pi, estimator.nphi, endpoint=False)
        ms = np.arange(data.lmax + 1)

        a_ell_m = np.zeros((data.npol, data.lmax + 1, data.lmax + 1), dtype=complex)
        y_ell_m = np.ascontiguousarray(np.transpose(
            legendre.normalized_associated_legendre_ms(ms, theta, data.lmax)))

        x_i_ell = np.ones((rb.nfact, data.npol, data.lmax + 1))
        y_i_ell = np.ones((rb.nfact, data.npol, data.lmax + 1))
        z_i_ell = np.ones((rb.nfact, data.npol, data.lmax + 1))

        x_i_phi = np.ones((rb.nfact, estimator.nphi)) * np.sin(phis)
        y_i_phi = np.ones_like(x_i_phi)
        z_i_phi = np.ones_like(x_i_phi)

        a_ell_m_exp = np.zeros_like(a_ell_m)
        a_ell_m_exp[:,:,0] = y_ell_m[:,0] * rb.nfact * np.pi * ct_weight / 3
        a_ell_m_exp[:,:,1] = y_ell_m[:,1] * -1j * rb.nfact * np.pi * ct_weight / 3

        estimator.forward(a_ell_m, x_i_ell, y_i_ell, z_i_ell,
                          x_i_phi, y_i_phi, z_i_phi, y_ell_m, ct_weight)

        np.testing.assert_array_almost_equal(a_ell_m, a_ell_m_exp)

    def test_ksw_forward_add(self):

        # Test if answer is correctly added and a_m_ell array is not overwritten

        data = self.FakeData()
        data.lmax = 350
        estimator = KSW(data)
        rb = estimator.cosmology.red_bispectra[0]
        theta = 0.1
        ct_weight = 0.1
        phis = np.linspace(0, 2 * np.pi, estimator.nphi, endpoint=False)
        ms = np.arange(data.lmax + 1)

        a_ell_m = np.zeros((data.npol, data.lmax + 1, data.lmax + 1), dtype=complex)
        a_ell_m += 10
        y_ell_m = np.ascontiguousarray(np.transpose(
            legendre.normalized_associated_legendre_ms(ms, theta, data.lmax)))

        x_i_ell = np.ones((rb.nfact, data.npol, data.lmax + 1))
        y_i_ell = np.ones((rb.nfact, data.npol, data.lmax + 1))
        z_i_ell = np.ones((rb.nfact, data.npol, data.lmax + 1))

        x_i_phi = np.ones((rb.nfact, estimator.nphi)) * np.sin(phis)
        y_i_phi = np.ones_like(x_i_phi)
        z_i_phi = np.ones_like(x_i_phi)

        a_ell_m_exp = np.zeros_like(a_ell_m)
        a_ell_m_exp[:,:,0] = y_ell_m[:,0] * rb.nfact * np.pi * ct_weight / 3
        a_ell_m_exp[:,:,1] = y_ell_m[:,1] * -1j * rb.nfact * np.pi * ct_weight / 3
        a_ell_m_exp += 10

        estimator.forward(a_ell_m, x_i_ell, y_i_ell, z_i_ell,
                          x_i_phi, y_i_phi, z_i_phi, y_ell_m, ct_weight)

        np.testing.assert_array_almost_equal(a_ell_m, a_ell_m_exp)

    def test_ksw_compute_fisher(self):

        data = self.FakeData()
        estimator = KSW(data)
        def icov(alm):
            alm *= 2
            return alm
        estimator.icov = icov

        self.assertIs(estimator.compute_fisher(), None)

        mc_gt = np.ones((data.npol, hp.Alm.getsize(data.lmax)),
                                      dtype=complex)
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
                                       [...,:data.lmax+1])))
        fisher_exp /= 3

        fisher = estimator.compute_fisher()
        self.assertEqual(fisher, fisher_exp)

        # I want internal quantities unchanged.
        self.assertEqual(estimator.mc_gt_sq, mc_gt_sq_copy)
        np.testing.assert_array_almost_equal(estimator.mc_gt, mc_gt_copy)

    def test_ksw_compute_linear_term(self):

        data = self.FakeData()
        estimator = KSW(data)
        def icov(alm):
            alm *= 2
            return alm
        estimator.icov = icov

        alm = np.zeros((data.npol, hp.Alm.getsize(data.lmax)),
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
                                       [...,:data.lmax+1]))
        lin_term = estimator.compute_linear_term(alm)

        self.assertAlmostEqual(lin_term, lin_term_exp)

        # I want internal quantities unchanged.
        self.assertEqual(estimator.mc_gt_sq, mc_gt_sq_copy)
        np.testing.assert_array_almost_equal(estimator.mc_gt, mc_gt_copy)

    def test_ksw_compute_estimate_decomposed(self):

        lmax = 2
        alm_in = np.asarray([1, 2, 3, 4j, 5j, 6])
        alm_in = alm_in.reshape((1, alm_in.size))

        npol = 1
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T')
        data.npol = npol

        # Create a reduced bispectrum that is just b_l1l2l3 = 1.
        rb = self.FakeReducedBispectrum
        rb.npol = 1
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((3, npol, lmax + 1))
        rb.rule = np.ones((1, 3), dtype=int)
        rb.weights = np.ones((1, 3, 1))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)
        estimator.mc_idx = 1

        # Make sure Fisher is 1 and linear term is 0.
        estimator.mc_gt_sq = 3.
        estimator.mc_gt = np.zeros_like(alm_in)

        alm = utils.alm2a_ell_m(alm_in)

        x_i_ell, y_i_ell, z_i_ell = estimator._init_reduced_bispectrum(rb)
        x_i_phi, y_i_phi, z_i_phi = estimator._init_rings(rb.nfact)

        ms = np.arange(estimator.data.lmax + 1)
        phis = np.linspace(0, 2 * np.pi, estimator.nphi, endpoint=False)

        t_cubic = 0
        t_cubic_exp = 0

        # Distribute rings over ranks.
        for tidx in range(len(estimator.thetas)):

            theta = estimator.thetas[tidx]
            ct_weight = estimator.theta_weights[tidx]
            ylm = np.ascontiguousarray(np.transpose(
                legendre.normalized_associated_legendre_ms(ms, theta, estimator.data.lmax)))

            estimator.backward(alm, x_i_ell, y_i_ell, z_i_ell,
                          x_i_phi, y_i_phi, z_i_phi, ylm)

            np.testing.assert_array_almost_equal(x_i_ell, y_i_ell)
            np.testing.assert_array_almost_equal(x_i_ell, z_i_ell)

            np.testing.assert_array_almost_equal(x_i_phi, y_i_phi)
            np.testing.assert_array_almost_equal(x_i_phi, z_i_phi)

            x_i_phi_exp = np.zeros((1, estimator.nphi), dtype=np.complex128) # nfact, nphi.

            # ell = 0.
            x_i_phi_exp += alm_in[0, hp.Alm.getidx(lmax, 0, 0)] * self.y00(theta, phis)

            # ell = 1.
            x_i_phi_exp += np.conj(alm_in[0, hp.Alm.getidx(lmax, 1, 1)] * self.y11(theta, phis))
            x_i_phi_exp += alm_in[0, hp.Alm.getidx(lmax, 1, 0)] * self.y10(theta, phis)
            x_i_phi_exp += alm_in[0, hp.Alm.getidx(lmax, 1, 1)] * self.y11(theta, phis)

            # ell = 2.
            x_i_phi_exp += np.conj(alm_in[0, hp.Alm.getidx(lmax, 2, 2)] * self.y22(theta, phis))
            x_i_phi_exp += np.conj(alm_in[0, hp.Alm.getidx(lmax, 2, 1)] * self.y21(theta, phis))
            x_i_phi_exp += alm_in[0, hp.Alm.getidx(lmax, 2, 0)] * self.y20(theta, phis)
            x_i_phi_exp += alm_in[0, hp.Alm.getidx(lmax, 2, 1)] * self.y21(theta, phis)
            x_i_phi_exp += alm_in[0, hp.Alm.getidx(lmax, 2, 2)] * self.y22(theta, phis)

            x_i_phi_exp *= 0.1 # Multiply by beam.

            np.testing.assert_array_almost_equal(x_i_phi, x_i_phi_exp)

            t_a = np.einsum('ij, ij, ij', x_i_phi, y_i_phi, z_i_phi, optimize=True)

            # Test einsum ans
            self.assertAlmostEqual(t_a, np.sum(x_i_phi_exp.real ** 3))

    def test_ksw_compute_estimate_cubic_I_simple(self):

        # Compare to direct 5 dimensional sum over (l,m).

        lmax = 5
        alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        alm += np.random.randn(alm.size)
        alm += np.random.randn(alm.size) * 1j
        alm[:lmax+1] = alm[:lmax+1].real # Make sure m=0 is real.
        alm = alm.reshape((1, alm.size))

        npol = 1
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T')
        data.npol = npol

        # Create a reduced bispectrum that is just b_l1l2l3 = 1.
        rb = self.FakeReducedBispectrum
        rb.npol = 1
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3, npol))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)
        estimator.mc_idx = 1

        # Make sure Fisher is 1 and linear term is 0.
        estimator.mc_gt_sq = 3.
        estimator.mc_gt = np.zeros_like(alm)

        self.assertEqual(estimator.compute_fisher(), 1)
        self.assertEqual(estimator.compute_linear_term(alm), 0.)

        estimate = estimator.compute_estimate(alm.copy())

        def red_bisp(ell1, ell2, ell3):
            return 1.

        alm = hp.almxfl(alm[0], data.b_ell[0])
        estimate_exp = self.cubic_term_direct(alm, alm, alm, red_bisp)

        self.assertAlmostEqual(estimate, estimate_exp)

    def test_ksw_compute_estimate_cubic_pol_simple(self):

        # Compare to direct 5 dimensional sum over (l,m).

        lmax = 5
        alm = np.zeros((2, hp.Alm.getsize(lmax)), dtype=np.complex128)
        alm += np.random.randn(alm.size).reshape(alm.shape)
        alm += np.random.randn(alm.size).reshape(alm.shape) * 1j
        alm[:,:lmax+1] = alm[:,:lmax+1].real # Make sure m=0 is real.

        npol = 2
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T', 'E')
        data.npol = npol

        # Create a reduced bispectrum with factors that are 1 for I and 2 for E.
        rb = self.FakeReducedBispectrum
        rb.npol = 2
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.factors[:,1,:] = 2
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3, npol))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)
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

        alm_I = hp.almxfl(alm[0], data.b_ell[0])
        alm_E = hp.almxfl(alm[1], data.b_ell[1])
        estimate_exp = self.cubic_term_direct(alm_I, alm_I, alm_I, red_bisp_III)
        estimate_exp += self.cubic_term_direct(alm_I, alm_I, alm_E, red_bisp_IIE)
        estimate_exp += self.cubic_term_direct(alm_I, alm_E, alm_I, red_bisp_IEI)
        estimate_exp += self.cubic_term_direct(alm_E, alm_I, alm_I, red_bisp_EII)
        estimate_exp += self.cubic_term_direct(alm_I, alm_E, alm_E, red_bisp_IEE)
        estimate_exp += self.cubic_term_direct(alm_E, alm_I, alm_E, red_bisp_EIE)
        estimate_exp += self.cubic_term_direct(alm_E, alm_E, alm_I, red_bisp_EEI)
        estimate_exp += self.cubic_term_direct(alm_E, alm_E, alm_E, red_bisp_EEE)

        self.assertAlmostEqual(estimate, estimate_exp)

    def test_ksw_compute_estimate_cubic_I_2d(self):

        # Compare to direct 5 dimensional sum over (l,m).
        # For 2 term reduced bispectrum.

        lmax = 5
        alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        alm += np.random.randn(alm.size)
        alm += np.random.randn(alm.size) * 1j
        alm[:lmax+1] = alm[:lmax+1].real # Make sure m=0 is real.
        alm = alm.reshape((1, alm.size))

        npol = 1
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T')
        data.npol = npol

        # Create a reduced bispectrum that is b_l1l2l3 = 1 * 1 * 1 + 2 * 2 * 2.
        rb = self.FakeReducedBispectrum
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
        rb.weights = np.ones((2, 3, npol))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)
        estimator.mc_idx = 1

        # Make sure Fisher is 1 and linear term is 0.
        estimator.mc_gt_sq = 3.
        estimator.mc_gt = np.zeros_like(alm)

        self.assertEqual(estimator.compute_fisher(), 1)
        self.assertEqual(estimator.compute_linear_term(alm), 0.)

        estimate = estimator.compute_estimate(alm.copy())

        def red_bisp(ell1, ell2, ell3):
            return 1 * 1 * 1 + 2 * 2 * 2

        alm = hp.almxfl(alm[0], data.b_ell[0])
        estimate_exp = self.cubic_term_direct(alm, alm, alm, red_bisp)

        self.assertAlmostEqual(estimate, estimate_exp)

    def test_ksw_compute_estimate_cubic_pol_2d(self):

        # Compare to direct 5 dimensional sum over (l,m).
        # For 2 term reduced bispectrum.

        lmax = 5
        alm = np.zeros((2, hp.Alm.getsize(lmax)), dtype=np.complex128)
        alm += np.random.randn(alm.size).reshape(alm.shape)
        alm += np.random.randn(alm.size).reshape(alm.shape) * 1j
        alm[:,:lmax+1] = alm[:,:lmax+1].real # Make sure m=0 is real.

        npol = 2
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T', 'E')
        data.npol = npol

        # Create a reduced bispectrum that is sum of 1 for I and 2 for E
        # and 3 for I and 6 for E.
        rb = self.FakeReducedBispectrum
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
        rb.weights = np.ones((2, 3, npol))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)
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

        alm_I = hp.almxfl(alm[0], data.b_ell[0])
        alm_E = hp.almxfl(alm[1], data.b_ell[1])
        estimate_exp = self.cubic_term_direct(alm_I, alm_I, alm_I, red_bisp_III)
        estimate_exp += self.cubic_term_direct(alm_I, alm_I, alm_E, red_bisp_IIE)
        estimate_exp += self.cubic_term_direct(alm_I, alm_E, alm_I, red_bisp_IEI)
        estimate_exp += self.cubic_term_direct(alm_E, alm_I, alm_I, red_bisp_EII)
        estimate_exp += self.cubic_term_direct(alm_I, alm_E, alm_E, red_bisp_IEE)
        estimate_exp += self.cubic_term_direct(alm_E, alm_I, alm_E, red_bisp_EIE)
        estimate_exp += self.cubic_term_direct(alm_E, alm_E, alm_I, red_bisp_EEI)
        estimate_exp += self.cubic_term_direct(alm_E, alm_E, alm_E, red_bisp_EEE)

        self.assertAlmostEqual(estimate, estimate_exp)

    def test_ksw_compute_estimate_cubic_local_I(self):

        # Compare to direct 5 dimensional sum over (l,m).
        # For local reduced bispectrum.

        lmax_transfer = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
        pars = camb.CAMBparams(**cosmo_opts)

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

        npol = 1
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T')
        data.npol = npol
        data.cosmology = cosmo

        estimator = KSW(data)
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

        alm = hp.almxfl(alm[0], data.b_ell[0])
        estimate_exp = self.cubic_term_direct(alm, alm, alm, red_bisp)

        self.assertAlmostEqual(estimate, estimate_exp)

    def test_ksw_compute_estimate_cubic_local_pol(self):

        # Compare to direct 5 dimensional sum over (l,m).
        # For local reduced bispectrum.

        lmax_transfer = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
        pars = camb.CAMBparams(**cosmo_opts)

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

        npol = 2
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T', 'E')
        data.npol = npol
        data.cosmology = cosmo

        estimator = KSW(data)
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
            
        alm_I = hp.almxfl(alm[0], data.b_ell[0])
        alm_E = hp.almxfl(alm[1], data.b_ell[1])
        estimate_exp = self.cubic_term_direct(alm_I, alm_I, alm_I, red_bisp_III)
        estimate_exp += self.cubic_term_direct(alm_I, alm_I, alm_E, red_bisp_IIE)
        estimate_exp += self.cubic_term_direct(alm_I, alm_E, alm_I, red_bisp_IEI)
        estimate_exp += self.cubic_term_direct(alm_E, alm_I, alm_I, red_bisp_EII)
        estimate_exp += self.cubic_term_direct(alm_I, alm_E, alm_E, red_bisp_IEE)
        estimate_exp += self.cubic_term_direct(alm_E, alm_I, alm_E, red_bisp_EIE)
        estimate_exp += self.cubic_term_direct(alm_E, alm_E, alm_I, red_bisp_EEI)
        estimate_exp += self.cubic_term_direct(alm_E, alm_E, alm_E, red_bisp_EEE)

        self.assertAlmostEqual(estimate, estimate_exp)

    def test_ksw_compute_estimate_cubic_equilateral_I(self):

        # Compare to direct 5 dimensional sum over (l,m).
        # For equilateral reduced bispectrum.

        lmax_transfer = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
        pars = camb.CAMBparams(**cosmo_opts)

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

        npol = 1
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T')
        data.npol = npol
        data.cosmology = cosmo

        estimator = KSW(data)
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

        alm = hp.almxfl(alm[0], data.b_ell[0])
        estimate_exp = self.cubic_term_direct(alm, alm, alm, red_bisp)

        self.assertAlmostEqual(estimate, estimate_exp)

    def test_ksw_step_I_simple(self):

        np.random.seed(1)

        # Compare to direct 4 dimensional sum over (l,m).
        lmax = 5
        alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        alm += np.random.randn(alm.size)
        alm += np.random.randn(alm.size) * 1j
        alm[:lmax+1] = alm[:lmax+1].real # Make sure m=0 is real.
        alm = alm.reshape((1, alm.size))

        npol = 1
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T')
        data.npol = npol

        # Create a reduced bispectrum that is just b_l1l2l3 = 1.
        rb = self.FakeReducedBispectrum
        rb.npol = 1
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3, npol))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)
        estimator.step(alm.copy())

        def red_bisp(ell1, ell2, ell3):
            return 1.

        alm = hp.almxfl(alm[0], data.b_ell[0])
        grad_exp = self.grad_direct(alm, alm, red_bisp)
        # Bispectrum in grad still has 3 b_ell terms, so 
        # add one term for fair comparison.
        grad_exp = hp.almxfl(grad_exp, data.b_ell[0])

        np.testing.assert_array_almost_equal(estimator.mc_gt[0], grad_exp)

        mc_gt_sq_exp = np.sum(2 * np.real(grad_exp * np.conj(grad_exp)))
        mc_gt_sq_exp -= np.sum(grad_exp[:lmax+1].real ** 2 )
        
        np.testing.assert_array_almost_equal(estimator.mc_gt_sq, mc_gt_sq_exp)

    def test_ksw_step_pol_simple(self):

        np.random.seed(1)

        lmax = 5
        alm = np.zeros((2, hp.Alm.getsize(lmax)), dtype=np.complex128)
        alm += np.random.randn(alm.size).reshape(alm.shape)
        alm += np.random.randn(alm.size).reshape(alm.shape) * 1j
        alm[:,:lmax+1] = alm[:,:lmax+1].real # Make sure m=0 is real.

        npol = 2
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T', 'E')
        data.npol = npol

        # Create a reduced bispectrum with factors that are 1 for I and 2 for E.
        rb = self.FakeReducedBispectrum
        rb.npol = 2
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.factors[:,1,:] = 2
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3, npol))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)
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

        alm_I = hp.almxfl(alm[0], data.b_ell[0])
        alm_E = hp.almxfl(alm[1], data.b_ell[1])
        
        # Grad is now 2d and is sum of all pol combinations.
        grad_exp_I = self.grad_direct(alm_I, alm_I, red_bisp_III)
        grad_exp_I += self.grad_direct(alm_I, alm_E, red_bisp_IIE)
        grad_exp_I += self.grad_direct(alm_E, alm_I, red_bisp_IEI)
        grad_exp_I += self.grad_direct(alm_E, alm_E, red_bisp_IEE)

        grad_exp_E = self.grad_direct(alm_I, alm_I, red_bisp_EII)
        grad_exp_E += self.grad_direct(alm_I, alm_E, red_bisp_EIE)
        grad_exp_E += self.grad_direct(alm_E, alm_I, red_bisp_EEI)
        grad_exp_E += self.grad_direct(alm_E, alm_E, red_bisp_EEE)

        # Bispectrum in grad still has 3 b_ell terms, so 
        # add one term for fair comparison.
        grad_exp_I = hp.almxfl(grad_exp_I, data.b_ell[0])
        grad_exp_E = hp.almxfl(grad_exp_E, data.b_ell[0])

        np.testing.assert_array_almost_equal(estimator.mc_gt[0], grad_exp_I)
        np.testing.assert_array_almost_equal(estimator.mc_gt[1], grad_exp_E)

        # We use diagonal cov in ell, m and pol in this example, so just sum of the two.
        mc_gt_sq_exp = np.sum(2 * np.real(grad_exp_I * np.conj(grad_exp_I)))
        mc_gt_sq_exp -= np.sum(grad_exp_I[:lmax+1].real ** 2 )

        mc_gt_sq_exp += np.sum(2 * np.real(grad_exp_E * np.conj(grad_exp_E)))
        mc_gt_sq_exp -= np.sum(grad_exp_E[:lmax+1].real ** 2 )
        
        np.testing.assert_array_almost_equal(estimator.mc_gt_sq, mc_gt_sq_exp)

    def test_ksw_step_I_simple_2d(self):

        np.random.seed(1)

        # Compare to direct 4 dimensional sum over (l,m).
        lmax = 5
        alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        alm += np.random.randn(alm.size)
        alm += np.random.randn(alm.size) * 1j
        alm[:lmax+1] = alm[:lmax+1].real # Make sure m=0 is real.
        alm = alm.reshape((1, alm.size))

        npol = 1
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T')
        data.npol = npol

        # Create a reduced bispectrum that is b_l1l2l3 = 1 * 1 * 1 + 2 * 2 * 2.
        rb = self.FakeReducedBispectrum
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
        rb.weights = np.ones((2, 3, npol))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)
        estimator.step(alm.copy())

        def red_bisp(ell1, ell2, ell3):
            return 1 * 1 * 1 + 2 * 2 * 2

        alm = hp.almxfl(alm[0], data.b_ell[0])
        grad_exp = self.grad_direct(alm, alm, red_bisp)
        # Bispectrum in grad still has 3 b_ell terms, so 
        # add one term for fair comparison.
        grad_exp = hp.almxfl(grad_exp, data.b_ell[0])

        np.testing.assert_array_almost_equal(estimator.mc_gt[0], grad_exp)

        mc_gt_sq_exp = np.sum(2 * np.real(grad_exp * np.conj(grad_exp)))
        mc_gt_sq_exp -= np.sum(grad_exp[:lmax+1].real ** 2 )
        
        np.testing.assert_array_almost_equal(estimator.mc_gt_sq, mc_gt_sq_exp)

    def test_ksw_step_pol_simple_2d(self):

        np.random.seed(1)

        lmax = 5
        alm = np.zeros((2, hp.Alm.getsize(lmax)), dtype=np.complex128)
        alm += np.random.randn(alm.size).reshape(alm.shape)
        alm += np.random.randn(alm.size).reshape(alm.shape) * 1j
        alm[:,:lmax+1] = alm[:,:lmax+1].real # Make sure m=0 is real.

        npol = 2
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T', 'E')
        data.npol = npol

        # Create a reduced bispectrum that is sum of 1 for I and 2 for E
        # and 3 for I and 6 for E.
        rb = self.FakeReducedBispectrum
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
        rb.weights = np.ones((2, 3, npol))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)
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

        alm_I = hp.almxfl(alm[0], data.b_ell[0])
        alm_E = hp.almxfl(alm[1], data.b_ell[1])
        
        # Grad is now 2d and is sum of all pol combinations.
        grad_exp_I = self.grad_direct(alm_I, alm_I, red_bisp_III)
        grad_exp_I += self.grad_direct(alm_I, alm_E, red_bisp_IIE)
        grad_exp_I += self.grad_direct(alm_E, alm_I, red_bisp_IEI)
        grad_exp_I += self.grad_direct(alm_E, alm_E, red_bisp_IEE)

        grad_exp_E = self.grad_direct(alm_I, alm_I, red_bisp_EII)
        grad_exp_E += self.grad_direct(alm_I, alm_E, red_bisp_EIE)
        grad_exp_E += self.grad_direct(alm_E, alm_I, red_bisp_EEI)
        grad_exp_E += self.grad_direct(alm_E, alm_E, red_bisp_EEE)

        # Bispectrum in grad still has 3 b_ell terms, so 
        # add one term for fair comparison.
        grad_exp_I = hp.almxfl(grad_exp_I, data.b_ell[0])
        grad_exp_E = hp.almxfl(grad_exp_E, data.b_ell[0])

        np.testing.assert_array_almost_equal(estimator.mc_gt[0], grad_exp_I)
        np.testing.assert_array_almost_equal(estimator.mc_gt[1], grad_exp_E)

        # We use diagonal cov in ell, m and pol in this example, so just sum of the two.
        mc_gt_sq_exp = np.sum(2 * np.real(grad_exp_I * np.conj(grad_exp_I)))
        mc_gt_sq_exp -= np.sum(grad_exp_I[:lmax+1].real ** 2 )

        mc_gt_sq_exp += np.sum(2 * np.real(grad_exp_E * np.conj(grad_exp_E)))
        mc_gt_sq_exp -= np.sum(grad_exp_E[:lmax+1].real ** 2 )
        
        np.testing.assert_array_almost_equal(estimator.mc_gt_sq, mc_gt_sq_exp)

    def test_ksw_step_local_I(self):

        # Compare to direct 4 dimensional sum over (l,m).
        # For local reduced bispectrum.

        lmax_transfer = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
        pars = camb.CAMBparams(**cosmo_opts)

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

        npol = 1
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T')
        data.npol = npol
        data.cosmology = cosmo

        estimator = KSW(data)
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

        alm = hp.almxfl(alm[0], data.b_ell[0])
        grad_exp = self.grad_direct(alm, alm, red_bisp)
        # Bispectrum in grad still has 3 b_ell terms, so 
        # add one term for fair comparison.
        grad_exp = hp.almxfl(grad_exp, data.b_ell[0])

        np.testing.assert_array_almost_equal(estimator.mc_gt[0], grad_exp)

        mc_gt_sq_exp = np.sum(2 * np.real(grad_exp * np.conj(grad_exp)))
        mc_gt_sq_exp -= np.sum(grad_exp[:lmax+1].real ** 2 )
        
        np.testing.assert_array_almost_equal(estimator.mc_gt_sq, mc_gt_sq_exp)

    def test_ksw_step_local_pol(self):

        # Compare to direct 4 dimensional sum over (l,m).
        # For local reduced bispectrum.

        np.random.seed(1)

        lmax_transfer = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
        pars = camb.CAMBparams(**cosmo_opts)

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

        npol = 2
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T', 'E')
        data.npol = npol
        data.cosmology = cosmo

        estimator = KSW(data)
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

        alm_I = hp.almxfl(alm[0], data.b_ell[0])
        alm_E = hp.almxfl(alm[1], data.b_ell[1])
        
        # Grad is now 2d and is sum of all pol combinations.
        grad_exp_I = self.grad_direct(alm_I, alm_I, red_bisp_III)
        grad_exp_I += self.grad_direct(alm_I, alm_E, red_bisp_IIE)
        grad_exp_I += self.grad_direct(alm_E, alm_I, red_bisp_IEI)
        grad_exp_I += self.grad_direct(alm_E, alm_E, red_bisp_IEE)

        grad_exp_E = self.grad_direct(alm_I, alm_I, red_bisp_EII)
        grad_exp_E += self.grad_direct(alm_I, alm_E, red_bisp_EIE)
        grad_exp_E += self.grad_direct(alm_E, alm_I, red_bisp_EEI)
        grad_exp_E += self.grad_direct(alm_E, alm_E, red_bisp_EEE)

        # Bispectrum in grad still has 3 b_ell terms, so 
        # add one term for fair comparison.
        grad_exp_I = hp.almxfl(grad_exp_I, data.b_ell[0])
        grad_exp_E = hp.almxfl(grad_exp_E, data.b_ell[0])

        np.testing.assert_array_almost_equal(estimator.mc_gt[0], grad_exp_I)
        np.testing.assert_array_almost_equal(estimator.mc_gt[1], grad_exp_E)

        # We use diagonal cov in ell, m and pol in this example, so just sum of the two.
        mc_gt_sq_exp = np.sum(2 * np.real(grad_exp_I * np.conj(grad_exp_I)))
        mc_gt_sq_exp -= np.sum(grad_exp_I[:lmax+1].real ** 2 )

        mc_gt_sq_exp += np.sum(2 * np.real(grad_exp_E * np.conj(grad_exp_E)))
        mc_gt_sq_exp -= np.sum(grad_exp_E[:lmax+1].real ** 2 )
        
        np.testing.assert_array_almost_equal(estimator.mc_gt_sq, mc_gt_sq_exp)

    def test_ksw_compute_fisher_nxn_I(self):
                
        npol = 1
        lmax = 2
        nfact = 2
        theta = np.radians(1)
        icov_ell = np.ones((1, 1, lmax + 1))
        icov_ell[0,0,0] = 1
        icov_ell[0,0,1] = 0.5
        icov_ell[0,0,2] = 0.1

        y_ell_0 = np.ones(lmax + 1)
        y_ell_0[:] = [self.y00(theta, 0), self.y10(theta, 0), self.y20(theta, 0)]

        x_i_ell = np.ones((nfact, 1, lmax + 1))
        y_i_ell = np.ones((nfact, 1, lmax + 1))
        z_i_ell = np.ones((nfact, 1, lmax + 1))

        x_i_ell *= np.random.randn(x_i_ell.size).reshape(x_i_ell.shape)
        y_i_ell *= np.random.randn(y_i_ell.size).reshape(y_i_ell.shape)
        z_i_ell *= np.random.randn(z_i_ell.size).reshape(z_i_ell.shape)
                
        ans = KSW._compute_fisher_nxn(icov_ell, y_ell_0, x_i_ell, y_i_ell, z_i_ell)

        ans_exp = np.zeros((nfact, nfact))
        ells = np.arange(lmax + 1)
        prefactor = y_ell_0 * np.sqrt((2 * ells + 1) / 4 / np.pi) 

        # xx yy zz.
        ans_exp[0,0] = np.sum(prefactor * x_i_ell[0] * icov_ell * x_i_ell[0]) * \
                       np.sum(prefactor * y_i_ell[0] * icov_ell * y_i_ell[0]) * \
                       np.sum(prefactor * z_i_ell[0] * icov_ell * z_i_ell[0])

        ans_exp[0,1] = np.sum(prefactor * x_i_ell[0] * icov_ell * x_i_ell[1]) * \
                       np.sum(prefactor * y_i_ell[0] * icov_ell * y_i_ell[1]) * \
                       np.sum(prefactor * z_i_ell[0] * icov_ell * z_i_ell[1])

        ans_exp[1,1] = np.sum(prefactor * x_i_ell[1] * icov_ell * x_i_ell[1]) * \
                       np.sum(prefactor * y_i_ell[1] * icov_ell * y_i_ell[1]) * \
                       np.sum(prefactor * z_i_ell[1] * icov_ell * z_i_ell[1])

        # xz yx zy.
        ans_exp[0,0] += np.sum(prefactor * x_i_ell[0] * icov_ell * z_i_ell[0]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_ell * x_i_ell[0]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_ell * y_i_ell[0])

        ans_exp[0,1] += np.sum(prefactor * x_i_ell[0] * icov_ell * z_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_ell * x_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_ell * y_i_ell[1])

        ans_exp[1,1] += np.sum(prefactor * x_i_ell[1] * icov_ell * z_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[1] * icov_ell * x_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[1] * icov_ell * y_i_ell[1])

        # xy yz zx.
        ans_exp[0,0] += np.sum(prefactor * x_i_ell[0] * icov_ell * y_i_ell[0]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_ell * z_i_ell[0]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_ell * x_i_ell[0])

        ans_exp[0,1] += np.sum(prefactor * x_i_ell[0] * icov_ell * y_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_ell * z_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_ell * x_i_ell[1])

        ans_exp[1,1] += np.sum(prefactor * x_i_ell[1] * icov_ell * y_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[1] * icov_ell * z_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[1] * icov_ell * x_i_ell[1])

        # xx yz zy.
        ans_exp[0,0] += np.sum(prefactor * x_i_ell[0] * icov_ell * x_i_ell[0]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_ell * z_i_ell[0]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_ell * y_i_ell[0])

        ans_exp[0,1] += np.sum(prefactor * x_i_ell[0] * icov_ell * x_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_ell * z_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_ell * y_i_ell[1])

        ans_exp[1,1] += np.sum(prefactor * x_i_ell[1] * icov_ell * x_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[1] * icov_ell * z_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[1] * icov_ell * y_i_ell[1])

        # xy yx zz.
        ans_exp[0,0] += np.sum(prefactor * x_i_ell[0] * icov_ell * y_i_ell[0]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_ell * x_i_ell[0]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_ell * z_i_ell[0])

        ans_exp[0,1] += np.sum(prefactor * x_i_ell[0] * icov_ell * y_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_ell * x_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_ell * z_i_ell[1])

        ans_exp[1,1] += np.sum(prefactor * x_i_ell[1] * icov_ell * y_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[1] * icov_ell * x_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[1] * icov_ell * z_i_ell[1])

        # xz yy zx.
        ans_exp[0,0] += np.sum(prefactor * x_i_ell[0] * icov_ell * z_i_ell[0]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_ell * y_i_ell[0]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_ell * x_i_ell[0])

        ans_exp[0,1] += np.sum(prefactor * x_i_ell[0] * icov_ell * z_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_ell * y_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_ell * x_i_ell[1])

        ans_exp[1,1] += np.sum(prefactor * x_i_ell[1] * icov_ell * z_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[1] * icov_ell * y_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[1] * icov_ell * x_i_ell[1])

        ans_exp[1,0] = ans_exp[0,1]
        ans_exp *= 2 * np.pi ** 2 / 9

        np.testing.assert_array_almost_equal(ans, ans_exp)
        
    def test_ksw_compute_fisher_nxn_pol(self):
         
        np.random.seed(1)

        npol = 2
        lmax = 2
        nfact = 2
        theta = np.radians(1)
        icov_ell = np.ones((npol, npol, lmax + 1))
        icov_ell[0,0,0] = 1
        icov_ell[0,0,1] = 0.5
        icov_ell[0,0,2] = 0.1
        icov_ell[0,1] *= 2
        icov_ell[1,0] *= 2
        icov_ell[1,1] *= 3
        
        y_ell_0 = np.ones(lmax + 1)
        y_ell_0[:] = [self.y00(theta, 0), self.y10(theta, 0), self.y20(theta, 0)]

        x_i_ell = np.ones((nfact, npol, lmax + 1))
        y_i_ell = np.ones((nfact, npol, lmax + 1))
        z_i_ell = np.ones((nfact, npol, lmax + 1))

        x_i_ell *= np.random.randn(x_i_ell.size).reshape(x_i_ell.shape)
        y_i_ell *= np.random.randn(y_i_ell.size).reshape(y_i_ell.shape)
        z_i_ell *= np.random.randn(z_i_ell.size).reshape(z_i_ell.shape)
                
        ans = KSW._compute_fisher_nxn(icov_ell, y_ell_0, x_i_ell, y_i_ell, z_i_ell)

        ans_exp = np.zeros((nfact, nfact))
        ells = np.arange(lmax + 1)
        prefactor = y_ell_0 * np.sqrt((2 * ells + 1) / 4 / np.pi)

        # Compute inverse-covariance weighted versions.
        icov_x_i_ell = np.zeros_like(x_i_ell)
        icov_y_i_ell = np.zeros_like(y_i_ell)
        icov_z_i_ell = np.zeros_like(z_i_ell)

        for fidx in range(nfact):
            for lidx in range(lmax + 1):
                icov_x_i_ell[fidx,:,lidx] = np.dot(icov_ell[:,:,lidx], x_i_ell[fidx,:,lidx])
                icov_y_i_ell[fidx,:,lidx] = np.dot(icov_ell[:,:,lidx], y_i_ell[fidx,:,lidx])
                icov_z_i_ell[fidx,:,lidx] = np.dot(icov_ell[:,:,lidx], z_i_ell[fidx,:,lidx])
            
        # xx yy zz.
        ans_exp[0,0] = np.sum(prefactor * x_i_ell[0] * icov_x_i_ell[0]) * \
                       np.sum(prefactor * y_i_ell[0] * icov_y_i_ell[0]) * \
                       np.sum(prefactor * z_i_ell[0] * icov_z_i_ell[0])

        ans_exp[0,1] = np.sum(prefactor * x_i_ell[0] * icov_x_i_ell[1]) * \
                       np.sum(prefactor * y_i_ell[0] * icov_y_i_ell[1]) * \
                       np.sum(prefactor * z_i_ell[0] * icov_z_i_ell[1])

        ans_exp[1,1] = np.sum(prefactor * x_i_ell[1] * icov_x_i_ell[1]) * \
                       np.sum(prefactor * y_i_ell[1] * icov_y_i_ell[1]) * \
                       np.sum(prefactor * z_i_ell[1] * icov_z_i_ell[1])

        # xz yx zy.
        ans_exp[0,0] += np.sum(prefactor * x_i_ell[0] * icov_z_i_ell[0]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_x_i_ell[0]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_y_i_ell[0])

        ans_exp[0,1] += np.sum(prefactor * x_i_ell[0] * icov_z_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_x_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_y_i_ell[1])

        ans_exp[1,1] += np.sum(prefactor * x_i_ell[1] * icov_z_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[1] * icov_x_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[1] * icov_y_i_ell[1])

        # xy yz zx.
        ans_exp[0,0] += np.sum(prefactor * x_i_ell[0] * icov_y_i_ell[0]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_z_i_ell[0]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_x_i_ell[0])

        ans_exp[0,1] += np.sum(prefactor * x_i_ell[0] * icov_y_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_z_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_x_i_ell[1])

        ans_exp[1,1] += np.sum(prefactor * x_i_ell[1] * icov_y_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[1] * icov_z_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[1] * icov_x_i_ell[1])

        # xx yz zy.
        ans_exp[0,0] += np.sum(prefactor * x_i_ell[0] * icov_x_i_ell[0]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_z_i_ell[0]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_y_i_ell[0])

        ans_exp[0,1] += np.sum(prefactor * x_i_ell[0] * icov_x_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_z_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_y_i_ell[1])

        ans_exp[1,1] += np.sum(prefactor * x_i_ell[1] * icov_x_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[1] * icov_z_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[1] * icov_y_i_ell[1])

        # xy yx zz.
        ans_exp[0,0] += np.sum(prefactor * x_i_ell[0] * icov_y_i_ell[0]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_x_i_ell[0]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_z_i_ell[0])

        ans_exp[0,1] += np.sum(prefactor * x_i_ell[0] * icov_y_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_x_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_z_i_ell[1])

        ans_exp[1,1] += np.sum(prefactor * x_i_ell[1] * icov_y_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[1] * icov_x_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[1] * icov_z_i_ell[1])

        # xz yy zx.
        ans_exp[0,0] += np.sum(prefactor * x_i_ell[0] * icov_z_i_ell[0]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_y_i_ell[0]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_x_i_ell[0])

        ans_exp[0,1] += np.sum(prefactor * x_i_ell[0] * icov_z_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[0] * icov_y_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[0] * icov_x_i_ell[1])

        ans_exp[1,1] += np.sum(prefactor * x_i_ell[1] * icov_z_i_ell[1]) * \
                        np.sum(prefactor * y_i_ell[1] * icov_y_i_ell[1]) * \
                        np.sum(prefactor * z_i_ell[1] * icov_x_i_ell[1])

        ans_exp[1,0] = ans_exp[0,1]
        ans_exp *= 2 * np.pi ** 2 / 9

        np.testing.assert_array_almost_equal(ans, ans_exp)

    def test_ksw_compute_fisher_isotropic_I_simple(self):
                
        lmax = 5
        npol = 1
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T')
        data.npol = npol
        data.icov_ell_nonlensed = np.ones((1, lmax + 1))

        # Create a reduced bispectrum that is just b_l1l2l3 = 1.
        rb = self.FakeReducedBispectrum
        rb.npol = 1
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3, npol))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)
        fisher = estimator.compute_fisher_isotropic()

        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            # Correct for 3 powers of beam.
            return 0.1 ** 3 * 1. 
            
        def icov(ell, pidx1, pidx2):
            return 1.

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov)

        self.assertAlmostEqual(fisher, fisher_exp)

    def test_ksw_compute_fisher_isotropic_pol_simple(self):
                
        lmax = 5
        npol = 2
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T', 'E')
        data.npol = npol
        data.icov_ell_nonlensed = np.ones((3, lmax + 1))

        # Create a reduced bispectrum that is just b_l1l2l3 = 1.
        rb = self.FakeReducedBispectrum
        rb.npol = npol
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3, npol))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)
        fisher = estimator.compute_fisher_isotropic()

        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            # Correct for 3 powers of beam.
            return 0.1 ** 3 * 1. 
            
        def icov(ell, pidx1, pidx2):
            return 1.

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov)

        self.assertAlmostEqual(fisher, fisher_exp)

    def test_ksw_compute_fisher_isotropic_I_simple_2d(self):
                
        # Compare to direct 4 dimensional sum over (l,m).
        lmax = 5
        npol = 1
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T')
        data.npol = npol
        data.icov_ell_nonlensed = np.ones((1, lmax + 1))

        # Create a reduced bispectrum that is b_l1l2l3 = 1 * 1 * 1 + 2 * 2 * 2.
        rb = self.FakeReducedBispectrum
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
        rb.weights = np.ones((2, 3, npol))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)
        fisher = estimator.compute_fisher_isotropic()

        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            # Correct for 3 powers of beam.
            return 0.1 ** 3 * (1 * 1 * 1 + 2 * 2 * 2)
            
        def icov(ell, pidx1, pidx2):
            return 1.

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov)

        self.assertAlmostEqual(fisher, fisher_exp)

    def test_ksw_compute_fisher_isotropic_pol_simple_2d(self):
                
        # Compare to direct 4 dimensional sum over (l,m).
        lmax = 5
        npol = 2
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T', 'E')
        data.npol = npol
        data.icov_ell_nonlensed = np.ones((3, lmax + 1))

        # Create a reduced bispectrum that is sum of 1 for I and 2 for E
        # and 3 for I and 6 for E.
        rb = self.FakeReducedBispectrum
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
        rb.weights = np.ones((2, 3, npol))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)
        fisher = estimator.compute_fisher_isotropic()

        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            # Correct for 3 powers of beam.
            b1_l1 = 2 if pidx1 else 1
            b1_l2 = 2 if pidx2 else 1
            b1_l3 = 2 if pidx3 else 1

            b2_l1 = 6 if pidx1 else 3
            b2_l2 = 6 if pidx2 else 3
            b2_l3 = 6 if pidx3 else 3

            return 0.1 ** 3 * (b1_l1 * b1_l2 * b1_l3 + b2_l1 * b2_l2 * b2_l3)
            
        def icov(ell, pidx1, pidx2):
            return 1.

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov)

        self.assertAlmostEqual(fisher, fisher_exp)

    def test_ksw_compute_fisher_isotropic_I_local(self):

        lmax_transfer = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
        pars = camb.CAMBparams(**cosmo_opts)

        cosmo = Cosmology(pars)
        cosmo.compute_transfer(lmax_transfer)

        prim_shape = Shape.prim_local(ns=1)

        self.assertTrue(len(cosmo.red_bispectra) == 0)
        cosmo.add_prim_reduced_bispectrum(prim_shape, radii)
        self.assertTrue(len(cosmo.red_bispectra) == 1)

        rb = cosmo.red_bispectra[0]

        # Lmax and pol of data should overrule those of bispectrum.
        lmax = 5
        npol = 1
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T')
        data.npol = npol
        data.icov_ell_nonlensed = np.ones((1, lmax + 1))
        data.cosmology = cosmo

        estimator = KSW(data)
        fisher = estimator.compute_fisher_isotropic()

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

            # Correct for beam.
            return ret * (0.1) ** 3

        def icov(ell, pidx1, pidx2):
            return 1.

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov)

        self.assertAlmostEqual(fisher, fisher_exp)

    def test_ksw_compute_fisher_isotropic_pol_local(self):

        lmax_transfer = 300
        radii = np.asarray([11000., 14000.])
        dr = ((radii[1] - radii[0]) / 2.)
        cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122,
                               mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
        pars = camb.CAMBparams(**cosmo_opts)

        cosmo = Cosmology(pars)
        cosmo.compute_transfer(lmax_transfer)

        prim_shape = Shape.prim_local(ns=1)

        self.assertTrue(len(cosmo.red_bispectra) == 0)
        cosmo.add_prim_reduced_bispectrum(prim_shape, radii)
        self.assertTrue(len(cosmo.red_bispectra) == 1)

        rb = cosmo.red_bispectra[0]

        # Lmax and pol of data should overrule those of bispectrum.
        lmax = 5
        npol = 2
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T', 'E')
        data.npol = npol
        data.icov_ell_nonlensed = np.ones((3, lmax + 1))
        data.cosmology = cosmo

        estimator = KSW(data)
        fisher = estimator.compute_fisher_isotropic()

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

            # Correct for beam.
            return ret * (0.1) ** 3

        def icov(ell, pidx1, pidx2):
            return 1.

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov)

        self.assertAlmostEqual(fisher, fisher_exp)

    def test_ksw_compute_fisher_isotropic_lens_nonlens(self):
                
        lmax = 5
        npol = 1
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T')
        data.npol = npol
        data.icov_ell_nonlensed = np.ones((1, lmax + 1))
        data.icov_ell_lensed = np.ones((1, lmax + 1)) * 0.5

        # Create a reduced bispectrum that is just b_l1l2l3 = 1.
        rb = self.FakeReducedBispectrum
        rb.npol = npol
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3, npol))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)

        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            # Correct for 3 powers of beam.
            return 0.1 ** 3 * 1. 
            
        def icov(ell, pidx1, pidx2):
            return 1.

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov)
        # Defaults to unlensed.
        fisher = estimator.compute_fisher_isotropic()
        self.assertAlmostEqual(fisher, fisher_exp)

        fisher = estimator.compute_fisher_isotropic(lensed=True)
        self.assertAlmostEqual(fisher, fisher_exp * 0.5 ** 3)

    def test_ksw_compute_fisher_isotropic_matrix(self):
                
        lmax = 5
        npol = 1
        data = self.FakeData()
        data.lmax = lmax
        data.pol = ('T')
        data.npol = npol
        data.icov_ell_nonlensed = np.ones((1, lmax + 1))

        # Create a reduced bispectrum that is just b_l1l2l3 = 1.
        rb = self.FakeReducedBispectrum
        rb.npol = npol
        rb.nfact = 1
        rb.ells_sparse = np.arange(lmax + 1)
        rb.ells_full = np.arange(lmax + 1)
        rb.lmax = lmax
        rb.lmin = 0
        rb.factors = np.ones((1, npol, lmax + 1))
        rb.rule = np.zeros((1, 3), dtype=int)
        rb.weights = np.ones((1, 3, npol))

        data.cosmology.red_bispectra[0] = rb

        estimator = KSW(data)

        def red_bisp(ell1, ell2, ell3, pidx1, pidx2, pidx3):
            # Correct for 3 powers of beam.
            return 0.1 ** 3 * 1. 
            
        def icov(ell, pidx1, pidx2):
            return 1.

        fisher_exp = self.fisher_direct(lmax, npol, red_bisp, icov)

        fisher, fisher_nxn = estimator.compute_fisher_isotropic(return_matrix=True)
        self.assertAlmostEqual(fisher, fisher_exp)
        self.assertAlmostEqual(fisher, np.sum(fisher_nxn))

