import unittest
import numpy as np

import healpy as hp

from ksw import KSW, legendre

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

        self.FakeData = FakeData

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
