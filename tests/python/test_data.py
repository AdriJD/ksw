'''A
Test the Data class.
'''
import unittest
import numpy as np

import healpy as hp

from ksw import Data

class TestData(unittest.TestCase):

    def setUp(self):
        # Is called before each test.

        self.lmax = 4
        self.nell = self.lmax + 1
        self.nelem = hp.Alm.getsize(self.lmax)

        self.alm_T = np.ones(self.nelem, dtype=complex)
        self.alm_T[0] = 1.
        self.alm_TplusE = np.ones((2, self.nelem), dtype=complex)
        self.alm_TplusE[:,:2] = 0.
        self.n_ell_T = np.arange(self.nell, dtype=float)
        self.n_ell_TplusE = np.arange(
            3 * self.nell, dtype=float).reshape(3, self.nell)
        self.b_ell_T = np.ones(self.nell)
        self.b_ell_TplusE = np.ones((2, self.nell))

        c_ell = np.ones((self.nell, 4))
        c_ell[:,0] *= 2.
        c_ell[:,1] *= 4.
        c_ell[:,2] *= 10. # BB.
        c_ell[:,3] *= 1.

        ells = np.arange(self.nell)
        lensed_scalar = {'c_ell' : c_ell, 'ells' : ells}
        unlensed_scalar = {'c_ell' : 2 * c_ell, 'ells' : ells}

        class FakeCosmology():
            def __init__(self):
                self.c_ell = {'lensed_scalar' : lensed_scalar,
                            'unlensed_scalar' : unlensed_scalar}
        self.FakeCosmology = FakeCosmology
        self.c_ell = c_ell

    def tearDown(self):
        # Is called after each test.
        pass

    def test_data_init_1d(self):

        pol = 'T'
        data = Data(self.alm_T, self.n_ell_T, self.b_ell_T, pol)

        # Test if original arrays do not share memory with
        # internal arrays. I want copies.
        self.assertFalse(np.shares_memory(data.alm_data, self.alm_T))
        self.assertFalse(np.shares_memory(data.b_ell, self.b_ell_T))
        self.assertFalse(np.shares_memory(data.n_ell, self.n_ell_T))

        # Check initialized attributes.
        self.assertEqual(data.pol, ('T',))
        self.assertEqual(data.lmax, self.lmax)
        self.assertEqual(data.npol, 1)

        # Test if 1d input is correctly changed to 2d input.
        np.testing.assert_almost_equal(data.alm_data,
                                       self.alm_T[np.newaxis,:])
        np.testing.assert_almost_equal(data.b_ell,
                                       self.b_ell_T[np.newaxis,:])
        np.testing.assert_almost_equal(data.n_ell,
                            self.n_ell_T[np.newaxis,:])

        # Check uninitialized attributes.
        self.assertIs(data.alm_sim, None)
        self.assertIs(data.totcov_diag, None)

    def test_data_init_2d(self):

        pol = ['T', 'E']
        data = Data(self.alm_TplusE, self.n_ell_TplusE,
                    self.b_ell_TplusE, pol)

        # Test if original arrays do not share memory with
        # internal arrays. I want copies.
        self.assertFalse(np.shares_memory(data.alm_data, self.alm_TplusE))
        self.assertFalse(np.shares_memory(data.b_ell, self.b_ell_TplusE))
        self.assertFalse(np.shares_memory(data.n_ell, self.n_ell_TplusE))

        # Check initialized attributes.
        self.assertEqual(data.pol, ('T', 'E'))
        self.assertEqual(data.lmax, self.lmax)
        self.assertEqual(data.npol, 2)

        np.testing.assert_almost_equal(data.alm_data,
                                       self.alm_TplusE)
        np.testing.assert_almost_equal(data.b_ell,
                                       self.b_ell_TplusE)
        np.testing.assert_almost_equal(data.n_ell,
                            self.n_ell_TplusE)

        # Check uninitialized attributes.
        self.assertIs(data.alm_sim, None)
        self.assertIs(data.totcov_diag, None)

    def test_data_init_err_pol(self):

        pol = ['E', 'T'] # Wrong order.
        self.assertRaises(ValueError, Data, self.alm_TplusE,
                          self.n_ell_TplusE, self.b_ell_TplusE, pol)

        pol = ['T', 'T'] # Duplicate.
        self.assertRaises(ValueError, Data, self.alm_TplusE,
                          self.n_ell_TplusE, self.b_ell_TplusE, pol)

        pol = ['T', 'E', 'B'] # Too many.
        self.assertRaises(ValueError, Data, self.alm_TplusE,
                          self.n_ell_TplusE, self.b_ell_TplusE, pol)

        pol = ['T', 'B'] # Cannot have B.
        self.assertRaises(ValueError, Data, self.alm_TplusE,
                          self.n_ell_TplusE, self.b_ell_TplusE, pol)

        pol = ['B'] # Cannot have B.
        self.assertRaises(ValueError, Data, self.alm_TplusE,
                          self.n_ell_TplusE, self.b_ell_TplusE, pol)

    def test_data_init_err_1d(self):

        pol = 'T'
        # Give 2d alm.
        self.assertRaises(ValueError, Data, self.alm_TplusE,
                          self.n_ell_T, self.b_ell_T, pol)

        # Give 2d beam.
        self.assertRaises(ValueError, Data, self.alm_T,
                          self.n_ell_T, self.b_ell_TplusE, pol)

        # Give 2d noise.
        self.assertRaises(ValueError, Data, self.alm_T,
                          self.n_ell_TplusE, self.b_ell_T, pol)

        # Give 2d pol.
        pol = ['T', 'E']
        self.assertRaises(ValueError, Data, self.alm_T,
                          self.n_ell_T, self.b_ell_T, pol)

    def test_data_init_err_2d(self):

        pol = ['T', 'E']
        # Give 1d alm.
        self.assertRaises(ValueError, Data, self.alm_T,
                          self.n_ell_TplusE, self.b_ell_TplusE, pol)

        # Give 1d beam.
        self.assertRaises(ValueError, Data, self.alm_TplusE,
                          self.n_ell_TplusE, self.b_ell_T, pol)

        # Give 1d noise.
        self.assertRaises(ValueError, Data, self.alm_TplusE,
                          self.n_ell_T, self.b_ell_TplusE, pol)

        # Give 1d pol.
        pol = ['T']
        self.assertRaises(ValueError, Data, self.alm_TplusE,
                          self.n_ell_TplusE, self.b_ell_TplusE, pol)

    def test_data_compute_totcov_diag_1d_T(self):

        cosmo = self.FakeCosmology()
        pol = ['T']
        data = Data(self.alm_T, self.n_ell_T,
                    self.b_ell_T, pol)

        data.compute_totcov_diag(cosmo)
        self.assertEqual(data.totcov_diag['unlensed'].shape,
                         (1, self.nell))
        self.assertEqual(data.totcov_diag['lensed'].shape,
                         (1, self.nell))

        expec_totcov = np.zeros((1, self.nell))
        expec_totcov[0] = self.n_ell_T + self.c_ell[:,0] * 2.
        np.testing.assert_almost_equal(data.totcov_diag['unlensed'],
                                       expec_totcov)
        # Test if inverse is also computed correctly.
        np.testing.assert_almost_equal(data.inv_totcov_diag['unlensed'],
                                       1/expec_totcov)

        # Again with lensing power.
        expec_totcov[0] = self.n_ell_T + self.c_ell[:,0]
        np.testing.assert_almost_equal(data.totcov_diag['lensed'],
                                       expec_totcov)

        np.testing.assert_almost_equal(data.inv_totcov_diag['lensed'],
                                       1/expec_totcov)

    def test_data_compute_totcov_diag_1d_E(self):

        cosmo = self.FakeCosmology()
        pol = ['E']
        data = Data(self.alm_T, self.n_ell_T,
                    self.b_ell_T, pol)

        data.compute_totcov_diag(cosmo)
        self.assertEqual(data.totcov_diag['unlensed'].shape,
                         (1, self.nell))
        self.assertEqual(data.totcov_diag['lensed'].shape,
                         (1, self.nell))


        expec_totcov = np.zeros((1, self.nell))
        expec_totcov[0] = self.n_ell_T + self.c_ell[:,1] * 2
        np.testing.assert_almost_equal(data.totcov_diag['unlensed'],
                                       expec_totcov)
        # Test if inverse is also computed correctly.
        np.testing.assert_almost_equal(data.inv_totcov_diag['unlensed'],
                                       1/expec_totcov)

        # Again with lensing power.
        expec_totcov[0] = self.n_ell_T + self.c_ell[:,1]
        np.testing.assert_almost_equal(data.totcov_diag['lensed'],
                                       expec_totcov)

        np.testing.assert_almost_equal(data.inv_totcov_diag['lensed'],
                                       1/expec_totcov)

    def test_data_compute_totcov_diag_2d(self):

        cosmo = self.FakeCosmology()
        pol = ['T', 'E']
        data = Data(self.alm_TplusE, self.n_ell_TplusE,
                    self.b_ell_TplusE, pol)

        data.compute_totcov_diag(cosmo)
        self.assertEqual(data.totcov_diag['lensed'].shape, (3, self.nell))
        self.assertEqual(data.totcov_diag['unlensed'].shape, (3, self.nell))

        expec_totcov = np.zeros((3, self.nell))
        expec_totcov[0] = self.n_ell_TplusE[0] + self.c_ell[:,0] * 2
        expec_totcov[1] = self.n_ell_TplusE[1] + self.c_ell[:,1] * 2
        expec_totcov[2] = self.n_ell_TplusE[2] + self.c_ell[:,3] * 2
        np.testing.assert_almost_equal(data.totcov_diag['unlensed'],
                                       expec_totcov)

        # Test if inverse is also computed correctly.
        expec_inv_totcov = np.zeros_like(expec_totcov)

        # Inverse of [[a,b],[b,d]] = [[d,-b],[-b,a]] / (ad - bb).
        expec_inv_totcov[0] = expec_totcov[1]
        expec_inv_totcov[1] = expec_totcov[0]
        expec_inv_totcov[2] = -expec_totcov[2]
        det = expec_totcov[0] * expec_totcov[1] - expec_totcov[2] ** 2
        expec_inv_totcov /= det
        np.testing.assert_almost_equal(data.inv_totcov_diag['unlensed'],
                                       expec_inv_totcov)

        # With lensing power.
        expec_totcov[0] = self.n_ell_TplusE[0] + self.c_ell[:,0]
        expec_totcov[1] = self.n_ell_TplusE[1] + self.c_ell[:,1]
        expec_totcov[2] = self.n_ell_TplusE[2] + self.c_ell[:,3]
        np.testing.assert_almost_equal(data.totcov_diag['lensed'],
                                       expec_totcov)

        expec_inv_totcov = np.zeros_like(expec_totcov)

        expec_inv_totcov[0] = expec_totcov[1]
        expec_inv_totcov[1] = expec_totcov[0]
        expec_inv_totcov[2] = -expec_totcov[2]
        det = expec_totcov[0] * expec_totcov[1] - expec_totcov[2] ** 2
        expec_inv_totcov /= det
        np.testing.assert_almost_equal(data.inv_totcov_diag['lensed'],
                                       expec_inv_totcov)

    def test_invert_totcov_diag_1d(self):

        pol = ['T']
        data = Data(self.alm_T, self.n_ell_T,
                    self.b_ell_T, pol)

        totcov_diag = np.ones_like(data.n_ell) # (nspec, nell).
        totcov_diag[0,:] = 2.
        inv_totcov_diag_expec = np.ones_like(totcov_diag) * 1/2.

        np.testing.assert_almost_equal(
            data._invert_totcov_diag(totcov_diag), inv_totcov_diag_expec)

    def test_invert_totcov_diag_2d(self):

        pol = ['T', 'E']
        data = Data(self.alm_TplusE, self.n_ell_TplusE,
                    self.b_ell_TplusE, pol)

        totcov_diag = np.ones_like(data.n_ell) # (nspec, nell).
        totcov_diag[:2,:] = 2.
        #  2  1  inv:  2/3 -1/3
        #  1  2       -1/3 2/3

        inv_totcov_diag_expec = np.ones_like(totcov_diag)
        inv_totcov_diag_expec[0] = 2/3.
        inv_totcov_diag_expec[1] = 2/3.
        inv_totcov_diag_expec[2] = -1/3.

        np.testing.assert_almost_equal(
            data._invert_totcov_diag(totcov_diag), inv_totcov_diag_expec)

    def test_data_get_c_inv_a_diag(self):

        pol = ['T', 'E']
        data = Data(self.alm_TplusE, self.n_ell_TplusE,
                    self.b_ell_TplusE, pol)

        inv_totcov_diag = np.ones_like(data.n_ell) # (nspec, nell).
        inv_totcov_diag[0] = 2/3.
        inv_totcov_diag[1] = 2/3.
        inv_totcov_diag[2] = -1/3.
        data.inv_totcov_diag = {'lensed' : 2 * inv_totcov_diag,
                                'unlensed' : inv_totcov_diag}
        c_inv_a = data.get_c_inv_a_diag()

        c_inv_a_expec = np.zeros_like(self.alm_TplusE)
        c_inv_a_expec[0,:] = self.alm_TplusE[0] * (2/3.)
        c_inv_a_expec[0,:] += self.alm_TplusE[1] * (-1/3.)
        c_inv_a_expec[1,:] = self.alm_TplusE[0] * (-1/3.)
        c_inv_a_expec[1,:] += self.alm_TplusE[1] * (2/3.)

        np.testing.assert_almost_equal(c_inv_a, c_inv_a_expec)

    def test_data_get_c_inv_a_diag_1d_T(self):

        pol = ['T']
        data = Data(self.alm_T, self.n_ell_T,
                    self.b_ell_T, pol)

        inv_totcov_diag = np.ones((1, self.nell))
        inv_totcov_diag[0,:] = 0.5
        data.inv_totcov_diag = {'lensed' : 2 * inv_totcov_diag,
                                'unlensed' : inv_totcov_diag}
        
        c_inv_a = data.get_c_inv_a_diag()

        c_inv_a_expec = np.zeros((1, self.nelem), dtype=complex)
        c_inv_a_expec[0,:] = self.alm_T[0] * (1/2.)

        np.testing.assert_almost_equal(c_inv_a, c_inv_a_expec)

    def test_data_get_c_inv_a_diag_1d_E(self):

        pol = ['E']
        data = Data(self.alm_T, self.n_ell_T,
                    self.b_ell_T, pol)

        inv_totcov_diag = np.ones((1, self.nell))
        inv_totcov_diag[0,:] = 0.5
        data.inv_totcov_diag = {'lensed' : 2 * inv_totcov_diag,
                                'unlensed' : inv_totcov_diag}
        
        c_inv_a = data.get_c_inv_a_diag()

        c_inv_a_expec = np.zeros((1, self.nelem), dtype=complex)
        c_inv_a_expec[0,:] = self.alm_T[0] * (1/2.)

        np.testing.assert_almost_equal(c_inv_a, c_inv_a_expec)

    def test_data_get_c_inv_a_diag_1d_lensed(self):

        pol = ['T']
        data = Data(self.alm_T, self.n_ell_T,
                    self.b_ell_T, pol)

        inv_totcov_diag = np.ones((1, self.nell))
        inv_totcov_diag[0,:] = 0.5
        data.inv_totcov_diag = {'lensed' : 4 * inv_totcov_diag,
                                'unlensed' : inv_totcov_diag}
        
        c_inv_a = data.get_c_inv_a_diag(lens_power=True)

        c_inv_a_expec = np.zeros((1, self.nelem), dtype=complex)
        c_inv_a_expec[0,:] = self.alm_T[0] * (2.)

        np.testing.assert_almost_equal(c_inv_a, c_inv_a_expec)
                
    def test_data_get_c_inv_a_diag_err(self):

        pol = ['E']
        data = Data(self.alm_T, self.n_ell_T,
                    self.b_ell_T, pol)

        # inv_totcov_diag is not set yet.
        self.assertRaises(AttributeError, data.get_c_inv_a_diag)

        inv_totcov_diag = np.ones((1, self.nell))
        inv_totcov_diag[0,:] = .5
        data.inv_totcov_diag = {'lensed' :  inv_totcov_diag,
                                'unlensed' : inv_totcov_diag}
        
        opts = {'sim' : True}
        # alm_sim is not set yet.
        self.assertRaises(AttributeError, data.get_c_inv_a_diag, **opts)

    def test_data_get_c_inv_a_diag_sim(self):

        pol = ['E']
        data = Data(self.alm_T, self.n_ell_T,
                    self.b_ell_T, pol)

        inv_totcov_diag = np.ones((1, self.nell))
        inv_totcov_diag[0,:] = 0.5
        data.inv_totcov_diag = {'lensed' :  inv_totcov_diag,
                                'unlensed' : inv_totcov_diag}

        data.alm_sim = self.alm_T[np.newaxis,:] + 1j
        c_inv_a = data.get_c_inv_a_diag(sim=True)

        c_inv_a_expec = np.zeros((1, self.nelem), dtype=complex)
        c_inv_a_expec[0,:] = self.alm_T[0] * (0.5 + 0.5j)

        np.testing.assert_almost_equal(c_inv_a, c_inv_a_expec)

    def test_data_compute_alm_sim(self):

        pol = ['T', 'E']
        data = Data(self.alm_TplusE, self.n_ell_TplusE,
                    self.b_ell_TplusE, pol)

        totcov_diag = np.ones_like(data.n_ell) # (nspec, nell).
        totcov_diag[:2,:] = 2.
        data.totcov_diag = {'lensed' : 2 * totcov_diag,
                            'unlensed' : totcov_diag}

        np.random.seed(10)
        data.compute_alm_sim()
        np.random.seed(10)

        c_ell = np.zeros((4, self.nell))
        c_ell[:2] = 2.
        c_ell[3] = 1.

        alm = hp.synalm(c_ell, new=True)
        alm_sim_expec = np.zeros((2, self.nelem), dtype=complex)
        alm_sim_expec[0] = alm[0]
        alm_sim_expec[1] = alm[1]

        np.testing.assert_almost_equal(data.alm_sim, alm_sim_expec)

    def test_data_compute_alm_sim_1d_E(self):

        pol = ['E']
        data = Data(self.alm_T, self.n_ell_T,
                    self.b_ell_T, pol)

        totcov_diag = np.ones((1, self.nell)) # (nspec, nell).
        totcov_diag[0,:] = 2.
        data.totcov_diag = {'lensed' : 2 * totcov_diag,
                            'unlensed' : totcov_diag}
        
        np.random.seed(10)
        data.compute_alm_sim()
        np.random.seed(10)

        c_ell = np.zeros((4, self.nell))
        c_ell[1] = 2.

        alm = hp.synalm(c_ell, new=True)
        alm_sim_expec = (alm[1])[np.newaxis,:]

        np.testing.assert_almost_equal(data.alm_sim, alm_sim_expec)

    def test_data_compute_alm_sim_1d_T(self):

        pol = ['T']
        data = Data(self.alm_T, self.n_ell_T,
                    self.b_ell_T, pol)

        totcov_diag = np.ones((1, self.nell)) # (nspec, nell).
        totcov_diag[0,:] = 2.
        data.totcov_diag = {'lensed' : 2 * totcov_diag,
                            'unlensed' : totcov_diag}
        
        np.random.seed(10)
        data.compute_alm_sim()
        np.random.seed(10)

        c_ell = np.ones(self.nell) * 2.

        alm = hp.synalm(c_ell, new=True)
        alm_sim_expec = alm[np.newaxis,:]

        np.testing.assert_almost_equal(data.alm_sim, alm_sim_expec)

    def test_data_compute_alm_sim_lensed(self):

        pol = ['T']
        data = Data(self.alm_T, self.n_ell_T,
                    self.b_ell_T, pol)

        totcov_diag = np.ones((1, self.nell)) # (nspec, nell).
        totcov_diag[0,:] = 2.
        data.totcov_diag = {'lensed' : 2 * totcov_diag,
                            'unlensed' : totcov_diag}
        
        np.random.seed(10)
        data.compute_alm_sim(lens_power=True)
        np.random.seed(10)

        c_ell = np.ones(self.nell) * 4. # Extra 2 from lensed spectrum.

        alm = hp.synalm(c_ell, new=True)
        alm_sim_expec = alm[np.newaxis,:]

        np.testing.assert_almost_equal(data.alm_sim, alm_sim_expec)
        
    def test_data_compute_alm_sim_err(self):

        pol = ['E']
        data = Data(self.alm_T, self.n_ell_T,
                    self.b_ell_T, pol)

        # totcov_diag is not set yet.
        self.assertRaises(AttributeError, data.compute_alm_sim)
