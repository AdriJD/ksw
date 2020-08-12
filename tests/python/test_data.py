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

        self.n_ell_T = np.arange(self.nell, dtype=float)
        self.n_ell_TplusE = np.arange(
            3 * self.nell, dtype=float).reshape(3, self.nell)
        self.b_ell_T = np.linspace(1, 0.8, self.nell)
        self.b_ell_TplusE = np.ones((2, self.nell))
        self.b_ell_TplusE[0] = self.b_ell_T
        self.b_ell_TplusE[1] = self.b_ell_T * 0.5

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
        cosmo = self.FakeCosmology()
        data = Data(self.lmax, self.n_ell_T, self.b_ell_T, pol,
                    cosmo)

        # Test if original arrays do not share memory with
        # internal arrays. I want copies.
        self.assertFalse(np.shares_memory(data.b_ell, self.b_ell_T))
        self.assertFalse(np.shares_memory(data.n_ell, self.n_ell_T))
                
        # Check initialized attributes.
        self.assertEqual(data.pol, ('T',))
        self.assertEqual(data.lmax, self.lmax)
        self.assertEqual(data.npol, 1)
        self.assertTrue(isinstance(data.cosmology, self.FakeCosmology))
        np.testing.assert_almost_equal(data.b_ell, self.b_ell_T[np.newaxis,:])
        np.testing.assert_almost_equal(data.n_ell, self.n_ell_T[np.newaxis,:])        
        
        # Test shapes.
        self.assertEqual(data.b_ell.shape, (1, self.lmax + 1))
        self.assertEqual(data.n_ell.shape, (1, self.lmax + 1))        
        self.assertEqual(data.cov_ell_lensed.shape, (1, self.lmax + 1))
        self.assertEqual(data.cov_ell_nonlensed.shape, (1, self.lmax + 1))
        self.assertEqual(data.icov_ell_lensed.shape, (1, self.lmax + 1))
        self.assertEqual(data.icov_ell_nonlensed.shape, (1, self.lmax + 1))        

    def test_data_init_2d(self):

        pol = ['T', 'E']
        cosmo = self.FakeCosmology()
        data = Data(self.lmax, self.n_ell_TplusE, self.b_ell_TplusE, pol,
                    cosmo)

        # Test if original arrays do not share memory with
        # internal arrays. I want copies.
        self.assertFalse(np.shares_memory(data.b_ell, self.b_ell_TplusE))
        self.assertFalse(np.shares_memory(data.n_ell, self.n_ell_TplusE))

        # Check initialized attributes.
        self.assertEqual(data.pol, ('T', 'E'))
        self.assertEqual(data.lmax, self.lmax)
        self.assertEqual(data.npol, 2)
        self.assertTrue(isinstance(data.cosmology, self.FakeCosmology))
        np.testing.assert_almost_equal(data.b_ell, self.b_ell_TplusE)
        np.testing.assert_almost_equal(data.n_ell, self.n_ell_TplusE)        

        # Test shapes.
        self.assertEqual(data.b_ell.shape, (2, self.lmax + 1))
        self.assertEqual(data.n_ell.shape, (3, self.lmax + 1))        
        self.assertEqual(data.cov_ell_lensed.shape, (3, self.lmax + 1))
        self.assertEqual(data.cov_ell_nonlensed.shape, (3, self.lmax + 1))
        self.assertEqual(data.icov_ell_lensed.shape, (3, self.lmax + 1))
        self.assertEqual(data.icov_ell_nonlensed.shape, (3, self.lmax + 1))        
                
    def test_data_init_err_pol(self):

        cosmo = self.FakeCosmology()        
        pol = ['E', 'T'] # Wrong order.
        self.assertRaises(ValueError, Data, self.lmax, self.n_ell_TplusE,
                          self.b_ell_TplusE, pol, cosmo)

        pol = ['T', 'T'] # Duplicate.
        self.assertRaises(ValueError, Data, self.lmax, self.n_ell_TplusE,
                          self.b_ell_TplusE, pol, cosmo)

        pol = ['T', 'E', 'B'] # Too many.
        self.assertRaises(ValueError, Data, self.lmax, self.n_ell_TplusE,
                          self.b_ell_TplusE, pol, cosmo)

        pol = ['T', 'B'] # Cannot have B.
        self.assertRaises(ValueError, Data, self.lmax, self.n_ell_TplusE,
                          self.b_ell_TplusE, pol, cosmo)

        pol = ['B'] # Cannot have B.
        self.assertRaises(ValueError, Data, self.lmax, self.n_ell_TplusE,
                          self.b_ell_TplusE, pol, cosmo)

    def test_data_init_err_1d(self):

        cosmo = self.FakeCosmology()                
        pol = 'T'
        
        # Give 2d beam.
        self.assertRaises(ValueError, Data, self.lmax, self.n_ell_T,
                          self.b_ell_TplusE, pol, cosmo)

        # Give 2d noise.
        self.assertRaises(ValueError, Data, self.lmax, self.n_ell_TplusE,
                          self.b_ell_T, pol, cosmo)

        # Give 2d pol.
        pol = ['T', 'E']
        self.assertRaises(ValueError, Data, self.lmax, self.n_ell_T,
                          self.b_ell_T, pol, cosmo)

    def test_data_init_err_2d(self):

        cosmo = self.FakeCosmology()                
        pol = ['T', 'E']

        # Give 1d beam.
        self.assertRaises(ValueError, Data, self.lmax, self.n_ell_TplusE,
                          self.b_ell_T, pol, cosmo)

        # Give 1d noise.
        self.assertRaises(ValueError, Data, self.lmax, self.n_ell_T,
                          self.b_ell_TplusE, pol, cosmo)

        # Give 1d pol.
        pol = ['T']
        self.assertRaises(ValueError, Data, self.lmax, self.n_ell_TplusE,
                          self.b_ell_TplusE, pol, cosmo)

    def test_data_compute_totcov_diag_1d_T(self):

        cosmo = self.FakeCosmology()
        pol = ['T']
        data = Data(self.lmax, self.n_ell_T, self.b_ell_T, pol, cosmo)

        cov_ell_exp = np.zeros((1, self.nell))
        cov_ell_exp[0] = self.n_ell_T + self.c_ell[:,0] * 2 * self.b_ell_T ** 2
        np.testing.assert_almost_equal(data.cov_ell_nonlensed,
                                       cov_ell_exp)
        # Test if inverse is also computed correctly.
        np.testing.assert_almost_equal(data.icov_ell_nonlensed,
                                       1 / cov_ell_exp)

        # Again with lensing power.
        cov_ell_exp[0] = self.n_ell_T + self.c_ell[:,0] * self.b_ell_T ** 2
        np.testing.assert_almost_equal(data.cov_ell_lensed,
                                       cov_ell_exp)
        # Test if inverse is also computed correctly.
        np.testing.assert_almost_equal(data.icov_ell_lensed,
                                       1/cov_ell_exp)

    def test_data_compute_totcov_diag_1d_E(self):

        cosmo = self.FakeCosmology()
        pol = ['E']
        data = Data(self.lmax, self.n_ell_T, self.b_ell_T, pol, cosmo)

        cov_ell_exp = np.zeros((1, self.nell))
        cov_ell_exp[0] = self.n_ell_T + self.c_ell[:,1] * 2 * self.b_ell_T ** 2
        np.testing.assert_almost_equal(data.cov_ell_nonlensed,
                                       cov_ell_exp)
        # Test if inverse is also computed correctly.
        np.testing.assert_almost_equal(data.icov_ell_nonlensed,
                                       1 / cov_ell_exp)

        # Again with lensing power.
        cov_ell_exp[0] = self.n_ell_T + self.c_ell[:,1] * self.b_ell_T ** 2
        np.testing.assert_almost_equal(data.cov_ell_lensed,
                                       cov_ell_exp)
        # Test if inverse is also computed correctly.
        np.testing.assert_almost_equal(data.icov_ell_lensed,
                                       1/cov_ell_exp)

    def test_data_compute_totcov_diag_2d(self):

        cosmo = self.FakeCosmology()
        pol = ['T', 'E']
        data = Data(self.lmax, self.n_ell_TplusE, self.b_ell_TplusE, pol, cosmo)        

        expec_totcov = np.zeros((3, self.nell))
        expec_totcov[0] = self.n_ell_TplusE[0] + self.c_ell[:,0] * 2 \
            * self.b_ell_TplusE[0] ** 2
        expec_totcov[1] = self.n_ell_TplusE[1] + self.c_ell[:,1] * 2 \
            * self.b_ell_TplusE[1] ** 2            
        expec_totcov[2] = self.n_ell_TplusE[2] + self.c_ell[:,3] * 2 \
            * self.b_ell_TplusE[0] * self.b_ell_TplusE[1]
        np.testing.assert_almost_equal(data.cov_ell_nonlensed, expec_totcov)

        # Test if inverse is also computed correctly.
        expec_inv_totcov = np.zeros_like(expec_totcov)

        # Inverse of [[a,b],[b,d]] = [[d,-b],[-b,a]] / (ad - bb).
        expec_inv_totcov[0] = expec_totcov[1]
        expec_inv_totcov[1] = expec_totcov[0]
        expec_inv_totcov[2] = -expec_totcov[2]
        det = expec_totcov[0] * expec_totcov[1] - expec_totcov[2] ** 2
        expec_inv_totcov /= det
        np.testing.assert_almost_equal(data.icov_ell_nonlensed, expec_inv_totcov)

        # With lensing power.
        expec_totcov = np.zeros((3, self.nell))
        expec_totcov[0] = self.n_ell_TplusE[0] + self.c_ell[:,0] \
            * self.b_ell_TplusE[0] ** 2
        expec_totcov[1] = self.n_ell_TplusE[1] + self.c_ell[:,1] \
            * self.b_ell_TplusE[1] ** 2            
        expec_totcov[2] = self.n_ell_TplusE[2] + self.c_ell[:,3] \
            * self.b_ell_TplusE[0] * self.b_ell_TplusE[1]
        np.testing.assert_almost_equal(data.cov_ell_lensed, expec_totcov)
        
        expec_inv_totcov = np.zeros_like(expec_totcov)

        expec_inv_totcov[0] = expec_totcov[1]
        expec_inv_totcov[1] = expec_totcov[0]
        expec_inv_totcov[2] = -expec_totcov[2]
        det = expec_totcov[0] * expec_totcov[1] - expec_totcov[2] ** 2
        expec_inv_totcov /= det
        np.testing.assert_almost_equal(data.icov_ell_lensed, expec_inv_totcov)

    def test_data_compute_alm_sim_2d(self):

        cosmo = self.FakeCosmology()        
        pol = ['T', 'E']
        data = Data(self.lmax, self.n_ell_TplusE,
                    self.b_ell_TplusE, pol, cosmo)

        np.random.seed(10)
        alm = data.compute_alm_sim(lens_power=False)
        np.random.seed(10)

        cl = np.zeros((4, self.nell))
        cl[:2] = data.cov_ell_nonlensed[:2]
        cl[3] = data.cov_ell_nonlensed[2]
        alm_exp = hp.synalm(cl, new=True)
        
        alm_sim_expec = np.zeros((2, self.nelem), dtype=complex)
        alm_sim_expec[0] = alm_exp[0]
        alm_sim_expec[1] = alm_exp[1]

        np.testing.assert_almost_equal(alm, alm_sim_expec)

        # Again for lensed.
        np.random.seed(10)
        alm = data.compute_alm_sim(lens_power=True)
        np.random.seed(10)

        cl = np.zeros((4, self.nell))
        cl[:2] = data.cov_ell_lensed[:2]
        cl[3] = data.cov_ell_lensed[2]
        alm_exp = hp.synalm(cl, new=True)
        
        alm_sim_expec = np.zeros((2, self.nelem), dtype=complex)
        alm_sim_expec[0] = alm_exp[0]
        alm_sim_expec[1] = alm_exp[1]

        np.testing.assert_almost_equal(alm, alm_sim_expec)

    def test_data_compute_alm_sim_1d_E(self):

        cosmo = self.FakeCosmology()        
        pol = ['E']
        data = Data(self.lmax, self.n_ell_T,
                    self.b_ell_T, pol, cosmo)

        np.random.seed(10)
        alm = data.compute_alm_sim(lens_power=False)
        np.random.seed(10)

        cl = np.zeros((4, self.nell))
        cl[1] = data.cov_ell_nonlensed[0]
        alm_exp = hp.synalm(cl, new=True)
        
        alm_sim_expec = np.zeros((1, self.nelem), dtype=complex)
        alm_sim_expec[0] = alm_exp[1]

        np.testing.assert_almost_equal(alm, alm_sim_expec)

        # Again for lensed.
        np.random.seed(10)
        alm = data.compute_alm_sim(lens_power=True)
        np.random.seed(10)

        cl = np.zeros((4, self.nell))
        cl[1] = data.cov_ell_lensed[0]
        alm_exp = hp.synalm(cl, new=True)
        
        alm_sim_expec = np.zeros((1, self.nelem), dtype=complex)
        alm_sim_expec[0] = alm_exp[1]

        np.testing.assert_almost_equal(alm, alm_sim_expec)
        
    def test_data_compute_alm_sim_1d_T(self):

        cosmo = self.FakeCosmology()        
        pol = ['T']
        data = Data(self.lmax, self.n_ell_T,
                    self.b_ell_T, pol, cosmo)

        np.random.seed(10)
        alm = data.compute_alm_sim(lens_power=False)
        np.random.seed(10)

        alm_exp = hp.synalm(data.cov_ell_nonlensed[0], new=True)
        
        alm_sim_expec = np.zeros((1, self.nelem), dtype=complex)
        alm_sim_expec[0] = alm_exp

        np.testing.assert_almost_equal(alm, alm_sim_expec)

        # Again for lensed.
        np.random.seed(10)
        alm = data.compute_alm_sim(lens_power=True)
        np.random.seed(10)

        alm_exp = hp.synalm(data.cov_ell_lensed[0], new=True)
        
        alm_sim_expec = np.zeros((1, self.nelem), dtype=complex)
        alm_sim_expec[0] = alm_exp

        np.testing.assert_almost_equal(alm, alm_sim_expec)
        
    def test_data_icov_diag_lensed_2d(self):
    
        cosmo = self.FakeCosmology()        
        pol = ['T', 'E']
        data = Data(self.lmax, self.n_ell_TplusE,
                    self.b_ell_TplusE, pol, cosmo)

        alm = np.random.randn(2 * hp.Alm.getsize(self.lmax))
        alm = np.reshape(alm.astype(complex), (2, self.nelem))
        alm_in = alm.copy()
        alm = data.icov_diag_lensed(alm)

        alm_exp = np.zeros_like(alm_in)
        alm_exp[0] = hp.almxfl(alm_in[0], data.icov_ell_lensed[0])
        alm_exp[0] += hp.almxfl(alm_in[1], data.icov_ell_lensed[2])        
        alm_exp[1] = hp.almxfl(alm_in[1], data.icov_ell_lensed[1])
        alm_exp[1] += hp.almxfl(alm_in[0], data.icov_ell_lensed[2])        

        np.testing.assert_array_almost_equal(alm, alm_exp)

    def test_data_icov_diag_lensed_1d_T(self):
    
        cosmo = self.FakeCosmology()        
        pol = ['T']
        data = Data(self.lmax, self.n_ell_T,
                    self.b_ell_T, pol, cosmo)

        alm = np.random.randn(hp.Alm.getsize(self.lmax))
        alm = np.reshape(alm.astype(complex), (1, self.nelem))
        alm_in = alm.copy()
        
        alm = data.icov_diag_lensed(alm)
        
        # Test if 1D input also works
        alm_1d = data.icov_diag_lensed(alm_in[0].copy())

        self.assertTrue(alm_1d.ndim == 1)
        np.testing.assert_array_almost_equal(alm_1d, alm[0])
        
        alm_exp = np.zeros_like(alm_in)
        alm_exp[0] = hp.almxfl(alm_in[0], data.icov_ell_lensed[0])

        np.testing.assert_array_almost_equal(alm, alm_exp)
        
    def test_data_icov_diag_lensed_1D_E(self):
    
        cosmo = self.FakeCosmology()        
        pol = ['E']
        data = Data(self.lmax, self.n_ell_T,
                    self.b_ell_T, pol, cosmo)

        alm = np.random.randn(hp.Alm.getsize(self.lmax))
        alm = np.reshape(alm.astype(complex), (1, self.nelem))
        alm_in = alm.copy()
        alm = data.icov_diag_lensed(alm)

        alm_exp = np.zeros_like(alm_in)
        alm_exp[0] = hp.almxfl(alm_in[0], data.icov_ell_lensed[0])

        np.testing.assert_array_almost_equal(alm, alm_exp)
        
    def test_data_icov_diag_nonlensed_2d(self):
    
        cosmo = self.FakeCosmology()        
        pol = ['T', 'E']
        data = Data(self.lmax, self.n_ell_TplusE,
                    self.b_ell_TplusE, pol, cosmo)

        alm = np.random.randn(2 * hp.Alm.getsize(self.lmax))
        alm = np.reshape(alm.astype(complex), (2, self.nelem))
        alm_in = alm.copy()
        alm = data.icov_diag_nonlensed(alm)

        alm_exp = np.zeros_like(alm_in)
        alm_exp[0] = hp.almxfl(alm_in[0], data.icov_ell_nonlensed[0])
        alm_exp[0] += hp.almxfl(alm_in[1], data.icov_ell_nonlensed[2])        
        alm_exp[1] = hp.almxfl(alm_in[1], data.icov_ell_nonlensed[1])
        alm_exp[1] += hp.almxfl(alm_in[0], data.icov_ell_nonlensed[2])        

        np.testing.assert_array_almost_equal(alm, alm_exp)

    def test_data_icov_diag_nonlensed_1d_T(self):
    
        cosmo = self.FakeCosmology()        
        pol = ['T']
        data = Data(self.lmax, self.n_ell_T,
                    self.b_ell_T, pol, cosmo)

        alm = np.random.randn(hp.Alm.getsize(self.lmax))
        alm = np.reshape(alm.astype(complex), (1, self.nelem))
        alm_in = alm.copy()
        
        alm = data.icov_diag_nonlensed(alm)
        
        # Test if 1D input also works
        alm_1d = data.icov_diag_nonlensed(alm_in[0].copy())

        self.assertTrue(alm_1d.ndim == 1)
        np.testing.assert_array_almost_equal(alm_1d, alm[0])
        
        alm_exp = np.zeros_like(alm_in)
        alm_exp[0] = hp.almxfl(alm_in[0], data.icov_ell_nonlensed[0])

        np.testing.assert_array_almost_equal(alm, alm_exp)
        
    def test_data_icov_diag_nonlensed_1D_E(self):
    
        cosmo = self.FakeCosmology()        
        pol = ['E']
        data = Data(self.lmax, self.n_ell_T,
                    self.b_ell_T, pol, cosmo)

        alm = np.random.randn(hp.Alm.getsize(self.lmax))
        alm = np.reshape(alm.astype(complex), (1, self.nelem))
        alm_in = alm.copy()
        alm = data.icov_diag_nonlensed(alm)

        alm_exp = np.zeros_like(alm_in)
        alm_exp[0] = hp.almxfl(alm_in[0], data.icov_ell_nonlensed[0])

        np.testing.assert_array_almost_equal(alm, alm_exp)

    def test_data_lmax(self):

        pol = 'T'
        cosmo = self.FakeCosmology()
        data = Data(self.lmax, self.n_ell_T, self.b_ell_T, pol,
                    cosmo)
        
        # Test shapes.
        self.assertEqual(data.b_ell.shape, (1, self.lmax + 1))
        self.assertEqual(data.n_ell.shape, (1, self.lmax + 1))        
        self.assertEqual(data.cov_ell_lensed.shape, (1, self.lmax + 1))
        self.assertEqual(data.cov_ell_nonlensed.shape, (1, self.lmax + 1))
        self.assertEqual(data.icov_ell_lensed.shape, (1, self.lmax + 1))
        self.assertEqual(data.icov_ell_nonlensed.shape, (1, self.lmax + 1))        
        
        lmax_new = self.lmax - 1
        data.lmax = lmax_new

        # Shapes should be updated
        self.assertEqual(data.b_ell.shape, (1, lmax_new + 1))
        self.assertEqual(data.n_ell.shape, (1, lmax_new + 1))        
        self.assertEqual(data.cov_ell_lensed.shape, (1, lmax_new + 1))
        self.assertEqual(data.cov_ell_nonlensed.shape, (1, lmax_new + 1))
        self.assertEqual(data.icov_ell_lensed.shape, (1, lmax_new + 1))
        self.assertEqual(data.icov_ell_nonlensed.shape, (1, lmax_new + 1))
        
        lmax_new = self.lmax + 1
        data.lmax = lmax_new
        self.assertRaises(IndexError, getattr, data, 'b_ell')
        self.assertRaises(IndexError, getattr, data, 'n_ell')
        self.assertRaises(IndexError, getattr, data, 'cov_ell_lensed')
        self.assertRaises(IndexError, getattr, data, 'icov_ell_lensed')
        self.assertRaises(IndexError, getattr, data, 'cov_ell_nonlensed')
        self.assertRaises(IndexError, getattr, data, 'icov_ell_nonlensed')

    def test_data_n_is_totcov_I(self):
        
        pol = ['T']
        cosmo = self.FakeCosmology()
        n_ell_T = self.n_ell_T + 1 # To make well conditioned.

        data = Data(self.lmax, n_ell_T, self.b_ell_T, pol, cosmo, n_is_totcov=True)

        cov_ell_exp = np.zeros((1, self.nell))
        cov_ell_exp[0] = n_ell_T
        print(n_ell_T)
        np.testing.assert_almost_equal(data.cov_ell_nonlensed,
                                       cov_ell_exp)
        # Test if inverse is also computed correctly.
        np.testing.assert_almost_equal(data.icov_ell_nonlensed,
                                       1 / cov_ell_exp)

        # Again with lensing power. This should be the same.
        cov_ell_exp[0] = n_ell_T 
        np.testing.assert_almost_equal(data.cov_ell_lensed,
                                       cov_ell_exp)
        # Test if inverse is also computed correctly.
        np.testing.assert_almost_equal(data.icov_ell_lensed,
                                       1 / cov_ell_exp)
                
    def test_data_n_is_totcov_E(self):

        cosmo = self.FakeCosmology()
        pol = ['E']
        n_ell_T = self.n_ell_T + 1 # To make well conditioned.
        data = Data(self.lmax, n_ell_T, self.b_ell_T, pol, cosmo, 
                    n_is_totcov=True)

        cov_ell_exp = np.zeros((1, self.nell))
        cov_ell_exp[0] = n_ell_T
        np.testing.assert_almost_equal(data.cov_ell_nonlensed,
                                       cov_ell_exp)
        # Test if inverse is also computed correctly.
        np.testing.assert_almost_equal(data.icov_ell_nonlensed,
                                       1 / cov_ell_exp)

        # Again with lensing power. This should be the same.
        cov_ell_exp[0] = n_ell_T
        np.testing.assert_almost_equal(data.cov_ell_lensed,
                                       cov_ell_exp)
        # Test if inverse is also computed correctly.
        np.testing.assert_almost_equal(data.icov_ell_lensed,
                                       1 / cov_ell_exp)

    def test_data_n_is_totcov_pol(self):

        cosmo = self.FakeCosmology()
        pol = ['T', 'E']
        n_ell_TplusE = self.n_ell_TplusE + 1
        print(n_ell_TplusE)
        data = Data(self.lmax, n_ell_TplusE, self.b_ell_TplusE, pol, cosmo,
                    n_is_totcov=True)        

        expec_totcov = np.zeros((3, self.nell))
        expec_totcov[0] = n_ell_TplusE[0]
        expec_totcov[1] = n_ell_TplusE[1]
        expec_totcov[2] = n_ell_TplusE[2]
        np.testing.assert_almost_equal(data.cov_ell_nonlensed, expec_totcov)

        # Test if inverse is also computed correctly.
        expec_inv_totcov = np.zeros_like(expec_totcov)

        # Inverse of [[a,b],[b,d]] = [[d,-b],[-b,a]] / (ad - bb).
        expec_inv_totcov[0] = expec_totcov[1]
        expec_inv_totcov[1] = expec_totcov[0]
        expec_inv_totcov[2] = -expec_totcov[2]
        det = expec_totcov[0] * expec_totcov[1] - expec_totcov[2] ** 2
        expec_inv_totcov /= det
        np.testing.assert_almost_equal(data.icov_ell_nonlensed, expec_inv_totcov)

        # With lensing power.
        expec_totcov = np.zeros((3, self.nell))
        expec_totcov[0] = n_ell_TplusE[0]
        expec_totcov[1] = n_ell_TplusE[1]
        expec_totcov[2] = n_ell_TplusE[2]
        np.testing.assert_almost_equal(data.cov_ell_lensed, expec_totcov)
        
        expec_inv_totcov = np.zeros_like(expec_totcov)

        expec_inv_totcov[0] = expec_totcov[1]
        expec_inv_totcov[1] = expec_totcov[0]
        expec_inv_totcov[2] = -expec_totcov[2]
        det = expec_totcov[0] * expec_totcov[1] - expec_totcov[2] ** 2
        expec_inv_totcov /= det
        np.testing.assert_almost_equal(data.icov_ell_lensed, expec_inv_totcov)
