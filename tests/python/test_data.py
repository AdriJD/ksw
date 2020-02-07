'''
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

        self.alm_T = np.zeros(self.nelem, dtype=complex)
        self.alm_TplusE = np.zeros((2, self.nelem), dtype=complex)
        self.n_ell_T = np.arange(self.nell, dtype=float)
        self.n_ell_TplusE = np.arange(
            3 * self.nell, dtype=float).reshape(3, self.nell)
        self.b_ell_T = np.ones(self.nell)
        self.b_ell_TplusE = np.ones((2, self.nell))

        cls = np.ones((self.nell, 4))
        cls[:,0] *= 2.
        cls[:,1] *= 4.
        cls[:,2] *= 10. # BB.
        cls[:,3] *= 1.        

        ells = np.arange(self.nell)
        lensed_scalar = {'cls' : cls, 'ells' : ells} 
        unlensed_scalar = {'cls' : 2 * cls, 'ells' : ells}       
        
        class FakeCosmology():
            def __init__(self):
                self.cls = {'lensed_scalar' : lensed_scalar,
                            'unlensed_scalar' : unlensed_scalar}
        self.FakeCosmology = FakeCosmology
        self.cls = cls
        
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
        
    def test_data_compute_totcov_diag(self):

        cosmo = self.FakeCosmology()
        pol = ['T', 'E']
        data = Data(self.alm_TplusE, self.n_ell_TplusE,
                    self.b_ell_TplusE, pol)

        # Without lensing power.
        data.compute_totcov_diag(cosmo, add_lens_power=False)
        self.assertEqual(data.totcov_diag.shape, (3, self.nell))

        expec_totcov = np.zeros((3, self.nell))
        expec_totcov[0] = self.n_ell_TplusE[0] + self.cls[:,0] * 2
        expec_totcov[1] = self.n_ell_TplusE[1] + self.cls[:,1] * 2
        expec_totcov[2] = self.n_ell_TplusE[2] + self.cls[:,3] * 2    
        np.testing.assert_almost_equal(data.totcov_diag, expec_totcov)

        # With lensing power.        
        data.compute_totcov_diag(cosmo, add_lens_power=True)
        self.assertEqual(data.totcov_diag.shape, (3, self.nell))

        expec_totcov[0] = self.n_ell_TplusE[0] + self.cls[:,0]
        expec_totcov[1] = self.n_ell_TplusE[1] + self.cls[:,1]
        expec_totcov[2] = self.n_ell_TplusE[2] + self.cls[:,3]    
        np.testing.assert_almost_equal(data.totcov_diag, expec_totcov)
        
    
    # test for sim
    # test invcov

