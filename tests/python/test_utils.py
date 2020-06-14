'''
Test the Shape class.
'''
import unittest
import numpy as np

from ksw import utils

class TestUtils(unittest.TestCase):

    def test_get_trapz_weights(self):

        x = np.asarray([3, 4, 5, 10])
        dx = utils.get_trapz_weights(x)
        
        dx_exp = np.asarray([0.5, 1, 3, 2.5])
        np.testing.assert_array_almost_equal(dx, dx_exp)

        y = np.asarray([6, 3, 7, 3])
        
        self.assertAlmostEqual(np.sum(y * dx), np.trapz(y, x))

    def test_get_trapz_weights_err(self):

        x = np.arange(9).reshape((3, 3))
        self.assertRaises(ValueError, utils.get_trapz_weights, x)        
        
        x = np.asarray([3, 4, -5, 10])
        self.assertRaises(ValueError, utils.get_trapz_weights, x)        
        
    def test_compute_fftlen_fftw(self):

        len_min = 734
        len_opt_exp = 735 # 2^0 3^1 5^1 7^2.
        len_opt_even_exp = 750 # 2^1 3^1 5^3 7^0.

        len_opt_even = utils.compute_fftlen_fftw(len_min)
        self.assertEqual(len_opt_even, len_opt_even_exp)
        
        len_opt = utils.compute_fftlen_fftw(len_min, even=False)
        self.assertEqual(len_opt, len_opt_exp)

        
