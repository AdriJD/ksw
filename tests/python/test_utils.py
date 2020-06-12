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
        
        
