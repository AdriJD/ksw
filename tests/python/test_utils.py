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

    def test_alm2a_m_ell(self):

        alm = np.ones((1, 2, 10), dtype=complex)
        alm *= np.arange(10, dtype=complex)

        arr = np.ones((1, 2, 4, 4), dtype=alm.dtype)
        arr_exp = np.ones((1, 2, 4, 4), dtype=alm.dtype)
        arr_exp *= np.asarray([[0, 1, 2, 3],
                              [0, 4, 5, 6],
                              [0, 0, 7, 8],
                              [0, 0, 0, 9]], dtype=complex)

        utils.alm2a_m_ell(alm, out=arr)
        np.testing.assert_array_equal(arr, arr_exp)

        self.assertTrue(arr.flags['OWNDATA'])

    def test_alm2a_m_ell_copy(self):

        alm = np.ones((1, 2, 10), dtype=complex)
        alm *= np.arange(10, dtype=complex)

        arr_exp = np.ones((1, 2, 4, 4), dtype=alm.dtype)
        arr_exp *= np.asarray([[0, 1, 2, 3],
                              [0, 4, 5, 6],
                              [0, 0, 7, 8],
                              [0, 0, 0, 9]], dtype=complex)

        arr = utils.alm2a_m_ell(alm)
        np.testing.assert_array_equal(arr, arr_exp)
        
    def test_alm2a_m_ell_err(self):

        alm = np.ones((1, 2, 10), dtype=complex)
        alm *= np.arange(10, dtype=complex)

        arr = np.ones((2, 2, 4, 4), dtype=alm.dtype) # Wrong dims.

        self.assertRaises(ValueError, utils.alm2a_m_ell, alm, **{'out' : arr})

        arr = np.ones((1, 2, 5, 5), dtype=alm.dtype) # Wrong last dims.

        self.assertRaises(ValueError, utils.alm2a_m_ell, alm, **{'out' : arr})

    def test_a_m_ell2alm(self):

        arr = np.ones((1, 2, 4, 4), dtype=complex)
        arr *= np.asarray([[0, 1, 2, 3],
                           [0, 4, 5, 6],
                           [0, 0, 7, 8],
                           [0, 0, 0, 9]], dtype=complex)

        alm = np.ones((1, 2, 10), dtype=complex) * np.nan
        alm_exp = np.ones_like(alm) * np.arange(10, dtype=complex)

        utils.a_m_ell2alm(arr, out=alm)
        np.testing.assert_array_equal(alm, alm_exp)

        self.assertTrue(alm.flags['OWNDATA'])

    def test_a_m_ell2alm_copy(self):

        arr = np.ones((1, 2, 4, 4), dtype=complex)
        arr *= np.asarray([[0, 1, 2, 3],
                           [0, 4, 5, 6],
                           [0, 0, 7, 8],
                           [0, 0, 0, 9]], dtype=complex)

        alm_exp = np.ones((1, 2, 10), dtype=complex) * \
            np.arange(10, dtype=complex)

        alm = utils.a_m_ell2alm(arr)
        np.testing.assert_array_equal(alm, alm_exp)
        
    def test_a_m_ell2alm_err(self):

        arr = np.ones((1, 2, 4, 4), dtype=complex)
        arr *= np.asarray([[0, 1, 2, 3],
                           [0, 4, 5, 6],
                           [0, 0, 7, 8],
                           [0, 0, 0, 9]], dtype=complex)

        alm = np.ones((2, 2, 10), dtype=complex) # Wrong dim.s

        self.assertRaises(ValueError, utils.a_m_ell2alm, arr, **{'out' : alm})

        alm = np.ones((1, 2, 12), dtype=alm.dtype) # Wrong last dims.

        self.assertRaises(ValueError, utils.a_m_ell2alm, arr, **{'out' : alm})

    def test_fakempicomm(self):

        comm = utils.FakeMPIComm()
        self.assertEqual(comm.Get_rank(), 0)
        self.assertEqual(comm.Get_size(), 1)
