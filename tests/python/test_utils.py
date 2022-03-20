import unittest
import numpy as np
import healpy as hp
from mpi4py import MPI

from ksw import utils

class TestUtils(unittest.TestCase):

    def test_utils_get_trapz_weights(self):

        x = np.asarray([3, 4, 5, 10])
        dx = utils.get_trapz_weights(x)

        dx_exp = np.asarray([0.5, 1, 3, 2.5])
        np.testing.assert_array_almost_equal(dx, dx_exp)

        y = np.asarray([6, 3, 7, 3])

        self.assertAlmostEqual(np.sum(y * dx), np.trapz(y, x))

    def test_utils_get_trapz_weights_err(self):

        x = np.arange(9).reshape((3, 3))
        self.assertRaises(ValueError, utils.get_trapz_weights, x)

        x = np.asarray([3, 4, -5, 10])
        self.assertRaises(ValueError, utils.get_trapz_weights, x)

    def test_utils_compute_fftlen_fftw(self):

        len_min = 734
        len_opt_exp = 735 # 2^0 3^1 5^1 7^2.
        len_opt_even_exp = 750 # 2^1 3^1 5^3 7^0.

        len_opt_even = utils.compute_fftlen_fftw(len_min)
        self.assertEqual(len_opt_even, len_opt_even_exp)

        len_opt = utils.compute_fftlen_fftw(len_min, even=False)
        self.assertEqual(len_opt, len_opt_exp)

    def test_utils_alm2a_ell_m(self):

        alm = np.ones((1, 2, 10), dtype=complex)
        alm *= np.arange(10, dtype=complex)

        arr = np.ones((1, 2, 4, 4), dtype=alm.dtype)
        arr_exp = np.ones((1, 2, 4, 4), dtype=alm.dtype)
        arr_exp *= np.asarray([[0, 0, 0, 0],
                              [1, 4, 0, 0],
                              [2, 5, 7, 0],
                              [3, 6, 8, 9]], dtype=complex)
        
        utils.alm2a_ell_m(alm, out=arr)
        np.testing.assert_array_equal(arr, arr_exp)

        self.assertTrue(arr.flags['OWNDATA'])

    def test_utils_alm2a_ell_m_copy(self):

        alm = np.ones((1, 2, 10), dtype=complex)
        alm *= np.arange(10, dtype=complex)

        arr_exp = np.ones((1, 2, 4, 4), dtype=alm.dtype)
        arr_exp *= np.asarray([[0, 0, 0, 0],
                              [1, 4, 0, 0],
                              [2, 5, 7, 0],
                              [3, 6, 8, 9]], dtype=complex)
        
        arr = utils.alm2a_ell_m(alm)
        np.testing.assert_array_equal(arr, arr_exp)
        
    def test_utils_alm2a_ell_m_err(self):

        alm = np.ones((1, 2, 10), dtype=complex)
        alm *= np.arange(10, dtype=complex)

        arr = np.ones((2, 2, 4, 4), dtype=alm.dtype) # Wrong dims.

        self.assertRaises(ValueError, utils.alm2a_ell_m, alm, **{'out' : arr})

        arr = np.ones((1, 2, 5, 5), dtype=alm.dtype) # Wrong last dims.

        self.assertRaises(ValueError, utils.alm2a_ell_m, alm, **{'out' : arr})

    def test_utils_a_ell_m2alm(self):

        arr = np.ones((1, 2, 4, 4), dtype=complex)
        arr *= np.asarray([[0, 0, 0, 0],
                           [1, 4, 0, 0],
                           [2, 5, 7, 0],
                           [3, 6, 8, 9]], dtype=complex)
        
        alm = np.ones((1, 2, 10), dtype=complex) * np.nan
        alm_exp = np.ones_like(alm) * np.arange(10, dtype=complex)

        utils.a_ell_m2alm(arr, out=alm)
        np.testing.assert_array_equal(alm, alm_exp)

        self.assertTrue(alm.flags['OWNDATA'])

    def test_utils_a_ell_m2alm_copy(self):

        arr = np.ones((1, 2, 4, 4), dtype=complex)
        arr *= np.asarray([[0, 0, 0, 0],
                           [1, 4, 0, 0],
                           [2, 5, 7, 0],
                           [3, 6, 8, 9]], dtype=complex)
        
        alm_exp = np.ones((1, 2, 10), dtype=complex) * \
            np.arange(10, dtype=complex)

        alm = utils.a_ell_m2alm(arr)
        np.testing.assert_array_equal(alm, alm_exp)
        
    def test_utils_a_ell_m2alm_err(self):

        arr = np.ones((1, 2, 4, 4), dtype=complex)
        arr *= np.asarray([[0, 0, 0, 0],
                           [1, 4, 0, 0],
                           [2, 5, 7, 0],
                           [3, 6, 8, 9]], dtype=complex)        

        alm = np.ones((2, 2, 10), dtype=complex) # Wrong dim.s

        self.assertRaises(ValueError, utils.a_ell_m2alm, arr, **{'out' : alm})

        alm = np.ones((1, 2, 12), dtype=alm.dtype) # Wrong last dims.

        self.assertRaises(ValueError, utils.a_ell_m2alm, arr, **{'out' : alm})

    def test_utils_contract_almxblm(self):
        
        alm = np.asarray([1, 1, 1, 1, 2j, 2j, 2j, 3j, 3j, 4j])
        blm = np.asarray([2, 2, 2, 2, 3, 3, 3, 4j, 4j, 5])

        ans_exp = -40

        ans = utils.contract_almxblm(alm, blm)

        self.assertEqual(ans, ans_exp)

    def test_utils_contract_almxblm_2d(self):
        
        alm = np.ones((3, 10), dtype=np.complex128)
        blm = np.ones((3, 10), dtype=np.complex128)

        alm *= np.asarray([1, 1, 1, 1, 2j, 2j, 2j, 3j, 3j, 4j])
        blm *= np.asarray([2, 2, 2, 2, 3, 3, 3, 4j, 4j, 5])

        ans_exp = -120

        ans = utils.contract_almxblm(alm, blm)

        self.assertEqual(ans, ans_exp)

    def test_utils_contract_almxblm_3d(self):
        
        alm = np.ones((2, 3, 10), dtype=np.complex128)
        blm = np.ones((2, 3, 10), dtype=np.complex128)

        alm *= np.asarray([1, 1, 1, 1, 2j, 2j, 2j, 3j, 3j, 4j])
        blm *= np.asarray([2, 2, 2, 2, 3, 3, 3, 4j, 4j, 5])

        ans_exp = -240

        ans = utils.contract_almxblm(alm, blm)

        self.assertEqual(ans, ans_exp)

    def test_utils_contract_almxblm_cl(self):

        # Check if contraction matches hp.alm2cl.
        
        alm = np.ones(10, dtype=np.complex128)
        alm += 1j * np.ones(10, dtype=np.complex128)
        alm[:4] = 1
        lmax = 3
        ells = np.asarray([0, 1, 2, 3])        

        cl = hp.alm2cl(alm)
        ans_exp = np.sum(cl * (2 * ells + 1))

        ans = utils.contract_almxblm(alm, np.conj(alm))
    
        self.assertEqual(ans, ans_exp)
        
    def test_utils_contract_almxblm_err(self):
        
        alm = np.ones((10), dtype=np.complex128)
        blm = np.ones((11), dtype=np.complex128)

        self.assertRaises(ValueError, utils.contract_almxblm, alm, blm)

    def test_utils_alm_return_2d_1d(self):

        npol = 1
        lmax = 5
        alm = np.ones(hp.Alm.getsize(lmax), dtype=np.complex128)

        alm_out = utils.alm_return_2d(alm, npol, lmax)        
        
        self.assertEqual(alm_out.shape, (1, hp.Alm.getsize(lmax)))
        self.assertTrue(np.shares_memory(alm, alm_out))

    def test_utils_alm_return_2d_2d(self):

        npol = 1
        lmax = 5
        alm = np.ones((1, hp.Alm.getsize(lmax)), dtype=np.complex128)

        alm_out = utils.alm_return_2d(alm, npol, lmax)        
        
        self.assertEqual(alm_out.shape, (1, hp.Alm.getsize(lmax)))
        self.assertTrue(np.shares_memory(alm, alm_out))

    def test_utils_alm_return_2d_2d_pol(self):

        npol = 2
        lmax = 5
        alm = np.ones((npol, hp.Alm.getsize(lmax)), dtype=np.complex128)

        alm_out = utils.alm_return_2d(alm, npol, lmax)        
        
        self.assertEqual(alm_out.shape, (2, hp.Alm.getsize(lmax)))
        self.assertTrue(np.shares_memory(alm, alm_out))

    def test_utils_alm_return_2d_err(self):

        npol = 2
        lmax = 5
        alm = np.ones((npol + 1, hp.Alm.getsize(lmax)), dtype=np.complex128)

        self.assertRaises(ValueError, utils.alm_return_2d, alm, npol, lmax)

        npol = 2
        lmax = 5
        alm = np.ones((npol, hp.Alm.getsize(lmax) + 1), dtype=np.complex128)

        self.assertRaises(ValueError, utils.alm_return_2d, alm, npol, lmax)

    def test_utils_fakempicomm(self):

        comm = utils.FakeMPIComm()
        self.assertEqual(comm.Get_rank(), 0)
        self.assertEqual(comm.Get_size(), 1)
        self.assertEqual(comm.rank, 0)
        self.assertEqual(comm.size, 1)
        self.assertTrue(callable(comm.Barrier))

    def test_utils_reduce_array_fake(self):

        arr = np.random.randn(100).reshape((5, 20)).astype(complex)
        arr_out = utils.reduce_array(arr, comm=utils.FakeMPIComm())

        np.testing.assert_array_equal(arr, arr_out)

    def test_utils_reduce_array(self):

        # Note, I can only test the n=1 case.        
        
        arr = np.random.randn(100).reshape((5, 20)).astype(complex)
        arr_out = utils.reduce_array(arr, comm=MPI.COMM_WORLD)

        np.testing.assert_array_equal(arr, arr_out)

    def test_utils_reduce_fake(self):

        obj = 2
        obj_out = utils.reduce(obj, comm=utils.FakeMPIComm())

        self.assertEqual(obj, obj_out)

    def test_utils_reduce(self):

        # Note, I can only test the n=1 case.        
        obj = 2
        obj_out = utils.reduce(obj, comm=MPI.COMM_WORLD)

        self.assertEqual(obj, obj_out)
        
