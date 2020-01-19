'''
test ksw
'''
import unittest
from ksw import radial_functional as rf
import numpy as np
from scipy.special import spherical_jn

class TestTools(unittest.TestCase):
    
    def test_check_and_return_shape_2d(self):

        arr_shape = (10, 5)
        arr = np.ones(arr_shape)
        s = rf.check_and_return_shape(arr, arr_shape)
        self.assertEqual(s, arr_shape)

    def test_check_and_return_shape_1d(self):

        arr_shape = (10,)
        arr = np.ones(arr_shape)
        s = rf.check_and_return_shape(arr, arr_shape)
        self.assertEqual(s, arr_shape)
        
    def test_check_and_return_shape_raise(self):

        arr_shape = (10, 5)
        arr = np.ones(arr_shape)
        arr_shape_wrong = (10,)
        self.assertRaises(ValueError, rf.check_and_return_shape,
                     arr, arr_shape_wrong)

    def test_check_and_return_shape_none(self):

        arr_shape = (10, 5)
        arr = np.ones(arr_shape)
        arr_shape_unknown = (None, 5)
        s = rf.check_and_return_shape(arr, arr_shape_unknown)
        self.assertEqual(s, arr_shape)
        
    def test_radial_func_shape(self):

        # Only test if output shapes make sense.
        nk = 1000
        ncomp = 4
        nell = 5
        nr = 3
        npol = 2
        
        f_k = np.ones((nk, ncomp), dtype=float)
        tr_ell_k = np.ones((nell, nk, npol), dtype=float)
        k = np.linspace(1e-6, 1e-2, nk, dtype=float)
        radii = np.asarray([10000, 12000, 13000], dtype=float)
        ells = np.arange(100, 100+nell, dtype=int)

        f_ell_r = rf.radial_func(f_k, tr_ell_k, k, radii, ells)
        self.assertEqual(f_ell_r.shape, (nr, nell, npol, ncomp))
        
    def test_radial_func_shape_err(self):

        # Only test if error is raised for wrong input
        # shapes.
        nk = 1000
        ncomp = 4
        nell = 5
        nr = 3
        npol = 2

        nk_wrong = 1001

        f_k = np.ones((nk_wrong, ncomp), dtype=float)
        tr_ell_k = np.ones((nell, nk, npol), dtype=float)
        k = np.linspace(1e-6, 1e-2, nk, dtype=float)
        radii = np.asarray([10000, 12000, 13000], dtype=float)
        ells = np.arange(100, 100+nell, dtype=int)

        self.assertRaises(ValueError, rf.radial_func,
                    f_k, tr_ell_k, k, radii, ells)
        
    def test_radial_func_inf_err(self):

        # Only test if error is raised for input with inf.
        nk = 1000
        ncomp = 4
        nell = 5
        nr = 3
        npol = 2

        f_k = np.ones((nk, ncomp), dtype=float)
        tr_ell_k = np.ones((nell, nk, npol), dtype=float)
        k = np.linspace(1e-6, 1e-2, nk, dtype=float)
        radii = np.asarray([10000, 12000, 13000], dtype=float)
        ells = np.arange(100, 100+nell, dtype=int)

        tr_ell_k[0, 100, 0] = np.inf
        
        self.assertRaises(ValueError, rf.radial_func,
                    f_k, tr_ell_k, k, radii, ells)

    def test_radial_func_nan_err(self):

        # Only test if error is raised for input with inf.
        nk = 1000
        ncomp = 4
        nell = 5
        nr = 3
        npol = 2

        f_k = np.ones((nk, ncomp), dtype=float)
        tr_ell_k = np.ones((nell, nk, npol), dtype=float)
        k = np.linspace(1e-6, 1e-2, nk, dtype=float)
        radii = np.asarray([10000, 12000, 13000], dtype=float)
        ells = np.arange(100, 100+nell, dtype=int)

        tr_ell_k[0, 100, 0] = np.nan
        
        self.assertRaises(ValueError, rf.radial_func,
                    f_k, tr_ell_k, k, radii, ells)
        

    def test_radial_func(self):

        # Test using orthogonality of spherical bessel functions.
        #
        # Int_0^inf k^2 dk jl(kr) jl(kr') = pi/(2r^2) * delta(r-r').

        nk = int(1e5)
        ncomp = 3
        nell = 1
        nr = 1
        npol = 1

        ell = 100
        radius = 10000.
        radii_prime = [9000., 10000., 11000.]
        
        k = np.linspace(1e-3, 3e0, nk, dtype=float)
        f_k = np.ones((nk, ncomp), dtype=float)
        
        for ridx, rp in enumerate(radii_prime):            
            f_k[:,ridx] = spherical_jn(ell, k * rp)
        
        radii = np.asarray([radius], dtype=float)        
        tr_ell_k = np.ones((nell, nk, npol), dtype=float)
        ells = np.asarray([ell], dtype=int)

        # Because this integral is so hard to get accurate to
        # Percent level using finite k samples, we compare the
        # result to the same integral calculated using numpy.
        # The actual integral (i.e. the one weighted by the
        # transfer function Tl(k)) is simpler because Tl(k)
        # falls of much faster with k than jl.
        
        exp_f_ell_r = np.zeros((nr, nell, npol, ncomp), dtype=float)
        for ridx, rp in enumerate(radii_prime):            
            exp_f_ell_r[0,0,0,ridx] = np.trapz(
             f_k[:,ridx] * spherical_jn(ell, k * radius) * k ** 2, k)
        
        f_ell_r = rf.radial_func(f_k, tr_ell_k, k, radii, ells)

        np.testing.assert_array_almost_equal(
            f_ell_r / exp_f_ell_r, np.ones_like(exp_f_ell_r),
            decimal=5)

        # Test if answer for r = r' is roughly right.
        exp_ans = 0.5 * np.pi * radius ** -2
        
        abs_ratio = np.abs(exp_ans / exp_f_ell_r[0,0,0,1])
        self.assertTrue(0.9 < abs_ratio)
        self.assertTrue(abs_ratio < 1.1)

        # Test if answers for r != r' are smaller than r = r' case.
        ratio = exp_f_ell_r[0,0,0,0] / exp_f_ell_r[0,0,0,1]
        self.assertTrue(ratio < 1e-3)

        ratio = exp_f_ell_r[0,0,0,2] / exp_f_ell_r[0,0,0,1]
        self.assertTrue(ratio < 1e-3)
