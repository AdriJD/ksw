import unittest
import numpy as np

from ksw import legendre

class TestLegendre(unittest.TestCase):

    def test_legendre_compute_normalized_associated_legendre(self):

        # Test output shape and values.
        thetas = np.linspace(0, np.pi, 10)
        lmax = 2

        # I expect values to be Y_lm(theta, 0).
        m = 0
        out_exp = np.ones((thetas.size, lmax + 1))
        out_exp[:,0] = np.sqrt(1 / 4. / np.pi) # ell = 0.
        out_exp[:,1] = np.sqrt(3 / 4. / np.pi) * np.cos(thetas) # ell = 1.
        out_exp[:,2] = np.sqrt(5 / 16. / np.pi) * (3 * np.cos(thetas) ** 2 - 1) # ell = 2.
        out = legendre.compute_normalized_associated_legendre(m, thetas, lmax)
        np.testing.assert_array_almost_equal(out, out_exp)

        m = 1
        out_exp = np.ones((thetas.size, lmax - m + 1))
        out_exp[:,0] = -np.sqrt(3 / 8. / np.pi) * np.sin(thetas) # ell = 1.
        out_exp[:,1] = -np.sqrt(15 / 8. / np.pi) * np.sin(thetas) * np.cos(thetas) # ell = 2.
        out = legendre.compute_normalized_associated_legendre(m, thetas, lmax)
        np.testing.assert_array_almost_equal(out, out_exp)

        m = 2
        out_exp = np.ones((thetas.size, lmax - m + 1))
        out_exp[:,0] = np.sqrt(15 / 32. / np.pi) * np.sin(thetas) ** 2 # ell = 2.
        out = legendre.compute_normalized_associated_legendre(m, thetas, lmax)
        np.testing.assert_array_almost_equal(out, out_exp)

    def test_legendre_compute_normalized_associated_legendre_out(self):

        thetas = np.linspace(0, np.pi, 10)
        lmax = 2

        m = 0
        out_exp = np.ones((thetas.size, lmax + 1))
        out_exp[:,0] = np.sqrt(1 / 4. / np.pi) # ell = 0.
        out_exp[:,1] = np.sqrt(3 / 4. / np.pi) * np.cos(thetas) # ell = 1.
        out_exp[:,2] = np.sqrt(5 / 16. / np.pi) * (3 * np.cos(thetas) ** 2 - 1) # ell = 2.
        out = np.empty_like(out_exp)
        legendre.compute_normalized_associated_legendre(m, thetas, lmax, out=out)
        np.testing.assert_array_almost_equal(out, out_exp)

    def test_legendre_compute_normalized_associated_legendre_err(self):

        thetas = np.linspace(0, np.pi, 10)
        lmax = 2

        m = -1
        self.assertRaises(ValueError, legendre.compute_normalized_associated_legendre,
                          m, thetas, lmax)
        m = 2
        lmax = 1
        self.assertRaises(ValueError, legendre.compute_normalized_associated_legendre,
                          m, thetas, lmax)

        lmax = 3
        out = np.empty((thetas.size, lmax + 1 - 1)) # Note
        self.assertRaises(ValueError, legendre.compute_normalized_associated_legendre,
                          m, thetas, lmax, out=out)

    def test_legendre_normalization(self):
        
        # Test if I understand the relation to the regular Legendre polynomials.

        thetas = np.linspace(0, np.pi, 10, endpoint=False)
        lmax = 2

        P0 = np.ones(thetas.size)
        P1 = np.cos(thetas)
        P2 = 0.5 * (3 * np.cos(thetas) ** 2 - 1)

        m = 0
        out_exp = np.ones((thetas.size, lmax + 1))

        # So Yl0(theta) = sqrt( (2l + 1) / 4pi) Pl(cos(theta)).
        out_exp[:,0] = np.sqrt(1 / 4. / np.pi) * P0
        out_exp[:,1] = np.sqrt((2 * 1 + 1) / 4. / np.pi) * P1
        out_exp[:,2] = np.sqrt((2 * 2 + 1) / 4. / np.pi) * P2
        
        out = np.empty_like(out_exp)
        legendre.compute_normalized_associated_legendre(m, thetas, lmax, out=out)
        np.testing.assert_array_almost_equal(out, out_exp)        

    def test_legendre_normalized_associated_legendre_ms(self):

        lmax = 2
        ms = np.arange(lmax + 1)
        out_exp = np.ones((ms.size, lmax + 1))

        theta = np.pi / 4.
        out_exp[0,0] = np.sqrt(1 / 4. / np.pi)
        out_exp[0,1] = np.sqrt(3 / 4. / np.pi) * np.cos(theta)
        out_exp[0,2] = np.sqrt(5 / 16. / np.pi) * (3 * np.cos(theta) ** 2 - 1)
        out_exp[1,0] = 0
        out_exp[1,1] = -np.sqrt(3 / 8. / np.pi) * np.sin(theta)
        out_exp[1,2] = -np.sqrt(15 / 8. / np.pi) * np.sin(theta) * np.cos(theta)
        out_exp[2,0] = 0
        out_exp[2,1] = 0
        out_exp[2,2] = np.sqrt(15 / 32. / np.pi) * np.sin(theta) ** 2

        out = legendre.normalized_associated_legendre_ms(ms, theta, lmax)
        np.testing.assert_array_almost_equal(out, out_exp)

        theta = np.pi / 5.4
        out_exp[0,0] = np.sqrt(1 / 4. / np.pi)
        out_exp[0,1] = np.sqrt(3 / 4. / np.pi) * np.cos(theta)
        out_exp[0,2] = np.sqrt(5 / 16. / np.pi) * (3 * np.cos(theta) ** 2 - 1)
        out_exp[1,0] = 0
        out_exp[1,1] = -np.sqrt(3 / 8. / np.pi) * np.sin(theta)
        out_exp[1,2] = -np.sqrt(15 / 8. / np.pi) * np.sin(theta) * np.cos(theta)
        out_exp[2,0] = 0
        out_exp[2,1] = 0
        out_exp[2,2] = np.sqrt(15 / 32. / np.pi) * np.sin(theta) ** 2

        out = legendre.normalized_associated_legendre_ms(ms, theta, lmax)
        np.testing.assert_array_almost_equal(out, out_exp)

    def test_legendre_normalized_associated_legendre_ms_out(self):

        lmax = 2
        ms = np.arange(lmax + 1)
        out_exp = np.ones((ms.size, lmax + 1))

        theta = np.pi / 4.
        out_exp[0,0] = np.sqrt(1 / 4. / np.pi)
        out_exp[0,1] = np.sqrt(3 / 4. / np.pi) * np.cos(theta)
        out_exp[0,2] = np.sqrt(5 / 16. / np.pi) * (3 * np.cos(theta) ** 2 - 1)
        out_exp[1,0] = 0
        out_exp[1,1] = -np.sqrt(3 / 8. / np.pi) * np.sin(theta)
        out_exp[1,2] = -np.sqrt(15 / 8. / np.pi) * np.sin(theta) * np.cos(theta)
        out_exp[2,0] = 0
        out_exp[2,1] = 0
        out_exp[2,2] = np.sqrt(15 / 32. / np.pi) * np.sin(theta) ** 2

        # Test if out is filled with zeros for m > ell entries.
        out = np.ones_like(out_exp) * np.nan
        legendre.normalized_associated_legendre_ms(ms, theta, lmax, out=out)
        np.testing.assert_array_almost_equal(out, out_exp)

    def test_legendre_normalized_associated_legendre_ms_err(self):

        lmax = 2
        ms = np.arange(lmax + 1)
        out_exp = np.ones((ms.size, lmax + 1))

        theta = -1.
        self.assertRaises(ValueError, legendre.normalized_associated_legendre_ms,
                          ms, theta, lmax)
        theta = np.pi + 1.
        self.assertRaises(ValueError, legendre.normalized_associated_legendre_ms,
                          ms, theta, lmax)        

        out = np.empty((ms.size, lmax + 1 - 1)) # Note
        self.assertRaises(ValueError, legendre.normalized_associated_legendre_ms,
                          ms, theta, lmax, out=out)
