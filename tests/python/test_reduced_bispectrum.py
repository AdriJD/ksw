import unittest
import numpy as np
import tempfile
import pathlib
import os

from ksw import ReducedBispectrum

class TestReducedBispectrum(unittest.TestCase):

    def test_reducedbispectrum_init(self):
        
        n_unique = 2
        nfact = 3
        npol = 2
        nell = 3

        factors = np.ones((n_unique, npol, nell))
        weights = np.ones((nfact, 3, npol))
        rule = np.ones((nfact, 3), dtype=int)
        ells = np.arange(nell)
        name = 'test_bispec'

        rb = ReducedBispectrum(factors, rule, weights, ells, name)

        self.assertEqual(rb.name, name)
        np.testing.assert_equal(rb.ells_sparse, ells)
        np.testing.assert_equal(rb.ells_full, ells)
        np.testing.assert_almost_equal(rb.factors, factors)
        np.testing.assert_almost_equal(rb.rule, rule)        
        np.testing.assert_almost_equal(rb.weights, weights)

    def test_reducedbispectrum_init_err_shape(self):

        n_unique = 2
        nfact = 3
        npol = 2
        nell = 3

        factors = np.ones((n_unique, npol, nell))
        rule = np.ones((nfact, 3), dtype=int)
        ells = np.arange(nell)
        name = 'test_bispec'

        weights = np.ones((nfact + 1, 3, npol))        
        self.assertRaises(ValueError, ReducedBispectrum,
                          factors, rule, weights, ells, name)

        weights = np.ones((nfact, 3, npol + 1))
        self.assertRaises(ValueError, ReducedBispectrum,
                          factors, rule, weights, ells, name)

        weights = np.ones((nfact, 3 + 1, npol))
        self.assertRaises(ValueError, ReducedBispectrum,
                          factors, rule, weights, ells, name)

        weights = np.ones((3, nfact, npol))
        ells = np.arange(nell + 1)
        self.assertRaises(ValueError, ReducedBispectrum,
                          factors, rule, weights, ells, name)

        ells = np.arange(nell)
        rule = np.ones((nfact + 1, 3), dtype=int)
        self.assertRaises(ValueError, ReducedBispectrum,
                          factors, rule, weights, ells, name)

        rule = np.ones((nfact, 3 + 1), dtype=int)
        self.assertRaises(ValueError, ReducedBispectrum,
                          factors, rule, weights, ells, name)

    def test_reducedbispectrum_init_err_rule(self):

        n_unique = 2
        nfact = 3
        npol = 2
        nell = 3

        factors = np.ones((n_unique, npol, nell))
        ells = np.arange(nell)
        name = 'test_bispec'
        weights = np.ones((nfact, 3, npol))

        rule = np.ones((nfact, 3), dtype=float)        
        self.assertRaises(TypeError, ReducedBispectrum,
                          factors, rule, weights, ells, name)

        rule = np.ones((nfact + 1, 3), dtype=int)        
        self.assertRaises(ValueError, ReducedBispectrum,
                          factors, rule, weights, ells, name)

        rule = np.ones((nfact, 3 + 1), dtype=int)        
        self.assertRaises(ValueError, ReducedBispectrum,
                          factors, rule, weights, ells, name)

        rule = np.ones((nfact, 3), dtype=int)
        rule[0,0] = -1
        self.assertRaises(ValueError, ReducedBispectrum,
                          factors, rule, weights, ells, name)

        rule = np.ones((nfact, 3), dtype=int)
        rule[0,0] = n_unique
        self.assertRaises(ValueError, ReducedBispectrum,
                          factors, rule, weights, ells, name)
        
    def test_reducedbispectrum_init_err_name(self):

        n_unique = 2
        nfact = 3
        npol = 2
        nell = 3

        factors = np.ones((n_unique, npol, nell))
        weights = np.ones((nfact, 3, npol))
        rule = np.ones((nfact, 3), dtype=int)
        ells = np.arange(nell)
        name = ''

        self.assertRaises(ValueError, ReducedBispectrum,
                          factors, rule, weights, ells, name)

        name = None
        self.assertRaises(ValueError, ReducedBispectrum,
                          factors, rule, weights, ells, name)

        name = ' '
        self.assertRaises(ValueError, ReducedBispectrum,
                          factors, rule, weights, ells, name)

    def test_reducedbispectrum_init_interp_bispec(self):

        n_unique = 2
        nfact = 2
        npol = 2
        nell_sparse = 7
        ells_sparse = np.asarray([3, 4, 5, 7, 10, 13, 15])
        
        weights = np.ones((nfact, 3, npol))
        rule = np.ones((nfact, 3), dtype=int)
        name = 'test_bispec'

        factors = np.ones((n_unique, npol, nell_sparse))
        factors[1] = 2.
        factors *= np.sin(0.1 * ells_sparse)
        
        ells_full = np.arange(3, 16)
        nell = ells_full.size

        factors_expec = np.ones((n_unique, npol, nell))
        factors_expec[1] = 2.
        factors_expec *= np.sin(0.1 * ells_full)

        rb = ReducedBispectrum(factors, rule, weights, ells_sparse, name)
        np.testing.assert_almost_equal(rb.factors, factors_expec,
                                       decimal=4)

class TestReducedBispectrumIO(unittest.TestCase):

    def setUp(self):

        # Get location of this script.
        self.path = pathlib.Path(__file__).parent.absolute()

    def tearDown(self):
        # Is called after each test.
        pass
        
    def test_read_write_red_bisp(self):

        n_unique = 2
        nfact = 4
        npol = 2
        ells_sparse = np.asarray([3, 5, 7, 10, 13, 15])
        ells_full = np.arange(3, 16)
        
        weights = np.ones((nfact, 3, npol))
        rule = np.ones((nfact, 3), dtype=int)
        factors = np.ones((n_unique, npol, ells_sparse.size))
        name = 'test_bispec'

        rb = ReducedBispectrum(factors, rule, weights, ells_sparse, name)
        
        with tempfile.TemporaryDirectory(dir=self.path) as tmpdirname:

            filename = os.path.join(tmpdirname, 'red_bisp')
            rb.write(filename)

            rb2 = ReducedBispectrum.init_from_file(filename)

        np.testing.assert_array_almost_equal(rb2.factors, rb.factors)
        np.testing.assert_array_equal(rb2.rule, rb.rule)
        np.testing.assert_array_almost_equal(rb2.weights, rb.weights)
        np.testing.assert_array_equal(rb2.ells_full, ells_full)
        np.testing.assert_array_equal(rb2.ells_sparse, ells_full)
        self.assertEqual(rb2.name, rb.name)        
        
