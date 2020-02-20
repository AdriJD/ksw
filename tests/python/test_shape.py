'''A
Test the Model class.
'''
import unittest
import numpy as np
import os

from ksw import Shape

class TestShape(unittest.TestCase):

    def setUp(self):
        # Is called before each test.

        # Create the local model.
        def f1(k):
            return k ** 0

        def f2(k):
            return k ** -3

        self.funcs = [f1, f2]
        self.rule = [(1,1,0)]
        self.amps = [1]
        self.name = 'mylocal'

    def tearDown(self):
        # Is called after each test.

        delattr(self, 'funcs')
        delattr(self, 'rule')
        delattr(self, 'amps')

    def test_shape_init(self):

        shape = Shape(self.funcs, self.rule, self.amps, self.name)

        self.assertIs(shape.funcs[0], self.funcs[0])
        self.assertIs(shape.funcs[1], self.funcs[1])

        self.assertIs(shape.rule, self.rule)
        self.assertIs(shape.amps, self.amps)
        self.assertEqual(shape.name, self.name)

    def test_shape_init_err(self):

        invalid_name = ''
        self.assertRaises(ValueError, Shape, self.funcs,
                          self.rule, self.amps, invalid_name)

        invalid_name = ' '
        self.assertRaises(ValueError, Shape, self.funcs,
                          self.rule, self.amps, invalid_name)

        invalid_name = 1
        self.assertRaises(ValueError, Shape, self.funcs,
                          self.rule, self.amps, invalid_name)

    def test_shape_get_f_k(self):

        k = np.asarray([1., 2., 3.])

        shape = Shape(self.funcs, self.rule, self.amps, self.name)

        f_k = shape.get_f_k(k)

        self.assertEqual(f_k.shape, (3, 2)) # (nk, ncomp)
        self.assertEqual(f_k.dtype, float) # (nk, ncomp)
        self.assertTrue(f_k.flags['C_CONTIGUOUS'])

        exp_f_k = np.empty((3, 2))
        exp_f_k[:,0] = [1., 1., 1.]
        exp_f_k[:,1] = [1. ** -3, 2. ** -3, 3. ** -3]

        np.testing.assert_almost_equal(f_k, exp_f_k)

    def test_power_law(self):

        exponent = -2.5
        f = Shape._power_law(exponent)
        k = np.arange(1, 10, dtype=float)

        np.testing.assert_almost_equal(f(k), k ** exponent)

    def test_shape_prim_local(self):

        local = Shape.prim_local()

        self.assertEqual(local.rule, self.rule)
        self.assertEqual(local.amps, self.amps)
        self.assertEqual(local.name, 'local')

        k = np.arange(1, 10, dtype=float)
        for f, f_expect in zip(local.funcs, self.funcs):
            np.testing.assert_almost_equal(f(k), f_expect(k))

    def test_shape_prim_local_ns(self):

        ns = 0.5
        local = Shape.prim_local(ns=ns)

        def f1(k):
            return k ** 0
        def f2(k):
            return k ** -(4-ns)

        funcs_expec = [f1, f2]

        k = np.arange(1, 10, dtype=float)
        for f, f_expect in zip(local.funcs, funcs_expec):
            np.testing.assert_almost_equal(f(k), f_expect(k))

    def test_shape_prim_local_name(self):

        name = 'myname'
        local = Shape.prim_local(name=name)
        self.assertEqual(local.name, name)

    def test_shape_prim_equilateral(self):

        ns = .9
        equilateral = Shape.prim_equilateral(ns=ns)

        def f1(k):
            return k ** 0
        def f2(k):
            return k ** -(4 - ns)
        def f3(k):
            return k ** -((4 - ns) / 3.)
        def f4(k):
            return k ** -(2 * (4 - ns) / 3)

        funcs_expec = [f1, f2, f3, f4]

        k = np.arange(1, 10, dtype=float)
        for f, f_expect in zip(equilateral.funcs, funcs_expec):
            np.testing.assert_almost_equal(f(k), f_expect(k))
        
        rule_expect = [(1,1,0), (3,3,3), (1,2,3)]
        self.assertEqual(equilateral.rule, rule_expect)
        amps_expect = [-3., -6., 3.]
        self.assertEqual(equilateral.amps, amps_expect)        
        self.assertEqual(equilateral.name, 'equilateral')

    def test_shape_prim_equilateral_name(self):

        name = 'myname'
        equilateral = Shape.prim_equilateral(name=name)
        self.assertEqual(equilateral.name, name)

    def test_shape_prim_orthogonal(self):

        ns = .9
        orthogonal = Shape.prim_orthogonal(ns=ns)

        def f1(k):
            return k ** 0
        def f2(k):
            return k ** -(4 - ns)
        def f3(k):
            return k ** -((4 - ns) / 3.)
        def f4(k):
            return k ** -(2 * (4 - ns) / 3.)

        funcs_expec = [f1, f2, f3, f4]

        k = np.arange(1, 10, dtype=float)
        for f, f_expect in zip(orthogonal.funcs, funcs_expec):
            np.testing.assert_almost_equal(f(k), f_expect(k))

        rule_expect = [(1,1,0), (3,3,3), (1,2,3)]
        self.assertEqual(orthogonal.rule, rule_expect)
        amps_expect = [-9., -24., 9.]
        self.assertEqual(orthogonal.amps, amps_expect)        
        self.assertEqual(orthogonal.name, 'orthogonal')

    def test_shape_prim_orthogonal_name(self):

        name = 'myname'
        orthogonal = Shape.prim_orthogonal(name=name)
        self.assertEqual(orthogonal.name, name)
        
