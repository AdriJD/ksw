from __future__ import annotations

"""Test the helper utilities in ksw.Afunctionals."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import healpy as hp
import numpy as np
from ducc0.misc import wigner3j_int

from ksw import Afunctionals  


class TestAfunctionals(unittest.TestCase):

    lmax = 100
    dtype = np.float32
    cdtype = np.complex64
    pol = ['T', 'E', 'B']

    def test_wigner_J(self):
        S = 2
        L_list = np.array([1, 2])
        deltaL_list = np.array([0, 1])
        Jindex = np.array([S, 0, 0])

        result = Afunctionals.wigner_J(S, L_list, deltaL_list, Jindex)
        expected = np.zeros_like(result)

        for iL, L in enumerate(L_list):
            for idL, dL in enumerate(deltaL_list):
                ell = L + dL
                if ell < 0:
                    continue
                l1_min, vals = wigner3j_int(L, ell, Jindex[1], Jindex[2])
                idx = S - l1_min
                if 0 <= idx < len(vals):
                    expected[iL, idL] = vals[idx]

        np.testing.assert_almost_equal(result, expected)

    def test_w3j_reproduces_ducc0_entries(self):
        S = 2
        n_val = 0
        L_list = np.array([1])
        deltaL_list = np.array([0])

        result = Afunctionals.w3j(S, n_val, L_list, deltaL_list)
        expected = np.zeros_like(result)
        Lmax = np.max(L_list)

        for idL, dL in enumerate(deltaL_list):
            for iL, L in enumerate(L_list):
                ell = L + dL
                if ell < 0:
                    continue
                for M in range(-L, L + 1):
                    m = -M - n_val
                    if abs(m) > ell:
                        continue
                    l1_min, vals = wigner3j_int(L, ell, M, m)
                    idx = S - l1_min
                    if 0 <= idx < len(vals):
                        expected[iL, M + Lmax, idL] = vals[idx]

        np.testing.assert_almost_equal(result, expected)

    def test_products_3j_array(self):
        S = 2
        n_val = 1
        L_list = np.array([2])
        deltaL_list = np.array([0, 1])
        Jindex = np.array([S, 0, 0])

        result = Afunctionals.products_3j_array(S, n_val, L_list, deltaL_list, Jindex)
        wJ = Afunctionals.wigner_J(S, L_list, deltaL_list, Jindex)
        w_vals = Afunctionals.w3j(S, n_val, L_list, deltaL_list)

        expected = np.zeros_like(result)
        for iL in range(len(L_list)):
            for idL in range(len(deltaL_list)):
                expected[iL, :, idL] = wJ[iL, idL] * w_vals[iL, :, idL]

        np.testing.assert_almost_equal(result, expected)

    def test_parity_x(self):
        self.assertEqual(Afunctionals.parity_x("t"), 0)
        self.assertEqual(Afunctionals.parity_x("E"), 0)
        self.assertEqual(Afunctionals.parity_x("B"), 1)
        self.assertIsNone(Afunctionals.parity_x("x"))

    def test_gamma_Z(self):
        self.assertEqual(Afunctionals.gamma_Z("T", "zeta", 2, 1), 1)
        expected = 1 + (-1) ** (1 + 2 * 3 + 1)
        self.assertEqual(Afunctionals.gamma_Z("B", "h", 3, 1), expected)

    def test_get_a_lm(self):
        alm = np.zeros((Afunctionals.npol, hp.Alm.getsize(Afunctionals.lmax)), dtype=Afunctionals.cdtype)
        idx = hp.Alm.getidx(Afunctionals.lmax, 2, 1)
        alm[0, idx] = 3 + 4j

        positive = Afunctionals.get_a_lm(alm, ell=2, m=1, pol="T")
        negative = Afunctionals.get_a_lm(alm, ell=2, m=-1, pol="T")

        np.testing.assert_almost_equal(positive, 3 + 4j)
        np.testing.assert_almost_equal(negative, -np.conjugate(3 + 4j))

    def test_A_LM(self):
        pol = ["T"]
        lmax = 1
        dtype = np.float32
        cdtype = np.complex64

        L_list = np.array([1])
        deltaL_list = np.array([0])
        Jindex = np.array([2, 0, 0])

        thetas = np.linspace(0.1, np.pi - 0.1, 40, dtype=dtype)
        theta_weights = np.ones_like(thetas)
        nphi = 8

        def fake_compute_ylm(thetas_batch, lmax, dtype):
            ntheta = len(thetas_batch)
            return np.full((ntheta, 2 * lmax + 1, len(L_list)), 2.0, dtype=dtype)

        def fake_products(S, n_val, L_vals, delta_vals, Jidx):
            Lmax = int(np.max(L_vals))
            return np.full((len(L_vals), 2 * Lmax + 1, len(delta_vals)), 0.5, dtype=dtype)

        def fake_get_a_lm(_alm, ell, m, pol=None):
            return (ell + 0.1 * m) + 1j * (ell - m)

        alm_input = np.zeros(1, dtype=cdtype)

        patch_kwargs = {
            "pol": pol,
            "npol": len(pol),
            "lmax": lmax,
            "dtype": dtype,
            "cdtype": cdtype,
            "thetas": thetas,
            "theta_weights": theta_weights,
            "nphi": nphi,
        }

        # mock.patch is used to replace the actual implementations and global settings
        # so that we can test the function in an isolated way
        with mock.patch.multiple(Afunctionals, **patch_kwargs):
            with mock.patch.object(Afunctionals.estimator_core, "compute_ylm", side_effect=fake_compute_ylm):
                with mock.patch.object(Afunctionals, "products_3j_array", side_effect=fake_products):
                    with mock.patch.object(Afunctionals, "get_a_lm", side_effect=fake_get_a_lm):
                        result = Afunctionals.A_LM(
                            deltaL_list,
                            L_list,
                            n=0,
                            x="T",
                            Z="h",
                            S=2,
                            Jindex=Jindex,
                            alm=alm_input,
                            theta_batch=1,
                        )

        expected_values = {
            -1: 2 * (1 + 0.1 * 1 + 1j * (1 - 1)),
            0: 2 * (1 + 0.1 * 0 + 1j * (1 - 0)),
            1: 2 * (1 + 0.1 * -1 + 1j * (1 - (-1))),
        }

        for M, value in expected_values.items():
            np.testing.assert_almost_equal(result[0, M + lmax, 0], value)


if __name__ == "__main__":
    unittest.main()


