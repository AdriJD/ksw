# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from ksw import cosmo, utils
from ksw import estimator
import camb
import healpy as hp

from ksw import Afunctionals

# %%
# Setup CAMB parameters
pars = camb.CAMBparams()
pars.WantTensors=True
pars.set_cosmology(H0=67.66, ombh2=0.02242, omch2=0.11933)
pars.InitPower.set_params(As=2.1056e-9, ns=0.9665, r=0.001)
cosmo = cosmo.Cosmology(pars, verbose=False)

# Compute transfer functions
print("Computing transfer functions...")
cosmo.compute_transfer(lmax=300)

# Compute angular power spectra
print("Computing angular power spectra...")
#cosmo.compute_c_ell()

# %%
from ksw.shape import Shape
# radii = np.logspace(0, 3, 10)  # Small number of radii for quick testing
radii = np.asarray([10000, 10100])

prim_shape = Shape.prim_local(ns=0.9665)
#radii = np.logspace(0, 3, 10)

cosmo.compute_transfer(lmax=300)
cosmo.add_prim_reduced_bispectrum_scalar_dL(prim_shape, radii)

cosmo.compute_transfer_tensor(lmax=300)
cosmo.add_prim_reduced_bispectrum_tensor_dL(prim_shape, radii)

# %%
# Create a estimator instance
red_bispectra = cosmo.red_bispectra
icov = lambda alm: alm
pol = ('T', 'E', 'B')
lmax = 5
estimator_con = estimator.KSW(red_bispectra, icov, lmax=lmax, pol=pol, precision='double')

# %%
import healpy as hp
lmax = estimator_con.lmax
nelem = hp.Alm.getsize(lmax+1)
print(nelem)

# %%
# Generate sample alm data for testing
lmax = estimator_con.lmax
print(f'{lmax=}')
npol = 3
#nelem = (lmax + 1) * (lmax + 2) // 2   # HEALPix alm element count
nelem = hp.Alm.getsize(lmax)

# Create random alm with correct shape
np.random.seed(20)
#alm_test = np.random.randn(npol, nelem) + 1j * np.random.randn(npol, nelem)
cls = np.zeros((4, lmax + 1))
cls[0] = np.ones(lmax + 1)
cls[1] = np.ones(lmax + 1)
cls[2] = np.ones(lmax + 1)
alm_test = hp.synalm((cls[0], cls[1], cls[2], cls[3]), lmax=lmax)
print(f'{alm_test[0,0]=}')
print(f'{alm_test.shape=}')
print(f'{alm_test.dtype=}')

# Try to call compute_estimate_sst

Lmax = lmax
L_list = np.arange(Lmax + 1)

print(alm_test.dtype)
print(L_list.dtype)

estimate, cubic, lin_term, fisher = estimator_con.compute_estimate_sst(
    alm_test,
    L_list,
    Lmax,
    theta_batch=25
)

# %%


# %%



