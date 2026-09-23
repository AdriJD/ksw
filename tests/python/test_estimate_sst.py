import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from ksw import cosmo, utils
from ksw import estimator
import camb
import healpy as hp

from pixell import curvedsky
from optweight import alm_utils

from ksw import Afunctionals

def cinv_filter_TEB(alms, clTT, clEE, clBB, clTE,lmax):
    """
    Apply isotropic inverse-covariance filtering to TEB alms.
    alms order : T, E, B
    
    Assumes covariance per cell:
        C_l = [[TT, TE, 0],
               [TE, EE, 0],
               [0, 0, BB]]
    """
    
    almT, almE, almB = alms

    almT_f = np.zeros_like(almT)
    almE_f = np.zeros_like(almE)                
    almB_f = np.zeros_like(almB)

    for ell in range(lmax + 1):
        # T/E covaraince block
        C_TE = np.array([
            [clTT[ell], clTE[ell]],
            [clTE[ell], clEE[ell]],
        ], dtype=float)

        Cinv_TE = np.linalg.pinv(C_TE)

        idx = hp.Alm.getidx(lmax, ell, np.arange(ell+1))

        aT = almT[idx]
        aE = almE[idx]

        almT_f[idx] = Cinv_TE[0, 0] * aT + Cinv_TE[0, 1] * aE
        almE_f[idx] = Cinv_TE[1, 0] * aT + Cinv_TE[1, 1] * aE

        # B 
        idxB = hp.Alm.getidx(lmax, ell, np.arange(ell+1))
        if clBB[ell] != 0 and np.isfinite(clBB[ell]):
            almB_f[idxB] = almB[idxB] / clBB[ell]
    
    return np.array([almT_f, almE_f, almB_f])

lmax = 600

# Setup CAMB parameters
pars = camb.CAMBparams()
pars.WantTensors=True
pars.set_cosmology(H0=67.32117, ombh2=0.0223828, omch2=0.1201075, tau=0.05430842, mnu=0.06)
pars.InitPower.set_params(As=2.100549e-9, ns=0.9660499, r=0)
cosmo_esti = cosmo.Cosmology(pars, verbose=False)
cosmo_esti.compute_transfer(lmax=lmax)


from ksw.shape import Shape
# radii = np.logspace(0, 3, 10)  # Small number of radii for quick testing
radii = np.asarray([13000, 14500])

prim_shape = Shape.prim_local(ns=0.9660499)
#radii = np.logspace(0, 3, 10)

cosmo_esti.compute_transfer(lmax=lmax)
cosmo_esti.add_prim_reduced_bispectrum_scalar_dL(prim_shape, radii)

cosmo_esti.compute_transfer_tensor(lmax=lmax)
cosmo_esti.add_prim_reduced_bispectrum_tensor_dL(prim_shape, radii)

# Create a estimator instance
red_bispectra = cosmo_esti.red_bispectra
icov = lambda alm: alm
pol = ('T', 'E', 'B')
estimator_con = estimator.KSW(red_bispectra, icov, lmax=lmax, pol=pol, precision='double')

'''
estimates = []

for sim in range(100):
    alm_file = f"/scratch/hb-CosmoGroup/bispectrum_benchmark/sims/lensed_cmb_sims/alms/lensed_cmb_alms_{sim:03d}.npy"
    alm_test = np.load(alm_file)

    Lmax = lmax
    L_list = np.arange(Lmax + 1)

    estimate, cubic, lin_term, fisher = estimator_con.compute_estimate_sst(
        alm_test,
        L_list,
        Lmax,
        theta_batch=25
    )

    estimates.append(estimate)

estimates = np.array(estimates)
np.save("estimates_000_99.npy", estimates)
'''

sim = int(os.environ["SLURM_ARRAY_TASK_ID"])

out_dir = f"estimates_sst_lmax{lmax}"
os.makedirs(out_dir, exist_ok=True)

Cl_th = np.load(
    f"/scratch/hb-CosmoGroup/bispectrum_benchmark/power_spectra/lensed_cmb_spectra_{lmax}.npz"
)

clTT = Cl_th["TT"] * 1e12
clEE = Cl_th["EE"] * 1e12
clBB = Cl_th["BB"] * 1e12
clTE = Cl_th["TE"] * 1e12

alm_file = f"/scratch/hb-CosmoGroup/bispectrum_benchmark/sims/lensed_cmb_sims/alms_{lmax}/lensed_cmb_alms_{sim:03d}.npy"
alm_test = np.load(alm_file) * 1e6

alm_T = alm_test[0]

alms_f = cinv_filter_TEB(
    alm_test,
    clTT,
    clEE,
    clBB,
    clTE,
    lmax=lmax
)

Lmax = lmax
L_list = np.arange(Lmax + 1)

estimate, cubic, lin_term, fisher = estimator_con.compute_estimate_sst(
    alms_f,
    L_list,
    Lmax,
    theta_batch=25
)

np.save(f"{out_dir}/estimate_{sim:03d}.npy", estimate)
