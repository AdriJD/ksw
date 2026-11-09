import sys
import os
import time
import numpy as np

from ksw import cosmo, utils
from ksw import estimator
from ksw.shape import Shape
import camb
import healpy as hp


def cinv_filter_TEB(alms, clTT, clEE, clBB, clTE,lmax):
    almT, almE, almB = alms

    almT_f = np.zeros_like(almT)
    almE_f = np.zeros_like(almE)
    almB_f = np.zeros_like(almB)

    for ell in range(lmax + 1):
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

        if clBB[ell] != 0 and np.isfinite(clBB[ell]):
            almB_f[idx] = almB[idx] / clBB[ell]

    return np.array([almT_f, almE_f, almB_f])

def get_radii_leo(oversample=1):
    low1 = np.linspace(1, 9377, num=int(24 * oversample), endpoint=False)
    rei1 = np.linspace(9377, 10007, num=int(4 * oversample), endpoint=False)
    rei2 = np.linspace(10007, 12632, num=int(8 * oversample), endpoint=False)
    rec1 = np.linspace(12632, 13682, num=int(12 * oversample), endpoint=False)
    rec2 = np.linspace(13682, 15500, num=int(75 * oversample), endpoint=False)
    rec3 = np.linspace(15500, 18000, num=int(5 * oversample), endpoint=False)

    radii = np.concatenate((low1, rei1, rei2, rec1, rec2, rec3))

    return radii

lmax = 200
n_mc = 3          # number of sims used for <grad T>
sim_data = 10     # sim used as "data"

# Setup CAMB parameters
pars = camb.CAMBparams()
pars.WantTensors=True
pars.set_cosmology(H0=67.32117, ombh2=0.0223828, omch2=0.1201075, tau=0.05430842, mnu=0.06)
pars.InitPower.set_params(As=2.100549e-9, ns=0.9660499, r=0)
cosmo_esti = cosmo.Cosmology(pars, verbose=False)

radii = get_radii_leo(oversample=1.5)
prim_shape = Shape.prim_local(ns=0.9660499)

cosmo_esti.compute_transfer(lmax=lmax)
cosmo_esti.add_prim_reduced_bispectrum_scalar_dL(prim_shape, radii)

cosmo_esti.compute_transfer_tensor(lmax=lmax)
cosmo_esti.add_prim_reduced_bispectrum_tensor_dL(prim_shape, radii)

red_bispectra = cosmo_esti.red_bispectra
icov = lambda alm: alm
pol = ('T', 'E', 'B')
estimator_con = estimator.KSW(red_bispectra, icov, lmax=lmax, pol=pol, precision='double')

Cl_th = np.load(
    f"/scratch/hb-CosmoGroup/bispectrum_benchmark/power_spectra/lensed_cmb_spectra_{lmax}.npz"
)
clTT = Cl_th["TT"] * 1e12
clEE = Cl_th["EE"] * 1e12
clBB = Cl_th["BB"] * 1e12
clTE = Cl_th["TE"] * 1e12

def load_filtered(sim):
    alm_file = f"/scratch/hb-CosmoGroup/bispectrum_benchmark/sims/lensed_cmb_sims/alms_{lmax}/lensed_cmb_alms_{sim:03d}.npy"
    alm = np.load(alm_file) * 1e6
    return cinv_filter_TEB(alm, clTT, clEE, clBB, clTE, lmax=lmax)

Lmax = lmax
L_list = np.arange(Lmax + 1)

# Monte Carlo: accumulate <grad T_sst>
for sim in range(n_mc):
    alms_f = load_filtered(sim)
    t0 = time.time()
    estimator_con.step_sst(alms_f, L_list, Lmax, theta_batch=25)
    print(f"step_sst sim {sim}: {time.time() - t0:.1f} s, mc_idx_sst = {estimator_con.mc_idx_sst}")

print("mc_gt_sst shape:", estimator_con.mc_gt_sst.shape)
print("mc_gt_sq_sst:", estimator_con.mc_gt_sq_sst)

# Linear term on the "data" sim
alms_data = load_filtered(sim_data)
lin_term = estimator_con.compute_linear_term_sst(alms_data)
print("lin_term:", lin_term)

# Cubic term with this linear term
#estimate, t_cubic, t_cubic_list, lin_term, fisher = estimator_con.compute_estimate_sst(
#    alms_data, L_list, Lmax, theta_batch=25, lin_term=lin_term)

