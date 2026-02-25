import numpy as np
from scipy.interpolate import CubicSpline
import inspect
import json

import camb
import h5py

from ksw import utils
from ksw import radial_functional as rf

from ksw import Cosmology
from ksw import Shape
import matplotlib.pyplot as plt

cosmo_opts = dict(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06, TCMB=2.7255)
pars = camb.CAMBparams()
pars.set_cosmology(**cosmo_opts)

lmax = 2000
cosmo = Cosmology(pars)
cosmo._setattr_camb('lSampleBoost', 3, subclass='Accuracy', verbose=True)
cosmo._setattr_camb('AccuracyBoost', 5, subclass='Accuracy', verbose=True)

cosmo.compute_transfer_tensor(lmax, k_eta_fac=5.)
tr_ell_k = cosmo.transfer['tr_ell_k_tensor']
k = cosmo.transfer['k_tensor']
ells_sparse = cosmo.transfer['ells_tensor']

prim_shape = Shape.prim_local(ns=1)
f_k = prim_shape.get_f_k(k)
amps = np.asarray(prim_shape.amps)
amps *= 2 * (2 * np.pi **2 * cosmo.camb_params.InitPower.As) ** 2 * (3 / 5)

radii = np.linspace(13000, 14500, num=len(tr_ell_k))
red_bisp = rf.radial_func_dL(f_k, tr_ell_k, k, radii, ells_sparse)
index = np.where(ells_sparse==60)[0][0]

# tensor to T 
delta_h_T1 = red_bisp[0, :, index, 0, 0]
delta_h_T2 = red_bisp[2, :, index, 0, 0]
delta_h_T3 = red_bisp[4, :, index, 0, 0]

plt.figure(figsize=(7,5))
plt.plot(radii, radii**2*delta_h_T1, label=f'L-l = {-2}', color='b', alpha=0.5, linestyle='--')  
plt.plot(radii, radii**2*delta_h_T2, label=f'L-l = {0}', color='b', alpha=1, linestyle='-')  
plt.plot(radii, radii**2*delta_h_T3, label=f'L-l = {2}', color='b', alpha=0.8, linestyle='-.')  
plt.xlabel("comoving radial distance r [Mpc]")
plt.ylabel("Radial Transfer Function")
plt.title("Radial Transfer Function for h to T")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("radial_hT.png", dpi=300)
plt.close()

# tensor to E
cosmo.compute_transfer_tensor(lmax, k_eta_fac=5.)
tr_ell_k = cosmo.transfer['tr_ell_k_tensor']
k = cosmo.transfer['k_tensor']
ells_sparse = cosmo.transfer['ells_tensor']

prim_shape = Shape.prim_local(ns=1)
f_k = prim_shape.get_f_k(k)
amps = np.asarray(prim_shape.amps)
amps *= 2 * (2 * np.pi **2 * cosmo.camb_params.InitPower.As) ** 2 * (3 / 5)

radii = np.linspace(13000, 14500, num=len(tr_ell_k))
red_bisp = rf.radial_func_dL(f_k, tr_ell_k, k, radii, ells_sparse)

delta_h_E1 = red_bisp[0, :, index, 1, 0]
delta_h_E2 = red_bisp[2, :, index, 1, 0]
delta_h_E3 = red_bisp[4, :, index, 1, 0]

plt.figure(figsize=(7,5))
plt.plot(radii, radii**2*delta_h_E1, label=f'L-l = {-2}', color='b', alpha=0.5, linestyle='--')  
plt.plot(radii, radii**2*delta_h_E2, label=f'L-l = {0}', color='b', alpha=1, linestyle='-')  
plt.plot(radii, radii**2*delta_h_E3, label=f'L-l = {2}', color='b', alpha=0.8, linestyle='-.')  
plt.xlabel("comoving radial distance r [Mpc]")
plt.ylabel("Radial Transfer Function")
plt.title("Radial Transfer Function for h to E")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("radial_hE.png", dpi=300)
plt.close()

# tensor to B
delta_h_B1 = red_bisp[1, :, index, 2, 0]
delta_h_B2 = red_bisp[3, :, index, 2, 0]

plt.figure(figsize=(7,5))
plt.plot(radii, radii**2*delta_h_B1, label=f'L-l = {-1}', color='b', alpha=0.5, linestyle='--')  
plt.plot(radii, radii**2*delta_h_B2, label=f'L-l = {1}', color='b', alpha=0.8, linestyle='-.')  
plt.xlabel("comoving radial distance r [Mpc]")
plt.ylabel("Radial Transfer Function")
plt.title("Radial Transfer Function for h to B")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("radial_hB.png", dpi=300)
plt.close()



