import numpy as np
from ducc0.misc import wigner3j_int


#deltaL_list = {-1, +1} for scalar
#deltaL_list = {-2, -1, 0, +1, +2} for tensor    
def wigner_J(S, L_list, deltaL_list, Jindex):
    L_list = np.asarray(L_list, dtype=int)
    deltaL_list = np.asarray(deltaL_list, dtype=int)

    out = np.zeros((len(L_list), len(deltaL_list)), dtype=np.float64)

    cJ = np.zeros((len(L_list), len(deltaL_list)), dtype=np.float64)
    for iL, L in enumerate(L_list):
        for idL, dL in enumerate(deltaL_list):
            ell = L + dL
            if ell < 0:
                continue

            # --- selection rules ---
            if abs(Jindex[1]) > L:
                continue
            if abs(Jindex[2]) > ell:
                continue

            l1_min_J, vals_J = wigner3j_int(L, ell, Jindex[1], Jindex[2])
            iS = S - l1_min_J
            if 0 <= iS < len(vals_J):
                out[iL, idL] = vals_J[iS]
    return out

def w3j(S, n, L_list, deltaL_list):

    L_list = np.asarray(L_list, dtype=int)
    deltaL_list = np.asarray(deltaL_list, dtype=int)

    Lmax = np.max(L_list)
    out = np.zeros((len(L_list), 2*Lmax+1, len(deltaL_list)), dtype=np.float64)

    for idL, dL in enumerate(deltaL_list):
        for iL, L in enumerate(L_list):
            ell = L + dL
            if ell < 0:
                continue

            for M in range(-L, L+1):
                m = - M - n
                if abs(m) > ell:
                    continue

                # --- selection rules ---
                if abs(M) > L:
                    continue
                if abs(m) > ell:
                    continue

                l1_min, vals = wigner3j_int(L, ell, M, m)
                iS = S - l1_min
                if 0 <= iS < len(vals):
                    out[iL, M + Lmax, idL] = vals[iS]
    return out

def products_3j_array(S, n , L_list, deltaL_list, Jindex):
    """
    Compute the product of two 3j symbols
    J_(S, L, L+deltaL)^(Jindex) * (S L L+deltaL; n M m)
    Parameters
    ----------
    S : int
        Total angular momentum quantum number
    n: int
        Magnetic quantum number
    L_list : int
        Orbital angular momentum quantum number
    deltaL_list : int
        choices for ell
    Jindex: an array of shape(3, )
        The indices (n, M, m) for the first 3j symbol

    Returns:
    -------
    np.ndarray
        Array of products of 3j symbols with shape(N_L, N_M, N_deltaL_list)
    """
    L_list = np.asarray(L_list, dtype=int)
    Lmax = int(np.max(L_list))

    w3j_J = wigner_J(S, L_list, deltaL_list, Jindex)
    w3j_vals = w3j(S, n, L_list, deltaL_list)

    out = np.zeros((len(L_list), 2*Lmax+1, len(deltaL_list)), dtype=np.float64)

    for iL in range(len(L_list)):
        for idL in range(len(deltaL_list)):
            out[iL, :, idL] = w3j_J[iL, idL] * w3j_vals[iL, :, idL]
    return out


def parity_x(x):
    """
    Map x in {T, E, B} to parity code:
    0 for T/E (parity even)
    1 for B (parity odd)
    """
    x = x.upper()
    if x == "T" or x == "E":
        return 0
    if x == "B":
        return 1
    
def gamma_Z(x, Z, L ,ell):
    """
    gamma^{(Z)}_{x, L, ell}:
      = 1,                        if Z = zeta
      = 1 + (-1)^{p_x + L + ell}, if Z = h

    where p_x = 0 for T/E and 1 for B.
    """
    Zlow = Z.lower()
    if Zlow == "zeta":
        return 1
    if Zlow == "h":
        px = parity_x(x)
        sgn = 1 + (-1) ** (px + L + ell)
        return sgn

def get_a_lm(a_ell_m, ell, m):
    """
    access a_{ell m} assuming a_ell_m stors m>=0 in the last axis.

    If m < 0, use reality condition:
    a_{ell, -m} = (-1)^m * conj(a_{ell, m})
    """
    alm = utils.alm_return_2d(a_ell_m)
    a_ell_m = utils.alm2a_ell_m(alm)
    a_ell_m = a_ell_m.astype(self.cdtype)

    if m >= 0:
        return a_ell_m[ell, m]
    mp = -m
    else:
        return ((-1)**mp) * np.conjugate(a_ell_m[ell, mp])

# thetas.max() = np.pi, not sure about len(self.thetas)

def A_LM(deltaL, n, x, Z, L, S, M, Jindex, a_ell_m_x, theta):
    ell = L + deltaL
    if ell < 0:
        continue

    # prefactors independent of M
    phase = (1j) ** deltaL
    gamma = gamma_Z(x, Z, L , ell)
    w3j_product = products_3j_array(S, n, deltaL, L, Jindex)
    prefactor = phase * w3j_product * gamma 

    alm = get_a_lm(a_ell_m_x, ell=L+deltaL, m=-M-n)
    for tidx_start in range(0, len(thetas), theta_batch):
        theta_batch = thetas[tidx_start:tidx_start+theta_batch]
        ct_weights_batch = theta_weights[tidx_start:tidx_start+theta_batch]
        y_m_ell = estimator_core.compute_ylm(thetas_batch, lmax)
        A_LM = prefactor * alm * 