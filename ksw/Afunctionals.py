import numpy as np
from ducc0.misc import wigner3j_int
from ksw import utils, estimator_core


lmax = 100
dtype = np.float32
cdtype = np.complex64
pol = ["T", "E", "B"]
npol = len(pol)

thetas = np.array([np.pi / 4], dtype=dtype)
theta_weights = np.array([1.0], dtype=dtype)
nphi = 1


#deltaL_list = {-1, +1} for scalar
#deltaL_list = {-2, -1, 0, +1, +2} for tensor 
 
   
def wigner_J(S, L_list, deltaL_list, Jindex):
    """Return the Wigner-3j coefficients that depend only on ``Jindex``.

    Parameters
    ----------
    S : int
        Total angular momentum of the first leg.
    L_list : array_like of int
        Base multipoles for the second leg.
    deltaL_list : array_like of int
        Offsets that shift each L to ell = L + deltaL.
    Jindex : array_like of shape (3,)
        (n, M, m) arguments for the first Wigner-3j symbol.

    Returns
    -------
    np.ndarray
        Array with shape (len(L_list), len(deltaL_list)) containing
        ``wigner3j(S, L, L+deltaL; n, Jindex[1], Jindex[2])``.
    """
    L_list = np.asarray(L_list, dtype=int)
    deltaL_list = np.asarray(deltaL_list, dtype=int)

    out = np.zeros((len(L_list), len(deltaL_list)), dtype=dtype)

    cJ = np.zeros((len(L_list), len(deltaL_list)), dtype=dtype)
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
    """Evaluate the Wigner-3j values that depend on ``M`` and ``m``.

    Parameters
    ----------
    S : int
        Total angular momentum of the first leg.
    n : int
        Magnetic number for the first leg.
    L_list : array_like of int
        Base multipoles for the second leg.
    deltaL_list : array_like of int
        Offsets that shift each L to ell = L + deltaL for the third leg.

    Returns
    -------
    np.ndarray
        Shape (N_L, 2*Lmax+1, N_deltaL) array that stores
        ``wigner3j(S, L, L+deltaL; n, M, m)`` for all M in [-L, L].
    """

    L_list = np.asarray(L_list, dtype=int)
    deltaL_list = np.asarray(deltaL_list, dtype=int)

    Lmax = np.max(L_list)
    out = np.zeros((len(L_list), 2*Lmax+1, len(deltaL_list)), dtype=dtype)

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
        shift between L and ell
    Jindex: an array of shape(3, )
        The indices (n, M, m) for the first 3j symbol

    Returns:
    -------
    np.ndarray
        Array of products of 3j symbols with shape(len(L_list), 2*Lmax+1, len(deltaL_list))
    """
    L_list = np.asarray(L_list, dtype=int)
    Lmax = int(np.max(L_list))

    w3j_J = wigner_J(S, L_list, deltaL_list, Jindex)
    w3j_vals = w3j(S, n, L_list, deltaL_list)

    out = np.zeros((len(L_list), 2*Lmax+1, len(deltaL_list)), dtype=dtype)

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
    
def gamma_Z(x, Z, L, deltaL):
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
        sgn = 1 + (-1) ** (px + 2 * L + deltaL)
        return sgn

def get_a_lm(alm, ell, m, pol=None):
    """
    access a_{lm} assuming a_ell_m stores m>=0 in the last axis.

    If m < 0, use reality condition:
    a_{ell, -m} = (-1)^m * conj(a_{ell, m})

    Parameters
    ----------
    alm : array_like
        Harmonic coefficients with polarization axis matching ``pol``.
    ell : int
        Multipole index.
    m : int
        Azimuthal index.
    pol : str, optional
        Which polarization channel to read ("T", "E", or "B"). If omitted, the
        function assumes ``npol == 1`` and uses that single channel.
    """
    alm = utils.alm_return_2d(alm, npol, lmax)
    channels = globals()['pol']

    if pol is None:
        if npol != 1:
            raise ValueError("When multiple polarizations are active, pass pol='T', 'E', or 'B'.")
        pol_idx = 0
    else:
        pol = pol.upper()
        try:
            pol_idx = channels.index(pol)
        except ValueError as exc:
            raise ValueError(f"Polarization {pol} is not enabled in pol={channels}") from exc

    a_ell_m = utils.alm2a_ell_m(alm)
    a_ell_m = a_ell_m.astype(cdtype)
    channel = a_ell_m[pol_idx]

    if m >= 0:
        return channel[ell, m]
    else:
        mp = -m
        return ((-1)**mp) * np.conjugate(channel[ell, mp])

# thetas.max() = np.pi

def A_LM(deltaL_list, L_list, n, x, Z, S, Jindex, alm, theta_batch=1):
    """Assemble the reduced ``A_{LM}`` tensors for the requested multipoles.

    Parameters
    ----------
    deltaL_list : array_like of int
        Offsets that shift each L to ell = L + deltaL.
    L_list : array_like of int
        Base multipoles for the second leg.
    n : int
        Magnetic number that couples to ``M``.
    x : {"T", "E", "B"}
        Polarization label.
    Z : {"zeta", "h"}
        Primordial type.
    S : int
        Total angular momentum.
    Jindex : array_like of shape (3,)
        Arguments (n, M, m) for the J-type Wigner-3j symbol.
    alm : np.ndarray
        Harmonic coefficients.
    theta_batch : int, optional
        Number of polar angles.

    Returns
    -------
     -------
    np.ndarray
        Array with shape(len(L_list), 2*Lmax+1, len(deltaL_list))
    """

    Lmax = np.max(L_list)
    alm_list = np.zeros((len(L_list), 2*Lmax+1, len(deltaL_list)), dtype=cdtype)
    out = np.zeros((len(L_list), 2*Lmax+1, len(deltaL_list)), dtype=cdtype)

    w3j_product = products_3j_array(S, n, L_list, deltaL_list, Jindex)

    thetas_batch = thetas
    y_m_ell = estimator_core.compute_ylm(thetas_batch, lmax, dtype=dtype)

    for iL, L in enumerate(L_list):
        for idL, dL in enumerate(deltaL_list):
            ell = L + dL
            gamma = gamma_Z(x, Z, L, dL)
            phase = (1j) ** dL
            if ell < 0:
                continue
            for M in range(-L, L+1):
                m = - M - n
                if abs(m) > ell:
                    continue
                if abs(M) > L:
                    continue
                alm = get_a_lm(alm, ell=L+dL, m=-M-n, pol=x)
                alm_list[iL, M+Lmax, idL] = alm
                # prefactors independent of M
                prefactor = phase * w3j_product[iL, Lmax+M, idL] * gamma
                y_val = y_m_ell[:, M+Lmax, iL]
                out[iL, M+Lmax, idL] = prefactor * alm * y_val.reshape(-1)[0]
    return out






    
    
    