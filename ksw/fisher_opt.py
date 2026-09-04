'''
The optimization scheme from Sec 5.1. from Smith & Zaldarriaga 2011.
'''

import numpy as np

def optimize_bispectrum(fisher_mat, threshold=1e-6, max_terms=None, verbose=False):
    '''
    Given the Nfact x Nfact Fisher matrix F_ij between the individual
    factorizable terms of an "oversampled" bispectrum representation
    (Eq. 29),

        b_{l1 l2 l3} = (1/6) sum_i X_l1^(i) Y_l2^(i) Z_l3^(i) + (symm.),

    this selects a subset of Nopt <= Nfact terms and weights w_i such that

        b'_{l1 l2 l3} = (1/6) sum_{i in S} w_i X_l1^(i) Y_l2^(i) Z_l3^(i) + (symm.)

    is indistinguishable from b (in the Fisher-distance sense, Eq. 33) to
    within `threshold`.

    The algorithm never explicitly forms F00^-1 (which would cost
    O(Nopt^3) or need re-inversion each step); instead it maintains the
    matrix G (Eq. 39) and updates it with the O(Nfact^2) sweep-type rule
    (Eq. 41), giving overall cost O(Nopt * Nfact^2).

    Parameters
    ----------
    fisher_mat : (nfact, nfact) array
        Fisher matrix between individual factorizable terms (Eq. 29).
        Must be symmetric and positive semi-definite.
    threshold : float, optional
        Target Fisher distance F(B, B'_opt) (Eq. 37) at which to stop.
        Default 1e-6, i.e. the input and output bispectra cannot be
        distinguished to better than 0.001 sigma (the criterion used in
        the paper).
    max_terms : int, optional
        Hard cap on Nopt in case the threshold can't be reached (e.g. due
        to numerical roundoff). Defaults to nfact.
    verbose : bool, optional
        Print the running Fisher distance after each term is added.

    Returns
    -------
    indices : (Nopt,) int array
        Indices into the original nfact terms, in the order selected.
    weights : (Nopt,) float array
        Optimal weights w_i (Eq. 43), in the same order as `indices`.
    score : float
        Final Fisher distance F(B, B'_opt) (Eq. 37).
    '''

    fisher_mat = np.asarray(fisher_mat, dtype=np.float64)
    nfact = fisher_mat.shape[0]
    if fisher_mat.shape != (nfact, nfact):
        raise ValueError('fisher_mat must be a square matrix')

    if max_terms is None:
        max_terms = nfact

    # Step 0: initialize G = F, score = sum of all entries of F.
    g_mat = fisher_mat.copy()
    score = float(np.sum(fisher_mat))

    selected = []
    is_selected = np.zeros(nfact, dtype=bool)

    if verbose:
        print(f'nfact = {nfact}, initial Fisher distance = {score:.6e}')

    while score > threshold and len(selected) < max_terms:

        remaining = np.where(~is_selected)[0]
        if remaining.size == 0:
            break

        # Restrict to the "not yet retained" block of G, which currently
        # equals (F11 - F01^T F00^-1 F01), the residual Fisher matrix
        # between the remaining candidate terms (bottom-right block of
        # Eq. 39/40).
        g_matsub = g_mat[np.ix_(remaining, remaining)]
        row_sums = np.sum(g_matsub, axis=1)
        diag = np.diag(g_matsub)

        # Eq. (42): score improvement from retaining candidate I.
        with np.errstate(divide='ignore', invalid='ignore'):
            delta = np.where(diag > 0, row_sums**2 / diag, -np.inf)

        best_local = np.argmax(delta)
        delta_best = delta[best_local]

        if not np.isfinite(delta_best) or delta_best <= 0:
            break

        p = remaining[best_local]

        # Update running Fisher distance, Eq. (38)/(42).
        score -= delta_best
        selected.append(p)
        is_selected[p] = True

        if verbose:
            print(f'  + term {p:5d}  (Nopt={len(selected):4d})  '
                  f'Fisher distance = {score:.6e}')

        # Sweep-update the full G matrix at pivot p, Eq. (41). This is
        # applied to the entire matrix (not just the remaining block)
        # so that the retained x retained and retained x remaining blocks
        # (needed for the final weights, Eq. 43) stay correct too.
        gpp = g_mat[p, p]
        gp = g_mat[:, p].copy()

        g_mat -= np.outer(gp, gp) / gpp
        g_mat[:, p] = -gp / gpp
        g_mat[p, :] = -gp / gpp
        g_mat[p, p] = -1.0 / gpp

    indices = np.array(selected, dtype=np.int64)
    not_selected = np.where(~is_selected)[0]

    # Optimal weights, Eq. (43): w_i = 1 - sum_{J not retained} G_{iJ}.
    if indices.size > 0:
        if not_selected.size > 0:
            weights = 1.0 - np.sum(g_mat[np.ix_(indices, not_selected)], axis=1)
        else:
            weights = np.ones(indices.size)
    else:
        weights = np.zeros(0)

    return indices, weights, score
