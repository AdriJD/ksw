import numpy as np

class Shape:
    '''
    A shape represents a f(k1, k2, k3) primordial shape
    function.

    Parameters
    ---------
    funcs : sequence of callable func
        Functions needed for primordial shape.
    rule : sequence of array-like
        Rule to combine functions into primordial shape.
    amps : array-like
        Amplitude for each term in primordial shape.

    Notes
    -----

    <zeta_k1, zeta_k2, zeta_k3> = (2pi)^3 delta(k1+k2+k3) * B(k1, k2, k3),

    with:

    B(k1, k2, k3)  = 2 * fNL * A_s^2 * f(k1, k2, k3)

    Examples
    --------
    >>> def f1(k):
    ...    return k ** 0
    >>> def f2(k):
    ...    return k ** -3

    >>> local = Shape([f1, f2], [(1,1,0)], [1, 1, 1])

    >>> def f3(k):
    ...    return k ** -2
    >>> def f4(k):
    ...    return k ** -1

    >>> orthogonal = Shape([f1, f2, f3, f4],
                           [(1,1,0), (2,2,2), (1,2,3)],
                           [-9, -24, 9])

    '''

    def __init__(self, funcs, rule, amps):

        self.funcs = funcs
        self.rule = rule
        self.amps = amps

    def get_f_k(self, k):
        '''
        Calculate f(k) for an array of wavenumbers k.

        Parameters
        ----------
        k : (nk) array
            Wavenumbers in 1/Mpc.

        Returns
        -------
        f_k : (nk, ncomp) array
            Factors of shape function.
        '''

        ncomp = len(self.funcs)
        nk = k.size
        f_k = np.empty((ncomp, nk), dtype=float)

        for cidx in range(ncomp):
            f_k[cidx,:] = self.funcs[cidx](k)

        f_k = f_k.transpose()
        f_k = np.ascontiguousarray(f_k)

        return f_k

    @staticmethod
    def prim_local(ns=1):
        '''
        Return input to Shape class for the Local model.

        Parameters
        ----------
        ns : float, optional
            Scalar spectral index.

        Returns
        -------
        funcs
        rule
        amps
        '''

        f1 = Shape._power_law(0)
        f2 = Shape._power_law(-4 + ns)
        
        funcs = [f1, f2]
        rule = [(1,1,0)]
        amps = [1, 1, 1]

        return funcs, rule, amps

    @staticmethod
    def prim_equilateral():

        # Call set_prim_model()
        pass

    @staticmethod
    def prim_orthogonal():

        # Call set_prim_model()
        pass

    @staticmethod
    def _power_law(exponent):
        '''
        Create a power law function f(k) = k^e.

        Parameters
        ----------
        exponent : float
            Exponent of the power law.

        Returns
        -------
        f : callable func
            Power law function f(k).
        '''

        def f(k):
            return k ** exponent
        return f
