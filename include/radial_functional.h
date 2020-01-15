

/*
 * Compute f_ell^X(r) = int k^2 dk f(k) transfer^X_ell(k) j_ell(k r), 
 * where f(k) is an arbitrary function of wavenumber k.
 *
 * Arguments
 * ---------
 * f_k      : (nk * ncomp) input functions.
 * tr_ell_k : (nell * nk * npol) transfer function.
 * k        : (nk) array of monotonically increasing wavenumbers.
 * radii    : (nr) array of radii.
 * f_r_ell  : (nr * nell * npol * ncomp) output array.
 * ells     : (nell) multipoles.
 * nk       : Number of wavenumbers.
 * nell     : Number of multipoles.
 * nr       : Number of radii.
 * npol     : Number of polarization components (1=T, 2=E).
 * ncomp    : Number of input functions.
*/

void compute_radial_func(double const *f_k,
			 double const *tr_ell_k,
			 double *k,
			 double const *radii,
			 double *f_r_ell,
			 int *ells,
			 int nk,
			 int nell,
			 int nr,
			 int npol,
			 int ncomp);

/*
 * Compute f_ell^X(r) = int k^2 dk f(k) transfer^X_ell(k) j_ell(k r), 
 * for fixed ell and r.
 *
 * Arguments
 * ---------
 * f_k      : (nk * ncomp) input functions.
 * tr_k     : (nk * npol) transfer function.
 * bessel_k : (nk) spherical bessel function.
 * w_k      : (nk) array of quadrature weight per wavenumber.
 * out      : (npol * ncomp) output array.
 * nk       : (nk) wavenumbers.
 * npol     : Number of polarization components (1=T, 2=E).
 * ncomp    : Number of input functions.
*/

void _integral_over_k(double const *f_k,
		      double const *tr_k,
		      double const *bessel_k,
		      double const *w_k,
		      double *out,
		      int nk,
		      int npol,
		      int ncomp);

/*
 * Compute quadrature weights for trapezoidal integration over 
 * monotonically increasing x values.
 *
 * Arguments
 * ---------
 * k   : (nk) array of wavenumbers.
 * w_k : (nk) output array of weights.
 * nk  : Number of wavenumbers.
 */

void _trapezoidal_weights(double const *k,
			  double *w_k,
			  int nk);

/*
 * Allocate memory block. Send error message to stderr
 * and exit when allocation fails.
 *
 * Arguments
 * ---------
 * size : Block size in bytes.
 */

void * _malloc_checked(size_t size);
