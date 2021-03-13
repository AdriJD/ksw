#include <ksw_common.h>

/*
 * Return min or max value for integer input.
 */

ptrdiff_t _max(ptrdiff_t a, ptrdiff_t b);
ptrdiff_t _min(ptrdiff_t a, ptrdiff_t b);

/*
 * Compute associated Legendre polynomials P^0_ell(cos(theta)).
 *
 * Arguments
 * ---------
 * thetas       : (ntheta) array thetas.
 * p_theta_ell  : (ntheta * nell) output array, will be overwritten.
 * ntheta       : number of thetas.
 * lmax         : Maximum multipole.
 */

void compute_associated_legendre_sp(const double *thetas, float *p_theta_ell,
				    int ntheta, int lmax);

void compute_associated_legendre_dp(const double *thetas, double *p_theta_ell,
				    int ntheta, int lmax);

/*
 * Compute zeta_XY (Smith & Zaldarriaga, Eq. 30) for all unique factors for single theta.
 *
 * Arguments
 * ---------
 * sqrt_icov_ell : (nell * npol * npol) array with sqrt of icov per ell (symmetric in pol).
 * f_ell_i       : (nell * npol * nufact) array of unique factors.
 * p_ell         : (nell) array with associated Legendre polynomials.
 * prefactor     : (nell) array
 * work_i        : (npol * nufact) array for internal work, will be overwritten!
 * unique_nxn    : (nufact * nufact) array for output.
 * nufact        : Number of unique factors.
 * nell          : Number of multipoles.
 * npol          : Number of polarizations.
 */

void unique_nxn_on_ring_sp(const float *sqrt_icov_ell, const float *f_ell_i, const float *p_ell, 
			   const float *prefactor, float *work_i, float *unique_nxn, int nufact,
			   int nell, int npol);

void unique_nxn_on_ring_dp(const double *sqrt_icov_ell, const double *f_ell_i, const double *p_ell, 
			   const double *prefactor, double *work_i, double *unique_nxn, int nufact,
			   int nell, int npol);

/*
 * Convert zeta to nrule x rnule fisher matrix (Eq. 29 in Smith & Zaldarriaga) for single theta.
 *
 * unique_nxn : (nufact * nufact) array with zeta.
 * rule       : (nrule * 3) Rule to combine unique factors to reduced bispectrum.
 * weights    : (nrule * 3) Amplitude for each element in rule.
 * fisher_nxn : (nrule * nrule) output fisher matrix.
 * nufact     : Number of unique factors.
 * nrule      : Number of rules.
 */

void fisher_nxn_on_ring_sp(const float *unique_nxn, const long long *rule,
			   const float *weights, float *fisher_nxn, double ct_weight, 
			   int nufact, int nrule);

void fisher_nxn_on_ring_dp(const double *unique_nxn, const long long *rule, 
			   const double *weights, double *fisher_nxn, double ct_weight, 
			   int nufact, int nrule);
