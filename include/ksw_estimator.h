
/*
 * Compute T[a] for a collection of rings.
 *
 * Arguments
 * ---------
 * ct_weights : (ntheta) array of quadrature weights for cos(theta).
 * rule       : (nrule, 3) array
 * weights    : (nrule, 3) array of weights for X_i Y_i Z_i.
 * f_i_ell    : (nufact * npol * nell) array with unique factors.
 * a_ell_m    : (npol * nell * nell) complex array with ell-major alms.
 * y_m_ell    : (ntheta * nell * nell) array with m-major Ylms for each ring.
 * ntheta     : Number of thetas (rings).
 * nrule      : Number of rules.
 * nell       : Number of multipoles.
 * npol       : Number of polarization dimensions.
 * nufact     : Number of unique factors.
 * nphi       : Number of phi per ring.
 */

float t_cubic_sp(const double *ct_weights, const long long *rule, const float *weights,
		 const float *f_i_ell, const float complex *a_ell_m,
		 const float *y_m_ell, int ntheta, int nrule,
		 int nell, int npol, int nufact, int nphi);

double t_cubic_dp(const double *ct_weights, const long long *rule, const double *weights,
		  const double *f_i_ell, const double complex *a_ell_m,
		  const double *y_m_ell, int ntheta, int nrule,
		  int nell, int npol, int nufact, int nphi);

/*
 * Compute forward and backward operation on collection of rings.
 *
 * Arguments
 * ---------
 * ct_weights : (ntheta) array of quadrature weights for cos(theta).
 * rule       : (nrule, 3) array
 * weights    : (nrule, 3) array of weights for X_i Y_i Z_i.
 * f_i_ell    : (nufact * npol * nell) array with unique factors.
 * a_ell_m    : (npol * nell * nell) complex array with ell-major alms.
 * y_m_ell    : (ntheta * nell * nell) array with m-major Ylms for each ring.
 * grad_t     : (npol * nell * nell) complex array with ell-major alms.
 * ntheta     : Number of thetas (rings).
 * nrule      : Number of rules.
 * nell       : Number of multipoles.
 * npol       : Number of polarization dimensions.
 * nufact     : Number of unique factors.
 * nphi       : Number of phi per ring.
 */

void step_sp(const double *ct_weights, const long long *rule, const float *weights,
	     const float *f_i_ell, const float complex *a_ell_m, const float *y_m_ell,
	     float complex *grad_t, int ntheta, int nrule, int nell, int npol, 
	     int nufact, int nphi);

void step_dp(const double *ct_weights, const long long *rule, const double *weights,
	     const double *f_i_ell, const double complex *a_ell_m, const double *y_m_ell,
	     double complex *grad_t, int ntheta, int nrule, int nell, int npol, 
	     int nufact, int nphi);

/*
 * Compute m-major Ylm(theta,0) for a range of thetas.
 *
 * Arguments
 * ---------
 * thetas  : (ntheta) array of theta values.
 * y_m_ell : (ntheta, nell, nell) output array.
 * ntheta  : number of theta values.
 * lmax    : Maximum multipole (determining nell=nm=lmax+1).
 */

void compute_ylm_sp(const double *thetas, float *y_m_ell, int ntheta, int lmax);

void compute_ylm_dp(const double *thetas, double *y_m_ell, int ntheta, int lmax);
