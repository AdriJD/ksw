
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

/*
 * Build reduced A_{LM} tensors for one or more polarization channels.
 *
 * Arguments
 * ---------
 * L_list      : (nL) array with base multipoles L.
 * deltaL_list : (ndeltaL) array with offsets such that ell = L + deltaL.
 * n           : Magnetic quantum number that couples to M.
 * a_ell_m     : (npol * nell * nell) complex array storing alm with m>=0.
 * y_m_ell     : (ntheta * nell * nell) real array with Y_{M,ell} samples for one ring.
 * w3j_product : (npol * ndeltaL * nL * m_dim) array with precomputed Wigner
 *                products, stored with ``npol`` leading and ``m`` last.
 * prefactors  : (npol * ndeltaL * nL) complex array containing gamma * phase
 *                factors for each polarization.
 * out         : (npol * ndeltaL * nL * m_dim) complex output array for A_{LM}
 *                using the same axis ordering as ``w3j_product``.
 * Lmax        : Maximum L considered (sets m_dim >= 2*Lmax+1).
 * nell        : Number of multipoles (size of ell dimension).
 * m_dim       : Number of available M samples (>= 2*Lmax+1).
 */

void compute_A_LM_sp(const long long *L_list, const long long *deltaL_list,
	           int nL, int ndeltaL, int npol, int n,
		       const float complex *a_ell_m,
		       const float *y_m_ell, const float *w3j_product,
		       const float complex *prefactors, float complex *out,
		       int Lmax, int nell, int m_dim);

void compute_A_LM_dp(const long long *L_list, const long long *deltaL_list,
	           int nL, int ndeltaL, int npol, int n,
		       const double complex *a_ell_m,
		       const double *y_m_ell, const double *w3j_product,
		       const double complex *prefactors, double complex *out,
		       int Lmax, int nell, int m_dim);

/*
 * Mixed backward pass that first builds reduced A_{LM} tensors via
 * compute_A_LM_sp using the provided L/deltaL geometry, then performs the
 * usual FFT/GEMM chain to project onto phi.
 */
void backward_sp_mixed(const float *f_i_ell, const float complex *a_ell_m,
			  const float *y_m_ell, const long long *L_list,
			  const long long *deltaL_list, int n,
			  const float *w3j_product,
			  const float complex *prefactors,
			  float complex *A_ell_m, float *n_ell_phi,
			  fftwf_plan plan_c2c, float *f_i_phi,
			  int ndeltaL, int nL, int npol, int nufact,
			  int nphi, int Lmax, int nell);

