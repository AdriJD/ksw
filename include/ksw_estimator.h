
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
* Compute T[a] for a collection of rings for sst bispectra.
*
* Arguments
*---------
* ct_weights : (ntheta) array of quadrature weights for cos(theta).
* rule       : (nrule, 3) array
* weights    : (nrule, 3) array of weights for X_i, Y_i, Z_i
* a_ell_m    : (npol, nell, nell) complex array with ell-major alms.
* y_M_L      : (ntheta, nL, nL) array with m-major Ylms for each ring.
* ntheta     : Number of thetas (rings).
* nrule      : Number of rules.
* nL         : Number of multipole values L.
* npol       : Number of polarization.
* nufact     : Number of unique factors
* nphi       : Number of phi per ring.
* L_list     : (nL) array of L values.
* n_scalar1/scalar2/tensor : 
* 				Magnetic quantum number that couples to S in w3j symbol for scalar1/scalar2/tensor A functionals.
* w3j_product_scalar1/scalar2/tensor : 
* 			   (ndeltaL, nL, 2*Lmax+1) array
* 			    Product of Wigner 3j symbols for scalar1/scalar2/tensor A functional.
* prefactors_scalar1/scalar2/tensor : 
* 			   (npol, ndeltaL, nL) complex array containing gamma * phase
* Lmax	    : Maximum value of L.
* nell	    : Number of ell values.
* kappa_i_L_scalar1/scalar2/tensor :
* 			   (nufact, npol, ndeltaL_scalar/tensor, nL) array, kappa functionals
*/
float t_cubic_sp_sst(const float *ct_weights, const long long *rule, const float *weights,
		  const float complex *a_ell_m,
		  const float *y_M_L, int ntheta, int nrule,
		  int nL, int npol,  
		  int nufact, int nphi,
		  const int *L_list,
		  int n_scalar1, int n_scalar2, int n_tensor,
		  const float *w3j_product_scalar1, const float *w3j_product_scalar2, const float *w3j_product_tensor,
		  const float complex *prefactors_scalar1, const float complex *prefactors_scalar2, const float complex *prefactors_tensor, 
		  int Lmax, int nell,
		  const float complex *kappa_i_L_scalar1, const float complex *kappa_i_L_scalar2, const float complex *kappa_i_L_tensor);

double t_cubic_dp_sst(const double *ct_weights, const long long *rule, const double *weights,
		  const double complex *a_ell_m,
		  const double *y_M_L, int ntheta, int nrule,
		  int nL, int npol, 
		  int nufact, int nphi,
		  const int *L_list,
		  int n_scalar1, int n_scalar2, int n_tensor,
		  const double *w3j_product_scalar1, const double *w3j_product_scalar2, const double *w3j_product_tensor,
		  const double complex *prefactors_scalar1, const double complex *prefactors_scalar2, const double complex *prefactors_tensor, 
		  int Lmax, int nell,
		  const double complex *kappa_i_L_scalar1, const double complex *kappa_i_L_scalar2, const double complex *kappa_i_L_tensor);


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
*/
void step_sst_sp(const int *L_list,
			  int nL, int npol, 
			  int n_scalar1, int n_scalar2, int n_tensor, int n_scalar,   
			  float complex *a_L_M_scalar, float complex *a_L_M_tensor,
			  const float *y_M_L,
			  const float *w3j_product_scalar1, const float *w3j_product_scalar2, const float *w3j_product_tensor,
			  const float complex *prefactors_scalar, const float complex *prefactors_tensor, 
			  int Lmax, int nell,
			  fftwf_plan plan_c2c_scalar, fftwf_plan plan_c2c_tensor,
			  float complex *f_i_phi_scalar1, float complex *f_i_phi_scalar2, float complex *f_i_phi_tensor, 
			  int nufact, int nphi,
			  const float complex *kappa_i_L_scalar1, const float complex *kappa_i_L_scalar2, const float complex *kappa_i_L_tensor,
			  float complex *work_i_L_scalar1, float complex *work_i_L_scalar2, float complex *work_i_L_tensor,
			  float complex *work_i_phi_scalar1, float complex *work_i_phi_scalar2, float complex *work_i_phi_tensor,
			  float complex *n_L_phi_scalar1, float complex *n_L_phi_scalar2, float complex *n_L_phi_scalar, float complex *n_L_phi_tensor,
			  float complex *m_L_M_scalar, float complex *m_L_M_tensor,
			  const long long *rule, const float *weights, const float ct_weight,
			  const float w3j, int nrule, int nw);

		

/*
 * Compute m-major Ylm(theta,0) for a range of thetas.
 *
 * Arguments
 * ---------
 * thetas  : (ntheta) array of theta values.
 * y_mell : (ntheta, nell, nell) output array.
 * ntheta  : number of theta values.
 * lmax    : Maximum multipole (determining nell=nm=lmax+1).
 */

void compute_ylm_sp(const double *thetas, float *y_m_ell, int ntheta, int lmax);

void compute_ylm_dp(const double *thetas, double *y_m_ell, int ntheta, int lmax);

/*
 * Compute A_{LM}(deltaL) = prefactor * w3j * alm * Y_{LM}(theta, 0) 
 * The shape of the output is (npol, ndeltaL, nL, nphi) with nphi >= 3*Lmax+1
 *
 * Arguments
 * ---------
 * L_list      : (nL) array of L values.
 * deltaL_list : (ndeltaL) array with offsets deltaL such that ell = L + deltaL.
 * n           :  Magnetic quantum number that couples to S.
 * a_ell_m     : (npol, nell, nell) complex array storing alm with m>=0.
 * y_M_L       : (ntheta, nL, nL) complex array with Y_{M, L} samples for one ring.
 * w3j_product : (ndeltaL, nL, 2*Lmax+1) array with precomputed Wigner
 *                products, stored with ``ndeltaL`` leading and ``m`` last.
 * prefactors  : (npol, ndeltaL, nL) complex array containing gamma * phase
 *                factors for each polarization.
 * out         : (npol, ndeltaL, nL, nphi) complex output array for A_{LM}.    
 * Lmax        : Maximum value of L.
 * nell        : Number of multipoles (size of ell dimension).
 * nphi        : Number of phi values (>= 3*Lmax+1).
 */

void compute_A_LM_sp(const int *L_list, const int *deltaL_list,
	           int nL, int ndeltaL, int npol, int n,
		       const float complex *a_ell_m,
		       const float *y_M_L, const float *w3j_product,
		       const float complex *prefactors, float complex *out,
		       int Lmax, int nell, int nphi);

void compute_A_LM_dp(const int *L_list, const int *deltaL_list,
	           int nL, int ndeltaL, int npol, int n,
		       const double complex *a_ell_m,
		       const double *y_M_L, const double *w3j_product,
		       const double complex *prefactors, double complex *out,
		       int Lmax, int nell, int nphi);



