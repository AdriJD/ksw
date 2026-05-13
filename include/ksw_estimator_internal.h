#include <ksw_common.h>
#include <fftw3.h>

/*
 * Compute sum_{i,phi} X_i_phi Y_i_phi Z_i_phi on single ring.
 *
 * Arguments
 * ---------
 * rule    : (nrule, 3) array of indices into f_i_phi that give X_i Y_i Z_i.
 * weights : (nrule, 3) array of weights for X_i Y_i Z_i.
 * f_i_phi : (nufact, nphi) array of unique factors on ring.
 * nrule   : Number of rules. 
 * nphi    : Number phi elements on ring.
 *
 */

float t_cubic_on_ring_sp(const long long *rule, const float *weights, const float *f_i_phi,
			 int nrule, int nphi);

double t_cubic_on_ring_dp(const long long *rule, const double *weights, const double *f_i_phi,
			  int nrule, int nphi);

/*
* Compute 3 * sum_{i, phi} A_i_phi(scalar) A_i_phi(scalar) A_i_phi(tensor) on single ring.
* 
* Arguments
* ---------
* rule    : (nrule, 3) array of indices into f_i_phi that give X_i Y_i Z_i.
* weights : (nrule, 3) array of weights for X_i Y_i Z_i.
* f_i_phi : (nufact, nphi) array of unique factors on ring.
* nrule   : Number of rules. 
* nphi    : Number phi elements on ring.
*
*/

float t_cubic_on_ring_sp_sst(const long long *rule, const float *weights, 
			const float complex *f_i_phi_scalar1, const float complex *f_i_phi_scalar2,
			const float complex *f_i_phi_tensor, int nrule, int nphi);

double t_cubic_on_ring_dp_sst(const long long *rule, const double *weights,
			const double complex *f_i_phi_scalar1, const double complex *f_i_phi_scalar2,
			const double complex *f_i_phi_tensor, int nrule, int nphi);


/* Access alm entries*/
float complex get_alm_entry_sp(const float complex *a_ell_m,
			      int nell, int ell, int m);

double complex get_alm_entry_dp(const double complex *a_ell_m,
			        int nell, int ell, int m);


/*
 * Convolve a single ring of the map with all unique bispectrum factors.
 *
 * Arguments
 * ---------
 * f_i_ell   : (nufact * npol * nell) array with unique factors.
 * a_ell_m   : (npol * nell * nell) complex array with ell-major alms.
 * y_m_ell   : (nell * nell) array with m-major Ylms.
 * m_ell_m   : (npol * nell * nm) complex array as input for ring fft, nm = nphi / 2 + 1.
 * n_ell_phi : (npol * nell * nphi) array as output for ring fft.
 * plan_c2r  : fftw plan for ring complex2real fft.
 * f_i_phi   : (nufact * nphi) array for output unique factors on ring.
 * nell      : Number of multipoles.
 * npol      : Number of polarization dimensions.
 * nufact    : Number of unique factors.
 * nphi      : Number of phi per ring.
 */

void backward_sp(const float *f_i_ell, const float complex *a_ell_m, 
		 const float *y_m_ell, float complex *m_ell_m, float *n_ell_phi,
		 fftwf_plan plan_c2r, float *f_i_phi, int nell, int npol, int nufact,
		 int nphi);

void backward_dp(const double *f_i_ell, const double complex *a_ell_m, 
		 const double *y_m_ell, double complex *m_ell_m, double *n_ell_phi,
		 fftw_plan plan_c2r, double *f_i_phi, int nell, int npol, int nufact,
		 int nphi);

/*
 * backward_sp_mixed_sst: convolve a single ring of the map for sst bispectra.
 *
 * Arguments
 * ---------
 * from L_list to m_dim: the same as compute_A_LM arguments, see ksw_estimator.h.
 * A_L_M: (npol, ndeltaL, nL, m_dim) array 
 * f_i_L : (nufact * npol * ndeltaL * nL) array with unique factors.
 * n_L_phi: (npol * ndeltaL * nL * nphi) array for output of ring fft.
 * plan_c2c: fftw plan for ring complex2complex fft.
 * f_i_phi: (nufact * nphi) array for output unique factors on ring.
 * nufact: Number of unique factors.
 * nphi: Number of phi elements on ring.
 * nell: Number of multipoles.
 * Kappa_i_L: (npol, ndeltaL, nL) array, kappa functionals
 */

void backward_sp_mixed_sst(const int *L_list,
			  int nL, int npol, 
			  int n_scalar1, int n_scalar2, int n_tensor,
			  const float complex *a_ell_m,
			  const float *y_M_L,
			  const float *w3j_product_scalar1, const float *w3j_product_scalar2, const float *w3j_product_tensor,
			  const float complex *prefactors_scalar1, const float complex *prefactors_scalar2, const float complex *prefactors_tensor, 
			  float complex *A_L_M_scalar1, float complex *A_L_M_scalar2, float complex *A_L_M_tensor,
			  int Lmax, int nell, int m_dim,
			  float complex *n_L_phi_scalar, float complex *n_L_phi_tensor,
			  fftwf_plan plan_c2c_scalar, fftwf_plan plan_c2c_tensor,
			  float complex *f_i_phi_scalar1, float complex *f_i_phi_scalar2, float complex *f_i_phi_tensor, 
			  int nufact, int nphi,
			  const float complex *kappa_i_L_scalar1, const float complex *kappa_i_L_scalar2, const float complex *kappa_i_L_tensor); 

void backward_dp_mixed_sst(const int *L_list,
			  int nL, int npol,
			  int n_scalar1, int n_scalar2, int n_tensor,	
			  const double complex *a_ell_m,
			  const double *y_M_L, 
			  const double *w3j_product_scalar1, const double *w3j_product_scalar2, const double *w3j_product_tensor,
			  const double complex *prefactors_scalar1, const double complex *prefactors_scalar2, const double complex *prefactors_tensor, 
			  double complex *A_L_M_scalar1, double complex *A_L_M_scalar2, double complex *A_L_M_tensor,
			  int Lmax, int nell, int m_dim,
			  double complex *n_L_phi_scalar, double complex *n_L_phi_tensor,
			  fftw_plan plan_c2c_scalar, fftw_plan plan_c2c_tensor,
			  double complex *f_i_phi_scalar1, double complex *f_i_phi_scalar2, double complex *f_i_phi_tensor, 
			  int nufact, int nphi,
			  const double complex *kappa_i_L_scalar1, const double complex *kappa_i_L_scalar2, const double complex *kappa_i_L_tensor); 

/*
 * Calculate the contribution of single ring to dT/dalm.
 *
 * Arguments
 * ---------
 * f_i_ell    : (nufact * npol * nell) array with unique factors.
 * a_ell_m    : (npol * nell * nell) complex array with ell-major alms.
 * y_m_ell    : (nell * nell) array with m-major Ylms.
 * m_ell_m    : (npol * nell * nm) complex array as input for ring fft, nm = nphi / 2 + 1.
 * n_ell_phi  : (npol * nell * nphi) array as output for ring fft.
 * plan_r2c   : fftw plan for ring rea2complex fft.
 * f_i_phi    : (nufact * nphi) array for output unique factors on ring.
 * work_i_ell : (nw, npol, nell) array for internal X_i_ell, Y_i_ell, Z_i_ell calculations.
 * work_i_phi : (nw, nphi) array for internal dT/dX_i_phi, dT/dY_i_phi, dT/dZ_i_phi calulations.
 * rule       : (nrule, 3) array of indices into f_i_phi that give X_i Y_i Z_i.
 * weights    : (nrule, 3) array of weights for X_i Y_i Z_i.
 * ct_weight  : Single quadruture weight (for cos(theta)) for this ring.
 * nrule      : Number of rules.
 * nw         : Number of elements in work arrays, see get_forward_array_size.
 * nell       : Number of multipoles.
 * npol       : Number of polarization dimensions.
 * nphi       : Number of phi per ring.
 */

void forward_sp(const float *f_i_ell, float complex *a_ell_m, const float *y_m_ell,
		float complex *m_ell_m, float *n_ell_phi, fftwf_plan plan_r2c,
		const float *f_i_phi, float *work_i_ell, float *work_i_phi,
		const long long *rule, const float *weights, const double ct_weight,
		int nrule, int nw, int nell, int npol, int nphi);

void forward_dp(const double *f_i_ell, double complex *a_ell_m, const double *y_m_ell,
		double complex *m_ell_m, double *n_ell_phi, fftw_plan plan_r2c,
		const double *f_i_phi, double *work_i_ell, double *work_i_phi,
		const long long *rule, const double *weights, const double ct_weight,
		int nrule, int nw, int nell, int npol, int nphi);

/*
 * Compute the number of elements needed for the arrays in the forward function.
 * The idea is that if rule = [[0, 0, 0]], i.e. X_ell = Y_ell = Z_ell, you also
 * have dT/dX_phi = dT/dY_phi= dT/dZ_phi, see Smith Eq. 76, 77. Therefore you 
 * can simply do 3 * X_ell dT/dX_ell to get dT/dN.
 *
 * Arguments
 * ---------
 * rule  : (nrule, 3) array
 * nrule : Number of rules.
 */

int get_forward_array_size(const long long *rule, int nrule);


