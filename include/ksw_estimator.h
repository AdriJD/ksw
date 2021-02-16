#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <omp.h>
#include <fftw3.h>
#include <mkl_cblas.h>

#define PI 3.14159265358979323846

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

float t_cubic_on_ring_sp(int *rule, float * weights, float *f_i_phi, int nrule,
			 int nphi);

/*
 * Convolve a single ring of the map with all unique bispectrum factors.
 *
 * Arguments
 * ---------
 * f_i_ell   : (nufact * npol * nell) array with unique factors.
 * a_ell_m   : (npol * nell * nell) complex array with ell-major alms.
 * y_m_ell   : (nell * nell) array with ell-major Ylms.
 * m_ell_m   : (npol * nell * nm) complex array as input for ring fft, nm = nphi / 2 + 1.
 * n_ell_phi : (npol * nell * nphi) array as output for ring fft.
 * plan_c2r  : fftw plan for ring complex2real fft.
 * f_i_phi   : (nufact * nphi) array for output unique factors on ring.
 * nell      : Number of multipoles.
 * npol      : Number of polarization dimensions.
 * nufact    : Number of unique factors.
 * nphi      : Number of phi per ring.
 */

void backward_sp(float *f_i_ell, float complex *a_ell_m, double *y_m_ell,
		 float complex *m_ell_m, float *n_ell_phi, fftwf_plan plan_c2r,
		 float *f_i_phi, int nell, int npol, int nufact, int nphi);

/*
 * Calculate the contribution of single ring to dT/dalm.
 *
 * Arguments
 * ---------
 * f_i_ell    : (nufact * npol * nell) array with unique factors.
 * a_ell_m    : (npol * nell * nell) complex array with ell-major alms.
 * y_m_ell    : (nell * nell) array with ell-major Ylms.
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

void forward_sp(float *f_i_ell, float complex *a_ell_m, double *y_m_ell,
		float complex *m_ell_m, float *n_ell_phi, fftwf_plan plan_r2c,
		float *f_i_phi, float *work_i_ell, float *work_i_phi,  int *rule,
		float *weights, float ct_weight, int nrule, int nw, int nell, int npol,
		int nphi);

/*
 * Compute T[a] for a collection of rings.
 *
 * Arguments
 * ---------
 * ct_weights : (ntheta) array of quadruture weights for cos(theta).
 * rule       : (nrule, 3) array
 * weights    : (nrule, 3) array of weights for X_i Y_i Z_i.
 * f_i_ell    : (nufact * npol * nell) array with unique factors.
 * a_ell_m    : (npol * nell * nell) complex array with ell-major alms.
 * y_m_ell    : (ntheta * nell * nell) array with ell-major Ylms for each ring.
 * ntheta     : Number of thetas (rings).
 * nrule      : Number of rules.
 * nell       : Number of multipoles.
 * npol       : Number of polarization dimensions.
 * nufact     : Number of unique factors.
 * nphi       : Number of phi per ring.
 */

float t_cubic_sp(float *ct_weights, int *rule, float *weights, float *f_i_ell,
		 float complex *a_ell_m, double *y_m_ell, int ntheta, int nrule,
		 int nell, int npol, int nufact, int nphi);

/*
 * Compute forward and backward operation on collection of rings.
 *
 * Arguments
 * ---------
 * ct_weights : (ntheta) array of quadruture weights for cos(theta).
 * rule       : (nrule, 3) array
 * weights    : (nrule, 3) array of weights for X_i Y_i Z_i.
 * f_i_ell    : (nufact * npol * nell) array with unique factors.
 * a_ell_m    : (npol * nell * nell) complex array with ell-major alms.
 * y_m_ell    : (ntheta * nell * nell) array with ell-major Ylms for each ring.
 * grad_t     : (npol * nell * nell) complex array with ell-major alms.
 * ntheta     : Number of thetas (rings).
 * nrule      : Number of rules.
 * nell       : Number of multipoles.
 * npol       : Number of polarization dimensions.
 * nufact     : Number of unique factors.
 * nphi       : Number of phi per ring.
 */

void step_sp(float *ct_weights, int *rule, float *weights, float *f_i_ell, 
	     float complex *a_ell_m, double *y_m_ell, float complex *grad_t, 
	     int ntheta, int nrule, int nell, int npol, int nufact, int nphi);

/*
 * Compute the number of elements needed for the arrays in the forward function.
 * The idea is that if rule = [[0, 0, 0]], i.e. X_ell = Y_ell = Z_ell, you also
 * have dT/dX_phi = dT/dY_phi= dT/dZ_phi, see Smith Eq. 76, 77. Therefore you 
 * can simply do 3 * X_ell dT/dX_ell to get dT/dN.
 *
 * Arguments
 * ---------
 * ct_weights : (ntheta) array of quadruture weights for cos(theta).
 * rule       : (nrule, 3) array
 */

int get_forward_array_size(int *rule, int nrule);
