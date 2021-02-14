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
 * f_i_phi : (nufact, nphi) array of unique factors on ring.
 * nrule   : Number of rules. 
 * nphi    : Number phi elements on ring.
 *
 */

float t_cubic_on_ring_sp(int *rule, float *f_i_phi, int nrule, int nphi);

/*
 * Convolve a single ring of the map with all unique bispectrum factors.
 *
 * Arguments
 * ---------
 * f_i_ell   : (nufact * npol * nell) array with unique factors.
 * a_m_ell   : (npol * nell * nell) complex array with ell-major alms.
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

void backward_sp(float *f_i_ell, float complex *a_m_ell, double *y_m_ell,
		 float complex *m_ell_m, float *n_ell_phi, fftwf_plan plan_c2r,
		 float *f_i_phi, int nell, int npol, int nufact, int nphi);

/*
 * Compute T[a] for a collection of rings.
 *
 * Arguments
 * ---------
 * ct_weights : (ntheta) array of quadruture weights for cos(theta).
 * rule       : (nrule, 3) array
 * f_i_ell    : (nufact * npol * nell) array with unique factors.
 * a_m_ell    : (npol * nell * nell) complex array with ell-major alms.
 * y_m_ell    : (ntheta * nell * nell) array with ell-major Ylms for each ring.
 * ntheta     : number of thetas (rings).
 * nrule      : number of rules.
 * nell       : Number of multipoles.
 * npol       : Number of polarization dimensions.
 * nufact     : Number of unique factors.
 * nphi       : Number of phi per ring.
 */

float t_cubic_sp(float *ct_weights, int *rule, float *f_i_ell, float complex *a_m_ell, 
		 double *y_m_ell, int ntheta, int nrule, int nell, int npol, int nufact,
		 int nphi);
