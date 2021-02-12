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


void backward_sp(float *f_i_ell, float complex *a_m_ell, double *y_m_ell,
		 float complex *m_ell_m, float *n_ell_phi, fftwf_plan plan_c2r,
		 float *f_i_phi, int nell, int npol, int nufact, int nphi);
