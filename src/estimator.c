#include <ksw_estimator.h>

// nufact = (ncomp * nr)

// rule : (nfact, 3)
// f_i_phi : (nufact, nphi)

float t_cubic_on_ring_sp(int *rule, float *f_i_phi, int nrule, int nphi){

    float t_cubic = 0.0;

    for (int ridx=0; ridx<nrule; ridx++){

	int rx = rule[ridx*3];
	int ry = rule[ridx*3+1];
	int rz = rule[ridx*3+2];

	for (int pidx=0; pidx<nphi; pidx++){
	    t_cubic += f_i_phi[rx*nphi+pidx] * f_i_phi[ry*nphi+pidx] * f_i_phi[rz*nphi+pidx];
	}
    }

    return t_cubic;
}

// backward for single theta.
// input f_i_ell, m_ell_m, n_ell_phi, a_m_ell, y_m_ell
// output f_i_phi

// Openmpd over theta, so each thread gets unique slice into y_ell_m
// Every thread needs private m_ell_m, f_i_ell, f_i_phi, n_ell_phi.
// a_m_ell is shared and accessed by everyone.
// So in function calling backward you need a parallel region where each
// allocates  m_ell_m, f_i_ell, f_i_phi arrays, then start a parallel loop
// imagine 5000 thetas, 200 threads : 25 passes through loop, that should be
// enough to dwarf the cost of allocating memory.
// If it's too slow, you can move the allocation outide the loop over step, like
// in the python code. but it's ugly because you need to know the number of threads.

/*
 * Convolve a single ring of the map with all unique bispectrum factors.
 *
 * Arguments
 * ---------
 * f_i_ell   : (nufact * npol, * nell) array with unique factors.
 * a_m_ell   : (npol * nell * nell) complex array with ell-major alms.
 * y_m_ell   : (nell * nell) array with ell-major Ylms.
 * m_ell_m   : (npol * nell * nm) complex array as input for ring fft.
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
		 float *f_i_phi, int nell, int npol, int nufact, int nphi){

    int nm = nphi / 2 + 1;

    // Place alm * Ylm into Mlm and correct for nphi.
    for (int pidx=0; pidx<npol; pidx++){
	for (int lidx=0; lidx<nell; lidx++){
	    for (int midx=0; midx<nm; midx++){

		complex double tmp;

		if (midx < nell){
		    tmp = (complex double) a_m_ell[pidx*npol*nell*nm+lidx*nell*nell+midx]
			* (complex double) y_m_ell[lidx*nell*nell+midx] * nphi;
		} else{
		    // Needed because m_ell_m array can be larger than alm.
		    tmp = 0. + 0.*I;
		}

		m_ell_m[pidx*npol*nell*nm+lidx*nell*nm+midx] = (complex float) tmp;

	    }
	}
    }

    // Backward fft.
    fftwf_execute(plan_c2r);
        
    // f_i_ell @ n_ell_phi -> f_i_phi.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * nell,
		1.0, f_i_ell, npol * nell,
		n_ell_phi, nphi,
		0.0, f_i_phi, nphi);    
}

//    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//		m, n, k, 1.0, mat_a, k, mat_b + tidx * k * n, n, 0.0, mat_c + tidx * m * n, n);


// needs all args to backward, t_cubic_on_ring, + theta, theta_weights. NOOO because you allocate a bunch.

/*
 * Compute T[a] for a collection of rings.
 *
 * Arguments
 * ---------
 * ct_weights : (ntheta) array of quadruture weights for cos(theta).
 * rule       : (nrule, 3) array
 * a_m_ell    : (npol * nell * nell) complex array with ell-major alms.
 * y_m_ell    : (nell * nell) array with ell-major Ylms.
 * ntheta     : number of thetas (rings).
 * nrule      : number of rules.
 * nell       : Number of multipoles.
 * npol       : Number of polarization dimensions.
 * nufact     : Number of unique factors.
 * nphi       : Number of phi per ring.
 */

float t_cubic_sp(float *ct_weights, int *rule, float complex *a_m_ell, double *y_m_ell,
		 int ntheta, int nrule, int nell, int npol, int nufact, int nphi){

    int nm = nphi / 2 + 1;
    float t_cubic = 0.0;

    // Parallel region
    
    // allocate m_ell_m, n_ell_phi, f_i_ell, f_i_phi arrays, then start a parallel loop
    
    // * m_ell_m   : (npol * nell * nm) complex array as input for ring fft.
    // * n_ell_phi : (npol * nell * nphi) array as output for ring fft.
    //* f_i_ell   : (nufact * npol, * nell) array with unique factors.
    //* f_i_phi   : (nufact * nphi) array for output unique factors on ring.
    
    int nffts[1] = {nphi};

    float complex *m_ell_m = fftwf_malloc(sizeof *m_ell_m * npol * nell * nm);
    float *n_ell_phi = fftwf_malloc(sizeof *n_ell_phi * npol * nell * nphi);

    float *f_i_ell = fftwf_malloc(sizeof *f_i_ell * nufact * npol * nell);
    float *f_i_phi = fftwf_malloc(sizeof *f_i_phi * nufact * nphi);
    
    if (m_ell_m == NULL || n_ell_phi == NULL || f_i_ell == NULL || f_i_phi == NULL){
	fftwf_free(m_ell_m);
	fftwf_free(n_ell_phi);
	fftwf_free(f_i_ell);
	fftwf_free(f_i_phi);
	exit(1);
    }
	
    // create plan
    //* plan_c2r  : fftw plan for ring complex2real fft.
    
    fftwf_plan plan_c2r =  fftwf_plan_many_dft_c2r(1, nffts, npol * nell,
						   m_ell_m, NULL,
						   1, npol * nell,
						   n_ell_phi, NULL,
						   1, npol * nell,
						   FFTW_ESTIMATE);
    						 
    
    for (int tidx=0; tidx<ntheta; tidx++){

	backward_sp(f_i_ell, a_m_ell, y_m_ell,
		    m_ell_m, n_ell_phi, plan_c2r,
		    f_i_phi, nell, npol, nufact, nphi);
	
	t_cubic += t_cubic_on_ring_sp(rule, f_i_phi, nrule, nphi) 
	    * PI * ct_weights[tidx] / 3. / nphi;
    }
    
    // Free plan.
    fftwf_destroy_plan(plan_c2r);

    // Free arrays    
    fftwf_free(m_ell_m);
    fftwf_free(n_ell_phi);
    fftwf_free(f_i_ell);
    fftwf_free(f_i_phi);
    
    
    // End parallel region
    
    return t_cubic;
}

// forward

// step
