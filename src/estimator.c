#include <ksw_estimator.h>

// nufact = (ncomp * nr)
// nrule = nfact

// rule : (nfact, 3)
// f_i_phi : (nufact, nphi)

float t_cubic_on_ring_sp(int *rule, float *weights, float *f_i_phi, int nrule, int nphi){

    float t_cubic = 0.0;

    for (int ridx=0; ridx<nrule; ridx++){

	int rx = rule[ridx*3];
	int ry = rule[ridx*3+1];
	int rz = rule[ridx*3+2];

	int wx = weights[ridx*3];
	int wy = weights[ridx*3+1];
	int wz = weights[ridx*3+2];

	for (int pidx=0; pidx<nphi; pidx++){
	    t_cubic += wx * wy * wz * f_i_phi[rx*nphi+pidx] * f_i_phi[ry*nphi+pidx]
		       * f_i_phi[rz*nphi+pidx];
	}
    }
    return t_cubic;
}

void backward_sp(float *f_i_ell, float complex *a_m_ell, double *y_m_ell,
		 float complex *m_ell_m, float *n_ell_phi, fftwf_plan plan_c2r,
		 float *f_i_phi, int nell, int npol, int nufact, int nphi){

    int nm = nphi / 2 + 1;

    // Place alm * Ylm into Mlm.
    for (int pidx=0; pidx<npol; pidx++){
	for (int lidx=0; lidx<nell; lidx++){
	    for (int midx=0; midx<nm; midx++){

		complex double tmp;

		if (midx < nell){
		    tmp = (complex double) a_m_ell[pidx*nell*nm+lidx*nell+midx]
			* (complex double) y_m_ell[lidx*nell+midx];
		} else{
		    // Needed because m_ell_m array can be larger than alm.
		    tmp = 0. + 0.*I;
		}
		m_ell_m[pidx*nell*nm+lidx*nm+midx] = (complex float) tmp;

	    }
	}
    }

    // Backward fft. Note that fftw has no normalization for backward
    // or forward. Unlike numpy and pyfftw which apply 1/nphi during 
    // backward (c2r). So in python version I multiply the result of 
    // the fft by nphi to compensate for that factor. Here it's not needed.
    fftwf_execute_dft_c2r(plan_c2r, m_ell_m, n_ell_phi);

    // f_i_ell @ n_ell_phi -> f_i_phi.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * nell,
		1.0, f_i_ell, npol * nell,
		n_ell_phi, nphi,
		0.0, f_i_phi, nphi);    
}

void forward_sp(float *f_i_ell, float complex *a_m_ell, double *y_m_ell,
		float complex *m_ell_m, float *n_ell_phi, fftwf_plan plan_r2c,
		float *f_i_phi, float *x_i_ell, float *y_i_ell, float *z_i_ell,
		float *dtdx_i_phi, float *dtdy_i_phi, float *dtdz_i_phi, int *rule,
		float *weights, int nrule, int nell, int npol, int nufact, int nphi){

    for (int ridx=0; ridx<nrule; ridx++){

	int rx = rule[ridx*3];
	int ry = rule[ridx*3+1];
	int rz = rule[ridx*3+2];

	float wx = weights[ridx*3];
	float wy = weights[ridx*3+1];
	float wz = weights[ridx*3+2];	

    }
}

float t_cubic_sp(float *ct_weights, int *rule, float *weights, float *f_i_ell, 
		 float complex *a_m_ell, double *y_m_ell, int ntheta, int nrule,
		 int nell, int npol, int nufact, int nphi){

    int nm = nphi / 2 + 1;
    float t_cubic = 0.0;
    int nffts[1] = {nphi};
    fftwf_plan plan_c2r;

    // Plan fft on temporary arrays now in order to avoid having to run the planner
    // in a omp critial region later.
    float complex *m_ell_m = fftwf_malloc(sizeof *m_ell_m * npol * nell * nm);
    float *n_ell_phi = fftwf_malloc(sizeof *n_ell_phi * npol * nell * nphi);

    plan_c2r = fftwf_plan_many_dft_c2r(1, nffts, npol * nell,
				       m_ell_m, NULL,
				       1, nm,
				       n_ell_phi, NULL,
				       1, nphi,
				       FFTW_MEASURE);
    fftwf_free(m_ell_m);
    fftwf_free(n_ell_phi);


    #pragma omp parallel 
    {
        
    float complex *m_ell_m = fftwf_malloc(sizeof *m_ell_m * npol * nell * nm);
    float *n_ell_phi = fftwf_malloc(sizeof *n_ell_phi * npol * nell * nphi);
    float *f_i_phi = fftwf_malloc(sizeof *f_i_phi * nufact * nphi);

    if (m_ell_m == NULL || n_ell_phi == NULL || f_i_phi == NULL){
	fftwf_free(m_ell_m);
	fftwf_free(n_ell_phi);
	fftwf_free(f_i_phi);
	exit(1);
    }

    #pragma omp for reduction (+:t_cubic)
    for (int tidx=0; tidx<ntheta; tidx++){

	backward_sp(f_i_ell, a_m_ell, y_m_ell + tidx * nell * nell,
		    m_ell_m, n_ell_phi, plan_c2r,
		    f_i_phi, nell, npol, nufact, nphi);
	
	t_cubic += t_cubic_on_ring_sp(rule, weights, f_i_phi, nrule, nphi) 
	    * PI * ct_weights[tidx] / 3. / (float)nphi;
    }

    fftwf_free(m_ell_m);
    fftwf_free(n_ell_phi);
    fftwf_free(f_i_phi);

    // End parallel region
    }
    
    fftwf_destroy_plan(plan_c2r);

    return t_cubic;
}

// step
