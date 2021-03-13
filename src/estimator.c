#include "ksw_estimator_internal.h"
#include <ksw_estimator.h>

int get_forward_array_size(const long long *rule, int nrule){

    int array_size = 0;

    for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

	long long rx = rule[ridx*3];
	long long ry = rule[ridx*3+1];
	long long rz = rule[ridx*3+2];

	array_size += 1;
	
	if (ry != rx){
	    array_size += 1;
	}
	if (ry != rz && rx != rz){
	    array_size += 1;
	}
    }
    return array_size;
}

/* Single precision versions */

float t_cubic_on_ring_sp(const long long *rule, const float *weights, const float *f_i_phi,
			 int nrule, int nphi){

    float t_cubic = 0.0;

    for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

	long long rx = rule[ridx*3];
	long long ry = rule[ridx*3+1];
	long long rz = rule[ridx*3+2];

	float wx = weights[ridx*3];
	float wy = weights[ridx*3+1];
	float wz = weights[ridx*3+2];

	for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
	    t_cubic += wx * wy * wz * f_i_phi[rx*nphi+phidx] * f_i_phi[ry*nphi+phidx]
		       * f_i_phi[rz*nphi+phidx];
	}
    }
    return t_cubic;
}

void backward_sp(const float *f_i_ell, const float complex *a_ell_m, 
		 const float *y_m_ell, float complex *m_ell_m, float *n_ell_phi,
		 fftwf_plan plan_c2r, float *f_i_phi, int nell, int npol, int nufact,
		 int nphi){

    int nm = nphi / 2 + 1;

    // Place alm * Ylm into Mlm.
    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
	for (ptrdiff_t lidx=0; lidx<nell; lidx++){
	    for (ptrdiff_t midx=0; midx<nm; midx++){

		complex float tmp;

		if (midx < nell){
		    tmp = a_ell_m[pidx*nell*nell+lidx*nell+midx]
			  * y_m_ell[midx*nell+lidx];
		} else{
		    // Needed because m_ell_m array can be larger than alm.
		    tmp = 0. + 0.*I;
		}

		m_ell_m[pidx*nell*nm+lidx*nm+midx] = tmp;
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

void forward_sp(const float *f_i_ell, float complex *a_ell_m, const float *y_m_ell,
		float complex *m_ell_m, float *n_ell_phi, fftwf_plan plan_r2c,
		const float *f_i_phi, float *work_i_ell, float *work_i_phi,
		const long long *rule, const float *weights, const double ct_weight,
		int nrule, int nw, int nell, int npol, int nphi){

    int widx = 0; // Index to work arrays.
    int nm = nphi / 2 + 1;

    for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

	long long rx = rule[ridx*3];
	long long ry = rule[ridx*3+1];
	long long rz = rule[ridx*3+2];

	float weight = weights[ridx*3] * weights[ridx*3+1] * weights[ridx*3+2]
	    * PI * ct_weight / 3. / (double) nphi;

	// Fill work arrays.
	if (rx == ry && rx == rz){ // Case: 000.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = 3. * weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    widx += 1;
	} else if (rx == ry && ry != rz){ // Case: 001.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = 2. * weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rz*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[ry*nphi+phidx];
	    }
	    widx += 2;
	} else if (rx != ry && ry == rz){ // Case: 100.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = 2. * weight
			* f_i_ell[ry*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    widx += 2;
	} else if (rx == rz && rx != ry){ // Case 010.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = 2. * weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[ry*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    widx += 2;
	} else { // Case: 012.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[ry*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+2)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rz*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+2)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[ry*nphi+phidx];
	    }
	    widx += 3;
	}
    }

    if (widx != nw){
	fprintf(stderr, "widx (%d) != nw (%d) \n", widx, nw);
	exit(1);
    }

    // Sum_i X_i_ell dT/dX_i_phi + Y_i_ell dT/dY_i_phi + Z_i_ell dT/dZ_i_phi -> n_ell_phi.
    // Implemented as work_i_ell.T @ work_i_phi -> n_ell_phi.
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		npol * nell, nphi, nw,
		1.0, work_i_ell, npol * nell,
		work_i_phi, nphi,
		0.0, n_ell_phi, nphi);    

    fftwf_execute_dft_r2c(plan_r2c, n_ell_phi, m_ell_m);

    // Multiply by ylm and add result to alm.
    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
	for (ptrdiff_t lidx=0; lidx<nell; lidx++){
	    for (ptrdiff_t midx=0; midx<=lidx; midx++){
    
		a_ell_m[pidx*nell*nell+lidx*nell+midx] += y_m_ell[midx*nell+lidx] 
		    * m_ell_m[pidx*nell*nm+lidx*nm+midx];

	    }
	}
    }        
}

float t_cubic_sp(const double *ct_weights, const long long *rule, const float *weights,
		 const float *f_i_ell, const float complex *a_ell_m,
		 const float *y_m_ell, int ntheta, int nrule,
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
    for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){

	backward_sp(f_i_ell, a_ell_m, y_m_ell + tidx * nell * nell,
		    m_ell_m, n_ell_phi, plan_c2r,
		    f_i_phi, nell, npol, nufact, nphi);
	
	t_cubic += t_cubic_on_ring_sp(rule, weights, f_i_phi, nrule, nphi) 
	    * PI * ct_weights[tidx] / 3. / (double)nphi;
    }

    fftwf_free(m_ell_m);
    fftwf_free(n_ell_phi);
    fftwf_free(f_i_phi);

    } // End of parallel region.
    
    fftwf_destroy_plan(plan_c2r);

    return t_cubic;
}

void step_sp(const double *ct_weights, const long long *rule, const float *weights,
	     const float *f_i_ell, const float complex *a_ell_m, const float *y_m_ell,
	     float complex *grad_t, int ntheta, int nrule, int nell, int npol, 
	     int nufact, int nphi){
 
    int nm = nphi / 2 + 1;
    float t_cubic = 0.0;
    int nffts[1] = {nphi};
    fftwf_plan plan_c2r, plan_r2c;
    int nw = get_forward_array_size(rule, nrule);

    // Plan ffts on temporary arrays now in order to avoid having to run the planner
    // in a omp critial region later.
    float complex *m_ell_m = fftwf_malloc(sizeof *m_ell_m * npol * nell * nm);
    float *n_ell_phi = fftwf_malloc(sizeof *n_ell_phi * npol * nell * nphi);

    plan_c2r = fftwf_plan_many_dft_c2r(1, nffts, npol * nell,
				       m_ell_m, NULL,
				       1, nm,
				       n_ell_phi, NULL,
				       1, nphi,
				       FFTW_MEASURE);
    plan_r2c = fftwf_plan_many_dft_r2c(1, nffts, npol * nell,
				       n_ell_phi, NULL,
				       1, nphi,
				       m_ell_m, NULL,
				       1, nm,
				       FFTW_MEASURE);
    
    fftwf_free(m_ell_m);
    fftwf_free(n_ell_phi);

    #pragma omp parallel 
    {
        
    float complex *m_ell_m = fftwf_malloc(sizeof *m_ell_m * npol * nell * nm);
    float *n_ell_phi = fftwf_malloc(sizeof *n_ell_phi * npol * nell * nphi);
    float *f_i_phi = fftwf_malloc(sizeof *f_i_phi * nufact * nphi);
    float *work_i_ell = fftwf_malloc(sizeof *work_i_ell * nw * npol * nell);    
    float *work_i_phi = fftwf_malloc(sizeof *work_i_phi * nw * nphi);    
    float complex *grad_t_priv = fftwf_malloc(sizeof *grad_t_priv * npol * nell * nell);	

    for (ptrdiff_t i=0; i<npol*nell*nell; i++){
	grad_t_priv[i] = 0;
    }

    if (m_ell_m == NULL || n_ell_phi == NULL || f_i_phi == NULL){
	fftwf_free(m_ell_m);
	fftwf_free(n_ell_phi);
	fftwf_free(f_i_phi);
	exit(1);
    }

    #pragma omp for
    for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){

	backward_sp(f_i_ell, a_ell_m, y_m_ell + tidx * nell * nell,
		    m_ell_m, n_ell_phi, plan_c2r,
		    f_i_phi, nell, npol, nufact, nphi);
	forward_sp(f_i_ell, grad_t_priv, y_m_ell + tidx * nell * nell,
		   m_ell_m, n_ell_phi, plan_r2c, f_i_phi, work_i_ell,
		   work_i_phi, rule, weights, ct_weights[tidx], nrule,
		   nw, nell, npol, nphi);		
    }

    #pragma omp critical
    {
	for (ptrdiff_t i=0; i<npol*nell*nell; i++){
	    grad_t[i] += grad_t_priv[i];
	}	
    }

    fftwf_free(m_ell_m);
    fftwf_free(n_ell_phi);
    fftwf_free(f_i_phi);
    fftwf_free(work_i_ell);
    fftwf_free(work_i_phi);
    fftwf_free(grad_t_priv);

    } // End of parallel region
    
    fftwf_destroy_plan(plan_c2r);
    fftwf_destroy_plan(plan_r2c);
}

void compute_ylm_sp(const double *thetas, float *y_m_ell, int ntheta, int lmax){

    int nell = lmax + 1;
    double epsilon = 1e-300;

    #pragma omp parallel
    {
    Ylmgen_C ygen;

    // Sse2 version not needed, ylm computation is subdominant to filling the ylm
    // array, i.e. it's really fast.
    Ylmgen_init(&ygen, lmax, lmax, 0, 0, epsilon);
    Ylmgen_set_theta(&ygen, thetas, ntheta);

    #pragma omp for schedule(dynamic, 10)
    for (ptrdiff_t midx=0; midx<nell; midx++){ 
	
	for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){
	
	    Ylmgen_prepare(&ygen, tidx, midx);
	    Ylmgen_recalc_Ylm(&ygen);

	    ptrdiff_t firstl = *ygen.firstl;

	    for (ptrdiff_t lidx=firstl; lidx<nell; lidx++){
		
		y_m_ell[tidx*nell*nell+midx*nell+lidx] = (float) ygen.ylm[lidx];
	    }
	}
    }

    Ylmgen_destroy(&ygen);
    } // End of parallel region.
}

/* Double precision versions */

double t_cubic_on_ring_dp(const long long *rule, const double *weights, const double *f_i_phi,
			  int nrule, int nphi){

    double t_cubic = 0.0;

    for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

	long long rx = rule[ridx*3];
	long long ry = rule[ridx*3+1];
	long long rz = rule[ridx*3+2];

	double wx = weights[ridx*3];
	double wy = weights[ridx*3+1];
	double wz = weights[ridx*3+2];

	for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
	    t_cubic += wx * wy * wz * f_i_phi[rx*nphi+phidx] * f_i_phi[ry*nphi+phidx]
		       * f_i_phi[rz*nphi+phidx];
	}
    }
    return t_cubic;
}

void backward_dp(const double *f_i_ell, const double complex *a_ell_m, 
		 const double *y_m_ell, double complex *m_ell_m, double *n_ell_phi,
		 fftw_plan plan_c2r, double *f_i_phi, int nell, int npol, int nufact,
		 int nphi){

    int nm = nphi / 2 + 1;

    // Place alm * Ylm into Mlm.
    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
	for (ptrdiff_t lidx=0; lidx<nell; lidx++){
	    for (ptrdiff_t midx=0; midx<nm; midx++){

		complex double tmp;

		if (midx < nell){
		    tmp = a_ell_m[pidx*nell*nell+lidx*nell+midx]
			  * y_m_ell[midx*nell+lidx];
		} else{
		    // Needed because m_ell_m array can be larger than alm.
		    tmp = 0. + 0.*I;
		}

		m_ell_m[pidx*nell*nm+lidx*nm+midx] = tmp;
	    }
	}
    }

    // Backward fft. Note that fftw has no normalization for backward
    // or forward. Unlike numpy and pyfftw which apply 1/nphi during 
    // backward (c2r). So in python version I multiply the result of 
    // the fft by nphi to compensate for that factor. Here it's not needed.
    fftw_execute_dft_c2r(plan_c2r, m_ell_m, n_ell_phi);

    // f_i_ell @ n_ell_phi -> f_i_phi.
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * nell,
		1.0, f_i_ell, npol * nell,
		n_ell_phi, nphi,
		0.0, f_i_phi, nphi);    
}

void forward_dp(const double *f_i_ell, double complex *a_ell_m, const double *y_m_ell,
		double complex *m_ell_m, double *n_ell_phi, fftw_plan plan_r2c,
		const double *f_i_phi, double *work_i_ell, double *work_i_phi,
		const long long *rule, const double *weights, const double ct_weight,
		int nrule, int nw, int nell, int npol, int nphi){

    int widx = 0; // Index to work arrays.
    int nm = nphi / 2 + 1;

    for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

	long long rx = rule[ridx*3];
	long long ry = rule[ridx*3+1];
	long long rz = rule[ridx*3+2];

	double weight = weights[ridx*3] * weights[ridx*3+1] * weights[ridx*3+2]
	    * PI * ct_weight / 3. / (double) nphi;

	// Fill work arrays.
	if (rx == ry && rx == rz){ // Case: 000.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = 3. * weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    widx += 1;
	} else if (rx == ry && ry != rz){ // Case: 001.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = 2. * weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rz*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[ry*nphi+phidx];
	    }
	    widx += 2;
	} else if (rx != ry && ry == rz){ // Case: 100.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = 2. * weight
			* f_i_ell[ry*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    widx += 2;
	} else if (rx == rz && rx != ry){ // Case 010.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = 2. * weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[ry*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    widx += 2;
	} else { // Case: 012.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[ry*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+2)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rz*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+2)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[ry*nphi+phidx];
	    }
	    widx += 3;
	}
    }

    if (widx != nw){
	fprintf(stderr, "widx (%d) != nw (%d) \n", widx, nw);
	exit(1);
    }

    // Sum_i X_i_ell dT/dX_i_phi + Y_i_ell dT/dY_i_phi + Z_i_ell dT/dZ_i_phi -> n_ell_phi.
    // Implemented as work_i_ell.T @ work_i_phi -> n_ell_phi.
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		npol * nell, nphi, nw,
		1.0, work_i_ell, npol * nell,
		work_i_phi, nphi,
		0.0, n_ell_phi, nphi);    

    fftw_execute_dft_r2c(plan_r2c, n_ell_phi, m_ell_m);

    // Multiply by ylm and add result to alm.
    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
	for (ptrdiff_t lidx=0; lidx<nell; lidx++){
	    for (ptrdiff_t midx=0; midx<=lidx; midx++){
    
		a_ell_m[pidx*nell*nell+lidx*nell+midx] += y_m_ell[midx*nell+lidx] 
		    * m_ell_m[pidx*nell*nm+lidx*nm+midx];

	    }
	}
    }        
}

double t_cubic_dp(const double *ct_weights, const long long *rule, const double *weights,
		  const double *f_i_ell, const double complex *a_ell_m,
		  const double *y_m_ell, int ntheta, int nrule,
		  int nell, int npol, int nufact, int nphi){

    int nm = nphi / 2 + 1;
    double t_cubic = 0.0;
    int nffts[1] = {nphi};
    fftw_plan plan_c2r;

    // Plan fft on temporary arrays now in order to avoid having to run the planner
    // in a omp critial region later.
    double complex *m_ell_m = fftw_malloc(sizeof *m_ell_m * npol * nell * nm);
    double *n_ell_phi = fftw_malloc(sizeof *n_ell_phi * npol * nell * nphi);

    plan_c2r = fftw_plan_many_dft_c2r(1, nffts, npol * nell,
				       m_ell_m, NULL,
				       1, nm,
				       n_ell_phi, NULL,
				       1, nphi,
				       FFTW_MEASURE);
    fftw_free(m_ell_m);
    fftw_free(n_ell_phi);

    #pragma omp parallel 
    {
        
    double complex *m_ell_m = fftw_malloc(sizeof *m_ell_m * npol * nell * nm);
    double *n_ell_phi = fftw_malloc(sizeof *n_ell_phi * npol * nell * nphi);
    double *f_i_phi = fftw_malloc(sizeof *f_i_phi * nufact * nphi);

    if (m_ell_m == NULL || n_ell_phi == NULL || f_i_phi == NULL){
	fftw_free(m_ell_m);
	fftw_free(n_ell_phi);
	fftw_free(f_i_phi);
	exit(1);
    }

    #pragma omp for reduction (+:t_cubic)
    for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){

	backward_dp(f_i_ell, a_ell_m, y_m_ell + tidx * nell * nell,
		    m_ell_m, n_ell_phi, plan_c2r,
		    f_i_phi, nell, npol, nufact, nphi);
	
	t_cubic += t_cubic_on_ring_dp(rule, weights, f_i_phi, nrule, nphi) 
	    * PI * ct_weights[tidx] / 3. / (double)nphi;
    }

    fftw_free(m_ell_m);
    fftw_free(n_ell_phi);
    fftw_free(f_i_phi);

    } // End of parallel region
    
    fftw_destroy_plan(plan_c2r);

    return t_cubic;
}

void step_dp(const double *ct_weights, const long long *rule, const double *weights,
	     const double *f_i_ell, const double complex *a_ell_m, const double *y_m_ell,
	     double complex *grad_t, int ntheta, int nrule, int nell, int npol, 
	     int nufact, int nphi){
 
    int nm = nphi / 2 + 1;
    double t_cubic = 0.0;
    int nffts[1] = {nphi};
    fftw_plan plan_c2r, plan_r2c;
    int nw = get_forward_array_size(rule, nrule);

    // Plan ffts on temporary arrays now in order to avoid having to run the planner
    // in a omp critial region later.
    double complex *m_ell_m = fftw_malloc(sizeof *m_ell_m * npol * nell * nm);
    double *n_ell_phi = fftw_malloc(sizeof *n_ell_phi * npol * nell * nphi);

    plan_c2r = fftw_plan_many_dft_c2r(1, nffts, npol * nell,
				       m_ell_m, NULL,
				       1, nm,
				       n_ell_phi, NULL,
				       1, nphi,
				       FFTW_MEASURE);
    plan_r2c = fftw_plan_many_dft_r2c(1, nffts, npol * nell,
				       n_ell_phi, NULL,
				       1, nphi,
				       m_ell_m, NULL,
				       1, nm,
				       FFTW_MEASURE);
    
    fftw_free(m_ell_m);
    fftw_free(n_ell_phi);

    #pragma omp parallel 
    {
        
    double complex *m_ell_m = fftw_malloc(sizeof *m_ell_m * npol * nell * nm);
    double *n_ell_phi = fftw_malloc(sizeof *n_ell_phi * npol * nell * nphi);
    double *f_i_phi = fftw_malloc(sizeof *f_i_phi * nufact * nphi);
    double *work_i_ell = fftw_malloc(sizeof *work_i_ell * nw * npol * nell);    
    double *work_i_phi = fftw_malloc(sizeof *work_i_phi * nw * nphi);    
    double complex *grad_t_priv = fftw_malloc(sizeof *grad_t_priv * npol * nell * nell);	

    for (ptrdiff_t i=0; i<npol*nell*nell; i++){
	grad_t_priv[i] = 0;
    }

    if (m_ell_m == NULL || n_ell_phi == NULL || f_i_phi == NULL){
	fftw_free(m_ell_m);
	fftw_free(n_ell_phi);
	fftw_free(f_i_phi);
	exit(1);
    }

    #pragma omp for
    for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){

	backward_dp(f_i_ell, a_ell_m, y_m_ell + tidx * nell * nell,
		    m_ell_m, n_ell_phi, plan_c2r,
		    f_i_phi, nell, npol, nufact, nphi);
	forward_dp(f_i_ell, grad_t_priv, y_m_ell + tidx * nell * nell,
		   m_ell_m, n_ell_phi, plan_r2c, f_i_phi, work_i_ell,
		   work_i_phi, rule, weights, ct_weights[tidx], nrule,
		   nw, nell, npol, nphi);		
    }

    #pragma omp critical
    {
	for (ptrdiff_t i=0; i<npol*nell*nell; i++){
	    grad_t[i] += grad_t_priv[i];
	}	
    }

    fftw_free(m_ell_m);
    fftw_free(n_ell_phi);
    fftw_free(f_i_phi);
    fftw_free(work_i_ell);
    fftw_free(work_i_phi);
    fftw_free(grad_t_priv);

    } // End of parallel region
    
    fftw_destroy_plan(plan_c2r);
    fftw_destroy_plan(plan_r2c);
}

void compute_ylm_dp(const double *thetas, double *y_m_ell, int ntheta, int lmax){

    int nell = lmax + 1;
    double epsilon = 1e-300;

    #pragma omp parallel
    {
    Ylmgen_C ygen;

    // Sse2 version not needed, ylm computation is subdominant to filling the ylm
    // array, i.e. it's really fast.
    Ylmgen_init(&ygen, lmax, lmax, 0, 0, epsilon);
    Ylmgen_set_theta(&ygen, thetas, ntheta);

    #pragma omp for schedule(dynamic, 10)
    for (ptrdiff_t midx=0; midx<nell; midx++){ 
	
	for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){
	
	    Ylmgen_prepare(&ygen, tidx, midx);
	    Ylmgen_recalc_Ylm(&ygen);

	    ptrdiff_t firstl = *ygen.firstl;

	    for (ptrdiff_t lidx=firstl; lidx<nell; lidx++){
		
		y_m_ell[tidx*nell*nell+midx*nell+lidx] = ygen.ylm[lidx];
	    }
	}
    }

    Ylmgen_destroy(&ygen);
    } // End of parallel region.
}

