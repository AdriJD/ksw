#include "ksw_estimator_internal.h"
#include <ksw_estimator.h>
#include <stdlib.h>

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
    mkl_set_num_threads_local(1);
        
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

    mkl_set_num_threads_local(0);
    } // End of parallel region.
    
    fftwf_destroy_plan(plan_c2r);

    return t_cubic;
}

void step_sp(const double *ct_weights, const long long *rule, const float *weights,
	     const float *f_i_ell, const float complex *a_ell_m, const float *y_m_ell,
	     float complex *grad_t, int ntheta, int nrule, int nell, int npol, 
	     int nufact, int nphi){
 
    int nm = nphi / 2 + 1;
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
    mkl_set_num_threads_local(1);        

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

    mkl_set_num_threads_local(0);
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
    mkl_set_num_threads_local(1);

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

    mkl_set_num_threads_local(0);
    } // End of parallel region
    
    fftw_destroy_plan(plan_c2r);

    return t_cubic;
}

void step_dp(const double *ct_weights, const long long *rule, const double *weights,
	     const double *f_i_ell, const double complex *a_ell_m, const double *y_m_ell,
	     double complex *grad_t, int ntheta, int nrule, int nell, int npol, 
	     int nufact, int nphi){
 
    int nm = nphi / 2 + 1;
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
    mkl_set_num_threads_local(1);
        
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

    mkl_set_num_threads_local(0);
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

/* 
*  Reduced A_{LM} construction                                           
*/

float complex get_alm_entry_sp(const float complex *a_ell_m,
		    int nell, int ell, int m){

	if (abs(m) >= nell){
		return 0.f + 0.f * I;
	}
	
    if (m >= 0){
		return a_ell_m[ell * nell + m];
    } 
	else{
		int mp = -m;
		float complex base = a_ell_m[ell * nell + mp];
		return (mp % 2) ? -conjf(base) : conjf(base);
    }
}

double complex get_alm_entry_dp(const double complex *a_ell_m,
		  int nell, int ell, int m){

	if (abs(m) >= nell){
		return 0. + 0. * I;
	}

    if (m >= 0){
		return a_ell_m[ell * nell + m];
    } 
	else{
		int mp = -m;
		double complex base = a_ell_m[ell * nell + mp];
		return (mp % 2) ? -conj(base) : conj(base);
    }
}

void compute_A_LM_sp(const long long *L_list, const long long *deltaL_list,
		     int nL, int ndeltaL, int npol, int n, 
			 const float complex *a_ell_m,
			 const float *y_m_ell, const float *w3j_product,
			 const float complex *prefactors, float complex *out,
			 int Lmax, int nell, int m_dim){

    int mdim_expected = 2 * Lmax + 1;
    if (m_dim < mdim_expected){
		fprintf(stderr, "compute_A_LM_sp: m_dim (%d) < 2*Lmax+1 (%d)\n",
		m_dim, mdim_expected);
		return;
    }

	ptrdiff_t total = (ptrdiff_t) npol * ndeltaL * nL * m_dim;

    for (ptrdiff_t idx=0; idx<total; idx++){
		out[idx] = 0.f + 0.f * I;
    }

    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		const float complex *alm_pol = a_ell_m + pidx * nell * nell;

		for (ptrdiff_t idL=0; idL<ndeltaL; idL++){
			long long deltaL = deltaL_list[idL];
			for (ptrdiff_t iL=0; iL<nL; iL++){
			long long L = L_list[iL];
			if (L < 0 || L >= nell){
				continue;
			}

			long long ell = L + deltaL;
			if (ell < 0 || ell >= nell){
				continue;
			}

			ptrdiff_t base = ((pidx * ndeltaL + idL) * nL + iL) * m_dim;
			float complex pref = prefactors[(pidx * ndeltaL + idL) * nL + iL];

			for (long long M=-L; M<=L; M++){
				int Moff = (int)(M + Lmax);
				if (Moff < 0 || Moff >= m_dim){
					continue;
				}

				long long m = - M - n;
				if (llabs(m) > ell || llabs(m) >= nell){
					continue;
				}

				float y_val = y_m_ell[Moff * nell + ell];
				float w3j = w3j_product[base + Moff];
				float complex alm_val = get_alm_entry_sp(alm_pol, nell, ell, m);

				float complex contrib = pref * w3j * alm_val * (y_val);
				out[base + Moff] += contrib;
			}
			}
		}
    }
}

void compute_A_LM_dp(const long long *L_list, const long long *deltaL_list,
		     int nL, int ndeltaL, int npol, int n,
			 const double complex *a_ell_m,
			 const double *y_m_ell, const double *w3j_product,
			 const double complex *prefactors, double complex *out,
			 int Lmax, int nell, int m_dim){

    int mdim_expected = 2 * Lmax + 1;
    if (m_dim < mdim_expected){
		fprintf(stderr, "compute_A_LM_dp: m_dim (%d) < 2*Lmax+1 (%d)\n",
		m_dim, mdim_expected);
		return;
    }

	ptrdiff_t total = (ptrdiff_t) npol * ndeltaL * nL * m_dim;

    for (ptrdiff_t idx=0; idx<total; idx++){
		out[idx] = 0. + 0. * I;
    }


    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		const double complex *alm_pol = a_ell_m + pidx * nell * nell;

		for (ptrdiff_t idL=0; idL<ndeltaL; idL++){
			long long deltaL = deltaL_list[idL];
			for (ptrdiff_t iL=0; iL<nL; iL++){
			long long L = L_list[iL];
			if (L < 0 || L >= nell){
				continue;
			}

			long long ell = L + deltaL;
			if (ell < 0 || ell >= nell){
				continue;
			}

			ptrdiff_t base = ((pidx * ndeltaL + idL) * nL + iL) * m_dim;
			double complex pref = prefactors[(pidx * ndeltaL + idL) * nL + iL];

			for (long long M=-L; M<=L; M++){
				int Moff = (int)(M + Lmax);
				if (Moff < 0 || Moff >= m_dim){
					continue;
				}

				long long m = - M - n;
				if (llabs(m) > ell || llabs(m) >= nell){
					continue;
				}

				double y_val = y_m_ell[Moff * nell + ell];
				double w3j = w3j_product[base + Moff];
				double complex alm_val = get_alm_entry_dp(alm_pol, nell, ell, m);

				double complex contrib = pref * w3j * alm_val * (y_val);
				out[base + Moff] += contrib;
			}
			}
		}
    }
}

float t_cubic_on_ring_sp_sst(const long long *rule, const float *weights,
			const float *f_i_phi_scalar1, const float *f_i_phi_scalar2,
			const float *f_i_phi_tensor, int nrule, int nphi){

	float t_cubic = 0.f;

	for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

		long long rx = rule[ridx*3];
		long long ry = rule[ridx*3+1];
		long long rz = rule[ridx*3+2];

		float wx = weights[ridx*3];
		float wy = weights[ridx*3+1];
		float wz = weights[ridx*3+2];

		for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
			t_cubic += 3 * wx * wy * wz * f_i_phi_scalar1[rx*nphi+phidx] * f_i_phi_scalar2[ry*nphi+phidx]
				* f_i_phi_tensor[rz*nphi+phidx];
		}
	}
	return t_cubic;
}

double t_cubic_on_ring_dp_sst(const long long *rule, const double *weights,
			const double *f_i_phi_scalar1, const double *f_i_phi_scalar2,
			const double *f_i_phi_tensor, int nrule, int nphi){

	double t_cubic = 0.f;

	for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

		long long rx = rule[ridx*3];
		long long ry = rule[ridx*3+1];
		long long rz = rule[ridx*3+2];

		double wx = weights[ridx*3];
		double wy = weights[ridx*3+1];
		double wz = weights[ridx*3+2];

		for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
			t_cubic += 3 * wx * wy * wz * f_i_phi_scalar1[rx*nphi+phidx] * f_i_phi_scalar2[ry*nphi+phidx]
				* f_i_phi_tensor[rz*nphi+phidx];
		}
	}
	return t_cubic;
}


void backward_sp_mixed_sst(const long long *L_list, const long long *deltaL_list,
			  int nL, int ndeltaL, int npol, int n,
			  const float complex *a_ell_m,
			  const float *y_M_L,
			  const float *w3j_product_scalar1, const float *w3j_product_scalar2, const float *w3j_product_tensor,
			  const float complex *prefactors_scalar1, const float complex *prefactors_scalar2, const float complex *prefactors_tensor, 
			  float complex *A_L_M_scalar1, float complex *A_L_M_scalar2, float complex *A_L_M_tensor,
			  int Lmax, int nell, int m_dim,
			  float complex *n_L_phi_scalar1, float complex *n_L_phi_scalar2, float complex *n_L_phi_tensor,
			  fftwf_plan plan_c2c,
			  float *f_i_phi_scalar1, float *f_i_phi_scalar2, float *f_i_phi_tensor, 
			  int nufact, int nphi,
			  const float *kappa_i_L_scalar1, const float *kappa_i_L_scalar2,const float *kappa_i_L_tensor){

	int nm = nphi;

	compute_A_LM_sp(L_list, deltaL_list,
		      nL, ndeltaL, npol, n,
		      a_ell_m, y_M_L, w3j_product_scalar1,
		      prefactors_scalar1, A_L_M_scalar1,
		      Lmax, nell, nm);

	compute_A_LM_sp(L_list, deltaL_list,
		      nL, ndeltaL, npol, n,
		      a_ell_m, y_M_L, w3j_product_scalar2,
		      prefactors_scalar2, A_L_M_scalar2,
		      Lmax, nell, nm);

	compute_A_LM_sp(L_list, deltaL_list,
		      nL, ndeltaL, npol, n,
		      a_ell_m, y_M_L, w3j_product_tensor,
		      prefactors_tensor, A_L_M_tensor,
		      Lmax, nell, nm);

	fftwf_execute_dft(plan_c2c, A_L_M_scalar1, n_L_phi_scalar1);
	fftwf_execute_dft(plan_c2c, A_L_M_scalar2, n_L_phi_scalar2);
	fftwf_execute_dft(plan_c2c, A_L_M_tensor, n_L_phi_tensor);

	const float complex alpha_sp = 1.0f + 0.0f * I;
	const float complex beta_sp = 0.0f + 0.0f * I;

	// f_i_L @ n_L_phi -> f_i_phi.
	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * ndeltaL * nL,
		&alpha_sp, kappa_i_L_scalar1, npol * ndeltaL * nL,
		n_L_phi_scalar1, nphi,
		&beta_sp, f_i_phi_scalar1, nphi);

	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * ndeltaL * nL,
		&alpha_sp, kappa_i_L_scalar2, npol * ndeltaL * nL,
		n_L_phi_scalar2, nphi,
		&beta_sp, f_i_phi_scalar2, nphi);

	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * ndeltaL * nL,
		&alpha_sp, kappa_i_L_tensor, npol * ndeltaL * nL,
		n_L_phi_tensor, nphi,
		&beta_sp, f_i_phi_tensor, nphi);  

}

void backward_dp_mixed_sst(const long long *L_list, const long long *deltaL_list,
			  int nL, int ndeltaL, int npol, int n,
			  const double complex *a_ell_m,
			  const double *y_M_L, 
			  const double *w3j_product_scalar1, const double *w3j_product_scalar2, const double *w3j_product_tensor,
			  const double complex *prefactors_scalar1, const double complex *prefactors_scalar2, const double complex *prefactors_tensor, 
			  double complex *A_L_M_scalar1, double complex *A_L_M_scalar2, double complex *A_L_M_tensor,
			  int Lmax, int nell, int m_dim,
			  double complex *n_L_phi_scalar1, double complex *n_L_phi_scalar2, double complex *n_L_phi_tensor,
			  fftw_plan plan_c2c,
			  double *f_i_phi_scalar1, double *f_i_phi_scalar2, double *f_i_phi_tensor, 
			  int nufact, int nphi,
			  const double *kappa_i_L_scalar1, const double *kappa_i_L_scalar2,const double *kappa_i_L_tensor){

	int nm = nphi;

	compute_A_LM_dp(L_list, deltaL_list,
		      nL, ndeltaL, npol, n,
		      a_ell_m, y_M_L, w3j_product_scalar1,
		      prefactors_scalar1, A_L_M_scalar1,
		      Lmax, nell, nm);

	compute_A_LM_dp(L_list, deltaL_list,
		      nL, ndeltaL, npol, n,
		      a_ell_m, y_M_L, w3j_product_scalar2,
		      prefactors_scalar2, A_L_M_scalar2,
		      Lmax, nell, nm);

	compute_A_LM_dp(L_list, deltaL_list,
		      nL, ndeltaL, npol, n,
		      a_ell_m, y_M_L, w3j_product_tensor,
		      prefactors_tensor, A_L_M_tensor,
		      Lmax, nell, nm);

	fftw_execute_dft(plan_c2c, A_L_M_scalar1, n_L_phi_scalar1);
	fftw_execute_dft(plan_c2c, A_L_M_scalar2, n_L_phi_scalar2);
	fftw_execute_dft(plan_c2c, A_L_M_tensor, n_L_phi_tensor);

	const double complex alpha_dp = 1.0 + 0.0 * I;
	const double complex beta_dp = 0.0 + 0.0 * I;

	// f_i_L @ n_L_phi -> f_i_phi.
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * ndeltaL * nL,
		&alpha_dp, kappa_i_L_scalar1, npol * ndeltaL * nL,
		n_L_phi_scalar1, nphi,
		&beta_dp, f_i_phi_scalar1, nphi);    

	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * ndeltaL * nL,
		&alpha_dp, kappa_i_L_scalar2, npol * ndeltaL * nL,
		n_L_phi_scalar2, nphi,
		&beta_dp, f_i_phi_scalar2, nphi);

	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * ndeltaL * nL,
		&alpha_dp, kappa_i_L_tensor, npol * ndeltaL * nL,
		n_L_phi_tensor, nphi,
		&beta_dp, f_i_phi_tensor, nphi);  

}


float t_cubic_sp_sst(const float *ct_weights, const long long *rule, const float *weights,
		  const float *f_i_L, const float complex *a_ell_m,
		  const float *y_M_L, int ntheta, int nrule,
		  int ndeltaL, int nL, int npol, int m_dim, 
		  int nufact, int nphi,
		  const long long *L_list, const long long *deltaL_list, int n,
		  const float *w3j_product_scalar1, const float *w3j_product_scalar2, const float *w3j_product_tensor,
		  const float complex *prefactors_scalar1, const float complex *prefactors_scalar2, const float complex *prefactors_tensor, 
		  float complex *A_L_M_scalar1, float complex *A_L_M_scalar2, float complex *A_L_M_tensor,
		  int Lmax, int nell,
		  float complex *n_L_phi_scalar1, float complex *n_L_phi_scalar2, float complex *n_L_phi_tensor,
		  float *f_i_phi_scalar1, float *f_i_phi_scalar2, float *f_i_phi_tensor, 
		  const float *kappa_i_L_scalar1, const float *kappa_i_L_scalar2, const float *kappa_i_L_tensor){

	int nm = nphi; 
	float t_cubic = 0.f;
	int nffts[1] = {nphi};
	fftwf_plan plan_c2c;

	// Plan fft on temporary arrays now in order to avoid having to run the planner
	// in a omp critical region later.
	float complex *A_L_M = fftwf_malloc(sizeof *A_L_M * npol * ndeltaL * nL * nm);
	float complex *n_L_phi = fftwf_malloc(sizeof *n_L_phi * npol * nL * nphi);

	plan_c2c = fftwf_plan_many_dft(1, nffts, npol * ndeltaL * nL,
				       A_L_M, NULL,
				       1, nm,
				       n_L_phi, NULL,
				       1, nphi,
					   FFTW_BACKWARD,
				       FFTW_MEASURE);
	fftwf_free(A_L_M);
	fftwf_free(n_L_phi);

	#pragma omp parallel 
	{
	mkl_set_num_threads_local(1);

	float complex *A_L_M_scalar1 = fftwf_malloc(sizeof *A_L_M_scalar1 * npol * ndeltaL * nL * nm);
	float complex *n_L_phi_scalar1 = fftwf_malloc(sizeof *n_L_phi_scalar1 * npol * ndeltaL * nL * nphi);
	float *f_i_phi_scalar1 = fftwf_malloc(sizeof *f_i_phi_scalar1 * nufact * nphi);

	float complex *A_L_M_scalar2 = fftwf_malloc(sizeof *A_L_M_scalar2 * npol * ndeltaL * nL * nm);
	float complex *n_L_phi_scalar2 = fftwf_malloc(sizeof *n_L_phi_scalar2 * npol * ndeltaL * nL * nphi);
	float *f_i_phi_scalar2 = fftwf_malloc(sizeof *f_i_phi_scalar2 * nufact * nphi);

	float complex *A_L_M_tensor = fftwf_malloc(sizeof *A_L_M_tensor * npol * ndeltaL * nL * nm);
	float complex *n_L_phi_tensor = fftwf_malloc(sizeof *n_L_phi_tensor * npol * ndeltaL * nL * nphi);
	float *f_i_phi_tensor = fftwf_malloc(sizeof *f_i_phi_tensor * nufact * nphi);


	if (A_L_M_scalar1 == NULL || n_L_phi_scalar1 == NULL || f_i_phi_scalar1 == NULL ||
	    A_L_M_scalar2 == NULL || n_L_phi_scalar2 == NULL || f_i_phi_scalar2 == NULL ||
	    A_L_M_tensor == NULL || n_L_phi_tensor == NULL || f_i_phi_tensor == NULL){

	    fftwf_free(A_L_M_scalar1);
	    fftwf_free(n_L_phi_scalar1);
	    fftwf_free(f_i_phi_scalar1);

	    fftwf_free(A_L_M_scalar2);
	    fftwf_free(n_L_phi_scalar2);
	    fftwf_free(f_i_phi_scalar2);

	    fftwf_free(A_L_M_tensor);
	    fftwf_free(n_L_phi_tensor);
	    fftwf_free(f_i_phi_tensor);
	    exit(1);
		}

	#pragma omp for reduction (+:t_cubic)
	for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){

		backward_sp_mixed_sst(L_list, deltaL_list,
			  nL, ndeltaL, npol, n,
			  a_ell_m, y_M_L + tidx * nL * nL,
			  w3j_product_scalar1, w3j_product_scalar2, w3j_product_tensor,
			  prefactors_scalar1, prefactors_scalar2, prefactors_tensor,
			  A_L_M_scalar1, A_L_M_scalar2, A_L_M_tensor,
			  Lmax, nell, m_dim,
			  n_L_phi_scalar1, n_L_phi_scalar2, n_L_phi_tensor,
			  plan_c2c, f_i_phi_scalar1, f_i_phi_scalar2, f_i_phi_tensor,
			  nufact, nphi,
			  kappa_i_L_scalar1, kappa_i_L_scalar2, kappa_i_L_tensor);

		t_cubic += t_cubic_on_ring_sp_sst(rule, weights, f_i_phi_scalar1, f_i_phi_scalar2, f_i_phi_tensor, nrule, nphi)
			*ct_weights[tidx];  

	}

	fftwf_free(A_L_M_scalar1);
	fftwf_free(n_L_phi_scalar1);
	fftwf_free(f_i_phi_scalar1);

	fftwf_free(A_L_M_scalar2);
	fftwf_free(n_L_phi_scalar2);
	fftwf_free(f_i_phi_scalar2);

	fftwf_free(A_L_M_tensor);
	fftwf_free(n_L_phi_tensor);
	fftwf_free(f_i_phi_tensor);

	mkl_set_num_threads_local(0);
	} // End of parallel region

	fftwf_destroy_plan(plan_c2c);
	return t_cubic;

}


double t_cubic_dp_sst(const double *ct_weights, const long long *rule, const double *weights,
		  const double *f_i_L, const double complex *a_ell_m,
		  const double *y_M_L, int ntheta, int nrule,
		  int ndeltaL, int nL, int npol, int m_dim, 
		  int nufact, int nphi,
		  const long long *L_list, const long long *deltaL_list, int n,
		  const double *w3j_product_scalar1, const double *w3j_product_scalar2, const double *w3j_product_tensor,
		  const double complex *prefactors_scalar1, const double complex *prefactors_scalar2, const double complex *prefactors_tensor, 
		  double complex *A_L_M_scalar1, double complex *A_L_M_scalar2, double complex *A_L_M_tensor,
		  int Lmax, int nell,
		  double complex *n_L_phi_scalar1, double complex *n_L_phi_scalar2, double complex *n_L_phi_tensor,
		  double *f_i_phi_scalar1, double *f_i_phi_scalar2, double *f_i_phi_tensor, 
		  const double *kappa_i_L_scalar1, const double *kappa_i_L_scalar2, const double *kappa_i_L_tensor){

	int nm = nphi; 
	double t_cubic = 0.;
	int nffts[1] = {nphi};
	fftw_plan plan_c2c;

	// Plan fft on temporary arrays now in order to avoid having to run the planner
	// in a omp critical region later.
	double complex *A_L_M = fftw_malloc(sizeof *A_L_M * npol * ndeltaL * nL * nm);
	double complex *n_L_phi = fftw_malloc(sizeof *n_L_phi * npol * nL * nphi);

	plan_c2c = fftw_plan_many_dft(1, nffts, npol * ndeltaL * nL,
				       A_L_M, NULL,
				       1, nm,
				       n_L_phi, NULL,
				       1, nphi,
					   FFTW_BACKWARD,
				       FFTW_MEASURE);
	fftw_free(A_L_M);
	fftw_free(n_L_phi);

	#pragma omp parallel 
	{
	mkl_set_num_threads_local(1);

	double complex *A_L_M_scalar1 = fftw_malloc(sizeof *A_L_M_scalar1 * npol * ndeltaL * nL * nm);
	double complex *n_L_phi_scalar1 = fftw_malloc(sizeof *n_L_phi_scalar1 * npol * ndeltaL * nL * nphi);
	double *f_i_phi_scalar1 = fftw_malloc(sizeof *f_i_phi_scalar1 * nufact * nphi);

	double complex *A_L_M_scalar2 = fftw_malloc(sizeof *A_L_M_scalar2 * npol * ndeltaL * nL * nm);
	double complex *n_L_phi_scalar2 = fftw_malloc(sizeof *n_L_phi_scalar2 * npol * ndeltaL * nL * nphi);
	double *f_i_phi_scalar2 = fftw_malloc(sizeof *f_i_phi_scalar2 * nufact * nphi);

	double complex *A_L_M_tensor = fftw_malloc(sizeof *A_L_M_tensor * npol * ndeltaL * nL * nm);
	double complex *n_L_phi_tensor = fftw_malloc(sizeof *n_L_phi_tensor * npol * ndeltaL * nL * nphi);
	double *f_i_phi_tensor = fftw_malloc(sizeof *f_i_phi_tensor * nufact * nphi);


	if (A_L_M_scalar1 == NULL || n_L_phi_scalar1 == NULL || f_i_phi_scalar1 == NULL ||
	    A_L_M_scalar2 == NULL || n_L_phi_scalar2 == NULL || f_i_phi_scalar2 == NULL ||
	    A_L_M_tensor == NULL || n_L_phi_tensor == NULL || f_i_phi_tensor == NULL){

	    fftw_free(A_L_M_scalar1);
	    fftw_free(n_L_phi_scalar1);
	    fftw_free(f_i_phi_scalar1);

		fftw_free(A_L_M_scalar2);
		fftw_free(n_L_phi_scalar2);
		fftw_free(f_i_phi_scalar2);

	    fftw_free(A_L_M_tensor);
	    fftw_free(n_L_phi_tensor);
	    fftw_free(f_i_phi_tensor);
	    exit(1);
		}

	#pragma omp for reduction (+:t_cubic)
	for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){

		backward_dp_mixed_sst(L_list, deltaL_list,
			  nL, ndeltaL, npol, n,
			  a_ell_m, y_M_L + tidx * nL * nL,
			  w3j_product_scalar1, w3j_product_scalar2, w3j_product_tensor,
			  prefactors_scalar1, prefactors_scalar2, prefactors_tensor, 
			  A_L_M_scalar1, A_L_M_scalar2, A_L_M_tensor,
			  Lmax, nell, m_dim,
			  n_L_phi_scalar1, n_L_phi_scalar2, n_L_phi_tensor,
			  plan_c2c, f_i_phi_scalar1, f_i_phi_scalar2, f_i_phi_tensor,
			  nufact, nphi,
			  kappa_i_L_scalar1, kappa_i_L_scalar2, kappa_i_L_tensor);

		t_cubic += t_cubic_on_ring_dp_sst(rule, weights, f_i_phi_scalar1, f_i_phi_scalar2, f_i_phi_tensor, nrule, nphi)
			*ct_weights[tidx];  

	}

	fftw_free(A_L_M_scalar1);
	fftw_free(n_L_phi_scalar1);
	fftw_free(f_i_phi_scalar1);

	fftw_free(A_L_M_scalar2);
	fftw_free(n_L_phi_scalar2);
	fftw_free(f_i_phi_scalar2);

	fftw_free(A_L_M_tensor);
	fftw_free(n_L_phi_tensor);
	fftw_free(f_i_phi_tensor);

	mkl_set_num_threads_local(0);
	} // End of parallel region

	fftw_destroy_plan(plan_c2c);
	return t_cubic;

}

