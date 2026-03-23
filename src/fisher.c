#include <ksw_fisher_internal.h>

inline ptrdiff_t _max(ptrdiff_t a, ptrdiff_t b){
    return((a) > (b) ? a : b);
}

inline ptrdiff_t _min(ptrdiff_t a, ptrdiff_t b){
    return((a) < (b) ? a : b);
}

void compute_associated_legendre_sp(const double *thetas, float *p_theta_ell,
				    int ntheta, int lmax){

    int nell = lmax + 1;
    double epsilon = 1e-300;

    #pragma omp parallel
    {
    Ylmgen_C ygen;

    Ylmgen_init(&ygen, lmax, 0, 0, 0, epsilon);
    Ylmgen_set_theta(&ygen, thetas, ntheta);

    #pragma omp for schedule(dynamic, 5)
    for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){
	
	Ylmgen_prepare(&ygen, tidx, 0);
	Ylmgen_recalc_Ylm(&ygen);

	ptrdiff_t firstl = *ygen.firstl;

	for (ptrdiff_t lidx=firstl; lidx<nell; lidx++){
		
	    // Convert from SH to associated Legendre.
	    p_theta_ell[tidx*nell+lidx] = (float) (ygen.ylm[lidx] 
	                           * sqrt(4 * PI / (double) (2 * lidx + 1)));
        }
    }

    Ylmgen_destroy(&ygen);
    } // End of parallel region.
}

void unique_nxn_on_ring_sp(const float *sqrt_icov_ell, const float *f_ell_i, const float *p_ell, 
			   const float *prefactor, float *work_i, float *unique_nxn, int nufact,
			   int nell, int npol){
    
    // Set output to zero.
    for (ptrdiff_t i=0; i<nufact*nufact; i++){
	unique_nxn[i] = 0.0;
    }

    // More accurate to loop through ell backwards for ISW-phi-like templates.
    for (ptrdiff_t lidx=0; lidx<nell; lidx++){
	//for (ptrdiff_t lidx=nell-1; lidx>=0; lidx--){		

	// sqrt_icov @ f_i -> work_i.
       cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper, npol, nufact,
		    1.0, sqrt_icov_ell + lidx * npol * npol, npol,
		    f_ell_i + lidx * npol * nufact, nufact, 0.0, work_i, nufact);
       
	// P * work^T x work -> unique_nxn.
       cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, nufact, npol,
	    p_ell[lidx] * prefactor[lidx], work_i, nufact, 1.0, unique_nxn, nufact);
    }
}

void fisher_nxn_on_ring_sp(const float *unique_nxn, const long long *rule, 
			   const float *weights, float *fisher_nxn, double ct_weight, 
			   int nufact, int nrule){

    for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

	long long rx = rule[ridx*3];
	long long ry = rule[ridx*3+1];
	long long rz = rule[ridx*3+2];

	float wx = weights[ridx*3];
	float wy = weights[ridx*3+1];
	float wz = weights[ridx*3+2];

	// We only fill upper triangular part.
	for (ptrdiff_t rjdx=ridx; rjdx<nrule; rjdx++){
	
	    long long rpx = rule[rjdx*3];
	    long long rpy = rule[rjdx*3+1];
	    long long rpz = rule[rjdx*3+2];

	    float wpx = weights[rjdx*3];
	    float wpy = weights[rjdx*3+1];
	    float wpz = weights[rjdx*3+2];
	    
	    // r and rp are indices into unique_nxn. Min/max to only acces uppper tri part.
	    float tmp_arr[6];
	    float tmp = 0.0;
	    
	    tmp_arr[0] = unique_nxn[_min(rx, rpx)*nufact+_max(rx, rpx)]
		* unique_nxn[_min(ry, rpy)*nufact+_max(ry, rpy)]
		* unique_nxn[_min(rz, rpz)*nufact+_max(rz, rpz)];
	    
	    // + 5 permutations.
	    tmp_arr[1] = unique_nxn[_min(rx, rpz)*nufact+_max(rx, rpz)]
		 * unique_nxn[_min(ry, rpx)*nufact+_max(ry, rpx)]
		 * unique_nxn[_min(rz, rpy)*nufact+_max(rz, rpy)];

	    tmp_arr[2] = unique_nxn[_min(rx, rpy)*nufact+_max(rx, rpy)]
		 * unique_nxn[_min(ry, rpz)*nufact+_max(ry, rpz)]
		 * unique_nxn[_min(rz, rpx)*nufact+_max(rz, rpx)];

	    tmp_arr[3] = unique_nxn[_min(rx, rpx)*nufact+_max(rx, rpx)]
		 * unique_nxn[_min(ry, rpz)*nufact+_max(ry, rpz)]
		 * unique_nxn[_min(rz, rpy)*nufact+_max(rz, rpy)];

	    tmp_arr[4] = unique_nxn[_min(rx, rpy)*nufact+_max(rx, rpy)]
		 * unique_nxn[_min(ry, rpx)*nufact+_max(ry, rpx)]
		 * unique_nxn[_min(rz, rpz)*nufact+_max(rz, rpz)];

	    tmp_arr[5] = unique_nxn[_min(rx, rpz)*nufact+_max(rx, rpz)]
		 * unique_nxn[_min(ry, rpy)*nufact+_max(ry, rpy)]
		 * unique_nxn[_min(rz, rpx)*nufact+_max(rz, rpx)];

	    // Use compensated summation to increase accuracy. Doesn't seem
	    // very important here, but can't hurt I guess.
	    float comp = 0.0;
	    for (ptrdiff_t idx=0; idx<6; idx++){
		
		float t = tmp + tmp_arr[idx];
		if (fabs(tmp) >= fabs(tmp_arr[idx])) {
		    comp += (tmp - t) + tmp_arr[idx];
		}
		else {
		    comp += (tmp_arr[idx] - t) + tmp;
		}
		tmp = t;
	    }
	    tmp += comp;
	    
	    fisher_nxn[ridx*nrule+rjdx] += tmp * wx * wy * wz
		* wpx * wpy * wpz * (float) (ct_weight * 2 * PI * PI / 9);
	}
    }
}

void fisher_nxn_sp(const float *sqrt_icov_ell, const float *f_ell_i, const double *thetas,
		   const double *ct_weights, const long long *rule, const float *weights, 
		   float *fisher_nxn, int nufact, int nrule, int ntheta, int lmax, int npol){

    int nell = lmax + 1;

    float *p_theta_ell = malloc(sizeof *p_theta_ell * ntheta * nell);
    float *prefactor = malloc(sizeof *prefactor * nell);

    if (p_theta_ell == NULL || prefactor == NULL){
	free(p_theta_ell);
	free(prefactor);
	exit(1);
    }
    
    for (ptrdiff_t lidx=0; lidx<nell; lidx++){
	prefactor[lidx] = (2 * lidx + 1) / 4. / PI;
    }

    compute_associated_legendre_sp(thetas, p_theta_ell, ntheta, lmax);
    
    #pragma omp parallel 
    {
    mkl_set_num_threads_local(1);

    float *work_i = malloc(sizeof *work_i * npol * nufact);
    float *unique_nxn = malloc(sizeof *unique_nxn * nufact * nufact);
    float *fisher_nxn_priv = calloc(nrule * nrule, sizeof *fisher_nxn_priv);
    float *fisher_nxn_priv_tmp = calloc(nrule * nrule, sizeof *fisher_nxn_priv);  
    float *comp = calloc(nrule * nrule, sizeof *fisher_nxn_priv);
    
    if (work_i == NULL || unique_nxn == NULL || fisher_nxn_priv == NULL || fisher_nxn_priv_tmp == NULL || comp == NULL){
	free(work_i);
	free(unique_nxn);
	free(fisher_nxn_priv);
	free(fisher_nxn_priv_tmp);
	free(comp);
	exit(1);
    }
    
    #pragma omp for schedule(dynamic)
    for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){

	unique_nxn_on_ring_sp(sqrt_icov_ell, f_ell_i, p_theta_ell + tidx * nell, 
	    prefactor, work_i, unique_nxn, nufact, nell, npol);

	// Set fisher_nxn_priv_tmp to zero. Needed for compensated summation later on.
	for (ptrdiff_t idx=0; idx<nrule; idx++){
           for (ptrdiff_t jdx=idx; jdx<nrule; jdx++){
               fisher_nxn_priv_tmp[idx*nrule+jdx] = 0.0;
           }
       }
       
	fisher_nxn_on_ring_sp(unique_nxn, rule, weights, fisher_nxn_priv_tmp,
			      ct_weights[tidx], nufact, nrule);

       // Now fisher_nxn_priv_tmp contains only the contribution from this ring.
       // Add this to fisher_nxn_priv using compensated summation for extra accuracy.
	for (ptrdiff_t idx=0; idx<nrule; idx++){
	    for (ptrdiff_t jdx=idx; jdx<nrule; jdx++){
               float t = fisher_nxn_priv[idx*nrule+jdx] + fisher_nxn_priv_tmp[idx*nrule+jdx];
               if (fabs(fisher_nxn_priv[idx*nrule+jdx]) >= fabs(fisher_nxn_priv_tmp[idx*nrule+jdx])){
                   comp[idx*nrule+jdx] += (fisher_nxn_priv[idx*nrule+jdx] - t) + fisher_nxn_priv_tmp[idx*nrule+jdx];
               }
               else {
                   comp[idx*nrule+jdx] += (fisher_nxn_priv_tmp[idx*nrule+jdx] - t) + fisher_nxn_priv[idx*nrule+jdx];
               }
               fisher_nxn_priv[idx*nrule+jdx] = t;
           }
       }
    }
    
    for (ptrdiff_t idx=0; idx<nrule; idx++){
	for (ptrdiff_t jdx=idx; jdx<nrule; jdx++){
	    fisher_nxn_priv[idx*nrule+jdx] += comp[idx*nrule+jdx];
	}
    }

    #pragma omp critical
    {
    for (ptrdiff_t idx=0; idx<nrule; idx++){
	for (ptrdiff_t jdx=idx; jdx<nrule; jdx++){
	    fisher_nxn[idx*nrule+jdx] += fisher_nxn_priv[idx*nrule+jdx];
        }
    }	
    }

    free(work_i);
    free(unique_nxn);
    free(fisher_nxn_priv);
    free(fisher_nxn_priv_tmp);
    free(comp);
    
    mkl_set_num_threads_local(0);
    } // End of parallel region

    free(p_theta_ell);
    free(prefactor);
}

/* Double precision versions */

void compute_associated_legendre_dp(const double *thetas, double *p_theta_ell,
				    int ntheta, int lmax){

    int nell = lmax + 1;
    double epsilon = 1e-300;

    #pragma omp parallel
    {
    Ylmgen_C ygen;

    Ylmgen_init(&ygen, lmax, 0, 0, 0, epsilon);
    Ylmgen_set_theta(&ygen, thetas, ntheta);

    #pragma omp for schedule(dynamic, 5)
    for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){
	
	Ylmgen_prepare(&ygen, tidx, 0);
	Ylmgen_recalc_Ylm(&ygen);

	ptrdiff_t firstl = *ygen.firstl;

	for (ptrdiff_t lidx=firstl; lidx<nell; lidx++){
		
	    // Convert from SH to associated Legendre.
	    p_theta_ell[tidx*nell+lidx] = (ygen.ylm[lidx] 
	                           * sqrt(4 * PI / (double) (2 * lidx + 1)));
        }
    }

    Ylmgen_destroy(&ygen);
    } // End of parallel region.
}

void unique_nxn_on_ring_dp(const double *sqrt_icov_ell, const double *f_ell_i, const double *p_ell, 
			   const double *prefactor, double *work_i, double *unique_nxn, int nufact,
			   int nell, int npol){
    
    // Set output to zero.
    for (ptrdiff_t i=0; i<nufact*nufact; i++){
	unique_nxn[i] = 0.0;
    }

    for (ptrdiff_t lidx=0; lidx<nell; lidx++){    
	//for (ptrdiff_t lidx=nell-1; lidx>=0; lidx--){	
	
	// sqrt_icov @ f_i -> work_i.
	cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, npol, nufact,
		    1.0, sqrt_icov_ell + lidx * npol * npol, npol,
		    f_ell_i + lidx * npol * nufact, nufact, 0.0, work_i, nufact);

	// P * work^T @ work -> unique_nxn.
	cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, nufact, npol,
	    p_ell[lidx] * prefactor[lidx], work_i, nufact, 1.0, unique_nxn, nufact);
    }
}

void unique_nxn_on_ring_batched_dp(const double *sqrt_icov_ell, const double *f_ell_i, const double *p_ell, 
				   const double *prefactor, double *work_i, double *unique_nxn, int nufact,
				   int nell, int npol){

    const int blocksize = 32;
    const size_t msize = (size_t)nufact * nufact;

    // Global long double accumulator.
    //long double *unique_ld = malloc(sizeof(long double) * msize);
    long double *unique_ld_even = calloc(msize, sizeof(long double));
    long double *unique_ld_odd = calloc(msize, sizeof(long double));    
    
    //if (!unique_ld) exit(1);
    
    /* for (ptrdiff_t i=0; i<msize; i++){ */
    /* 	unique_ld[i] = 0.0; */
    /* } */

    // Temporary buffers
    double *tmp = malloc(sizeof(double) * msize);
    
    //long double *block_ld = malloc(sizeof(long double) * msize);
    long double *block_ld_odd = malloc(sizeof(long double) * msize);        
    long double *block_ld_even = malloc(sizeof(long double) * msize);
    
    //if (!tmp || !block_ld) exit(1);
    
    // Loop over ℓ in blocks
    for (int l0 = 0; l0 < nell; l0 += blocksize) {

	int lend = (l0 + blocksize < nell) ? l0 + blocksize : nell;

	// Reset block accumulator
        for (size_t i = 0; i < msize; i++){
            //block_ld[i] = 0.0L;
            block_ld_even[i] = 0.0L;
            block_ld_odd[i] = 0.0L;		
	}
	// Process ℓ within block
        //for (int lidx = l0; lidx < lend; lidx++) {
	//for (int lidx = l0; lidx < lend; lidx += 2) {
	
	for (int lidx = l0; lidx < lend; lidx++) {

	    long double *block_target = (lidx % 2 == 0) ? block_ld_even : block_ld_odd;
	    
            // work_i = sqrt_icov @ f.
            cblas_dsymm(
                CblasRowMajor, CblasLeft, CblasUpper,
                npol, nufact,
                1.0,
                sqrt_icov_ell + lidx * npol * npol, npol,
                f_ell_i + lidx * npol * nufact, nufact,
                0.0,
                work_i, nufact
            );
		    
            // Zero tmp matrix.
            //memset(tmp, 0, sizeof(double) * msize);
	    for (size_t i = 0; i < msize; i++){
		tmp[i] = 0.0;
	    }

	    // tmp = alpha * work^T work.
            double alpha = p_ell[lidx] * prefactor[lidx];

	    //printf("%d %.10e\n", lidx, (double)alpha);
	    
            cblas_dsyrk(
                CblasRowMajor, CblasUpper, CblasTrans,
                nufact, npol,
                alpha,
                work_i, nufact,
                0.0,
                tmp, nufact
            );
	    
            // Accumulate this ℓ into block (in long double).
	    for (size_t i = 0; i < msize; i++){
                //block_ld[i] += (long double)tmp[i];
                block_target[i] += (long double)tmp[i];
	    }
	    //printf("%d %.10e\n", lidx, (double)block_target[0]);
	}

        for (size_t i = 0; i < msize; i++) {
            unique_ld_even[i] += block_ld_even[i];
            unique_ld_odd[i]  += block_ld_odd[i];
        }
	
        // Add completed block into global accumulator
	//for (size_t i = 0; i < msize; i++){
        //    unique_ld[i] += block_ld[i];
	//}

        // ODD
        /* for (int lidx = l0 + 1; lidx < lend; lidx += 2) { */

        /*     // work_i = sqrt_icov @ f. */
        /*     cblas_dsymm( */
        /*         CblasRowMajor, CblasLeft, CblasUpper, */
        /*         npol, nufact, */
        /*         1.0, */
        /*         sqrt_icov_ell + lidx * npol * npol, npol, */
        /*         f_ell_i + lidx * npol * nufact, nufact, */
        /*         0.0, */
        /*         work_i, nufact */
        /*     ); */
		    
        /*     // Zero tmp matrix. */
        /*     //memset(tmp, 0, sizeof(double) * msize); */
	/*     for (size_t i = 0; i < msize; i++){ */
	/* 	tmp[i] = 0.0; */
	/*     } */

	/*     // tmp = alpha * work^T work. */
        /*     double alpha = p_ell[lidx] * prefactor[lidx]; */

	/*     //printf("%d %.10e\n", lidx, (double)alpha); */
	    
        /*     cblas_dsyrk( */
        /*         CblasRowMajor, CblasUpper, CblasTrans, */
        /*         nufact, npol, */
        /*         alpha, */
        /*         work_i, nufact, */
        /*         0.0, */
        /*         tmp, nufact */
        /*     ); */
	    
        /*     // Accumulate this ℓ into block (in long double). */
	/*     for (size_t i = 0; i < msize; i++){ */
        /*         block_ld[i] += (long double)tmp[i]; */
	/*     } */
	/* } */
        /* // Add completed block into global accumulator */
	/* for (size_t i = 0; i < msize; i++){ */
        /*     unique_ld[i] += block_ld[i]; */
	/* } */

	
    }


    /* long double sum_even = 0.0L, sum_odd = 0.0L, sum_total = 0.0L; */
    /* for (size_t i = 0; i < msize; i++) { */
    /* 	sum_even += fabsl(unique_ld_even[i]); */
    /* 	sum_odd  += fabsl(unique_ld_odd[i]); */
    /* 	sum_total += fabsl(unique_ld_even[i] + unique_ld_odd[i]); */
    /* } */
    /* printf("unique_nxn cancellation ratio: %.6e  (even=%.6e odd=%.6e result=%.6e)\n", */
    /* 	   (double)((sum_even + sum_odd) / sum_total), */
    /* 	   (double)sum_even, (double)sum_odd, (double)sum_total); */

    
    for (size_t i = 0; i < msize; i++){
        unique_nxn[i] = (double)(unique_ld_even[i] + unique_ld_odd[i]);
    }

    
    
    
    // Cast back to double
    //for (size_t i = 0; i < msize; i++)
    //    unique_nxn[i] = (double)unique_ld[i];

    free(tmp);
    //free(block_ld);
    free(block_ld_even);
    free(block_ld_odd);    
    //free(unique_ld);
    free(unique_ld_even);
    free(unique_ld_odd);        
}

//void fisher_nxn_on_ring_dp(const double *unique_nxn, const long long *rule, 
//			   const double *weights, long double *fisher_nxn, double ct_weight, 
//			   int nufact, int nrule){
void fisher_nxn_on_ring_dp(const double *unique_nxn, const long long *rule, 
			   const double *weights, long double *fisher_nxn_pos,  long double *fisher_nxn_neg,
			   double ct_weight, 
			   int nufact, int nrule){

    for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

	long long rx = rule[ridx*3];
	long long ry = rule[ridx*3+1];
	long long rz = rule[ridx*3+2];

	double wx = weights[ridx*3];
	double wy = weights[ridx*3+1];
	double wz = weights[ridx*3+2];

	// We only fill upper triangular part.
	for (ptrdiff_t rjdx=ridx; rjdx<nrule; rjdx++){
	
	    long long rpx = rule[rjdx*3];
	    long long rpy = rule[rjdx*3+1];
	    long long rpz = rule[rjdx*3+2];

	    double wpx = weights[rjdx*3];
	    double wpy = weights[rjdx*3+1];
	    double wpz = weights[rjdx*3+2];
	    
	    // r and rp are indices into unique_nxn. Min/max to only acces uppper tri part.
	    //double tmp_arr[6];
	    long double tmp_arr[6];	    
	    //double tmp = 0.0;
	    long double tmp = 0.0;	    

	    /* tmp_arr[0] = unique_nxn[_min(rx, rpx)*nufact+_max(rx, rpx)] */
	    /* 	* unique_nxn[_min(ry, rpy)*nufact+_max(ry, rpy)] */
	    /* 	* unique_nxn[_min(rz, rpz)*nufact+_max(rz, rpz)]; */
	    	    
	    /* //+ 5 permutations. */
	    /* tmp_arr[1] = unique_nxn[_min(rx, rpz)*nufact+_max(rx, rpz)] */
	    /* 	 * unique_nxn[_min(ry, rpx)*nufact+_max(ry, rpx)] */
	    /* 	 * unique_nxn[_min(rz, rpy)*nufact+_max(rz, rpy)]; */

	    /* tmp_arr[2] = unique_nxn[_min(rx, rpy)*nufact+_max(rx, rpy)] */
	    /* 	 * unique_nxn[_min(ry, rpz)*nufact+_max(ry, rpz)] */
	    /* 	 * unique_nxn[_min(rz, rpx)*nufact+_max(rz, rpx)]; */

	    /* tmp_arr[3] = unique_nxn[_min(rx, rpx)*nufact+_max(rx, rpx)] */
	    /* 	 * unique_nxn[_min(ry, rpz)*nufact+_max(ry, rpz)] */
	    /* 	 * unique_nxn[_min(rz, rpy)*nufact+_max(rz, rpy)]; */

	    /* tmp_arr[4] = unique_nxn[_min(rx, rpy)*nufact+_max(rx, rpy)] */
	    /* 	 * unique_nxn[_min(ry, rpx)*nufact+_max(ry, rpx)] */
	    /* 	 * unique_nxn[_min(rz, rpz)*nufact+_max(rz, rpz)]; */

	    /* tmp_arr[5] = unique_nxn[_min(rx, rpz)*nufact+_max(rx, rpz)] */
	    /* 	 * unique_nxn[_min(ry, rpy)*nufact+_max(ry, rpy)] */
	    /* 	 * unique_nxn[_min(rz, rpx)*nufact+_max(rz, rpx)]; */



	    
	    tmp_arr[0] = (long double) unique_nxn[_min(rx, rpx)*nufact+_max(rx, rpx)]
		* (long double) unique_nxn[_min(ry, rpy)*nufact+_max(ry, rpy)]
		* (long double) unique_nxn[_min(rz, rpz)*nufact+_max(rz, rpz)];
	    	    
	    //+ 5 permutations.
	    tmp_arr[1] = (long double) unique_nxn[_min(rx, rpz)*nufact+_max(rx, rpz)]
		 * (long double) unique_nxn[_min(ry, rpx)*nufact+_max(ry, rpx)]
		 * (long double) unique_nxn[_min(rz, rpy)*nufact+_max(rz, rpy)];

	    tmp_arr[2] = (long double) unique_nxn[_min(rx, rpy)*nufact+_max(rx, rpy)]
		 * (long double) unique_nxn[_min(ry, rpz)*nufact+_max(ry, rpz)]
		 * (long double) unique_nxn[_min(rz, rpx)*nufact+_max(rz, rpx)];

	    tmp_arr[3] = (long double) unique_nxn[_min(rx, rpx)*nufact+_max(rx, rpx)]
		 * (long double) unique_nxn[_min(ry, rpz)*nufact+_max(ry, rpz)]
		 * (long double) unique_nxn[_min(rz, rpy)*nufact+_max(rz, rpy)];

	    tmp_arr[4] = (long double) unique_nxn[_min(rx, rpy)*nufact+_max(rx, rpy)]
		 * (long double) unique_nxn[_min(ry, rpx)*nufact+_max(ry, rpx)]
		 * (long double) unique_nxn[_min(rz, rpz)*nufact+_max(rz, rpz)];

	    tmp_arr[5] = (long double) unique_nxn[_min(rx, rpz)*nufact+_max(rx, rpz)]
		 * (long double) unique_nxn[_min(ry, rpy)*nufact+_max(ry, rpy)]
		 * (long double) unique_nxn[_min(rz, rpx)*nufact+_max(rz, rpx)];





	    
	    //for (ptrdiff_t idx=0; idx<6; idx++){
	    //printf("%.10e\n", tmp_arr[idx]);
	    //}
	    //printf("\n");


	    /* long double pos_sum = 0.0L, neg_sum = 0.0L; */
	    /* for (int idx = 0; idx < 6; idx++) { */
	    /* 	if (tmp_arr[idx] >= 0) pos_sum += tmp_arr[idx]; */
	    /* 	else                   neg_sum -= tmp_arr[idx];  // make positive */
	    /* } */
	    /* long double result = pos_sum - neg_sum; */
	    /* if (fabsl(result) > 0 && (pos_sum + neg_sum) / fabsl(result) > 1.) { */
	    /* 	printf("ring cancellation ratio: %.6e\n", */
	    /* 	       (double)((pos_sum + neg_sum) / fabsl(result))); */
	    /* } */

	    
	    // Use compensated summation to increase accuracy. Doesn't seem
	    // very important here, but can't hurt I guess.	    
	    //double comp = 0.0;
	    long double comp = 0.0;	    
	    for (ptrdiff_t idx=0; idx<6; idx++){
		
		//double t = tmp + tmp_arr[idx];
		long double t = tmp + tmp_arr[idx];		
		//if (abs(tmp) >= abs(tmp_arr[idx])) {
		if (fabsl(tmp) >= fabsl(tmp_arr[idx])) {		    
		    comp += (tmp - t) + tmp_arr[idx];
		}
		else {
		    comp += (tmp_arr[idx] - t) + tmp;
		}
		tmp = t;
	    }
	    tmp += comp;
	    
	    //fisher_nxn[ridx*nrule+rjdx] += tmp * wx * wy * wz
	    //	* wpx * wpy * wpz * (ct_weight * 2 * PI * PI / 9);
	    //printf("%.10e\n", (double) tmp);
	    if (tmp >= 0){
		fisher_nxn_pos[ridx*nrule+rjdx] += tmp * wx * wy * wz
			* wpx * wpy * wpz * (ct_weight * 2 * PI * PI / 9);
	    }
	    else {
		fisher_nxn_neg[ridx*nrule+rjdx] += tmp * wx * wy * wz
			* wpx * wpy * wpz * (ct_weight * 2 * PI * PI / 9);		
	    }
	    
	    //if (ridx*nrule+rjdx == 8){
	    //printf("tmp_nxn[3,3] = %.10e \n", tmp * wx * wy * wz
	    //	       * wpx * wpy * wpz * (ct_weight * 2 * PI * PI / 9));
	    //}
	}
    }
    //printf("fisher_nxn_pos[3,3] = %.10e \n", (double)fisher_nxn_pos[8]);
    //printf("fisher_nxn_neg[3,3] = %.10e \n", (double)fisher_nxn_neg[8]);    
}

void fisher_nxn_dp(const double *sqrt_icov_ell, const double *f_ell_i, const double *thetas,
		   const double *ct_weights, const long long *rule, const double *weights, 
		   double *fisher_nxn, int nufact, int nrule, int ntheta, int lmax, int npol){

    int nell = lmax + 1;

    double *p_theta_ell = malloc(sizeof *p_theta_ell * ntheta * nell);
    double *prefactor = malloc(sizeof *prefactor * nell);

    if (p_theta_ell == NULL || prefactor == NULL){
	free(p_theta_ell);
	free(prefactor);
	exit(1);
    }
    
    for (ptrdiff_t lidx=0; lidx<nell; lidx++){
	prefactor[lidx] = (2 * lidx + 1) / 4. / PI;
    }

    compute_associated_legendre_dp(thetas, p_theta_ell, ntheta, lmax);

    #pragma omp parallel 
    {
    mkl_set_num_threads_local(1);

    double *work_i = malloc(sizeof *work_i * npol * nufact);
    double *unique_nxn = malloc(sizeof *unique_nxn * nufact * nufact);
    //double *fisher_nxn_priv = calloc(nrule * nrule, sizeof *fisher_nxn_priv);
    //long double *fisher_nxn_priv = calloc(nrule * nrule, sizeof *fisher_nxn_priv);
    long double *fisher_nxn_priv_pos = calloc(nrule * nrule, sizeof *fisher_nxn_priv_pos);
    long double *fisher_nxn_priv_neg = calloc(nrule * nrule, sizeof *fisher_nxn_priv_neg);    

    //if (work_i == NULL || unique_nxn == NULL || fisher_nxn_priv == NULL){
    if (work_i == NULL || unique_nxn == NULL || fisher_nxn_priv_pos == NULL || fisher_nxn_priv_neg == NULL){	
	free(work_i);
	free(unique_nxn);
	//free(fisher_nxn_priv);
	free(fisher_nxn_priv_pos);
	free(fisher_nxn_priv_neg);		
	exit(1);
    }


    //NOTE
    long double ring_pos_total = 0.0L, ring_neg_total = 0.0L;    
    
    #pragma omp for schedule(dynamic)
    
    for (ptrdiff_t tidx=ntheta-1; tidx>=0; tidx--){	

	//unique_nxn_on_ring_dp(sqrt_icov_ell, f_ell_i, p_theta_ell + tidx * nell, 
	// prefactor, work_i, unique_nxn, nufact, nell, npol);
	unique_nxn_on_ring_batched_dp(sqrt_icov_ell, f_ell_i, p_theta_ell + tidx * nell, 
				      prefactor, work_i, unique_nxn, nufact, nell, npol);

	//fisher_nxn_on_ring_dp(unique_nxn, rule, weights, fisher_nxn_priv,
	//		      ct_weights[tidx], nufact, nrule);
	fisher_nxn_on_ring_dp(unique_nxn, rule, weights, fisher_nxn_priv_pos, fisher_nxn_priv_neg,
			      ct_weights[tidx], nufact, nrule);


	// NOTE
	ring_pos_total += fisher_nxn_priv_pos[0];  // track element [0,0] as proxy
	ring_neg_total += fisher_nxn_priv_neg[0];
	
    }

    printf("theta-sum cancellation [0,0]: pos=%.6e neg=%.6e ratio=%.6e\n",
	   (double)ring_pos_total, (double)ring_neg_total,
	   (double)fabsl(ring_pos_total + ring_neg_total) > 0 ?
	   (double)((fabsl(ring_pos_total) + fabsl(ring_neg_total)) /
		    fabsl(ring_pos_total + ring_neg_total)) : 0.0);
    
    #pragma omp critical
    {
    for (ptrdiff_t idx=0; idx<nrule; idx++){
	for (ptrdiff_t jdx=idx; jdx<nrule; jdx++){
	    //printf("%ld %ld %.10e\n", idx, jdx, fisher_nxn_priv[idx*nrule+jdx]);
	    //fisher_nxn[idx*nrule+jdx] += (double)fisher_nxn_priv[idx*nrule+jdx];
	    fisher_nxn[idx*nrule+jdx] += (double)fisher_nxn_priv_pos[idx*nrule+jdx];
	    fisher_nxn[idx*nrule+jdx] += (double)fisher_nxn_priv_neg[idx*nrule+jdx];	    	    
        }
    }	
    }

    free(work_i);
    free(unique_nxn);
    //free(fisher_nxn_priv);
    free(fisher_nxn_priv_pos);
    free(fisher_nxn_priv_neg);    

    mkl_set_num_threads_local(0);
    } // End of parallel region

    free(p_theta_ell);
    free(prefactor);
}
