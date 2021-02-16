#include "ksw_estimator.h"
#include "seatest.h"

void test_t_cubic_on_ring_sp(void){

    float t_cubic;
    int nrule = 2;
    int nufact = 4;
    int nphi = 4;
    float delta = 1e-6;

    int *rule = malloc(sizeof rule * nrule * 3);
    float *weights = malloc(sizeof weights * nrule * 3);
    float *f_i_phi = malloc(sizeof f_i_phi * nufact * nphi);

    rule[0] = 0;
    rule[1] = 0;
    rule[2] = 0;
    rule[3] = 0;
    rule[4] = 1;
    rule[5] = 2;

    weights[0] = 1.;
    weights[1] = 1.;
    weights[2] = 1.;
    weights[3] = 2.;
    weights[4] = 2.;
    weights[5] = 2.;

    for (int i=0; i<nufact*nphi; i++){
	f_i_phi[i] = (float) i;
    }
    
    t_cubic = t_cubic_on_ring_sp(rule, weights, f_i_phi, nrule, nphi);

    assert_float_equal(3204., t_cubic, delta);

    free(rule);
    free(weights);
    free(f_i_phi);
}

void test_backward_sp(void){

    int nufact = 4;
    int npol = 2;
    int nell = 3;
    int nphi = 5;
    int nm = nphi / 2 + 1;
    int nffts[1] = {nphi};
    float delta = 1e-6;

    float *f_i_ell = malloc(sizeof f_i_ell * nufact * npol * nell);
    complex float *a_ell_m = malloc(sizeof a_ell_m * npol * nell * nell);
    double *y_m_ell = malloc(sizeof y_m_ell * nell * nell);
    float *f_i_phi = malloc(sizeof f_i_phi * nufact * nphi);

    for (int i=0; i<npol*nell*nell; i++){
	a_ell_m[i] = 0.;
    }
    // Set ell=2, m=0 elements for both polarizations.
    a_ell_m[6] = 1.;
    a_ell_m[15] = 7.;

    for (int i=0; i<nell*nell; i++){
	y_m_ell[i] = 0.;
    }
    // Set ell=2, m=0 element.
    y_m_ell[6] = 10.;

    // Set all f_i_ells to 1.
    for (int i=0; i<nufact*npol*nell; i++){
	f_i_ell[i] = 1.;
    }

    float complex *m_ell_m = fftwf_malloc(sizeof *m_ell_m * npol * nell * nm);
    float *n_ell_phi = fftwf_malloc(sizeof *n_ell_phi * npol * nell * nphi);

    fftwf_plan plan_c2r =  fftwf_plan_many_dft_c2r(1, nffts, npol * nell,
						   m_ell_m, NULL,
						   1, nm,
						   n_ell_phi, NULL,
						   1, nphi,
						   FFTW_ESTIMATE);

    backward_sp(f_i_ell, a_ell_m, y_m_ell, m_ell_m, n_ell_phi, plan_c2r,
		f_i_phi, nell, npol, nufact, nphi);

    // for pidx=0, the fft of [10, 0, 0] should give [10, 10, 10, 10, 10]
    // for pidx=1, you get [70, 70, 70, 70, 70].
    // These are then simply added because f_i_ell is all ones.
    assert_float_equal(80., f_i_phi[0], delta);
    assert_float_equal(80., f_i_phi[1], delta);
    assert_float_equal(80., f_i_phi[2], delta);
    assert_float_equal(80., f_i_phi[3], delta);
    assert_float_equal(80., f_i_phi[4], delta);
    assert_float_equal(80., f_i_phi[5], delta);
    assert_float_equal(80., f_i_phi[6], delta);
    assert_float_equal(80., f_i_phi[7], delta);
    assert_float_equal(80., f_i_phi[8], delta);
    assert_float_equal(80., f_i_phi[9], delta);
    assert_float_equal(80., f_i_phi[10], delta);
    assert_float_equal(80., f_i_phi[11], delta);
    assert_float_equal(80., f_i_phi[12], delta);
    assert_float_equal(80., f_i_phi[13], delta);
    assert_float_equal(80., f_i_phi[14], delta);
    assert_float_equal(80., f_i_phi[15], delta);

    fftwf_destroy_plan(plan_c2r);
    free(f_i_ell);
    free(a_ell_m);
    free(y_m_ell);
    free(f_i_phi);
    fftwf_free(m_ell_m);
    fftwf_free(n_ell_phi);
}

void test_t_cubic_sp(void){

    float t_cubic;
    int nrule = 2;
    int nufact = 4;
    int npol = 2;
    int nell = 3;
    int nphi = 5;
    int ntheta = 2;    

    float delta = 1e-6;
    float exp_ans;

    float *ct_weights = malloc(sizeof ct_weights * ntheta);
    int *rule = malloc(sizeof rule * nrule * 3);
    float *weights = malloc(sizeof weights * nrule * 3);
    float *f_i_ell = malloc(sizeof f_i_ell * nufact * npol * nell);
    complex float *a_ell_m = malloc(sizeof a_ell_m * npol * nell * nell);
    double *y_m_ell = malloc(sizeof y_m_ell * ntheta * nell * nell);

    ct_weights[0] = 1.;
    ct_weights[1] = 2.;
    
    rule[0] = 0;
    rule[1] = 0;
    rule[2] = 0;
    rule[3] = 0;
    rule[4] = 1;
    rule[5] = 2;

    weights[0] = 1.;
    weights[1] = 1.;
    weights[2] = 1.;
    weights[3] = 1.;
    weights[4] = 1.;
    weights[5] = 1.;

    // Set all f_i_ells to 1.
    for (int i=0; i<nufact*npol*nell; i++){
	f_i_ell[i] = 1.;
    }

    for (int i=0; i<npol*nell*nell; i++){
	a_ell_m[i] = 0.;
    }
    // Set ell=2, m=0 elements for both polarizations.
    a_ell_m[6] = 1.;
    a_ell_m[15] = 7.;

    for (int i=0; i<ntheta*nell*nell; i++){
	y_m_ell[i] = 0.;
    }
    // Set ell=2, m=0 elements.
    y_m_ell[6] = 10.;
    y_m_ell[nell*nell+6] = 20.;


    t_cubic = t_cubic_sp(ct_weights, rule, weights, f_i_ell, a_ell_m, y_m_ell,
			 ntheta, nrule, nell, npol, nufact, nphi);

    exp_ans = 2 * 80 * 80 * 80 * 5 * (1. + 16.) * PI / 3. / (float)nphi;    
    assert_float_equal(exp_ans, t_cubic, delta);

    free(ct_weights);
    free(rule);
    free(weights);
    free(f_i_ell);
    free(a_ell_m);
    free(y_m_ell);
}

void test_get_forward_array_size(void){

    int array_size;
    int nrule = 1;
    int *rule = malloc(sizeof rule * nrule * 3);

    rule[0] = 0;
    rule[1] = 0;
    rule[2] = 0;

    array_size = get_forward_array_size(rule, nrule);
    assert_int_equal(1, array_size);

    rule[0] = 0;
    rule[1] = 0;
    rule[2] = 1;

    array_size = get_forward_array_size(rule, nrule);
    assert_int_equal(2, array_size);

    rule[0] = 0;
    rule[1] = 1;
    rule[2] = 0;

    array_size = get_forward_array_size(rule, nrule);
    assert_int_equal(2, array_size);

    rule[0] = 1;
    rule[1] = 0;
    rule[2] = 0;

    array_size = get_forward_array_size(rule, nrule);
    assert_int_equal(2, array_size);

    rule[0] = 0;
    rule[1] = 1;
    rule[2] = 2;

    array_size = get_forward_array_size(rule, nrule);
    assert_int_equal(3, array_size);

    free(rule);

    nrule = 3;
    rule = malloc(sizeof rule * nrule * 3);

    rule[0] = 0;
    rule[1] = 0;
    rule[2] = 0;

    rule[3] = 0;
    rule[4] = 0;
    rule[5] = 1;

    rule[6] = 0;
    rule[7] = 1;
    rule[8] = 2;

    array_size = get_forward_array_size(rule, nrule);
    assert_int_equal(6, array_size);

    free(rule);
}

void test_forward_sp(void){

    int nufact = 3;
    int npol = 2;
    int nell = 3;
    int nphi = 5;
    int nm = nphi / 2 + 1;
    int nrule = 3;
    int nw = 6;
    float ct_weight = 2.;

    float delta = 1e-6;

    float *f_i_ell = malloc(sizeof f_i_ell * nufact * npol * nell);
    float complex *a_ell_m = malloc(sizeof a_ell_m * npol * nell * nell);
    double *y_m_ell = malloc(sizeof y_m_ell * nell * nell);
    float complex *m_ell_m = fftwf_malloc(sizeof *m_ell_m * npol * nell * nm);
    float *n_ell_phi = fftwf_malloc(sizeof *n_ell_phi * npol * nell * nphi);
    float *f_i_phi = malloc(sizeof f_i_phi * nufact * nphi);
    float *work_i_ell = malloc(sizeof *work_i_ell * nw * npol * nell);    
    float *work_i_phi = malloc(sizeof *work_i_phi * nw * nphi);    
    int *rule = malloc(sizeof rule * nrule * 3);
    float *weights = malloc(sizeof weights * nrule * 3);

    int nffts[1] = {nphi};
    fftwf_plan plan_r2c = fftwf_plan_many_dft_r2c(1, nffts, npol * nell,
						  n_ell_phi, NULL,
						  1, nphi,
						  m_ell_m, NULL,
						  1, nm,
						  FFTW_MEASURE);

    rule[0] = 0;
    rule[1] = 0;
    rule[2] = 0;

    rule[3] = 0;
    rule[4] = 0;
    rule[5] = 1;

    rule[6] = 0;
    rule[7] = 1;
    rule[8] = 2;

    weights[0] = 1;
    weights[1] = 2;
    weights[2] = 3;

    weights[3] = 4;
    weights[4] = 5;
    weights[5] = 6;

    weights[6] = 7;
    weights[7] = 8;
    weights[8] = 9;
    
    for (int fidx=0; fidx<nufact; fidx++){
	for (int phidx=0; phidx<nphi; phidx++){	    
	    f_i_phi[fidx*nphi+phidx] = fidx + 1;
	}
    }

    for (int i=0; i<npol*nell*nell; i++){
	a_ell_m[i] = 1 + 1*I;
    }

    for (int i=0; i<nell*nell; i++){
	y_m_ell[i] = 1;
    }


    for (int fidx=0; fidx<nufact; fidx++){
	for (int pidx=0; pidx<npol; pidx++){
	    for (int lidx=0; lidx<nell; lidx++){
		f_i_ell[fidx*npol*nell+pidx*nell+lidx] = pidx + 1;
	    }
	}
    }
    	   
    forward_sp(f_i_ell, a_ell_m, y_m_ell, m_ell_m, n_ell_phi, plan_r2c, f_i_phi,
	       work_i_ell, work_i_phi,rule, weights, ct_weight, nrule, nw, nell, 
	       npol, nphi);

    // a_ell_m should have received dT/da from this ring.
    // We have an f_i_phi that is constant with phi, so only the 
    // m=0 elements should have been affected. Pol=1 should be 
    // multiplied by 2.
    assert_float_equal(1.0, creal(a_ell_m[0]) / 12906.6626, delta);
    assert_float_equal(creal(a_ell_m[1]), 1, delta);    
    assert_float_equal(creal(a_ell_m[2]), 1, delta);    
    assert_float_equal(1.0, creal(a_ell_m[3]) / 12906.6626, delta);
    assert_float_equal(creal(a_ell_m[4]), 1, delta);    
    assert_float_equal(creal(a_ell_m[5]), 1, delta);    
    assert_float_equal(1.0, creal(a_ell_m[6]) / 12906.6626, delta);
    assert_float_equal(creal(a_ell_m[7]), 1, delta);    
    assert_float_equal(creal(a_ell_m[8]), 1, delta);    
    assert_float_equal(1.0, creal(a_ell_m[9]) / 25812.3252, delta);
    assert_float_equal(creal(a_ell_m[10]), 1, delta);    
    assert_float_equal(creal(a_ell_m[11]), 1, delta);    
    assert_float_equal(1.0, creal(a_ell_m[12]) / 25812.3252, delta);
    assert_float_equal(creal(a_ell_m[13]), 1, delta);    
    assert_float_equal(creal(a_ell_m[14]), 1, delta);    
    assert_float_equal(1.0, creal(a_ell_m[15]) / 25812.3252, delta);
    assert_float_equal(creal(a_ell_m[16]), 1, delta);    
    assert_float_equal(creal(a_ell_m[17]), 1, delta);    

    assert_float_equal(cimag(a_ell_m[0]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[1]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[2]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[3]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[4]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[5]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[6]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[7]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[8]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[9]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[10]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[11]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[12]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[13]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[14]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[15]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[16]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[17]), 1, delta);    

    // Again, but now with different factors.
    for (int fidx=0; fidx<nufact; fidx++){
	for (int pidx=0; pidx<npol; pidx++){
	    for (int lidx=0; lidx<nell; lidx++){
		f_i_ell[fidx*npol*nell+pidx*nell+lidx] = fidx + 4;
	    }
	}
    }

    for (int i=0; i<npol*nell*nell; i++){
	a_ell_m[i] = 1 + 1*I;
    }
    	   
    forward_sp(f_i_ell, a_ell_m, y_m_ell, m_ell_m, n_ell_phi, plan_r2c, f_i_phi,
	       work_i_ell, work_i_phi,rule, weights, ct_weight, nrule, nw, nell, 
	       npol, nphi);

    assert_float_equal(1.0, creal(a_ell_m[0]) / 59264.0038, delta);
    assert_float_equal(creal(a_ell_m[1]), 1, delta);    
    assert_float_equal(creal(a_ell_m[2]), 1, delta);    
    assert_float_equal(1.0, creal(a_ell_m[3]) / 59264.0038, delta);
    assert_float_equal(creal(a_ell_m[4]), 1, delta);    
    assert_float_equal(creal(a_ell_m[5]), 1, delta);    
    assert_float_equal(1.0, creal(a_ell_m[6]) / 59264.0038, delta);
    assert_float_equal(creal(a_ell_m[7]), 1, delta);    
    assert_float_equal(creal(a_ell_m[8]), 1, delta);    
    assert_float_equal(1.0, creal(a_ell_m[9]) / 59264.0038, delta);
    assert_float_equal(creal(a_ell_m[10]), 1, delta);    
    assert_float_equal(creal(a_ell_m[11]), 1, delta);    
    assert_float_equal(1.0, creal(a_ell_m[12]) / 59264.0038, delta);
    assert_float_equal(creal(a_ell_m[13]), 1, delta);    
    assert_float_equal(creal(a_ell_m[14]), 1, delta);    
    assert_float_equal(1.0, creal(a_ell_m[15]) / 59264.0038, delta);
    assert_float_equal(creal(a_ell_m[16]), 1, delta);    
    assert_float_equal(creal(a_ell_m[17]), 1, delta);    
    
    assert_float_equal(cimag(a_ell_m[0]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[1]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[2]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[3]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[4]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[5]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[6]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[7]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[8]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[9]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[10]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[11]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[12]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[13]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[14]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[15]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[16]), 1, delta);    
    assert_float_equal(cimag(a_ell_m[17]), 1, delta);    

    free(f_i_ell);
    free(a_ell_m);
    free(y_m_ell);
    fftwf_free(m_ell_m);
    fftwf_free(n_ell_phi);
    free(f_i_phi);
    free(work_i_ell);
    free(work_i_phi);
    free(rule);
    free(weights);
}

void test_fixture_estimator(void){

  test_fixture_start();

  run_test(test_t_cubic_on_ring_sp);
  run_test(test_backward_sp);
  run_test(test_t_cubic_sp);
  run_test(test_get_forward_array_size);
  run_test(test_forward_sp);

  test_fixture_end();
}
