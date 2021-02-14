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
    complex float *a_m_ell = malloc(sizeof a_m_ell * npol * nell * nell);
    double *y_m_ell = malloc(sizeof y_m_ell * nell * nell);
    float *f_i_phi = malloc(sizeof f_i_phi * nufact * nphi);

    for (int i=0; i<npol*nell*nell; i++){
	a_m_ell[i] = 0.;
    }
    // Set ell=2, m=0 elements for both polarizations.
    a_m_ell[6] = 1.;
    a_m_ell[15] = 7.;

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

    backward_sp(f_i_ell, a_m_ell, y_m_ell, m_ell_m, n_ell_phi, plan_c2r,
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
    free(a_m_ell);
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
    complex float *a_m_ell = malloc(sizeof a_m_ell * npol * nell * nell);
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
	a_m_ell[i] = 0.;
    }
    // Set ell=2, m=0 elements for both polarizations.
    a_m_ell[6] = 1.;
    a_m_ell[15] = 7.;

    for (int i=0; i<ntheta*nell*nell; i++){
	y_m_ell[i] = 0.;
    }
    // Set ell=2, m=0 elements.
    y_m_ell[6] = 10.;
    y_m_ell[nell*nell+6] = 20.;


    t_cubic = t_cubic_sp(ct_weights, rule, weights, f_i_ell, a_m_ell, y_m_ell,
			 ntheta, nrule, nell, npol, nufact, nphi);

    exp_ans = 2 * 80 * 80 * 80 * 5 * (1. + 16.) * PI / 3. / (float)nphi;    
    assert_float_equal(exp_ans, t_cubic, delta);

    free(ct_weights);
    free(rule);
    free(weights);
    free(f_i_ell);
    free(a_m_ell);
    free(y_m_ell);
}

void test_fixture_estimator(void){

  test_fixture_start();

  run_test(test_t_cubic_on_ring_sp);
  run_test(test_backward_sp);
  run_test(test_t_cubic_sp);

  test_fixture_end();
}
