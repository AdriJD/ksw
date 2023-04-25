#include <ksw_fisher_internal.h>
#include <ksw_fisher.h>
#include <seatest.h>

void test_compute_associated_legendre_sp(void){

    int ntheta = 2;
    int lmax = 2;
    int nell = lmax + 1;    
    float delta = 1e-6;

    double *thetas = malloc(sizeof *thetas * ntheta);
    float *p_theta_ell = calloc(ntheta * nell * nell, sizeof *p_theta_ell);

    thetas[0] = 0.1;
    thetas[1] = 0.5;

    compute_associated_legendre_sp(thetas, p_theta_ell, ntheta, lmax);

    assert_float_equal(1., p_theta_ell[0], delta);
    assert_float_equal(cos(thetas[0]), p_theta_ell[1], delta);
    assert_float_equal(0.5 * (3 * cos(thetas[0]) * cos(thetas[0]) - 1), p_theta_ell[2], delta);
    assert_float_equal(1., p_theta_ell[3], delta);
    assert_float_equal(cos(thetas[1]), p_theta_ell[4], delta);
    assert_float_equal(0.5 * (3 * cos(thetas[1]) * cos(thetas[1]) - 1), p_theta_ell[5], delta);
    
    free(thetas);
    free(p_theta_ell);
}

void test_unique_nxn_on_ring_sp(void){

    int nufact = 3;
    int npol = 2;
    int nell = 2;
    float delta = 1e-6;

    float *sqrt_icov_ell = calloc(nell * npol * npol, sizeof *sqrt_icov_ell);
    float *f_ell_i = calloc(nell * npol * nufact, sizeof *f_ell_i);
    float *p_ell = calloc(nell, sizeof *p_ell);
    float *prefactor = calloc(nell, sizeof *prefactor);
    float *work_i = malloc(sizeof *work_i * npol * nufact);
    float *unique_nxn = calloc(nufact * nufact, sizeof *unique_nxn);
    
    p_ell[0] = 1.0;
    p_ell[1] = -0.5;

    prefactor[0] = 1.0;
    prefactor[1] = 1.0;

    sqrt_icov_ell[0] = 10.;
    sqrt_icov_ell[1] = 3.;
    sqrt_icov_ell[2] = 3.;
    sqrt_icov_ell[3] = 20.;

    sqrt_icov_ell[4] = 20.;
    sqrt_icov_ell[5] = 6.;
    sqrt_icov_ell[6] = 6.;
    sqrt_icov_ell[7] = 40.;

    f_ell_i[0] = 2.;
    f_ell_i[1] = 2.5;
    f_ell_i[2] = 3.;
    f_ell_i[3] = 5.;
    f_ell_i[4] = -0.5;
    f_ell_i[5] = 4.;

    f_ell_i[6] = 4.;
    f_ell_i[7] = 5.;
    f_ell_i[8] = 6.;
    f_ell_i[9] = 10.;
    f_ell_i[10] = -1.;
    f_ell_i[11] = 8.;

    unique_nxn_on_ring_sp(sqrt_icov_ell, f_ell_i, p_ell, prefactor, work_i, 
			  unique_nxn, nufact, nell, npol);
        
    assert_float_equal(-87227., unique_nxn[0], delta);
    assert_float_equal(-3902.5, unique_nxn[1], delta);
    assert_float_equal(-76328., unique_nxn[2], delta);
    assert_float_equal(0., unique_nxn[3], delta);
    assert_float_equal(-3909.5, unique_nxn[4], delta);
    assert_float_equal(-5351.5, unique_nxn[5], delta);
    assert_float_equal(0., unique_nxn[6], delta);
    assert_float_equal(0., unique_nxn[7], delta);
    assert_float_equal(-67795., unique_nxn[8], delta);

    free(sqrt_icov_ell);
    free(f_ell_i);
    free(p_ell);
    free(prefactor);
    free(work_i);
    free(unique_nxn);    
}

void test_min(void){

    int a, b, c; 
    long long al, bl, cl;
    ptrdiff_t ap, bp, cp;
    
    a = 4;
    b = -5;
    c = _min(a, b);
    assert_int_equal(c, -5);

    a = 4;
    b = 5;
    c = _min(a, b);
    assert_int_equal(c, 4);

    a = 0;
    b = 0;
    c = _min(a, b);
    assert_int_equal(c, 0);

    al = 4;
    bl = -5;
    cl = _min(al, bl);
    assert_int_equal(cl, -5);

    al = 4;
    bl = 5;
    cl = _min(al, bl);
    assert_int_equal(cl, 4);

    al = 0;
    bl = 0;
    cl = _min(al, bl);
    assert_int_equal(cl, 0);

    ap = 4;
    bp = -5;
    cp = _min(ap, bp);
    assert_int_equal(cp, -5);

    ap = 4;
    bp = 5;
    cp = _min(ap, bp);
    assert_int_equal(cp, 4);

    ap = 0;
    bp = 0;
    cp = _min(ap, bp);
    assert_int_equal(cp, 0);
}

void test_max(void){

    int a, b, c; 
    long long al, bl, cl;
    ptrdiff_t ap, bp, cp;
    
    a = 4;
    b = -5;
    c = _max(a, b);
    assert_int_equal(c, 4);

    a = 4;
    b = 5;
    c = _max(a, b);
    assert_int_equal(c, 5);

    a = 0;
    b = 0;
    c = _min(a, b);
    assert_int_equal(c, 0);

    al = 4;
    bl = -5;
    cl = _max(al, bl);
    assert_int_equal(cl, 4);

    al = 4;
    bl = 5;
    cl = _max(al, bl);
    assert_int_equal(cl, 5);

    al = 0;
    bl = 0;
    cl = _max(al, bl);
    assert_int_equal(cl, 0);

    ap = 4;
    bp = -5;
    cp = _max(ap, bp);
    assert_int_equal(cp, 4);

    ap = 4;
    bp = 5;
    cp = _max(ap, bp);
    assert_int_equal(cp, 5);

    ap = 0;
    bp = 0;
    cp = _max(ap, bp);
    assert_int_equal(cp, 0);
}

void test_fisher_nxn_on_ring_sp(void){

    int nufact = 3;
    int nrule = 4;
    int npol = 2;
    int nell = 2;
    float delta = 1e-6;

    float prefac = 2 * PI * PI / 9;
    double ct_weight = 1.;

    float *unique_nxn = calloc(nufact * nufact, sizeof *unique_nxn);
    long long *rule = calloc(nrule * 3, sizeof *rule);
    float *weights = calloc(nrule * 3, sizeof *weights);
    float *fisher_nxn = calloc(nrule * nrule, sizeof *fisher_nxn);

    unique_nxn[0] = 1.;
    unique_nxn[1] = 2.;
    unique_nxn[2] = 3.;
    unique_nxn[4] = 4.;
    unique_nxn[5] = 5.;
    unique_nxn[8] = 6.;

    for (int i=0; i<nrule*3; i++){
	rule[i] = 1;
    }

    rule[0] = 0;
    rule[1] = 1;
    rule[2] = 1;

    rule[3] = 0;
    rule[4] = 1;
    rule[5] = 2;

    rule[6] = 0;
    rule[7] = 0;
    rule[8] = 1;

    rule[9] = 0;
    rule[10] = 0;
    rule[11] = 0;

    for (int i=0; i<nrule*3; i++){
	weights[i] = 1.;
    }

    weights[0] = 0.1;
    weights[1] = 0.2;
    weights[2] = 0.3;

    weights[3] = 0.5;
    weights[4] = 0.6;
    weights[5] = 0.7;

    fisher_nxn_on_ring_sp(unique_nxn, rule, weights, fisher_nxn, ct_weight, 
			  nufact, nrule);

    assert_float_equal(1.0, fisher_nxn[0] / (prefac * 96 * 0.000036), delta);
    assert_float_equal(1.0, fisher_nxn[1] / (prefac * 128 * 0.00126), delta);
    assert_float_equal(1.0, fisher_nxn[2] / (prefac * 0.288), delta);
    assert_float_equal(1.0, fisher_nxn[3] / (prefac * 0.144), delta);    
    assert_float_equal(0, fisher_nxn[4], delta);
    assert_float_equal(1.0, fisher_nxn[5] / (prefac * 7.452901), delta);    
    assert_float_equal(1.0, fisher_nxn[6] / (prefac * 14.28), delta);    
    assert_float_equal(1.0, fisher_nxn[7] / (prefac * 7.56), delta);    
    assert_float_equal(0, fisher_nxn[8], delta);
    assert_float_equal(0, fisher_nxn[9], delta);
    assert_float_equal(1.0, fisher_nxn[10] / (prefac * 24), delta);    
    assert_float_equal(1.0, fisher_nxn[11] / (prefac * 12), delta);    
    assert_float_equal(0, fisher_nxn[12], delta);
    assert_float_equal(0, fisher_nxn[13], delta);
    assert_float_equal(0, fisher_nxn[14], delta);
    assert_float_equal(1.0, fisher_nxn[15] / (prefac * 6), delta);    

    free(unique_nxn);
    free(rule);
    free(weights);
    free(fisher_nxn);
}

void test_fisher_nxn_sp(void){

    int nufact = 3;
    int nrule = 4;
    int npol = 2;
    int lmax = 1;
    int nell = lmax + 1;
    int ntheta = 2;
    float delta = 1e-6;

    float *sqrt_icov_ell = calloc(nell * npol * npol, sizeof *sqrt_icov_ell);
    float *f_ell_i = calloc(nell * npol * nufact, sizeof *f_ell_i);
    double *thetas = malloc(sizeof *thetas * ntheta);
    double *ct_weights = malloc(sizeof *ct_weights * ntheta);
    long long *rule = calloc(nrule * 3, sizeof *rule);
    float *weights = calloc(nrule * 3, sizeof *weights);
    float *fisher_nxn = calloc(nrule * nrule, sizeof *fisher_nxn);
    
    thetas[0] = 0.1;
    thetas[1] = 0.5;

    fisher_nxn_sp(sqrt_icov_ell, f_ell_i, thetas, ct_weights, rule, weights,
		  fisher_nxn, nufact, nrule, ntheta, lmax, npol);

    // No actual test here... I'll use the python wrapper for testing.
    
    assert_float_equal(0, fisher_nxn[0], delta);
    assert_float_equal(0, fisher_nxn[1], delta);
    assert_float_equal(0, fisher_nxn[2], delta);
    assert_float_equal(0, fisher_nxn[3], delta);
    assert_float_equal(0, fisher_nxn[4], delta);
    assert_float_equal(0, fisher_nxn[5], delta);
    assert_float_equal(0, fisher_nxn[6], delta);
    assert_float_equal(0, fisher_nxn[7], delta);
    assert_float_equal(0, fisher_nxn[8], delta);
    assert_float_equal(0, fisher_nxn[9], delta);
    assert_float_equal(0, fisher_nxn[10], delta);
    assert_float_equal(0, fisher_nxn[11], delta);
    assert_float_equal(0, fisher_nxn[12], delta);
    assert_float_equal(0, fisher_nxn[13], delta);
    assert_float_equal(0, fisher_nxn[14], delta);
    assert_float_equal(0, fisher_nxn[15], delta);
    
    free(sqrt_icov_ell);
    free(f_ell_i);
    free(thetas);
    free(ct_weights);
    free(rule);
    free(weights);
    free(fisher_nxn);
}

/* Double precision functions */

void test_compute_associated_legendre_dp(void){

    int ntheta = 2;
    int lmax = 2;
    int nell = lmax + 1;    
    double delta = 1e-6;

    double *thetas = malloc(sizeof *thetas * ntheta);
    double *p_theta_ell = calloc(ntheta * nell * nell, sizeof *p_theta_ell);

    thetas[0] = 0.1;
    thetas[1] = 0.5;

    compute_associated_legendre_dp(thetas, p_theta_ell, ntheta, lmax);

    assert_double_equal(1., p_theta_ell[0], delta);
    assert_double_equal(cos(thetas[0]), p_theta_ell[1], delta);
    assert_double_equal(0.5 * (3 * cos(thetas[0]) * cos(thetas[0]) - 1), p_theta_ell[2], delta);
    assert_double_equal(1., p_theta_ell[3], delta);
    assert_double_equal(cos(thetas[1]), p_theta_ell[4], delta);
    assert_double_equal(0.5 * (3 * cos(thetas[1]) * cos(thetas[1]) - 1), p_theta_ell[5], delta);
    
    free(thetas);
    free(p_theta_ell);
}

void test_unique_nxn_on_ring_dp(void){

    int nufact = 3;
    int npol = 2;
    int nell = 2;
    double delta = 1e-6;

    double *sqrt_icov_ell = calloc(nell * npol * npol, sizeof *sqrt_icov_ell);
    double *f_ell_i = calloc(nell * npol * nufact, sizeof *f_ell_i);
    double *p_ell = calloc(nell, sizeof *p_ell);
    double *prefactor = calloc(nell, sizeof *prefactor);
    double *work_i = malloc(sizeof *work_i * npol * nufact);
    double *unique_nxn = calloc(nufact * nufact, sizeof *unique_nxn);
    
    p_ell[0] = 1.0;
    p_ell[1] = -0.5;

    prefactor[0] = 1.0;
    prefactor[1] = 1.0;

    sqrt_icov_ell[0] = 10.;
    sqrt_icov_ell[1] = 3.;
    sqrt_icov_ell[2] = 3.;
    sqrt_icov_ell[3] = 20.;

    sqrt_icov_ell[4] = 20.;
    sqrt_icov_ell[5] = 6.;
    sqrt_icov_ell[6] = 6.;
    sqrt_icov_ell[7] = 40.;

    f_ell_i[0] = 2.;
    f_ell_i[1] = 2.5;
    f_ell_i[2] = 3.;
    f_ell_i[3] = 5.;
    f_ell_i[4] = -0.5;
    f_ell_i[5] = 4.;

    f_ell_i[6] = 4.;
    f_ell_i[7] = 5.;
    f_ell_i[8] = 6.;
    f_ell_i[9] = 10.;
    f_ell_i[10] = -1.;
    f_ell_i[11] = 8.;

    unique_nxn_on_ring_dp(sqrt_icov_ell, f_ell_i, p_ell, prefactor, work_i, 
			  unique_nxn, nufact, nell, npol);
        
    assert_double_equal(-87227., unique_nxn[0], delta);
    assert_double_equal(-3902.5, unique_nxn[1], delta);
    assert_double_equal(-76328., unique_nxn[2], delta);
    assert_double_equal(0., unique_nxn[3], delta);
    assert_double_equal(-3909.5, unique_nxn[4], delta);
    assert_double_equal(-5351.5, unique_nxn[5], delta);
    assert_double_equal(0., unique_nxn[6], delta);
    assert_double_equal(0., unique_nxn[7], delta);
    assert_double_equal(-67795., unique_nxn[8], delta);

    free(sqrt_icov_ell);
    free(f_ell_i);
    free(p_ell);
    free(prefactor);
    free(work_i);
    free(unique_nxn);    
}

void test_fisher_nxn_on_ring_dp(void){

    int nufact = 3;
    int nrule = 4;
    int npol = 2;
    int nell = 2;
    double delta = 1e-6;

    double prefac = 2 * PI * PI / 9;
    double ct_weight = 1.;

    double *unique_nxn = calloc(nufact * nufact, sizeof *unique_nxn);
    long long *rule = calloc(nrule * 3, sizeof *rule);
    double *weights = calloc(nrule * 3, sizeof *weights);
    double *fisher_nxn = calloc(nrule * nrule, sizeof *fisher_nxn);

    unique_nxn[0] = 1.;
    unique_nxn[1] = 2.;
    unique_nxn[2] = 3.;
    unique_nxn[4] = 4.;
    unique_nxn[5] = 5.;
    unique_nxn[8] = 6.;

    for (int i=0; i<nrule*3; i++){
	rule[i] = 1;
    }

    rule[0] = 0;
    rule[1] = 1;
    rule[2] = 1;

    rule[3] = 0;
    rule[4] = 1;
    rule[5] = 2;

    rule[6] = 0;
    rule[7] = 0;
    rule[8] = 1;

    rule[9] = 0;
    rule[10] = 0;
    rule[11] = 0;

    for (int i=0; i<nrule*3; i++){
	weights[i] = 1.;
    }

    weights[0] = 0.1;
    weights[1] = 0.2;
    weights[2] = 0.3;

    weights[3] = 0.5;
    weights[4] = 0.6;
    weights[5] = 0.7;

    fisher_nxn_on_ring_dp(unique_nxn, rule, weights, fisher_nxn, ct_weight, 
			  nufact, nrule);

    assert_double_equal(1.0, fisher_nxn[0] / (prefac * 96 * 0.000036), delta);
    assert_double_equal(1.0, fisher_nxn[1] / (prefac * 128 * 0.00126), delta);
    assert_double_equal(1.0, fisher_nxn[2] / (prefac * 0.288), delta);
    assert_double_equal(1.0, fisher_nxn[3] / (prefac * 0.144), delta);    
    assert_double_equal(0, fisher_nxn[4], delta);
    assert_double_equal(1.0, fisher_nxn[5] / (prefac * 7.452901), delta);    
    assert_double_equal(1.0, fisher_nxn[6] / (prefac * 14.28), delta);    
    assert_double_equal(1.0, fisher_nxn[7] / (prefac * 7.56), delta);    
    assert_double_equal(0, fisher_nxn[8], delta);
    assert_double_equal(0, fisher_nxn[9], delta);
    assert_double_equal(1.0, fisher_nxn[10] / (prefac * 24), delta);    
    assert_double_equal(1.0, fisher_nxn[11] / (prefac * 12), delta);    
    assert_double_equal(0, fisher_nxn[12], delta);
    assert_double_equal(0, fisher_nxn[13], delta);
    assert_double_equal(0, fisher_nxn[14], delta);
    assert_double_equal(1.0, fisher_nxn[15] / (prefac * 6), delta);    

    free(unique_nxn);
    free(rule);
    free(weights);
    free(fisher_nxn);
}

void test_fisher_nxn_dp(void){

    int nufact = 3;
    int nrule = 4;
    int npol = 2;
    int lmax = 1;
    int nell = lmax + 1;
    int ntheta = 2;
    double delta = 1e-6;

    double *sqrt_icov_ell = calloc(nell * npol * npol, sizeof *sqrt_icov_ell);
    double *f_ell_i = calloc(nell * npol * nufact, sizeof *f_ell_i);
    double *thetas = malloc(sizeof *thetas * ntheta);
    double *ct_weights = malloc(sizeof *ct_weights * ntheta);
    long long *rule = calloc(nrule * 3, sizeof *rule);
    double *weights = calloc(nrule * 3, sizeof *weights);
    double *fisher_nxn = calloc(nrule * nrule, sizeof *fisher_nxn);
    
    thetas[0] = 0.1;
    thetas[1] = 0.5;

    fisher_nxn_dp(sqrt_icov_ell, f_ell_i, thetas, ct_weights, rule, weights,
		  fisher_nxn, nufact, nrule, ntheta, lmax, npol);

    // No actual test here... I'll use the python wrapper for testing.
    
    assert_double_equal(0, fisher_nxn[0], delta);
    assert_double_equal(0, fisher_nxn[1], delta);
    assert_double_equal(0, fisher_nxn[2], delta);
    assert_double_equal(0, fisher_nxn[3], delta);
    assert_double_equal(0, fisher_nxn[4], delta);
    assert_double_equal(0, fisher_nxn[5], delta);
    assert_double_equal(0, fisher_nxn[6], delta);
    assert_double_equal(0, fisher_nxn[7], delta);
    assert_double_equal(0, fisher_nxn[8], delta);
    assert_double_equal(0, fisher_nxn[9], delta);
    assert_double_equal(0, fisher_nxn[10], delta);
    assert_double_equal(0, fisher_nxn[11], delta);
    assert_double_equal(0, fisher_nxn[12], delta);
    assert_double_equal(0, fisher_nxn[13], delta);
    assert_double_equal(0, fisher_nxn[14], delta);
    assert_double_equal(0, fisher_nxn[15], delta);
    
    free(sqrt_icov_ell);
    free(f_ell_i);
    free(thetas);
    free(ct_weights);
    free(rule);
    free(weights);
    free(fisher_nxn);
}

void test_fixture_fisher(void){

  test_fixture_start();

  run_test(test_min);
  run_test(test_max);

  run_test(test_compute_associated_legendre_sp);
  run_test(test_unique_nxn_on_ring_sp);
  run_test(test_fisher_nxn_on_ring_sp);
  run_test(test_fisher_nxn_sp);

  run_test(test_compute_associated_legendre_dp);
  run_test(test_unique_nxn_on_ring_dp);
  run_test(test_fisher_nxn_on_ring_dp);
  run_test(test_fisher_nxn_dp);

  test_fixture_end();
}
