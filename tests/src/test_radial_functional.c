#include "common.h"
#include "radial_functional.h"
#include "seatest.h"

void test_compute_radial_func(void){
  
  // Use the following analytic relation to test the integral:
  //
  // int_0^inf j_ell(x) dx = sqrt(pi) G((ell+1) / 2) / (2 G(1 + ell/2)).
  // 
  // See e.g. 1703.06428. G denotes the Gamma function. When we substitue
  // x = kr, we have to multiply the LHS by 1/r.

  int npol = 2;
  int ncomp = 3;
  int nell = 2;
  int nr = 2;
  int nk = 10000;  
  int kidx;
  int cidx;
  int lidx;
  int pidx;
  int ridx;
  double kmin = 1.0e-7;
  double kmax = 1.0;
  double kstep = (kmax - kmin) / (double) nk;
  double delta;
  double exp_ans;
  
  double * k = malloc(nk * sizeof(double));
  double * f_k = malloc(nk * ncomp * sizeof(double));
  double * tr_ell_k = malloc(nell * nk * npol * sizeof(double));
  double * radii = malloc(nr * sizeof(double));
  double * f_r_ell = malloc(nr * nell * npol * ncomp * sizeof(double));
  double * exp_answers = malloc(nr * nell * sizeof(double));
  int * ells = malloc(nell * sizeof(int));
  
  if (k == NULL || f_k == NULL || tr_ell_k == NULL ||
      radii == NULL || f_r_ell == NULL || ells == NULL){
    // Make test fail.
    assert_true(1 == 0);
    return;
  }

  // Test function for these multipoles and radii.
  ells[0] = 2;
  ells[1] = 10;

  radii[0] = 11000.0;
  radii[1] = 14000.0;

  // Calculated using above analytic relation.
  exp_answers[0] = 7.139983303613165e-05; // r=11000, ell=2.
  exp_answers[1] = 3.514210532247105e-05; // r=11000, ell=10.
  exp_answers[2] = 5.609986881410344e-05; // r=14000, ell=2.
  exp_answers[3] = 2.761165418194154e-05; // r=14000, ell=10
  
  // Initialize linearly spaced k array.
  for (kidx=0; kidx<nk; kidx++){
    k[kidx] = kmin + kstep * (double) kidx;
  }
  
  // Initialize other input arrays.
  for (kidx=0; kidx<nk; kidx++){

    for(cidx=0; cidx<ncomp; cidx++){
      // Use k^-2 (pi/2) input function to cancel k^2 (2/pi)
      // weighting in the integral.
      f_k[kidx*ncomp+cidx] = 0.5 * pow(k[kidx], -2) * _PI_;
    }    
  }

  // Set all transfer functions to one.
  for (lidx=0; lidx<nell; lidx++){
    for (kidx=0; kidx<nk; kidx++){
      for (pidx=0; pidx<npol; pidx++){
	
	tr_ell_k[lidx*nk*npol+kidx*npol+pidx] = 1.0;
      }
    }
  }

  compute_radial_func(f_k, tr_ell_k, k, radii, f_r_ell, ells,
  		      nk, nell, nr, npol, ncomp);

  // Because we made f_k and tr_k the same for all polarizations and components
  // the answer should only depend on r and ell.
  
  delta = 3e-4; // Relative error. Integral is not perfect, but this is good
                // enough for testing.
  
  for (ridx=0; ridx<nr; ridx++){
    for (lidx=0; lidx<nell; lidx++){
      
      exp_ans = exp_answers[ridx*nell+lidx];
	
      for (pidx=0; pidx<npol; pidx++){
	for (cidx=0; cidx<ncomp; cidx++){

	  assert_double_equal(1.0, 
	    exp_ans / f_r_ell[ridx*nell*npol*ncomp+lidx*npol*ncomp+pidx*ncomp+cidx],
	    delta);
	}
      }
    }
  }
  
  free(k);
  free(f_k);
  free(tr_ell_k);
  free(radii);
  free(f_r_ell);
  free(ells);
  free(exp_answers);
}

void test_trapezoidal_weights(void){

  int n = 5;
  int kidx;
  double delta = 1e-14;

  double * k = malloc(n * sizeof(double));
  double * w_k = malloc(n * sizeof(double));
  if (k == NULL || w_k == NULL){
    // Make test fail.
    assert_true(1 == 0);
    return;
  }

  k[0] = 4.;
  k[1] = 6.;
  k[2] = 8.;
  k[3] = 10.;
  k[4] = 20.;

  double exp_w_k[5] = {1., 2., 2., 6., 5.};

  _trapezoidal_weights(k, w_k, n);

  for (kidx=0; kidx<n; kidx++){
    assert_double_equal(exp_w_k[kidx], w_k[kidx], delta);
  }

  free(k);
  free(w_k);
}

void test_integral_over_k(void){

  int nk = 10;
  int ncomp = 3;
  int npol = 2;
  int kidx;
  int pidx;
  int cidx;
  int i;
  double delta = 1e-30;

  double * k = malloc(nk * sizeof(double));
  double * w_k = malloc(nk * sizeof(double));
  double * f_k = malloc(nk * ncomp * sizeof(double));
  double * tr_k = malloc(nk * npol * sizeof(double));
  double * bessel_k = malloc(nk * sizeof(double));
  double * out = malloc(npol * ncomp * sizeof(double));
  
  if (k == NULL || w_k == NULL || f_k == NULL || tr_k == NULL ||
      bessel_k == NULL || out == NULL){
    // Make test fail.
    assert_true(1 == 0);
    return;
  }
  
  k[0] = 1.0e-7;
  k[1] = 1.2e-6;
  k[2] = 2.3e-6;
  k[3] = 3.4e-6;
  k[4] = 4.5e-6;
  k[5] = 5.6e-6;
  k[6] = 6.7e-6;
  k[7] = 7.8e-6;
  k[8] = 8.9e-6;
  k[9] = 1.0e-5;

  w_k[0] = 5.5e-7;
  w_k[1] = 1.1e-6;
  w_k[2] = 1.1e-6;
  w_k[3] = 1.1e-6;
  w_k[4] = 1.1e-6;
  w_k[5] = 1.1e-6;
  w_k[6] = 1.1e-6;
  w_k[7] = 1.1e-6;
  w_k[8] = 1.1e-6;
  w_k[9] = 5.5e-7;

  // Multiply weights by k^2.
  for (kidx=0; kidx<nk; kidx++){
    w_k[kidx] *= pow(k[kidx], 2);
  }

  // Init arrays with simple values for which I manually calculated the answer.
  for (kidx=0; kidx<nk; kidx++){

    for(pidx=0; pidx<npol; pidx++){
      tr_k[kidx*npol+pidx] = 1. + (double) pidx;
    }

    for(cidx=0; cidx<ncomp; cidx++){
      f_k[kidx*ncomp+cidx] = 1 + (double) cidx;
    }

    bessel_k[kidx] = 1.0;
  }

  double * exp_out = malloc(npol * ncomp * sizeof(double));
  if (exp_out == NULL){
    // Make test fail.
    assert_true(1 == 0);
    return;
  }
  // Manually calculated answers.
  exp_out[0] = 3.3532950e-16;
  exp_out[1] = 6.7065900e-16;
  exp_out[2] = 1.0059885e-15;
  exp_out[3] = 6.7065900e-16;
  exp_out[4] = 1.3413180e-15;
  exp_out[5] = 2.0119770e-15;
  
  // Perform the actual integral we want to test.
  _integral_over_k(f_k, tr_k, bessel_k, w_k, out, nk, npol, ncomp);

  for (i=0; i<6; i++){
    assert_double_equal(exp_out[i], out[i], delta);
  }

  free(k);
  free(w_k);
  free(tr_k);
  free(bessel_k);
  free(f_k);
  free(out);
  free(exp_out);
}

void test_malloc_checked(void){

  int n = 2;
  double delta = 1e-14;
  double * arr = _malloc_checked(n * sizeof(double));

  assert_true(arr != NULL);

  arr[0] = 1.;
  arr[1] = 2.;

  assert_double_equal(1., arr[0], delta);
  assert_double_equal(2., arr[1], delta);

  free(arr);
}

void test_fixture_radial_functional(void){

  test_fixture_start();

  run_test(test_compute_radial_func);
  run_test(test_trapezoidal_weights);
  run_test(test_integral_over_k);
  run_test(test_malloc_checked);

  test_fixture_end();
}
