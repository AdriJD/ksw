#include "common.h"
#include "hyperspherical.h"
#include "radial_functional.h"


void compute_radial_func(double const *f_k,
			 double const *tr_ell_k,
			 double *k,
			 double const *radii,
			 double *f_r_ell,
			 int *ells,
			 int nk,
			 int nell,
			 int nr,
			 int npol,
			 int ncomp){

  int kidx;
  int ridx;
  int lidx;
  int lmax;
  double kmin;
  double kmax;
  double radius;
  double *bessel_k;
  double *w_k;
  double const *tr_k;
  double *out;
  HyperInterpStruct HIS;
  ErrorMsg error_message; // Just a dummy array of char. I'm not using
                          // the error macros of CLASS.
  
  kmin = k[0];
  kmax = k[nk-1];
  lmax = ells[nell-1];

  // Allocate space for interpolated bessel function.
  bessel_k = _malloc_checked(sizeof(double) * nk);  

  // Compute trapezoidal quadrature weights.
  w_k = _malloc_checked(sizeof(double) * nk);
  _trapezoidal_weights(k, w_k, nk);

  // Multiply weights by k^2.
  for (kidx=0; kidx<nk; kidx++){
    w_k[kidx] *= pow(k[kidx], 2);
  }

  // Loop over radii.
  for (ridx=0; ridx<nr; ridx++){

    radius = radii[ridx];

    if (radius < 1e-10){
      fprintf(stderr, "radius is too small (r=%e Mpc < 1e-10 Mpc)\n", radius);
      exit(_FAILURE_);
    }

    // Create HIS object.
    int retval = hyperspherical_HIS_create(0,        // sgnK
					   radius,   // nu = r for sgnK=0.
					   nell,
					   ells,
					   kmin,     // xmin.
					   kmax,     // xmax
					   6.0,      // sampling of x.
					   lmax + 1, // l_WKB.
					   1e-20,    // phiminabs.
					   &HIS,
					   error_message);
    if (retval == _FAILURE_){
      fprintf(stderr, "Failed to init HIS object");
      exit(_FAILURE_);
    }

    // Loop over ell.
    for (lidx=0; lidx<nell; lidx++){
      
      // Interpolate bessel functions.
      retval = hyperspherical_Hermite6_interpolation_vector_Phi(&HIS,
								nk,
								lidx,
								k,
								bessel_k,
								error_message);
    
      if (retval == _FAILURE_){
	fprintf(stderr, "Interpolation failed");
	exit(_FAILURE_);
      }

      // Derive pointers to slices in input and output arrays.
      tr_k = tr_ell_k + lidx * nk * npol;
      out = f_r_ell + ridx * (nell * npol * ncomp) + lidx * (npol * ncomp);

      // Do the actual integral over k for all polarizations and input function
      // components.
      _integral_over_k(f_k, tr_k, bessel_k, w_k, out, nk, npol, ncomp);
    }
    
      // End loop over ell, free HIS object.
    retval = hyperspherical_HIS_free(&HIS, error_message);
    if (retval == _FAILURE_){
      fprintf(stderr, "Failed to free HIS object");
      exit(_FAILURE_);
    }

  // End loop over radii.
  }
  // Free weights and bessel array.  
  free(w_k);
  free(bessel_k);
}

void _integral_over_k(double const *f_k,
		      double const *tr_k,
		      double const *bessel_k,
		      double const *w_k,
		      double *out,
		      int nk,
		      int npol,
		      int ncomp){

  int kidx;
  int pidx;
  int cidx;
  double weight;
  double bessel;
  double tr;

  // Set output array to zero.
  for (pidx=0; pidx<npol; pidx++){
    for (cidx=0; cidx<ncomp; cidx++){
      out[pidx*ncomp+cidx] = 0.0;
    }
  }
  
  // Loop over wavenumber.
  for (kidx=0; kidx<nk; kidx++){

    weight = w_k[kidx];
    bessel = bessel_k[kidx];

    // Loop over polarizations.
    for (pidx=0; pidx<npol; pidx++){

      tr = tr_k[kidx*npol+pidx];

      // Loop over components in input function.
      for (cidx=0; cidx<ncomp; cidx++){

	// Multiply all and store in output array.
	out[pidx*ncomp+cidx] += weight * bessel * f_k[kidx*ncomp+cidx] * tr;
      }
    }
  }
}

void _trapezoidal_weights(double const *k,
			  double *w_k,
			  int nk){

  int kidx;

  if (nk == 1){
    fprintf(stderr, "Cannot do integral with single point.");
    exit(_FAILURE_);
  }

  // First weight.
  w_k[0] = 0.5 * (k[1] - k[0]);

  // Middle weights.
  for (kidx=1; kidx<(nk-1); kidx++){
    w_k[kidx] = 0.5 * (k[kidx+1] - k[kidx-1]);
  }

  // Last weight.
  w_k[nk-1] = 0.5 * (k[nk-1] - k[nk-2]);
}

void * _malloc_checked(size_t size){

  void * buffer = malloc(size);
  
  if (buffer == NULL){
    fprintf(stderr, "Allocation error");
    exit(_FAILURE_);
  }

  return buffer;
}
