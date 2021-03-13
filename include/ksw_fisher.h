
// sqrt_icov_ell (nell, npol, npol) array symmetric
// f_ell_i (nell, npol, nufact)
// thetas : (ntheta) array thetas.
// ct_weights
// rule (nrule, 3)
// weights (nrule, 3)
// fisher_nxn (nrule, nrule) output array
// nufact
// nrule
// ntheta : number of thetas.
// lmax   : Maximum multipole.
// npol

void fisher_nxn_sp(const float *sqrt_icov_ell, const float *f_ell_i, const double *thetas,
		   const double *ct_weights, const long long *rule, const float *weights, 
		   float *fisher_nxn, int nufact, int nrule, int ntheta, int lmax, int npol);

void fisher_nxn_dp(const double *sqrt_icov_ell, const double *f_ell_i, const double *thetas,
		   const double *ct_weights, const long long *rule, const double *weights, 
		   double *fisher_nxn, int nufact, int nrule, int ntheta, int lmax, int npol);
