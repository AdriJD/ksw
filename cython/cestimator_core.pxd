cdef extern from "complex.h":
	pass

cdef extern from "ksw_estimator.h":

	# Basic cubic and step functions
	float t_cubic_sp(const double *ct_weights, const long long *rule, const float *weights,
		 const float *f_i_ell, const float complex *a_ell_m,
		 const float *y_m_ell, int ntheta, int nrule,
		 int nell, int npol, int nufact, int nphi)

	double t_cubic_dp(const double *ct_weights, const long long *rule, const double *weights,
		 const double *f_i_ell, const double complex *a_ell_m,
		 const double *y_m_ell, int ntheta, int nrule,
		 int nell, int npol, int nufact, int nphi)

	void step_sp(const double *ct_weights, const long long *rule, const float *weights,
		 const float *f_i_ell, const float complex *a_ell_m, const float *y_m_ell,
		 float complex *grad_t, int ntheta, int nrule, int nell, int npol,
		 int nufact, int nphi)

	void step_dp(const double *ct_weights, const long long *rule, const double *weights,
		 const double *f_i_ell, const double complex *a_ell_m, const double *y_m_ell,
		 double complex *grad_t, int ntheta, int nrule, int nell, int npol,
		 int nufact, int nphi)

	void step_sst_sp(const int *L_list,
			  int nL, int npol, 
			  int n_scalar1, int n_scalar2, int n_tensor, int n_scalar,   
			  float complex *a_L_M_scalar, float complex *a_L_M_tensor,
			  const float complex *a_ell_m,
			  const float *y_M_L,
			  const float *w3j_product_scalar1, const float *w3j_product_scalar2, const float *w3j_product_tensor,
			  const float complex *prefactors_scalar1, const float complex *prefactors_scalar2, const float complex *prefactors_tensor, 
			  float complex *A_L_M_scalar1, float complex *A_L_M_scalar2, float complex *A_L_M_tensor,
			  int Lmax, int nell,
			  int nufact, int nphi,
			  const float complex *kappa_i_L_scalar1, const float complex *kappa_i_L_scalar2, const float complex *kappa_i_L_tensor,
			  float complex *grad_t_zeta, float complex *grad_t_h,
			  const long long *rule, const float *weights, const float ct_weight,
			  const float w3j, int nrule, int ntheta);

	void step_sst_dp(const int *L_list,
			  int nL, int npol, 
			  int n_scalar1, int n_scalar2, int n_tensor, int n_scalar,   
			  double complex *a_L_M_scalar, double complex *a_L_M_tensor,
			  const double complex *a_ell_m,
			  const double *y_M_L,
			  const double *w3j_product_scalar1, const double *w3j_product_scalar2, const double *w3j_product_tensor,
			  const double complex *prefactors_scalar1, const double complex *prefactors_scalar2, const double complex *prefactors_tensor, 
			  double complex *A_L_M_scalar1, double complex *A_L_M_scalar2, double complex *A_L_M_tensor,
			  int Lmax, int nell,
			  int nufact, int nphi,
			  const double complex *kappa_i_L_scalar1, const double complex *kappa_i_L_scalar2, const double complex *kappa_i_L_tensor,
			  double complex *grad_t_zeta, double complex *grad_t_h,
			  const long long *rule, const double *weights, const double ct_weight,
			  const double w3j, int nrule, int ntheta);


	# ylm computation
	void compute_ylm_sp(const double *thetas, float *y_m_ell, int ntheta, int lmax)

	void compute_ylm_dp(const double *thetas, double *y_m_ell, int ntheta, int lmax)

	# Cubic terms for scalar-scalar-tensor bispectra
	float t_cubic_sp_sst(const float *ct_weights, const long long *rule, const float *weights,
		 const float complex *a_ell_m,
		 const float *y_M_L, int ntheta, int nrule,
		 int nL, int npol,
		 int nufact, int nphi,
		 const int *L_list,
		 int n_scalar1, int n_scalar2, int n_tensor,
		 const float *w3j_product_scalar1, const float *w3j_product_scalar2, const float *w3j_product_tensor,
		 const float complex *prefactors_scalar1, const float complex *prefactors_scalar2, const float complex *prefactors_tensor,
		 int Lmax, int nell,
		 const float complex *kappa_i_L_scalar1, const float complex *kappa_i_L_scalar2, const float complex *kappa_i_L_tensor)

	double t_cubic_dp_sst(const double *ct_weights, const long long *rule, const double *weights,
		 const double complex *a_ell_m,
		 const double *y_M_L, int ntheta, int nrule,
		 int nL, int npol, 
		 int nufact, int nphi,
		 const int *L_list,
		 int n_scalar1, int n_scalar2, int n_tensor,
		 const double *w3j_product_scalar1, const double *w3j_product_scalar2, const double *w3j_product_tensor,
		 const double complex *prefactors_scalar1, const double complex *prefactors_scalar2, const double complex *prefactors_tensor,
		 int Lmax, int nell,
		 const double complex *kappa_i_L_scalar1, const double complex *kappa_i_L_scalar2, const double complex *kappa_i_L_tensor)
