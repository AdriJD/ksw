cdef extern from "complex.h":
     pass

cdef extern from "ksw_estimator.h":

    float t_cubic_sp(const double *ct_weights, const long long *rule, const float *weights,
                     const float *f_i_ell, const float complex *a_ell_m,
                     const float *y_m_ell, int ntheta, int nrule,
                     int nell, int npol, int nufact, int nphi);

    double t_cubic_dp(const double *ct_weights, const long long *rule, const double *weights,
      		      const double *f_i_ell, const double complex *a_ell_m,
		      const double *y_m_ell, int ntheta, int nrule,
		      int nell, int npol, int nufact, int nphi);


    void step_sp(const double *ct_weights, const long long *rule, const float *weights,
                 const float *f_i_ell, const float complex *a_ell_m, const float *y_m_ell,
                 float complex *grad_t, int ntheta, int nrule, int nell, int npol, 
                 int nufact, int nphi);

    void step_dp(const double *ct_weights, const long long *rule, const double *weights,
 	         const double *f_i_ell, const double complex *a_ell_m, const double *y_m_ell,
	         double complex *grad_t, int ntheta, int nrule, int nell, int npol, 
	         int nufact, int nphi);

    void compute_ylm_sp(const double *thetas, float *y_m_ell, int ntheta, int lmax);

    void compute_ylm_dp(const double *thetas, double *y_m_ell, int ntheta, int lmax);