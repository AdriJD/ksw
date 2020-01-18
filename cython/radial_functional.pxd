cdef extern from "radial_functional.h":
     void compute_radial_func(const double *f_k,
			 const double *tr_ell_k,
			 double *k,
			 const double *radii,
			 double *f_r_ell,
			 int *ells,
			 int nk,
			 int nell,
			 int nr,
			 int npol,
			 int ncomp);
