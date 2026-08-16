#include "ksw_estimator_internal.h"
#include <ksw_estimator.h>
#include <stdlib.h>
#include <complex.h>

int get_forward_array_size(const long long *rule, int nrule){

    int array_size = 0;

    for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

	long long rx = rule[ridx*3];
	long long ry = rule[ridx*3+1];
	long long rz = rule[ridx*3+2];

	array_size += 1;
	
	if (ry != rx){
	    array_size += 1;
	}
	if (ry != rz && rx != rz){
	    array_size += 1;
	}
    }
    return array_size;
}

/* Single precision versions */

float t_cubic_on_ring_sp(const long long *rule, const float *weights, const float *f_i_phi,
			 int nrule, int nphi){

    float t_cubic = 0.0;

    for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

	long long rx = rule[ridx*3];
	long long ry = rule[ridx*3+1];
	long long rz = rule[ridx*3+2];

	float wx = weights[ridx*3];
	float wy = weights[ridx*3+1];
	float wz = weights[ridx*3+2];

	for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
	    t_cubic += wx * wy * wz * f_i_phi[rx*nphi+phidx] * f_i_phi[ry*nphi+phidx]
		       * f_i_phi[rz*nphi+phidx];
	}
    }
    return t_cubic;
}

void backward_sp(const float *f_i_ell, const float complex *a_ell_m, 
		 const float *y_m_ell, float complex *m_ell_m, float *n_ell_phi,
		 fftwf_plan plan_c2r, float *f_i_phi, int nell, int npol, int nufact,
		 int nphi){

    int nm = nphi / 2 + 1;

    // Place alm * Ylm into Mlm.
    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
	for (ptrdiff_t lidx=0; lidx<nell; lidx++){
	    for (ptrdiff_t midx=0; midx<nm; midx++){

		complex float tmp;

		if (midx < nell){
		    tmp = a_ell_m[pidx*nell*nell+lidx*nell+midx]
			  * y_m_ell[midx*nell+lidx];
		} else{
		    // Needed because m_ell_m array can be larger than alm.
		    tmp = 0. + 0.*I;
		}

		m_ell_m[pidx*nell*nm+lidx*nm+midx] = tmp;
	    }
	}
    }

    // Backward fft. Note that fftw has no normalization for backward
    // or forward. Unlike numpy and pyfftw which apply 1/nphi during 
    // backward (c2r). So in python version I multiply the result of 
    // the fft by nphi to compensate for that factor. Here it's not needed.
    fftwf_execute_dft_c2r(plan_c2r, m_ell_m, n_ell_phi);

    // f_i_ell @ n_ell_phi -> f_i_phi.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * nell,
		1.0, f_i_ell, npol * nell,
		n_ell_phi, nphi,
		0.0, f_i_phi, nphi);    
}

void forward_sp(const float *f_i_ell, float complex *a_ell_m, const float *y_m_ell,
		float complex *m_ell_m, float *n_ell_phi, fftwf_plan plan_r2c,
		const float *f_i_phi, float *work_i_ell, float *work_i_phi,
		const long long *rule, const float *weights, const double ct_weight,
		int nrule, int nw, int nell, int npol, int nphi){

    int widx = 0; // Index to work arrays.
    int nm = nphi / 2 + 1;

    for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

	long long rx = rule[ridx*3];
	long long ry = rule[ridx*3+1];
	long long rz = rule[ridx*3+2];

	float weight = weights[ridx*3] * weights[ridx*3+1] * weights[ridx*3+2]
	    * PI * ct_weight / 3. / (double) nphi;

	// Fill work arrays.
	if (rx == ry && rx == rz){ // Case: 000.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = 3. * weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    widx += 1;
	} else if (rx == ry && ry != rz){ // Case: 001.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = 2. * weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rz*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[ry*nphi+phidx];
	    }
	    widx += 2;
	} else if (rx != ry && ry == rz){ // Case: 100.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = 2. * weight
			* f_i_ell[ry*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    widx += 2;
	} else if (rx == rz && rx != ry){ // Case 010.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = 2. * weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[ry*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    widx += 2;
	} else { // Case: 012.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[ry*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+2)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rz*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+2)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[ry*nphi+phidx];
	    }
	    widx += 3;
	}
    }

    if (widx != nw){
	fprintf(stderr, "widx (%d) != nw (%d) \n", widx, nw);
	exit(1);
    }

    // Sum_i X_i_ell dT/dX_i_phi + Y_i_ell dT/dY_i_phi + Z_i_ell dT/dZ_i_phi -> n_ell_phi.
    // Implemented as work_i_ell.T @ work_i_phi -> n_ell_phi.
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		npol * nell, nphi, nw,
		1.0, work_i_ell, npol * nell,
		work_i_phi, nphi,
		0.0, n_ell_phi, nphi);    

    fftwf_execute_dft_r2c(plan_r2c, n_ell_phi, m_ell_m);

    // Multiply by ylm and add result to alm.
    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
	for (ptrdiff_t lidx=0; lidx<nell; lidx++){
	    for (ptrdiff_t midx=0; midx<=lidx; midx++){
    
		a_ell_m[pidx*nell*nell+lidx*nell+midx] += y_m_ell[midx*nell+lidx] 
		    * m_ell_m[pidx*nell*nm+lidx*nm+midx];

	    }
	}
    }        
}

void forward_sst_sp(const int *L_list,
			  int nL, int npol, 
			  int n_scalar1, int n_scalar2, int n_tensor, int n_scalar,   
			  float complex *a_L_M_scalar, float complex *a_L_M_tensor,
			  const float *y_M_L,
			  const float *w3j_product_scalar1, const float *w3j_product_scalar2, const float *w3j_product_tensor,
			  const float complex *prefactors_scalar, const float complex *prefactors_tensor, 
			  int Lmax, int nell,
			  fftwf_plan plan_c2c_scalar, fftwf_plan plan_c2c_tensor,
			  float complex *f_i_phi_scalar1, float complex *f_i_phi_scalar2, float complex *f_i_phi_tensor, 
			  int nufact, int nphi,
			  const float complex *kappa_i_L_scalar1, const float complex *kappa_i_L_scalar2, const float complex *kappa_i_L_tensor,
			  float complex *work_i_L_scalar1, float complex *work_i_L_scalar2, float complex *work_i_L_tensor,
			  float complex *work_i_phi_scalar1, float complex *work_i_phi_scalar2, float complex *work_i_phi_tensor,
			  float complex *n_L_phi_scalar1, float complex *n_L_phi_scalar2, float complex *n_L_phi_scalar, float complex *n_L_phi_tensor,
			  float complex *m_L_M_scalar, float complex *m_L_M_tensor,
			  const long long *rule, const float *weights, const float ct_weight,
			  const float w3j, int nrule, int nw){

			int ndeltaL_scalar = 2;                               // deltaL = -1, 1 for scalar
			int ndeltaL_tensor = 5;                               // deltaL = -2, -1, 0, 1, 2 for tensor

			//const int deltaL_list_scalar[] = {-1, 1};             //scalar
			//const int deltaL_list_tensor[] = {-2, -1, 0, 1, 2};	  //tensor
					
			int widx_scalar1 = 0; // Index to work arrays.
			int widx_scalar2 = 0; // Index to work arrays.
			int widx_tensor = 0; // Index to work arrays.

			int nM = nphi;
			for (ptrdiff_t ridx=0; ridx<nrule; ridx++){
			
			long long rx = rule[ridx*3];
			long long ry = rule[ridx*3+1];
			long long rz = rule[ridx*3+2];

			float weight = weights[ridx*3] * weights[ridx*3+1] * weights[ridx*3+2]
			    * sqrt(2) * w3j * ct_weight / 54 / nphi;

			// Fill work arrays.
			for (ptrdiff_t pidx=0; pidx<npol; pidx++){
				for (ptrdiff_t deltaLidx=0; deltaLidx<ndeltaL_scalar; deltaLidx++){
					for (ptrdiff_t Lidx=0; Lidx<nL; Lidx++){
						work_i_L_scalar1[widx_scalar1*npol*ndeltaL_scalar*nL+pidx*ndeltaL_scalar*nL+deltaLidx*nL+Lidx] = weight
						* kappa_i_L_scalar1[rx*npol*ndeltaL_scalar*nL+pidx*ndeltaL_scalar*nL+deltaLidx*nL+Lidx];
					}
	   		}
			}
			for (ptrdiff_t pidx=0; pidx<npol; pidx++){
				for (ptrdiff_t deltaLidx=0; deltaLidx<ndeltaL_scalar; deltaLidx++){
					for (ptrdiff_t Lidx=0; Lidx<nL; Lidx++){
						work_i_L_scalar1[(widx_scalar1+1)*npol*ndeltaL_scalar*nL+pidx*ndeltaL_scalar*nL+deltaLidx*nL+Lidx] = weight
						* kappa_i_L_scalar1[ry*npol*ndeltaL_scalar*nL+pidx*ndeltaL_scalar*nL+deltaLidx*nL+Lidx];
					}
	   		}
			}
			for (ptrdiff_t pidx=0; pidx<npol; pidx++){
				for (ptrdiff_t deltaLidx=0; deltaLidx<ndeltaL_scalar; deltaLidx++){
					for (ptrdiff_t Lidx=0; Lidx<nL; Lidx++){
						work_i_L_scalar1[(widx_scalar1+2)*npol*ndeltaL_scalar*nL+pidx*ndeltaL_scalar*nL+deltaLidx*nL+Lidx] = weight
						* kappa_i_L_scalar1[rz*npol*ndeltaL_scalar*nL+pidx*ndeltaL_scalar*nL+deltaLidx*nL+Lidx];
					}
	   		}
			}
			for (ptrdiff_t pidx=0; pidx<npol; pidx++){

				for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
				work_i_phi_scalar1[widx_scalar1*nphi+phidx] = f_i_phi_scalar2[ry*nphi+phidx] * f_i_phi_tensor[rz*nphi+phidx];
				}
				for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
				work_i_phi_scalar1[(widx_scalar1+1)*nphi+phidx] = f_i_phi_scalar2[rx*nphi+phidx] * f_i_phi_tensor[rz*nphi+phidx];
				}
				for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
				work_i_phi_scalar1[(widx_scalar1+2)*nphi+phidx] = f_i_phi_scalar2[rx*nphi+phidx] * f_i_phi_tensor[ry*nphi+phidx];
				}
				widx_scalar1 += 3;
			}

			for (ptrdiff_t pidx=0; pidx<npol; pidx++){
				for (ptrdiff_t deltaLidx=0; deltaLidx<ndeltaL_scalar; deltaLidx++){
					for (ptrdiff_t Lidx=0; Lidx<nL; Lidx++){
						work_i_L_scalar2[widx_scalar2*npol*ndeltaL_scalar*nL+pidx*ndeltaL_scalar*nL+deltaLidx*nL+Lidx] = weight
						* kappa_i_L_scalar2[rx*npol*ndeltaL_scalar*nL+pidx*ndeltaL_scalar*nL+deltaLidx*nL+Lidx];
					}
	   		}
			}
			for (ptrdiff_t pidx=0; pidx<npol; pidx++){
				for (ptrdiff_t deltaLidx=0; deltaLidx<ndeltaL_scalar; deltaLidx++){
					for (ptrdiff_t Lidx=0; Lidx<nL; Lidx++){
						work_i_L_scalar2[(widx_scalar2+1)*npol*ndeltaL_scalar*nL+pidx*ndeltaL_scalar*nL+deltaLidx*nL+Lidx] = weight
						* kappa_i_L_scalar2[ry*npol*ndeltaL_scalar*nL+pidx*ndeltaL_scalar*nL+deltaLidx*nL+Lidx];
					}
	   		}
			}
			for (ptrdiff_t pidx=0; pidx<npol; pidx++){
				for (ptrdiff_t deltaLidx=0; deltaLidx<ndeltaL_scalar; deltaLidx++){
					for (ptrdiff_t Lidx=0; Lidx<nL; Lidx++){
						work_i_L_scalar2[(widx_scalar2+2)*npol*ndeltaL_scalar*nL+pidx*ndeltaL_scalar*nL+deltaLidx*nL+Lidx] = weight
						* kappa_i_L_scalar2[rz*npol*ndeltaL_scalar*nL+pidx*ndeltaL_scalar*nL+deltaLidx*nL+Lidx];
					}
	   		}
			}
			for (ptrdiff_t pidx=0; pidx<npol; pidx++){

				for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
				work_i_phi_scalar2[widx_scalar2*nphi+phidx] = f_i_phi_scalar1[ry*nphi+phidx] * f_i_phi_tensor[rz*nphi+phidx];
				}
				for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
				work_i_phi_scalar2[(widx_scalar2+1)*nphi+phidx] = f_i_phi_scalar1[rx*nphi+phidx] * f_i_phi_tensor[rz*nphi+phidx];
				}
				for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
				work_i_phi_scalar2[(widx_scalar2+2)*nphi+phidx] = f_i_phi_scalar1[rx*nphi+phidx] * f_i_phi_tensor[ry*nphi+phidx];
				}
				widx_scalar2 += 3;
			}

			for (ptrdiff_t pidx=0; pidx<npol; pidx++){
				for (ptrdiff_t deltaLidx=0; deltaLidx<ndeltaL_tensor; deltaLidx++){
					for (ptrdiff_t Lidx=0; Lidx<nL; Lidx++){
						work_i_L_tensor[widx_tensor*npol*ndeltaL_tensor*nL+pidx*ndeltaL_tensor*nL+deltaLidx*nL+Lidx] = weight
						* kappa_i_L_tensor[rx*npol*ndeltaL_tensor*nL+pidx*ndeltaL_tensor*nL+deltaLidx*nL+Lidx];
					}
	   		}
			}
			for (ptrdiff_t pidx=0; pidx<npol; pidx++){
				for (ptrdiff_t deltaLidx=0; deltaLidx<ndeltaL_tensor; deltaLidx++){
					for (ptrdiff_t Lidx=0; Lidx<nL; Lidx++){
						work_i_L_tensor[(widx_tensor+1)*npol*ndeltaL_tensor*nL+pidx*ndeltaL_tensor*nL+deltaLidx*nL+Lidx] = weight
						* kappa_i_L_tensor[ry*npol*ndeltaL_tensor*nL+pidx*ndeltaL_tensor*nL+deltaLidx*nL+Lidx];
					}
	   		}
			}
			for (ptrdiff_t pidx=0; pidx<npol; pidx++){
				for (ptrdiff_t deltaLidx=0; deltaLidx<ndeltaL_tensor; deltaLidx++){
					for (ptrdiff_t Lidx=0; Lidx<nL; Lidx++){
						work_i_L_tensor[(widx_tensor+2)*npol*ndeltaL_tensor*nL+pidx*ndeltaL_tensor*nL+deltaLidx*nL+Lidx] = weight
						* kappa_i_L_tensor[rz*npol*ndeltaL_tensor*nL+pidx*ndeltaL_tensor*nL+deltaLidx*nL+Lidx];
					}
	   		}
			}
			for (ptrdiff_t pidx=0; pidx<npol; pidx++){

				for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
				work_i_phi_tensor[widx_tensor*nphi+phidx] = f_i_phi_scalar1[ry*nphi+phidx] * f_i_phi_scalar2[rz*nphi+phidx];
				}
				for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
				work_i_phi_tensor[(widx_tensor+1)*nphi+phidx] = f_i_phi_scalar1[rx*nphi+phidx] * f_i_phi_scalar2[rz*nphi+phidx];
				}
				for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
				work_i_phi_tensor[(widx_tensor+2)*nphi+phidx] = f_i_phi_scalar1[rx*nphi+phidx] * f_i_phi_scalar2[ry*nphi+phidx];
				}
				widx_tensor += 3;
			}
			}

			const float complex alpha_sp = 1.0f + 0.0f * I;
			const float complex beta_sp = 0.0f + 0.0f * I;

			cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				npol * ndeltaL_scalar * nL, nphi, nw,
				&alpha_sp, work_i_L_scalar1, npol * ndeltaL_scalar * nL,
				work_i_phi_scalar1, nphi, 			
				&beta_sp, n_L_phi_scalar1, nphi);

			cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				npol * ndeltaL_scalar * nL, nphi, nw,
				&alpha_sp, work_i_L_scalar2, npol * ndeltaL_scalar * nL,
				work_i_phi_scalar2, nphi, 			
				&beta_sp, n_L_phi_scalar2, nphi);

			cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				npol * ndeltaL_tensor * nL, nphi, nw,
				&alpha_sp, work_i_L_tensor, npol * ndeltaL_tensor * nL,
				work_i_phi_tensor, nphi, 			
				&beta_sp, n_L_phi_tensor, nphi);

			long long size_scalar = npol * ndeltaL_scalar * nL * nphi;

			if (n_scalar1==n_scalar2){
				for (ptrdiff_t i = 0; i < size_scalar; i++) {
        			n_L_phi_scalar[i] = n_L_phi_scalar1[i] + n_L_phi_scalar2[i];
				}

			} else if (n_scalar==n_scalar1 && n_scalar!=n_scalar2){
				for (ptrdiff_t i = 0; i < size_scalar; i++) {
					n_L_phi_scalar[i] = n_L_phi_scalar1[i];
				}
			 } else if (n_scalar!=n_scalar1 && n_scalar==n_scalar2){
						for (ptrdiff_t i = 0; i < size_scalar; i++) {
							n_L_phi_scalar[i] = n_L_phi_scalar2[i];
						}
					}
		

			fftwf_execute_dft(plan_c2c_scalar, n_L_phi_scalar, m_L_M_scalar);

			fftwf_execute_dft(plan_c2c_tensor, n_L_phi_tensor, m_L_M_tensor);

			for (ptrdiff_t deltaLidx=0; deltaLidx<ndeltaL_scalar; deltaLidx++){
				for (ptrdiff_t Lidx=0; Lidx<nL; Lidx++){
					for (ptrdiff_t Midx=0; Midx<=Lidx; Midx++){

					a_L_M_scalar[0*ndeltaL_scalar*nL*nL+deltaLidx*nL*nL+Lidx*nL+Midx] += y_M_L[Midx*nL+Lidx] 
					* m_L_M_scalar[0*ndeltaL_scalar*nL+deltaLidx*nL+Lidx] * prefactors_scalar[0*ndeltaL_scalar*nL+deltaLidx*nL+Lidx];

					a_L_M_scalar[1*ndeltaL_scalar*nL*nL+deltaLidx*nL*nL+Lidx*nL+Midx] += y_M_L[Midx*nL+Lidx] 
					* m_L_M_scalar[1*ndeltaL_scalar*nL+deltaLidx*nL+Lidx] * prefactors_scalar[1*ndeltaL_scalar*nL+deltaLidx*nL+Lidx];

					a_L_M_scalar[2*ndeltaL_scalar*nL*nL+deltaLidx*nL*nL+Lidx*nL+Midx] += y_M_L[Midx*nL+Lidx] 
					* m_L_M_scalar[2*ndeltaL_scalar*nL+deltaLidx*nL+Lidx] * prefactors_scalar[2*ndeltaL_scalar*nL+deltaLidx*nL+Lidx];

					}
				} 
			}


			for (ptrdiff_t deltaLidx=0; deltaLidx<ndeltaL_tensor; deltaLidx++){
				for (ptrdiff_t Lidx=0; Lidx<nL; Lidx++){
					for (ptrdiff_t Midx=0; Midx<=Lidx; Midx++){

					a_L_M_tensor[0*ndeltaL_tensor*nL*nL+deltaLidx*nL*nL+Lidx*nL+Midx] += y_M_L[Midx*nL+Lidx] 
					* m_L_M_tensor[0*ndeltaL_tensor*nL+deltaLidx*nL+Lidx] * prefactors_tensor[0*ndeltaL_tensor*nL+deltaLidx*nL+Lidx];

					a_L_M_tensor[1*ndeltaL_tensor*nL*nL+deltaLidx*nL*nL+Lidx*nL+Midx] += y_M_L[Midx*nL+Lidx] 
					* m_L_M_tensor[1*ndeltaL_tensor*nL+deltaLidx*nL+Lidx] * prefactors_tensor[1*ndeltaL_tensor*nL+deltaLidx*nL+Lidx];

					a_L_M_tensor[2*ndeltaL_tensor*nL*nL+deltaLidx*nL*nL+Lidx*nL+Midx] += y_M_L[Midx*nL+Lidx] 
					* m_L_M_tensor[2*ndeltaL_tensor*nL+deltaLidx*nL+Lidx] * prefactors_tensor[2*ndeltaL_tensor*nL+deltaLidx*nL+Lidx];

					}
				}
			}

	   	}
			
	
float t_cubic_sp(const double *ct_weights, const long long *rule, const float *weights,
		 const float *f_i_ell, const float complex *a_ell_m,
		 const float *y_m_ell, int ntheta, int nrule,
		 int nell, int npol, int nufact, int nphi){

    int nm = nphi / 2 + 1;
    float t_cubic = 0.0;
    int nffts[1] = {nphi};
    fftwf_plan plan_c2r;

    // Plan fft on temporary arrays now in order to avoid having to run the planner
    // in a omp critial region later.
    float complex *m_ell_m = fftwf_malloc(sizeof *m_ell_m * npol * nell * nm);
    float *n_ell_phi = fftwf_malloc(sizeof *n_ell_phi * npol * nell * nphi);

    plan_c2r = fftwf_plan_many_dft_c2r(1, nffts, npol * nell,
				       m_ell_m, NULL,
				       1, nm,
				       n_ell_phi, NULL,
				       1, nphi,
				       FFTW_MEASURE);
    fftwf_free(m_ell_m);
    fftwf_free(n_ell_phi);

    #pragma omp parallel 
    {
    mkl_set_num_threads_local(1);
        
    float complex *m_ell_m = fftwf_malloc(sizeof *m_ell_m * npol * nell * nm);
    float *n_ell_phi = fftwf_malloc(sizeof *n_ell_phi * npol * nell * nphi);
    float *f_i_phi = fftwf_malloc(sizeof *f_i_phi * nufact * nphi);

    if (m_ell_m == NULL || n_ell_phi == NULL || f_i_phi == NULL){
	fftwf_free(m_ell_m);
	fftwf_free(n_ell_phi);
	fftwf_free(f_i_phi);
	exit(1);
    }

    #pragma omp for reduction (+:t_cubic)
    for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){

	backward_sp(f_i_ell, a_ell_m, y_m_ell + tidx * nell * nell,
		    m_ell_m, n_ell_phi, plan_c2r,
		    f_i_phi, nell, npol, nufact, nphi);
	
	t_cubic += t_cubic_on_ring_sp(rule, weights, f_i_phi, nrule, nphi) 
	    * PI * ct_weights[tidx] / 3. / (double)nphi;
    }

    fftwf_free(m_ell_m);
    fftwf_free(n_ell_phi);
    fftwf_free(f_i_phi);

    mkl_set_num_threads_local(0);
    } // End of parallel region.
    
    fftwf_destroy_plan(plan_c2r);

    return t_cubic;
}

void step_sp(const double *ct_weights, const long long *rule, const float *weights,
	     const float *f_i_ell, const float complex *a_ell_m, const float *y_m_ell,
	     float complex *grad_t, int ntheta, int nrule, int nell, int npol, 
	     int nufact, int nphi){
 
    int nm = nphi / 2 + 1;
    int nffts[1] = {nphi};
    fftwf_plan plan_c2r, plan_r2c;
    int nw = get_forward_array_size(rule, nrule);

    // Plan ffts on temporary arrays now in order to avoid having to run the planner
    // in a omp critial region later.
    float complex *m_ell_m = fftwf_malloc(sizeof *m_ell_m * npol * nell * nm);
    float *n_ell_phi = fftwf_malloc(sizeof *n_ell_phi * npol * nell * nphi);

    plan_c2r = fftwf_plan_many_dft_c2r(1, nffts, npol * nell,
				       m_ell_m, NULL,
				       1, nm,
				       n_ell_phi, NULL,
				       1, nphi,
				       FFTW_MEASURE);
    plan_r2c = fftwf_plan_many_dft_r2c(1, nffts, npol * nell,
				       n_ell_phi, NULL,
				       1, nphi,
				       m_ell_m, NULL,
				       1, nm,
				       FFTW_MEASURE);
    
    fftwf_free(m_ell_m);
    fftwf_free(n_ell_phi);

    #pragma omp parallel 
    {
    mkl_set_num_threads_local(1);        

    float complex *m_ell_m = fftwf_malloc(sizeof *m_ell_m * npol * nell * nm);
    float *n_ell_phi = fftwf_malloc(sizeof *n_ell_phi * npol * nell * nphi);
    float *f_i_phi = fftwf_malloc(sizeof *f_i_phi * nufact * nphi);
    float *work_i_ell = fftwf_malloc(sizeof *work_i_ell * nw * npol * nell);    
    float *work_i_phi = fftwf_malloc(sizeof *work_i_phi * nw * nphi);    
    float complex *grad_t_priv = fftwf_malloc(sizeof *grad_t_priv * npol * nell * nell);	

    for (ptrdiff_t i=0; i<npol*nell*nell; i++){
	grad_t_priv[i] = 0;
    }

    if (m_ell_m == NULL || n_ell_phi == NULL || f_i_phi == NULL){
	fftwf_free(m_ell_m);
	fftwf_free(n_ell_phi);
	fftwf_free(f_i_phi);
	exit(1);
    }

    #pragma omp for
    for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){

	backward_sp(f_i_ell, a_ell_m, y_m_ell + tidx * nell * nell,
		    m_ell_m, n_ell_phi, plan_c2r,
		    f_i_phi, nell, npol, nufact, nphi);
	forward_sp(f_i_ell, grad_t_priv, y_m_ell + tidx * nell * nell,
		   m_ell_m, n_ell_phi, plan_r2c, f_i_phi, work_i_ell,
		   work_i_phi, rule, weights, ct_weights[tidx], nrule,
		   nw, nell, npol, nphi);		
    }

    #pragma omp critical
    {
	for (ptrdiff_t i=0; i<npol*nell*nell; i++){
	    grad_t[i] += grad_t_priv[i];
	}	
    }

    fftwf_free(m_ell_m);
    fftwf_free(n_ell_phi);
    fftwf_free(f_i_phi);
    fftwf_free(work_i_ell);
    fftwf_free(work_i_phi);
    fftwf_free(grad_t_priv);

    mkl_set_num_threads_local(0);
    } // End of parallel region
    
    fftwf_destroy_plan(plan_c2r);
    fftwf_destroy_plan(plan_r2c);
}

void step_sst_sp(const int *L_list,
			  int nL, int npol, 
			  int n_scalar1, int n_scalar2, int n_tensor, int n_scalar,   
			  float complex *a_L_M_scalar, float complex *a_L_M_tensor,
			  const float *y_M_L,
			  const float *w3j_product_scalar1, const float *w3j_product_scalar2, const float *w3j_product_tensor,
			  const float complex *prefactors_scalar, const float complex *prefactors_tensor, 
			  int Lmax, int nell,
			  float complex *f_i_phi_scalar1, float complex *f_i_phi_scalar2, float complex *f_i_phi_tensor, 
			  int nufact, int nphi,
			  const float complex *kappa_i_L_scalar1, const float complex *kappa_i_L_scalar2, const float complex *kappa_i_L_tensor,
			  float complex *work_i_L_scalar1, float complex *work_i_L_scalar2, float complex *work_i_L_tensor,
			  float complex *work_i_phi_scalar1, float complex *work_i_phi_scalar2, float complex *work_i_phi_tensor,
			  float complex *n_L_phi_scalar1, float complex *n_L_phi_scalar2, float complex *n_L_phi_scalar, float complex *n_L_phi_tensor,
			  float complex *m_L_M_scalar, float complex *m_L_M_tensor,
			  float complex *grad_t_zeta, float complex *grad_t_h,
			  const long long *rule, const float *weights, const float ct_weight,
			  const float w3j, int nrule, int nw){


			int ndeltaL_scalar = 2;                               
			int ndeltaL_tensor = 5;                               

			int nm = nphi;
			int nffts[1] = {nphi};
			fftwf_plan plan_c2c_scalar, plan_c2c_tensor;
			int nw = get_forward_array_size(rule, nrule);


			float complex *m_L_M_scalarp = fftwf_malloc(sizeof *m_L_M_scalarp * npol * ndeltaL_scalar * nL * nphi);
    		float *n_L_phi_scalarp = fftwf_malloc(sizeof *n_L_phi_scalarp * npol * ndeltaL_scalar * nL * nphi);

			plan_c2c_scalar = fftwf_plan_many_dft(1, nffts, npol * ndeltaL_scalar * nL,
				       m_L_M_scalarp, NULL,
				       1, nm,
				       n_L_phi_scalarp, NULL,
				       1, nphi,
				       FFTW_MEASURE);
			plan_c2c_scalar_backward = fftwf_plan_many_dft(1, nffts, npol * ndeltaL_scalar * nL,
				       n_L_phi_scalarp, NULL,
				       1, nphi,
				       m_L_M_scalarp, NULL,
				       1, nm,
				       FFTW_MEASURE);	
			fftwf_free(m_L_M_scalarp);
			fftwf_free(n_L_phi_scalarp);

			float complex *m_L_M_tensorp = fftwf_malloc(sizeof *m_L_M_tensorp * npol * ndeltaL_tensor * nL * nphi);
			float complex *n_L_phi_tensorp = fftwf_malloc(sizeof *n_L_phi_tensorp * npol * ndeltaL_tensor * nL * nphi);

			plan_c2c_tensor = fftwf_plan_many_dft(1, nffts, npol * ndeltaL_tensor * nL,
							m_L_M_tensorp, NULL,
							1, nm,
							n_L_phi_tensorp, NULL,
							1, nphi,
							FFTW_MEASURE);
			plan_c2c_tensor_backward = fftwf_plan_many_dft(1, nffts, npol * ndeltaL_tensor * nL,
							n_L_phi_tensorp, NULL,
							1, nphi,
							m_L_M_tensorp, NULL,
							1, nm,
							FFTW_MEASURE);				

			fftwf_free(m_L_M_tensorp);
			fftwf_free(n_L_phi_tensorp);

			#pragma omp parallel
			{
			mkl_set_num_threads_local(1);

			float complex *m_L_M_scalar = fftwf_malloc(sizeof *m_L_M_scalar * npol * ndeltaL_scalar * nL * nphi);
			float complex *m_L_M_tensor = fftwf_malloc(sizeof *m_L_M_tensor * npol * ndeltaL_tensor * nL * nphi);

			float *n_L_phi_scalar1 = fftwf_malloc(sizeof *n_L_phi_scalar1 * npol * ndeltaL_scalar * nL * nphi);
			float *n_L_phi_scalar2 = fftwf_malloc(sizeof *n_L_phi_scalar2 * npol * ndeltaL_scalar * nL * nphi);
			float *n_L_phi_scalar = fftwf_malloc(sizeof *n_L_phi_scalar * npol * ndeltaL_scalar * nL * nphi);
			float *n_L_phi_tensor = fftwf_malloc(sizeof *n_L_phi_tensor * npol * ndeltaL_tensor * nL * nphi);

			float *f_i_phi_scalar1 = fftwf_malloc(sizeof *f_i_phi_scalar1 * nufact * nphi);
			float *f_i_phi_scalar2 = fftwf_malloc(sizeof *f_i_phi_scalar2 * nufact * nphi);
			float *f_i_phi_tensor = fftwf_malloc(sizeof *f_i_phi_tensor * nufact * nphi);

			float *work_i_L_scalar1 = fftwf_malloc(sizeof *work_i_L_scalar1 * nw * npol * ndeltaL_scalar * nL);    
			float *work_i_L_scalar2 = fftwf_malloc(sizeof *work_i_L_scalar2 * nw * npol * ndeltaL_scalar * nL);    
			float *work_i_L_tensor = fftwf_malloc(sizeof *work_i_L_tensor * nw * npol * ndeltaL_tensor * nL); 

			float *work_i_phi_scalar1 = fftwf_malloc(sizeof *work_i_phi_scalar1 * nw * nphi);    
			float *work_i_phi_scalar2 = fftwf_malloc(sizeof *work_i_phi_scalar2 * nw * nphi);    
			float *work_i_phi_tensor = fftwf_malloc(sizeof *work_i_phi_tensor * nw * nphi);    

			float complex *grad_t_zeta_priv = fftwf_malloc(sizeof *grad_t_zeta_priv * npol * ndeltaL_scalar * nL * nL);
			float complex *grad_t_h_priv = fftwf_malloc(sizeof *grad_t_h_priv * npol * ndeltaL_tensor * nL * nL);

			for (ptrdiff_t i=0; i<npol*ndeltaL_scalar*nL*nL; i++){
			grad_t_zeta_priv[i] = 0;
    		}
			for (ptrdiff_t i=0; i<npol*ndeltaL_tensor*nL*nL; i++){
			grad_t_h_priv[i] = 0;
    		}

			#pragma omp for
    		for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){

			backward_sp_mixed_sst(const int *L_list,
				int nL, int npol,
				int n_scalar1, int n_scalar2, int n_tensor,
				const float complex *a_ell_m,
				const float *y_M_L,
				const float *w3j_product_scalar1, const float *w3j_product_scalar2, const float *w3j_product_tensor,
				const float complex *prefactors_scalar1, const float complex *prefactors_scalar2, const float complex *prefactors_tensor, 
				float complex *A_L_M_scalar1, float complex *A_L_M_scalar2, float complex *A_L_M_tensor,
				int Lmax, int nell,
				float complex *n_L_phi_scalar, float complex *n_L_phi_tensor,
				fftwf_plan plan_c2c_scalar, fftwf_plan plan_c2c_tensor,
				float complex*f_i_phi_scalar1, float complex *f_i_phi_scalar2, float complex *f_i_phi_tensor, 
				int nufact, int nphi,
				const float complex *kappa_i_L_scalar1, const float complex *kappa_i_L_scalar2,const float complex *kappa_i_L_tensor);

			 forward_sst_sp(const int *L_list,
				int nL, int npol, 
				int n_scalar1, int n_scalar2, int n_tensor, int n_scalar,   
				float complex *grad_t_zeta, float complex *grad_t_h,
				const float *y_M_L,
				const float *w3j_product_scalar1, const float *w3j_product_scalar2, const float *w3j_product_tensor,
				const float complex *prefactors_scalar, const float complex *prefactors_tensor, 
				int Lmax, int nell,
				fftwf_plan plan_c2c_scalar, fftwf_plan plan_c2c_tensor,
				float complex *f_i_phi_scalar1, float complex *f_i_phi_scalar2, float complex *f_i_phi_tensor, 
				int nufact, int nphi,
				const float complex *kappa_i_L_scalar1, const float complex *kappa_i_L_scalar2, const float complex *kappa_i_L_tensor,
				float complex *work_i_L_scalar1, float complex *work_i_L_scalar2, float complex *work_i_L_tensor,
				float complex *work_i_phi_scalar1, float complex *work_i_phi_scalar2, float complex *work_i_phi_tensor,
				float complex *n_L_phi_scalar1, float complex *n_L_phi_scalar2, float complex *n_L_phi_scalar, float complex *n_L_phi_tensor,
				float complex *m_L_M_scalar, float complex *m_L_M_tensor,
				const long long *rule, const float *weights, const float ct_weight,
				const float w3j, int nrule, int nw)	
    		}

			#pragma omp critical

			{
			for (ptrdiff_t i=0; i<npol*ndeltaL_scalar*nL*nL; i++){
	    			grad_t_zeta[i] += grad_t_zeta_priv[i];
			}	
			for (ptrdiff_t i=0; i<npol*ndeltaL_tensor*nL*nL; i++){
	    			grad_t_h[i] += grad_t_h_priv[i];
			}	
			}

			fftwf_free(m_L_M_scalar);
			fftwf_free(m_L_M_tensor);

			fftwf_free(n_L_phi_scalar1);
			fftwf_free(n_L_phi_scalar2);
			fftwf_free(n_L_phi_scalar);
			fftwf_free(n_L_phi_tensor);

			fftwf_free(f_i_phi_scalar1);
			fftwf_free(f_i_phi_scalar2);
			fftwf_free(f_i_phi_tensor);

			fftwf_free(work_i_L_scalar1);
			fftwf_free(work_i_L_scalar2);
			fftwf_free(work_i_L_tensor);

			fftwf_free(work_i_phi_scalar1);
			fftwf_free(work_i_phi_scalar2);
			fftwf_free(work_i_phi_tensor);
			
			fftwf_free(grad_t_zeta_priv);
			fftwf_free(grad_t_h_priv);
		
			mkl_set_num_threads_local(0);
   			} // End of parallel region
    
    		fftwf_destroy_plan(plan_c2c_scalar);
    		fftwf_destroy_plan(plan_c2c_tensor);	
}

void compute_ylm_sp(const double *thetas, float *y_m_ell, int ntheta, int lmax){

    int nell = lmax + 1;
    double epsilon = 1e-300;

    #pragma omp parallel
    {
    Ylmgen_C ygen;

    // Sse2 version not needed, ylm computation is subdominant to filling the ylm
    // array, i.e. it's really fast.
    Ylmgen_init(&ygen, lmax, lmax, 0, 0, epsilon);
    Ylmgen_set_theta(&ygen, thetas, ntheta);

    #pragma omp for schedule(dynamic, 10)
    for (ptrdiff_t midx=0; midx<nell; midx++){ 
	
	for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){
	
	    Ylmgen_prepare(&ygen, tidx, midx);
	    Ylmgen_recalc_Ylm(&ygen);

	    ptrdiff_t firstl = *ygen.firstl;

	    for (ptrdiff_t lidx=firstl; lidx<nell; lidx++){
		
		y_m_ell[tidx*nell*nell+midx*nell+lidx] = (float) ygen.ylm[lidx];
	    }
	}
    }

    Ylmgen_destroy(&ygen);
    } // End of parallel region.
}

/* Double precision versions */

double t_cubic_on_ring_dp(const long long *rule, const double *weights, const double *f_i_phi,
			  int nrule, int nphi){

    double t_cubic = 0.0;

    for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

	long long rx = rule[ridx*3];
	long long ry = rule[ridx*3+1];
	long long rz = rule[ridx*3+2];

	double wx = weights[ridx*3];
	double wy = weights[ridx*3+1];
	double wz = weights[ridx*3+2];

	for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
	    t_cubic += wx * wy * wz * f_i_phi[rx*nphi+phidx] * f_i_phi[ry*nphi+phidx]
		       * f_i_phi[rz*nphi+phidx];
	}
    }
    return t_cubic;
}

void backward_dp(const double *f_i_ell, const double complex *a_ell_m, 
		 const double *y_m_ell, double complex *m_ell_m, double *n_ell_phi,
		 fftw_plan plan_c2r, double *f_i_phi, int nell, int npol, int nufact,
		 int nphi){

    int nm = nphi / 2 + 1;

    // Place alm * Ylm into Mlm.
    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
	for (ptrdiff_t lidx=0; lidx<nell; lidx++){
	    for (ptrdiff_t midx=0; midx<nm; midx++){

		complex double tmp;

		if (midx < nell){
		    tmp = a_ell_m[pidx*nell*nell+lidx*nell+midx]
			  * y_m_ell[midx*nell+lidx];
		} else{
		    // Needed because m_ell_m array can be larger than alm.
		    tmp = 0. + 0.*I;
		}

		m_ell_m[pidx*nell*nm+lidx*nm+midx] = tmp;
	    }
	}
    }

    // Backward fft. Note that fftw has no normalization for backward
    // or forward. Unlike numpy and pyfftw which apply 1/nphi during 
    // backward (c2r). So in python version I multiply the result of 
    // the fft by nphi to compensate for that factor. Here it's not needed.
    fftw_execute_dft_c2r(plan_c2r, m_ell_m, n_ell_phi);

    // f_i_ell @ n_ell_phi -> f_i_phi.
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * nell,
		1.0, f_i_ell, npol * nell,
		n_ell_phi, nphi,
		0.0, f_i_phi, nphi);    
}



void forward_dp(const double *f_i_ell, double complex *a_ell_m, const double *y_m_ell,
		double complex *m_ell_m, double *n_ell_phi, fftw_plan plan_r2c,
		const double *f_i_phi, double *work_i_ell, double *work_i_phi,
		const long long *rule, const double *weights, const double ct_weight,
		int nrule, int nw, int nell, int npol, int nphi){

    int widx = 0; // Index to work arrays.
    int nm = nphi / 2 + 1;

    for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

	long long rx = rule[ridx*3];
	long long ry = rule[ridx*3+1];
	long long rz = rule[ridx*3+2];

	double weight = weights[ridx*3] * weights[ridx*3+1] * weights[ridx*3+2]
	    * PI * ct_weight / 3. / (double) nphi;

	// Fill work arrays.
	if (rx == ry && rx == rz){ // Case: 000.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = 3. * weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    widx += 1;
	} else if (rx == ry && ry != rz){ // Case: 001.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = 2. * weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rz*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[ry*nphi+phidx];
	    }
	    widx += 2;
	} else if (rx != ry && ry == rz){ // Case: 100.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = 2. * weight
			* f_i_ell[ry*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    widx += 2;
	} else if (rx == rz && rx != ry){ // Case 010.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = 2. * weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[ry*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    widx += 2;
	} else { // Case: 012.
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[widx*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rx*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+1)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[ry*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		for (ptrdiff_t lidx=0; lidx<nell; lidx++){
		    work_i_ell[(widx+2)*npol*nell+pidx*nell+lidx] = weight
			* f_i_ell[rz*npol*nell+pidx*nell+lidx];
		}
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[widx*nphi+phidx] = f_i_phi[ry*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+1)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[rz*nphi+phidx];
	    }
	    for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
		work_i_phi[(widx+2)*nphi+phidx] = f_i_phi[rx*nphi+phidx] * f_i_phi[ry*nphi+phidx];
	    }
	    widx += 3;
	}
    }

    if (widx != nw){
	fprintf(stderr, "widx (%d) != nw (%d) \n", widx, nw);
	exit(1);
    }

    // Sum_i X_i_ell dT/dX_i_phi + Y_i_ell dT/dY_i_phi + Z_i_ell dT/dZ_i_phi -> n_ell_phi.
    // Implemented as work_i_ell.T @ work_i_phi -> n_ell_phi.
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		npol * nell, nphi, nw,
		1.0, work_i_ell, npol * nell,
		work_i_phi, nphi,
		0.0, n_ell_phi, nphi);    

    fftw_execute_dft_r2c(plan_r2c, n_ell_phi, m_ell_m);

    // Multiply by ylm and add result to alm.
    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
	for (ptrdiff_t lidx=0; lidx<nell; lidx++){
	    for (ptrdiff_t midx=0; midx<=lidx; midx++){
    
		a_ell_m[pidx*nell*nell+lidx*nell+midx] += y_m_ell[midx*nell+lidx] 
		    * m_ell_m[pidx*nell*nm+lidx*nm+midx];

	    }
	}
    }        
}

double t_cubic_dp(const double *ct_weights, const long long *rule, const double *weights,
		  const double *f_i_ell, const double complex *a_ell_m,
		  const double *y_m_ell, int ntheta, int nrule,
		  int nell, int npol, int nufact, int nphi){

    int nm = nphi / 2 + 1;
    double t_cubic = 0.0;
    int nffts[1] = {nphi};
    fftw_plan plan_c2r;

    // Plan fft on temporary arrays now in order to avoid having to run the planner
    // in a omp critial region later.
    double complex *m_ell_m = fftw_malloc(sizeof *m_ell_m * npol * nell * nm);
    double *n_ell_phi = fftw_malloc(sizeof *n_ell_phi * npol * nell * nphi);

    plan_c2r = fftw_plan_many_dft_c2r(1, nffts, npol * nell,
				       m_ell_m, NULL,
				       1, nm,
				       n_ell_phi, NULL,
				       1, nphi,
				       FFTW_MEASURE);
    fftw_free(m_ell_m);
    fftw_free(n_ell_phi);

    #pragma omp parallel 
    {
    mkl_set_num_threads_local(1);

    double complex *m_ell_m = fftw_malloc(sizeof *m_ell_m * npol * nell * nm);
    double *n_ell_phi = fftw_malloc(sizeof *n_ell_phi * npol * nell * nphi);
    double *f_i_phi = fftw_malloc(sizeof *f_i_phi * nufact * nphi);

    if (m_ell_m == NULL || n_ell_phi == NULL || f_i_phi == NULL){
	fftw_free(m_ell_m);
	fftw_free(n_ell_phi);
	fftw_free(f_i_phi);
	exit(1);
    }

    #pragma omp for reduction (+:t_cubic)
    for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){

	backward_dp(f_i_ell, a_ell_m, y_m_ell + tidx * nell * nell,
		    m_ell_m, n_ell_phi, plan_c2r,
		    f_i_phi, nell, npol, nufact, nphi);
	
	t_cubic += t_cubic_on_ring_dp(rule, weights, f_i_phi, nrule, nphi) 
	    * PI * ct_weights[tidx] / 3. / (double)nphi;
    }

    fftw_free(m_ell_m);
    fftw_free(n_ell_phi);
    fftw_free(f_i_phi);

    mkl_set_num_threads_local(0);
    } // End of parallel region
    
    fftw_destroy_plan(plan_c2r);

    return t_cubic;
}

void step_dp(const double *ct_weights, const long long *rule, const double *weights,
	     const double *f_i_ell, const double complex *a_ell_m, const double *y_m_ell,
	     double complex *grad_t, int ntheta, int nrule, int nell, int npol, 
	     int nufact, int nphi){
 
    int nm = nphi / 2 + 1;
    int nffts[1] = {nphi};
    fftw_plan plan_c2r, plan_r2c;
    int nw = get_forward_array_size(rule, nrule);

    // Plan ffts on temporary arrays now in order to avoid having to run the planner
    // in a omp critial region later.
    double complex *m_ell_m = fftw_malloc(sizeof *m_ell_m * npol * nell * nm);
    double *n_ell_phi = fftw_malloc(sizeof *n_ell_phi * npol * nell * nphi);

    plan_c2r = fftw_plan_many_dft_c2r(1, nffts, npol * nell,
				       m_ell_m, NULL,
				       1, nm,
				       n_ell_phi, NULL,
				       1, nphi,
				       FFTW_MEASURE);
    plan_r2c = fftw_plan_many_dft_r2c(1, nffts, npol * nell,
				       n_ell_phi, NULL,
				       1, nphi,
				       m_ell_m, NULL,
				       1, nm,
				       FFTW_MEASURE);
    
    fftw_free(m_ell_m);
    fftw_free(n_ell_phi);

    #pragma omp parallel 
    {
    mkl_set_num_threads_local(1);
        
    double complex *m_ell_m = fftw_malloc(sizeof *m_ell_m * npol * nell * nm);
    double *n_ell_phi = fftw_malloc(sizeof *n_ell_phi * npol * nell * nphi);
    double *f_i_phi = fftw_malloc(sizeof *f_i_phi * nufact * nphi);
    double *work_i_ell = fftw_malloc(sizeof *work_i_ell * nw * npol * nell);    
    double *work_i_phi = fftw_malloc(sizeof *work_i_phi * nw * nphi);    
    double complex *grad_t_priv = fftw_malloc(sizeof *grad_t_priv * npol * nell * nell);	

    for (ptrdiff_t i=0; i<npol*nell*nell; i++){
	grad_t_priv[i] = 0;
    }

    if (m_ell_m == NULL || n_ell_phi == NULL || f_i_phi == NULL){
	fftw_free(m_ell_m);
	fftw_free(n_ell_phi);
	fftw_free(f_i_phi);
	exit(1);
    }

    #pragma omp for
    for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){

	backward_dp(f_i_ell, a_ell_m, y_m_ell + tidx * nell * nell,
		    m_ell_m, n_ell_phi, plan_c2r,
		    f_i_phi, nell, npol, nufact, nphi);
	forward_dp(f_i_ell, grad_t_priv, y_m_ell + tidx * nell * nell,
		   m_ell_m, n_ell_phi, plan_r2c, f_i_phi, work_i_ell,
		   work_i_phi, rule, weights, ct_weights[tidx], nrule,
		   nw, nell, npol, nphi);		
    }

    #pragma omp critical
    {
	for (ptrdiff_t i=0; i<npol*nell*nell; i++){
	    grad_t[i] += grad_t_priv[i];
	}	
    }

    fftw_free(m_ell_m);
    fftw_free(n_ell_phi);
    fftw_free(f_i_phi);
    fftw_free(work_i_ell);
    fftw_free(work_i_phi);
    fftw_free(grad_t_priv);

    mkl_set_num_threads_local(0);
    } // End of parallel region
    
    fftw_destroy_plan(plan_c2r);
    fftw_destroy_plan(plan_r2c);
}

void compute_ylm_dp(const double *thetas, double *y_m_ell, int ntheta, int lmax){

    int nell = lmax + 1;
    double epsilon = 1e-300;

    #pragma omp parallel
    {
    Ylmgen_C ygen;

    // Sse2 version not needed, ylm computation is subdominant to filling the ylm
    // array, i.e. it's really fast.
    Ylmgen_init(&ygen, lmax, lmax, 0, 0, epsilon);
    Ylmgen_set_theta(&ygen, thetas, ntheta);

    #pragma omp for schedule(dynamic, 10)
    for (ptrdiff_t midx=0; midx<nell; midx++){ 
	
	for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){
	
	    Ylmgen_prepare(&ygen, tidx, midx);
	    Ylmgen_recalc_Ylm(&ygen);

	    ptrdiff_t firstl = *ygen.firstl;

	    for (ptrdiff_t lidx=firstl; lidx<nell; lidx++){
		
		y_m_ell[tidx*nell*nell+midx*nell+lidx] = ygen.ylm[lidx];
	    }
	}
    }

    Ylmgen_destroy(&ygen);
    } // End of parallel region.
}

/* 
*  Reduced A_{LM} construction                                           
*/

float complex get_alm_entry_sp(const float complex *a_ell_m,
		    int nell, int ell, int m){

	if (abs(m) >= nell){
		return 0.f + 0.f * I;
	}
	
    if (m >= 0){
		return a_ell_m[ell * nell + m];
    } 
	else{
		int mp = -m;
		float complex base = a_ell_m[ell * nell + mp];
		return (mp % 2) ? -conjf(base) : conjf(base);
    }
}

double complex get_alm_entry_dp(const double complex *a_ell_m,
		  int nell, int ell, int m){

	if (abs(m) >= nell){
		return 0. + 0. * I;
	}

    if (m >= 0){
		return a_ell_m[ell * nell + m];
    } 
	else{
		int mp = -m;
		double complex base = a_ell_m[ell * nell + mp];
		return (mp % 2) ? -conj(base) : conj(base);
    }
}

float get_ylm_entry_sp(const float *y_m_ell,
		    int nell, int ell, int m){

	if (abs(m) >= nell){
		return 0.f + 0.f * I;
	}
	
    if (m >= 0){
		return y_m_ell[m*nell + ell];
    } 
	else{
		int mp = -m;
		float base = y_m_ell[mp*nell + ell];
		return (mp % 2) ? -base : base;
    }
}

double get_ylm_entry_dp(const double *y_m_ell,
		    int nell, int ell, int m){

	if (abs(m) >= nell){
		return 0. + 0. * I;
	}
	
    if (m >= 0){
		return y_m_ell[m*nell + ell];
    } 
	else{
		int mp = -m;
		double base = y_m_ell[mp*nell + ell];
		return (mp % 2) ? -base : base;
    }
}

void compute_A_LM_sp(const int *L_list, const int *deltaL_list,
		     int nL, int ndeltaL, int npol, int n, 
			 const float complex *a_ell_m,
			 const float *y_M_L, const float *w3j_product,
			 const float complex *prefactors, float complex *out,
			 int Lmax, int nell, int nphi){

	ptrdiff_t total = (ptrdiff_t) npol * ndeltaL * nL * nphi;

    for (ptrdiff_t idx=0; idx<total; idx++){
		out[idx] = 0.f + 0.f * I;
    }

    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		const float complex *alm_pol = a_ell_m + pidx * nell * nell;

		for (ptrdiff_t idL=0; idL<ndeltaL; idL++){
			int deltaL = deltaL_list[idL];
			for (ptrdiff_t iL=0; iL<nL; iL++){
				int L = L_list[iL];
				if (L < 0 || L >= nell){
					continue;
				}

				int ell = L + deltaL;
				if (ell < 0 || ell >= nell){
					continue;
				}

				ptrdiff_t base = ((pidx * ndeltaL + idL) * nL + iL) * nphi;
				float complex pref = prefactors[(pidx * ndeltaL + idL) * nL + iL];

				for (int M=-L; M<=L; M++){
					int Midx = (int)(M + Lmax);

					int m = - M - n;
					if (abs(m) > ell || abs(m) >= nell){
						continue;
					}

					float w3j = w3j_product[(idL * nL + iL) * (2 * Lmax + 1) + Midx];
					float complex alm_val = get_alm_entry_sp(alm_pol, nell, ell, m);
					float y_val = get_ylm_entry_sp(y_M_L, nL, L, M);

					float complex contrib = pref * w3j * alm_val * y_val;

					if (M>=0){
						out[base+M] += contrib;
					}
					else{
						out[base + (nphi + M)] += contrib;
					}
			}
			}
		}
	}
}

void compute_A_LM_dp(const int *L_list, const int *deltaL_list,
		     int nL, int ndeltaL, int npol, int n,
			 const double complex *a_ell_m,
			 const double *y_M_L, const double *w3j_product,
			 const double complex *prefactors, double complex *out,
			 int Lmax, int nell, int nphi){

	ptrdiff_t total = (ptrdiff_t) npol * ndeltaL * nL * nphi;

    for (ptrdiff_t idx=0; idx<total; idx++){
		out[idx] = 0. + 0. * I;
    }

    for (ptrdiff_t pidx=0; pidx<npol; pidx++){
		const double complex *alm_pol = a_ell_m + pidx * nell * nell;

		for (ptrdiff_t idL=0; idL<ndeltaL; idL++){
			int deltaL = deltaL_list[idL];
			for (ptrdiff_t iL=0; iL<nL; iL++){
				int L = L_list[iL];
				if (L < 0 || L >= nell){
					continue;
			}

				int ell = L + deltaL;
				if (ell < 0 || ell >= nell){
					continue;
				}

				ptrdiff_t base = ((pidx * ndeltaL + idL) * nL + iL) * nphi;
				double complex pref = prefactors[(pidx * ndeltaL + idL) * nL + iL];

				for (int M=-L; M<=L; M++){
					int Midx = (int)(M + Lmax);

					int m = - M - n;
					if (abs(m) > ell || abs(m) >= nell){
						continue;
					}
					double w3j = w3j_product[(idL * nL + iL) * (2 * Lmax + 1) + Midx];
					double complex alm_val = get_alm_entry_dp(alm_pol, nell, ell, m);
					double y_val = get_ylm_entry_dp(y_M_L, nL, L, M);

					double complex contrib = pref * w3j * alm_val * y_val;
					if (M>=0){
						out[base+M] += contrib;
					}
					else{
						out[base + (nphi + M)] += contrib;
					}
				}
			}
		}
    }
}

float t_cubic_on_ring_sp_sst(const long long *rule, const float *weights,
			const float complex *f_i_phi_scalar1, const float complex *f_i_phi_scalar2,
			const float complex *f_i_phi_tensor, int nrule, int nphi){

	float t_cubic = 0.f;

	for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

		long long rx = rule[ridx*3];
		long long ry = rule[ridx*3+1];
		long long rz = rule[ridx*3+2];

		float wx = weights[ridx*3];
		float wy = weights[ridx*3+1];
		float wz = weights[ridx*3+2];

		// 3 cyclic permutations.
		// 2 / 6 Take real value and divide by 6.
		for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
			t_cubic += wx * wy * wz * 2./6. * creal(
			  f_i_phi_scalar1[rx*nphi+phidx] * f_i_phi_scalar2[ry*nphi+phidx]
				* f_i_phi_tensor[rz*nphi+phidx]
			+ f_i_phi_scalar2[rx*nphi+phidx] * f_i_phi_tensor[ry*nphi+phidx]
				* f_i_phi_scalar1[rz*nphi+phidx]
			+ f_i_phi_tensor[rx*nphi+phidx] * f_i_phi_scalar1[ry*nphi+phidx]
				* f_i_phi_scalar2[rz*nphi+phidx]);
		}
	}
	return t_cubic;
}

double t_cubic_on_ring_dp_sst(const long long *rule, const double *weights,
			const double complex *f_i_phi_scalar1, const double complex *f_i_phi_scalar2,
			const double complex *f_i_phi_tensor, int nrule, int nphi){

	double t_cubic = 0.f;

	for (ptrdiff_t ridx=0; ridx<nrule; ridx++){

		long long rx = rule[ridx*3];
		long long ry = rule[ridx*3+1];
		long long rz = rule[ridx*3+2];

		double wx = weights[ridx*3];
		double wy = weights[ridx*3+1];
		double wz = weights[ridx*3+2];

		// 3 cyclic permutations.
		// 2 / 6 Take real value and divide by 6.
		for (ptrdiff_t phidx=0; phidx<nphi; phidx++){
			t_cubic += wx * wy * wz * 2./6. * creal(
			  f_i_phi_scalar1[rx*nphi+phidx] * f_i_phi_scalar2[ry*nphi+phidx]
				* f_i_phi_tensor[rz*nphi+phidx]
			+ f_i_phi_scalar2[rx*nphi+phidx] * f_i_phi_tensor[ry*nphi+phidx]
				* f_i_phi_scalar1[rz*nphi+phidx]
			+ f_i_phi_tensor[rx*nphi+phidx] * f_i_phi_scalar1[ry*nphi+phidx]
				* f_i_phi_scalar2[rz*nphi+phidx]);
		}

	}
	return t_cubic;
}


void backward_sp_mixed_sst(const int *L_list,
			  int nL, int npol,
			  int n_scalar1, int n_scalar2, int n_tensor,
			  const float complex *a_ell_m,
			  const float *y_M_L,
			  const float *w3j_product_scalar1, const float *w3j_product_scalar2, const float *w3j_product_tensor,
			  const float complex *prefactors_scalar1, const float complex *prefactors_scalar2, const float complex *prefactors_tensor, 
			  float complex *A_L_M_scalar1, float complex *A_L_M_scalar2, float complex *A_L_M_tensor,
			  int Lmax, int nell,
			  float complex *n_L_phi_scalar, float complex *n_L_phi_tensor,
			  fftwf_plan plan_c2c_scalar, fftwf_plan plan_c2c_tensor,
			  float complex*f_i_phi_scalar1, float complex *f_i_phi_scalar2, float complex *f_i_phi_tensor, 
			  int nufact, int nphi,
			  const float complex *kappa_i_L_scalar1, const float complex *kappa_i_L_scalar2,const float complex *kappa_i_L_tensor){

	int ndeltaL_scalar = 2;                               // deltaL = -1, 1 for scalar
	int ndeltaL_tensor = 5;                               // deltaL = -2, -1, 0, 1, 2 for tensor

	const int deltaL_list_scalar[] = {-1, 1};             //scalar
	const int deltaL_list_tensor[] = {-2, -1, 0, 1, 2};	  //tensor

	compute_A_LM_sp(L_list, deltaL_list_scalar,
		      nL, ndeltaL_scalar, npol, n_scalar1,
		      a_ell_m, y_M_L, w3j_product_scalar1,
		      prefactors_scalar1, A_L_M_scalar1,
		      Lmax, nell, nphi);
	
	fftwf_execute_dft(plan_c2c_scalar, A_L_M_scalar1, n_L_phi_scalar);

	const float complex alpha_sp = 1.0f + 0.0f * I;
	const float complex beta_sp = 0.0f + 0.0f * I;


	// Consume scalar1 immediately so scalar buffers can be safely reused.
	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * ndeltaL_scalar * nL,
		&alpha_sp, kappa_i_L_scalar1, npol * ndeltaL_scalar * nL,
		n_L_phi_scalar, nphi,
		&beta_sp, f_i_phi_scalar1, nphi);

	compute_A_LM_sp(L_list, deltaL_list_scalar,
		      nL, ndeltaL_scalar, npol, n_scalar2,
		      a_ell_m, y_M_L, w3j_product_scalar2,
		      prefactors_scalar2, A_L_M_scalar2,
		      Lmax, nell, nphi);

	fftwf_execute_dft(plan_c2c_scalar, A_L_M_scalar2, n_L_phi_scalar);

	// Consume scalar2 immediately so scalar buffers can be safely reused.
	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * ndeltaL_scalar * nL,
		&alpha_sp, kappa_i_L_scalar2, npol * ndeltaL_scalar * nL,
		n_L_phi_scalar, nphi,
		&beta_sp, f_i_phi_scalar2, nphi);

	compute_A_LM_sp(L_list, deltaL_list_tensor,
		      nL, ndeltaL_tensor, npol, n_tensor,
		      a_ell_m, y_M_L, w3j_product_tensor,
		      prefactors_tensor, A_L_M_tensor,
		      Lmax, nell, nphi);

	fftwf_execute_dft(plan_c2c_tensor, A_L_M_tensor, n_L_phi_tensor);

	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * ndeltaL_tensor * nL,
		&alpha_sp, kappa_i_L_tensor, npol * ndeltaL_tensor * nL,
		n_L_phi_tensor, nphi,
		&beta_sp, f_i_phi_tensor, nphi);  

}

void backward_dp_mixed_sst(const int *L_list, 
			  int nL, int npol,
			  int n_scalar1, int n_scalar2, int n_tensor,
			  const double complex *a_ell_m,
			  const double *y_M_L, 
			  const double *w3j_product_scalar1, const double *w3j_product_scalar2, const double *w3j_product_tensor,
			  const double complex *prefactors_scalar1, const double complex *prefactors_scalar2, const double complex *prefactors_tensor, 
			  double complex *A_L_M_scalar1, double complex *A_L_M_scalar2, double complex *A_L_M_tensor,
			  int Lmax, int nell,
			  double complex *n_L_phi_scalar, double complex *n_L_phi_tensor,
			  fftw_plan plan_c2c_scalar, fftw_plan plan_c2c_tensor,
			  double complex *f_i_phi_scalar1, double complex *f_i_phi_scalar2, double complex *f_i_phi_tensor, 
			  int nufact, int nphi,
			  const double complex *kappa_i_L_scalar1, const double complex *kappa_i_L_scalar2,const double complex *kappa_i_L_tensor){

	int ndeltaL_scalar = 2; // deltaL = -1, 1 for scalar
	int ndeltaL_tensor = 5; // deltaL = -2, -1, 0, 1, 2 for tensor

	const int deltaL_list_scalar[] = {-1, 1};             //scalar
	const int deltaL_list_tensor[] = {-2, -1, 0, 1, 2};	//tensor

	compute_A_LM_dp(L_list, deltaL_list_scalar,
		      nL, ndeltaL_scalar, npol, n_scalar1,
		      a_ell_m, y_M_L, w3j_product_scalar1,
		      prefactors_scalar1, A_L_M_scalar1,
		      Lmax, nell, nphi);

	fftw_execute_dft(plan_c2c_scalar, A_L_M_scalar1, n_L_phi_scalar);

	const double complex alpha_dp = 1.0 + 0.0 * I;
	const double complex beta_dp = 0.0 + 0.0 * I;

	// Consume scalar1 immediately so scalar buffers can be safely reused.
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * ndeltaL_scalar * nL,
		&alpha_dp, kappa_i_L_scalar1, npol * ndeltaL_scalar * nL,
		n_L_phi_scalar, nphi,
		&beta_dp, f_i_phi_scalar1, nphi);


	compute_A_LM_dp(L_list, deltaL_list_scalar,
		      nL, ndeltaL_scalar, npol, n_scalar2,
		      a_ell_m, y_M_L, w3j_product_scalar2,
		      prefactors_scalar2, A_L_M_scalar2,
		      Lmax, nell, nphi);

	fftw_execute_dft(plan_c2c_scalar, A_L_M_scalar2, n_L_phi_scalar);

	// Consume scalar2 immediately so scalar buffers can be safely reused.
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * ndeltaL_scalar * nL,
		&alpha_dp, kappa_i_L_scalar2, npol * ndeltaL_scalar * nL,
		n_L_phi_scalar, nphi,
		&beta_dp, f_i_phi_scalar2, nphi);

	compute_A_LM_dp(L_list, deltaL_list_tensor,
		      nL, ndeltaL_tensor, npol, n_tensor,
		      a_ell_m, y_M_L, w3j_product_tensor,
		      prefactors_tensor, A_L_M_tensor,
		      Lmax, nell, nphi);


	fftw_execute_dft(plan_c2c_tensor, A_L_M_tensor, n_L_phi_tensor);

	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		nufact, nphi, npol * ndeltaL_tensor * nL,
		&alpha_dp, kappa_i_L_tensor, npol * ndeltaL_tensor * nL,
		n_L_phi_tensor, nphi,
		&beta_dp, f_i_phi_tensor, nphi);  

}


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
		  const float complex *kappa_i_L_scalar1, const float complex *kappa_i_L_scalar2, const float complex *kappa_i_L_tensor){
 
	float t_cubic = 0.f;
	int nffts[1] = {nphi};
	fftwf_plan plan_c2c_scalar, plan_c2c_tensor;

	int ndeltaL_scalar = 2; // deltaL = -1, 1 for scalar
	int ndeltaL_tensor = 5; // deltaL = -2, -1, 0, 1, 2 for tensor

	//const int deltaL_list_scalar[] = {-1, 1};             //scalar
	//const int deltaL_list_tensor[] = {-2, -1, 0, 1, 2};   //tensor

	// Plan fft on temporary arrays now in order to avoid having to run the planner
	// in a omp critical region later.
	float complex *A_L_M_scalarp = fftwf_malloc(sizeof *A_L_M_scalarp * npol * ndeltaL_scalar * nL * nphi);
	float complex *n_L_phi_scalarp = fftwf_malloc(sizeof *n_L_phi_scalarp * npol * ndeltaL_scalar * nL * nphi);

	plan_c2c_scalar = fftwf_plan_many_dft(1, nffts, npol * ndeltaL_scalar * nL,
				       A_L_M_scalarp, NULL,
				       1, nphi,
				       n_L_phi_scalarp, NULL,
				       1, nphi,
					   FFTW_BACKWARD,
				       FFTW_MEASURE);
	fftwf_free(A_L_M_scalarp);
	fftwf_free(n_L_phi_scalarp);

	float complex *A_L_M_tensorp = fftwf_malloc(sizeof *A_L_M_tensorp * npol * ndeltaL_tensor * nL * nphi);
	float complex *n_L_phi_tensorp = fftwf_malloc(sizeof *n_L_phi_tensorp * npol * ndeltaL_tensor * nL * nphi);

	plan_c2c_tensor = fftwf_plan_many_dft(1, nffts, npol * ndeltaL_tensor * nL,
				       A_L_M_tensorp, NULL,
				       1, nphi,
				       n_L_phi_tensorp, NULL,
				       1, nphi,
					   FFTW_BACKWARD,
				       FFTW_MEASURE);
	fftwf_free(A_L_M_tensorp);
	fftwf_free(n_L_phi_tensorp);

	#pragma omp parallel 
	{
	mkl_set_num_threads_local(1);

	float complex *A_L_M_scalar1 = fftwf_malloc(sizeof *A_L_M_scalar1 * npol * ndeltaL_scalar * nL * nphi);
	float complex *n_L_phi_scalar = fftwf_malloc(sizeof *n_L_phi_scalar * npol * ndeltaL_scalar * nL * nphi);
	float complex *f_i_phi_scalar1 = fftwf_malloc(sizeof *f_i_phi_scalar1 * nufact * nphi);

	float complex *A_L_M_scalar2 = fftwf_malloc(sizeof *A_L_M_scalar2 * npol * ndeltaL_scalar * nL * nphi);
	float complex *f_i_phi_scalar2 = fftwf_malloc(sizeof *f_i_phi_scalar2 * nufact * nphi);

	float complex *A_L_M_tensor = fftwf_malloc(sizeof *A_L_M_tensor * npol * ndeltaL_tensor * nL * nphi);
	float complex *n_L_phi_tensor = fftwf_malloc(sizeof *n_L_phi_tensor * npol * ndeltaL_tensor * nL * nphi);
	float complex *f_i_phi_tensor = fftwf_malloc(sizeof *f_i_phi_tensor * nufact * nphi);


	if (A_L_M_scalar1 == NULL || n_L_phi_scalar == NULL || f_i_phi_scalar1 == NULL ||
	    A_L_M_scalar2 == NULL || f_i_phi_scalar2 == NULL ||
	    A_L_M_tensor == NULL || n_L_phi_tensor == NULL || f_i_phi_tensor == NULL){

	    fftwf_free(A_L_M_scalar1);
	    fftwf_free(n_L_phi_scalar);
	    fftwf_free(f_i_phi_scalar1);

	    fftwf_free(A_L_M_scalar2);
	    fftwf_free(f_i_phi_scalar2);

	    fftwf_free(A_L_M_tensor);
	    fftwf_free(n_L_phi_tensor);
	    fftwf_free(f_i_phi_tensor);
	    exit(1);
		}

	#pragma omp for reduction (+:t_cubic)
	for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){

		backward_sp_mixed_sst(L_list,
			  nL, npol,
			  n_scalar1, n_scalar2, n_tensor,
			  a_ell_m, y_M_L + tidx * nL * nL,
			  w3j_product_scalar1, w3j_product_scalar2, w3j_product_tensor,
			  prefactors_scalar1, prefactors_scalar2, prefactors_tensor,
			  A_L_M_scalar1, A_L_M_scalar2, A_L_M_tensor,
			  Lmax, nell, 
			  n_L_phi_scalar, n_L_phi_tensor,
			  plan_c2c_scalar, plan_c2c_tensor,
			  f_i_phi_scalar1, f_i_phi_scalar2, f_i_phi_tensor,
			  nufact, nphi,
			  kappa_i_L_scalar1, kappa_i_L_scalar2, kappa_i_L_tensor);

		t_cubic += t_cubic_on_ring_sp_sst(rule, weights, f_i_phi_scalar1, f_i_phi_scalar2, f_i_phi_tensor, nrule, nphi)
			*ct_weights[tidx]; 
	}

	fftwf_free(A_L_M_scalar1);
	fftwf_free(n_L_phi_scalar);
	fftwf_free(f_i_phi_scalar1);

	fftwf_free(A_L_M_scalar2);
	fftwf_free(f_i_phi_scalar2);

	fftwf_free(A_L_M_tensor);
	fftwf_free(n_L_phi_tensor);
	fftwf_free(f_i_phi_tensor);

	mkl_set_num_threads_local(0);
	} // End of parallel region

	fftwf_destroy_plan(plan_c2c_scalar);
	fftwf_destroy_plan(plan_c2c_tensor);
	return t_cubic;

}


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
		  const double complex *kappa_i_L_scalar1, const double complex *kappa_i_L_scalar2, const double complex *kappa_i_L_tensor){
 
	double t_cubic = 0.;
	int nffts[1] = {nphi};
	fftw_plan plan_c2c_scalar, plan_c2c_tensor;

	int ndeltaL_scalar = 2; // deltaL = -1, 1 for scalar
	int ndeltaL_tensor = 5; // deltaL = -2, -1, 0, 1, 2 for tensor

	//const int deltaL_list_scalar[] = {-1, 1};             //scalar
	//const int deltaL_list_tensor[] = {-2, -1, 0, 1, 2};	  //tensor


	// Plan fft on temporary arrays now in order to avoid having to run the planner
	// in a omp critical region later.
	double complex *A_L_M_scalarp = fftw_malloc(sizeof *A_L_M_scalarp * npol * ndeltaL_scalar * nL * nphi);
	double complex *n_L_phi_scalarp = fftw_malloc(sizeof *n_L_phi_scalarp * npol * ndeltaL_scalar * nL * nphi);

	plan_c2c_scalar = fftw_plan_many_dft(1, nffts, npol * ndeltaL_scalar * nL,
				       A_L_M_scalarp, NULL,
				       1, nphi,
				       n_L_phi_scalarp, NULL,
				       1, nphi,
					   FFTW_BACKWARD,
				       FFTW_MEASURE);
	fftw_free(A_L_M_scalarp);
	fftw_free(n_L_phi_scalarp);

	double complex *A_L_M_tensorp = fftw_malloc(sizeof *A_L_M_tensorp * npol * ndeltaL_tensor * nL * nphi);
	double complex *n_L_phi_tensorp = fftw_malloc(sizeof *n_L_phi_tensorp * npol * ndeltaL_tensor * nL * nphi);

	plan_c2c_tensor = fftw_plan_many_dft(1, nffts, npol * ndeltaL_tensor * nL,
				       A_L_M_tensorp, NULL,
				       1, nphi,
				       n_L_phi_tensorp, NULL,
				       1, nphi,
					   FFTW_BACKWARD,
				       FFTW_MEASURE);
	fftw_free(A_L_M_tensorp);
	fftw_free(n_L_phi_tensorp);

	#pragma omp parallel 
	{
	mkl_set_num_threads_local(1);

	double complex *A_L_M_scalar1 = fftw_malloc(sizeof *A_L_M_scalar1 * npol * ndeltaL_scalar * nL * nphi);
	double complex *n_L_phi_scalar = fftw_malloc(sizeof *n_L_phi_scalar * npol * ndeltaL_scalar * nL * nphi);
	double complex *f_i_phi_scalar1 = fftw_malloc(sizeof *f_i_phi_scalar1 * nufact * nphi);

	double complex *A_L_M_scalar2 = fftw_malloc(sizeof *A_L_M_scalar2 * npol * ndeltaL_scalar * nL * nphi);
	double complex*f_i_phi_scalar2 = fftw_malloc(sizeof *f_i_phi_scalar2 * nufact * nphi);

	double complex *A_L_M_tensor = fftw_malloc(sizeof *A_L_M_tensor * npol * ndeltaL_tensor * nL * nphi);
	double complex *n_L_phi_tensor = fftw_malloc(sizeof *n_L_phi_tensor * npol * ndeltaL_tensor * nL * nphi);
	double complex *f_i_phi_tensor = fftw_malloc(sizeof *f_i_phi_tensor * nufact * nphi);


	if (A_L_M_scalar1 == NULL || n_L_phi_scalar == NULL || f_i_phi_scalar1 == NULL ||
	    A_L_M_scalar2 == NULL || f_i_phi_scalar2 == NULL ||
	    A_L_M_tensor == NULL || n_L_phi_tensor == NULL || f_i_phi_tensor == NULL){

	    fftw_free(A_L_M_scalar1);
	    fftw_free(n_L_phi_scalar);
	    fftw_free(f_i_phi_scalar1);

		fftw_free(A_L_M_scalar2);
		fftw_free(f_i_phi_scalar2);

	    fftw_free(A_L_M_tensor);
	    fftw_free(n_L_phi_tensor);
	    fftw_free(f_i_phi_tensor);
	    exit(1);
		}

	#pragma omp for reduction (+:t_cubic)
	for (ptrdiff_t tidx=0; tidx<ntheta; tidx++){

		backward_dp_mixed_sst(L_list,
			  nL, npol, 
			  n_scalar1, n_scalar2, n_tensor,
			  a_ell_m, y_M_L + tidx * nL * nL,
			  w3j_product_scalar1, w3j_product_scalar2, w3j_product_tensor,
			  prefactors_scalar1, prefactors_scalar2, prefactors_tensor, 
			  A_L_M_scalar1, A_L_M_scalar2, A_L_M_tensor,
			  Lmax, nell,
			  n_L_phi_scalar, n_L_phi_tensor,
			  plan_c2c_scalar, plan_c2c_tensor, 
			  f_i_phi_scalar1, f_i_phi_scalar2, f_i_phi_tensor,
			  nufact, nphi,
			  kappa_i_L_scalar1, kappa_i_L_scalar2, kappa_i_L_tensor);

		t_cubic += t_cubic_on_ring_dp_sst(rule, weights, f_i_phi_scalar1, f_i_phi_scalar2, f_i_phi_tensor, nrule, nphi)
			*ct_weights[tidx];  
	}

	fftw_free(A_L_M_scalar1);
	fftw_free(n_L_phi_scalar);
	fftw_free(f_i_phi_scalar1);

	fftw_free(A_L_M_scalar2);
	fftw_free(f_i_phi_scalar2);

	fftw_free(A_L_M_tensor);
	fftw_free(n_L_phi_tensor);
	fftw_free(f_i_phi_tensor);

	mkl_set_num_threads_local(0);
	} // End of parallel region

	fftw_destroy_plan(plan_c2c_scalar);
	fftw_destroy_plan(plan_c2c_tensor);
	return t_cubic;

}

