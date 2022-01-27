#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#ifndef PI
#define PI 3.14159265358979323846
#endif
#define TWOPI 6.283185307179586476925287
#define ABS_ERR 1.49e-08 
#define REL_ERR 1.49e-08

double R1_re(const double alpha,const double psi){
  const double cos_psi = cos(psi);
  const double X = 1+alpha*alpha-2*alpha*cos_psi;
  const double denom = sqrt(X*X*X);
  const double direct =  0.5 * (alpha * alpha - alpha * cos_psi )  / denom;
  const double indirect = 0.5 * alpha * cos_psi;
  return direct + indirect;
}

double R1_im(const double alpha,const double psi){
  const double cos_psi = cos(psi);
  const double sin_psi = sin(psi);
  const double X = 1+alpha*alpha-2*alpha*cos_psi;
  const double denom = sqrt(X*X*X);
  const double direct =  -1 * alpha * sin_psi  / denom;
  const double indirect = alpha * sin_psi;
  return direct + indirect;
}

double integrand_re(const double nt,const double alpha,const double psi0,const double l0){
  const double psi = psi0 + nt * ( sqrt(alpha*alpha*alpha) -  1.0 );
  const double l = nt + l0;
  const double cosl = cos(l);
  const double sinl = sin(l);
  return -2 * alpha * (sinl * R1_re(alpha,psi) + cosl * R1_im(alpha,psi));
}

double integrand_im(const double nt,const double alpha,const double psi0,const double l0){
  const double psi = psi0 + nt * ( sqrt(alpha*alpha*alpha) -  1.0 );
  const double l = nt + l0;
  const double cosl = cos(l);
  const double sinl = sin(l);
  return 2 * alpha * (cosl * R1_re(alpha,psi) - sinl * R1_im(alpha,psi));
}

struct integrand_params {double alpha,psi0,l0}; 
double integRe(double t, void * params){
	struct integrand_params * p = (struct integrand_params *) params;
	return integrand_re(t,p->alpha,p->psi0,p->l0);
}
double integIm(double t, void * params){
	struct integrand_params * p = (struct integrand_params *) params;
	return integrand_im(t,p->alpha,p->psi0,p->l0);
}

double dzRe(const double alpha,const double t, const double t0, const double psi0, const double l0){
	gsl_function F;
	F.function = &integRe;
	F.params = &(struct integrand_params){alpha,psi0,l0};
	double result,error;
	size_t neval;
	gsl_set_error_handler_off();
	gsl_integration_qng(&F,t0,t,ABS_ERR,REL_ERR,&result,&error,&neval);
	return result;
}

double dzIm(const double alpha,const double t, const double t0, const double psi0, const double l0){
	gsl_function F;
	F.function = &integIm;
	F.params = &(struct integrand_params){alpha,psi0,l0};
	double result,error;
	size_t neval;
	gsl_set_error_handler_off();
	gsl_integration_qng(&F,t0,t,ABS_ERR,REL_ERR,&result,&error,&neval);
	return result;
}
// 
double Rp1_re(const double alpha,const double psi){
  const double cos_psi = cos(psi);
  const double X = 1+alpha*alpha-2*alpha*cos_psi;
  const double denom = sqrt(X*X*X);
  const double direct =  0.5 * (1 - alpha * cos_psi)  / denom;
  const double indirect = 0.5 * cos_psi / alpha / alpha;
  return direct + indirect;
}

double Rp1_im(const double alpha,const double psi){
  const double cos_psi = cos(psi);
  const double sin_psi = sin(psi);
  const double X = 1+alpha*alpha-2*alpha*cos_psi;
  const double denom = sqrt(X*X*X);
  const double direct =   alpha * sin_psi  / denom;
  const double indirect = -1 * sin_psi / alpha / alpha;
  return direct + indirect;
}

double integrand1_re(const double n1t,const double alpha,const double psi0,const double l0){
  const double psi = psi0 + n1t * (1 - 1 / sqrt(alpha*alpha*alpha)); 
  const double l = n1t + l0;
  const double cosl = cos(l);
  const double sinl = sin(l);
  return -2 * (sinl * Rp1_re(alpha,psi) + cosl * Rp1_im(alpha,psi));
}

double integrand1_im(const double n1t,const double alpha,const double psi0,const double l0){
  const double psi = psi0 + n1t * (1 - 1 / sqrt(alpha*alpha*alpha)); 
  const double l = n1t + l0;
  const double cosl = cos(l);
  const double sinl = sin(l);
  return 2 * (cosl * Rp1_re(alpha,psi) - sinl * Rp1_im(alpha,psi));
}

double integ1Re(double t, void * params){
	struct integrand_params * p = (struct integrand_params *) params;
	return integrand1_re(t,p->alpha,p->psi0,p->l0);
}
double integ1Im(double t, void * params){
	struct integrand_params * p = (struct integrand_params *) params;
	return integrand1_im(t,p->alpha,p->psi0,p->l0);
}

double dz1Re(const double alpha,const double t, const double t0, const double psi0, const double l0){
	gsl_function F;
	F.function = &integ1Re;
	F.params = &(struct integrand_params){alpha,psi0,l0};
	double result,error;
	size_t neval;
	gsl_set_error_handler_off();
	gsl_integration_qng(&F,t0,t,ABS_ERR,REL_ERR,&result,&error,&neval);
	return result;
}

double dz1Im(const double alpha,const double t, const double t0, const double psi0, const double l0){
	gsl_function F;
	F.function = &integ1Im;
	F.params = &(struct integrand_params){alpha,psi0,l0};
	double result,error;
	size_t neval;
	gsl_set_error_handler_off();
	gsl_integration_qng(&F,t0,t,ABS_ERR,REL_ERR,&result,&error,&neval);
	return result;
}
