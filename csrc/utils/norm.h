/*
  normal distribution helpers
*/

#ifndef NORM_H
#define NORM_H

#ifdef __cplusplus
extern "C" {
#endif

  double norm_phi(double z);
  double norm_Q(double z);
  double norm_logpdf(double x, double mu, double sig);

#ifdef __cplusplus
}
#endif

#endif
