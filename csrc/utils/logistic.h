/*
  logistic distribution helpers
*/

#ifndef LOGISTIC_H
#define LOGISTIC_H

#ifdef __cplusplus
extern "C" {
#endif

  double logistic_cdf(double x, double loc, double scale);
  double logistic_sf(double x, double loc, double scale);
  double logistic_logcdf(double x, double loc, double scale);
  double logistic_logsf(double x, double loc, double scale);
  double logistic_logpdf(double x, double loc, double scale);

#ifdef __cplusplus
}
#endif

#endif
