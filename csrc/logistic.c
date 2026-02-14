/*
  logistic distribution helpers
*/

#include <math.h>
#include "logistic.h"

static double logistic_z(double x, double loc, double scale) {
  return (x - loc) / scale;
}

double logistic_cdf(double x, double loc, double scale) {
  const double z = logistic_z(x, loc, scale);
  if (z >= 0.0) {
    const double ez = exp(-z);
    return 1.0 / (1.0 + ez);
  }
  const double ez = exp(z);
  return ez / (1.0 + ez);
}

double logistic_sf(double x, double loc, double scale) {
  const double z = logistic_z(x, loc, scale);
  if (z >= 0.0) {
    const double ez = exp(-z);
    return ez / (1.0 + ez);
  }
  const double ez = exp(z);
  return 1.0 / (1.0 + ez);
}

double logistic_logcdf(double x, double loc, double scale) {
  const double z = logistic_z(x, loc, scale);
  if (z >= 0.0) {
    return -log1p(exp(-z));
  }
  return z - log1p(exp(z));
}

double logistic_logsf(double x, double loc, double scale) {
  const double z = logistic_z(x, loc, scale);
  if (z >= 0.0) {
    return -z - log1p(exp(-z));
  }
  return -log1p(exp(z));
}

double logistic_logpdf(double x, double loc, double scale) {
  return -log(scale) + logistic_logcdf(x, loc, scale) + logistic_logsf(x, loc, scale);
}
