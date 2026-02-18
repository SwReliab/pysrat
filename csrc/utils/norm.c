/*
  normal distribution helpers
*/

#include <math.h>
#include "norm.h"

static const double SQRT2 = 1.41421356237309504880168872420969807857;
static const double INV_SQRT2PI = 0.39894228040143267793994605993438186848;

static double log_sqrt2pi(void) {
  const double pi = acos(-1.0);
  return 0.5 * log(2.0 * pi);
}

double norm_phi(double z) {
  return INV_SQRT2PI * exp(-0.5 * z * z);
}

double norm_Q(double z) {
  return 0.5 * erfc(z / SQRT2);
}

double norm_logpdf(double x, double mu, double sig) {
  const double z = (x - mu) / sig;
  return -log(sig) - log_sqrt2pi() - 0.5 * z * z;
}
