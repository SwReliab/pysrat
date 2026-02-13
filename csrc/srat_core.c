#include "srat_core.h"

double srat_sum(const double* x, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += x[i];
  return s;
}
