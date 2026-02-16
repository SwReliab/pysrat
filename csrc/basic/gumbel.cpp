#include "gumbel.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace Revd {

static inline void check_scale(double scale) {
  if (!(scale > 0.0) || !std::isfinite(scale)) {
    throw std::invalid_argument("scale must be > 0 and finite.");
  }
}

double dgumbel(double x, double loc, double scale, bool log) {
  check_scale(scale);
  const double z = (x - loc) / scale;
  const double y = std::exp(-z);
  if (!log) {
    return y * std::exp(-y) / scale;
  }
  return -z - y - std::log(scale);
}

double dgumbel_min(double x, double loc, double scale, bool log) {
  return dgumbel(-x, loc, scale, log);
}

double pgumbel(double q, double loc, double scale, bool lower, bool log) {
  check_scale(scale);
  const double z = (q - loc) / scale;
  const double y = std::exp(-z);

  if (lower) {
    if (!log) {
      return std::exp(-y);
    }
    return -y;
  }

  if (!log) {
    return 1.0 - std::exp(-y);
  }
  return std::log1p(-std::exp(-y));
}

double pgumbel_min(double q, double loc, double scale, bool lower, bool log) {
  return pgumbel(-q, loc, scale, !lower, log);
}

double qgumbel(double p, double loc, double scale, bool lower, bool log) {
  check_scale(scale);

  if (log) {
    p = std::exp(p);
  }
  if (!lower) {
    p = 1.0 - p;
  }

  if (!(p > 0.0) || !(p < 1.0)) {
    if (p == 0.0) return -std::numeric_limits<double>::infinity();
    if (p == 1.0) return std::numeric_limits<double>::infinity();
    return std::numeric_limits<double>::quiet_NaN();
  }

  return -scale * std::log(-std::log(p)) + loc;
}

double qgumbel_min(double p, double loc, double scale, bool lower, bool log) {
  check_scale(scale);

  if (log) {
    p = std::exp(p);
  }
  if (lower) {
    p = 1.0 - p;
  }

  if (!(p > 0.0) || !(p < 1.0)) {
    if (p == 0.0) return -std::numeric_limits<double>::infinity();
    if (p == 1.0) return std::numeric_limits<double>::infinity();
    return std::numeric_limits<double>::quiet_NaN();
  }

  return scale * std::log(-std::log(p)) - loc;
}

} // namespace Revd
