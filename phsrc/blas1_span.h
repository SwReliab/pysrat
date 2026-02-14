#pragma once

#include <span>
#include <cmath>
#include <cassert>
#include <cstddef>

namespace marlib {

// dot product
inline double ddot(std::span<const double> x, std::span<const double> y) {
  assert(x.size() == y.size());
  double s = 0.0;
  for (size_t i = 0; i < x.size(); ++i) s += x[i] * y[i];
  return s;
}

// sum of absolute values
inline double dasum(std::span<const double> x) {
  double s = 0.0;
  for (double v : x) s += std::abs(v);
  return s;
}

// index of max absolute value (0-based)
inline int idamax(std::span<const double> x) {
  assert(!x.empty());
  int maxi = 0;
  double maxv = std::abs(x[0]);
  for (int i = 1; i < static_cast<int>(x.size()); ++i) {
    const double v = std::abs(x[static_cast<size_t>(i)]);
    if (v > maxv) { maxv = v; maxi = i; }
  }
  return maxi;
}

// fill
inline void dfill(std::span<double> x, double v) {
  for (double& e : x) e = v;
}

// copy
inline void dcopy(std::span<const double> x, std::span<double> y) {
  assert(x.size() == y.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] = x[i];
}

// scale
inline void dscal(double alpha, std::span<double> x) {
  for (double& e : x) e *= alpha;
}

// y += alpha * x
inline void daxpy(double alpha, std::span<const double> x, std::span<double> y) {
  assert(x.size() == y.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] += alpha * x[i];
}

// -------------------------------------------------
// optional convenience overloads (mutable x treated as const)
// -------------------------------------------------
inline double ddot(std::span<double> x, std::span<double> y) {
  return ddot(std::span<const double>(x.data(), x.size()),
              std::span<const double>(y.data(), y.size()));
}
inline void dcopy(std::span<double> x, std::span<double> y) {
  dcopy(std::span<const double>(x.data(), x.size()), y);
}
inline void daxpy(double alpha, std::span<double> x, std::span<double> y) {
  daxpy(alpha, std::span<const double>(x.data(), x.size()), y);
}

} // namespace marlib
