#pragma once

#include <span>
#include <cassert>
#include <cstddef>

namespace marlib {

struct trans {};
struct notrans {};

struct cf1_matrix {}; // coefficient vector A of length n

// -------------------------------------------------
// dgemv for cf1_matrix
// A: size n
// x,y: size n
//
// trans:
//   for i=n-1..1: y[i] = alpha*( x[i]*(1-A[i]) + x[i-1]*A[i-1] ) + beta*y[i]
//   y[0] = alpha*( x[0]*(1-A[0]) ) + beta*y[0]
//
// notrans:
//   for i=0..n-2: y[i] = alpha*( x[i]*(1-A[i]) + x[i+1]*A[i] ) + beta*y[i]
//   y[n-1] = alpha*( x[n-1]*(1-A[n-1]) ) + beta*y[n-1]
// -------------------------------------------------

inline void dgemv(cf1_matrix, trans,
                  double alpha,
                  std::span<const double> A,
                  std::span<const double> x,
                  double beta,
                  std::span<double> y) {
  const int n = static_cast<int>(A.size());
  assert(static_cast<int>(x.size()) == n);
  assert(static_cast<int>(y.size()) == n);
  if (n == 0) return;

  for (int i = n - 1; i >= 1; --i) {
    y[static_cast<size_t>(i)] =
      alpha * (x[static_cast<size_t>(i)] * (1.0 - A[static_cast<size_t>(i)]) +
               x[static_cast<size_t>(i - 1)] * A[static_cast<size_t>(i - 1)]) +
      beta * y[static_cast<size_t>(i)];
  }
  y[0] = alpha * (x[0] * (1.0 - A[0])) + beta * y[0];
}

inline void dgemv(cf1_matrix, notrans,
                  double alpha,
                  std::span<const double> A,
                  std::span<const double> x,
                  double beta,
                  std::span<double> y) {
  const int n = static_cast<int>(A.size());
  assert(static_cast<int>(x.size()) == n);
  assert(static_cast<int>(y.size()) == n);
  if (n == 0) return;

  for (int i = 0; i < n - 1; ++i) {
    y[static_cast<size_t>(i)] =
      alpha * (x[static_cast<size_t>(i)] * (1.0 - A[static_cast<size_t>(i)]) +
               x[static_cast<size_t>(i + 1)] * A[static_cast<size_t>(i)]) +
      beta * y[static_cast<size_t>(i)];
  }
  y[static_cast<size_t>(n - 1)] =
    alpha * (x[static_cast<size_t>(n - 1)] * (1.0 - A[static_cast<size_t>(n - 1)])) +
    beta * y[static_cast<size_t>(n - 1)];
}

// -------------------------------------------------
// dger for cf1_matrix (same packed layout as original)
// Atri: length (2*n - 1)
// updates:
//   for i=0..n-2 with p=2*i:
//     Atri[p]   += alpha*x[i]*y[i]
//     Atri[p+1] += alpha*x[i]*y[i+1]
//   Atri[2*n-2] += alpha*x[n-1]*y[n-1]
// -------------------------------------------------

inline void dger(cf1_matrix,
                 double alpha,
                 std::span<const double> x,
                 std::span<const double> y,
                 std::span<double> Atri) {
  const int n = static_cast<int>(x.size());
  assert(static_cast<int>(y.size()) == n);
  assert(static_cast<int>(Atri.size()) == 2 * n - 1);
  if (n == 0) return;

  for (int i = 0, p = 0; i < n - 1; ++i, p += 2) {
    Atri[static_cast<size_t>(p)]     += alpha * x[static_cast<size_t>(i)] * y[static_cast<size_t>(i)];
    Atri[static_cast<size_t>(p + 1)] += alpha * x[static_cast<size_t>(i)] * y[static_cast<size_t>(i + 1)];
  }
  Atri[static_cast<size_t>(2 * n - 2)] += alpha * x[static_cast<size_t>(n - 1)] * y[static_cast<size_t>(n - 1)];
}

} // namespace marlib
