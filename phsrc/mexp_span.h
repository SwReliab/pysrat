#pragma once

#include <span>
#include <vector>
#include <cassert>

#include "blas1_span.h"
#include "blas2_span.h"

namespace marlib {

//====================================================
// mexpv
//   y = ( sum_{k=0..right} poi[k] * P^k x ) / weight
// where multiplication by P is done via dgemv(cf1_matrix,...)
//
// P: size n
// poi: size >= right+1
// x,y,xi: size n (xi is work vector)
//====================================================
template<typename MatT, typename TR>
inline void mexpv(MatT, TR,
                  std::span<const double> P,
                  std::span<const double> poi,
                  int right,
                  double weight,
                  std::span<const double> x,
                  std::span<double> y,
                  std::span<double> xi) {
  const size_t n = P.size();
  assert(x.size() == n);
  assert(y.size() == n);
  assert(xi.size() == n);
  assert(right >= 0);
  assert(poi.size() >= static_cast<size_t>(right + 1));
  assert(weight != 0.0);

  dcopy(x, xi);
  dfill(y, 0.0);

  // k=0
  daxpy(poi[0], std::span<const double>(xi.data(), xi.size()), y);

  for (int k = 1; k <= right; ++k) {
    dgemv(MatT{}, TR{}, 1.0, P,
          std::span<const double>(xi.data(), xi.size()),
          0.0, xi);
    daxpy(poi[static_cast<size_t>(k)],
          std::span<const double>(xi.data(), xi.size()),
          y);
  }

  dscal(1.0 / weight, y);
}

//====================================================
// not_ helper to switch trans <-> notrans
//====================================================
template<typename TR> struct not_;
template<> struct not_<trans>   { using Type = notrans; };
template<> struct not_<notrans> { using Type = trans; };

//====================================================
// mexp_conv
//
// This computes (z, H) given x, y and poisson weights, using vc workspace.
//
// Inputs:
//   P: size n
//   poi: size >= right+2  (note the code uses poi[right+1])
//   x,y,z,xi: size n
//   H: size (2*n-1)  (packed format compatible with dger(cf1_matrix))
//   vc: vector of (right+2) vectors, each size n  (workspace)
//
// IMPORTANT:
//   - vc is treated as WORKSPACE. Contents overwritten.
//   - vc must already be allocated with at least (right+2) vectors of size n.
//
//====================================================
template<typename MatT, typename TR>
inline void mexp_conv(MatT, TR,
                      std::span<const double> P,
                      double qv,
                      std::span<const double> poi,
                      int right,
                      double weight,
                      std::span<const double> x,
                      std::span<const double> y,
                      std::span<double> z,
                      std::span<double> H,
                      std::span<double> xi,
                      std::vector<std::vector<double>>& vc) {
  using notTR = typename not_<TR>::Type;

  const size_t n = P.size();
  assert(x.size() == n);
  assert(y.size() == n);
  assert(z.size() == n);
  assert(xi.size() == n);
  assert(H.size() == 2 * n - 1);
  assert(right >= 0);
  assert(weight != 0.0);
  assert(qv != 0.0);

  // need poi[0..right+1]
  assert(poi.size() >= static_cast<size_t>(right + 2));

  // vc workspace must be (right+2) x n
  assert(vc.size() >= static_cast<size_t>(right + 2));
  for (int i = 0; i <= right + 1; ++i) {
    assert(vc[static_cast<size_t>(i)].size() == n);
    dfill(std::span<double>(vc[static_cast<size_t>(i)].data(), n), 0.0);
  }

  // -------------------------------------------------
  // Backward build: vc[l] for l=1..right+1
  // -------------------------------------------------
  // vc[right+1] += poi[right+1] * y
  {
    auto v = std::span<double>(vc[static_cast<size_t>(right + 1)].data(), n);
    daxpy(poi[static_cast<size_t>(right + 1)], y, v);
  }

  for (int l = right; l >= 1; --l) {
    auto v_next = std::span<const double>(vc[static_cast<size_t>(l + 1)].data(), n);
    auto v_cur  = std::span<double>(vc[static_cast<size_t>(l)].data(), n);

    dgemv(MatT{}, notTR{}, 1.0, P, v_next, 0.0, v_cur);
    daxpy(poi[static_cast<size_t>(l)], y, v_cur);
  }

  // -------------------------------------------------
  // Forward accumulation for z and H
  // -------------------------------------------------
  dcopy(x, xi);
  dfill(z, 0.0);

  // z += poi[0] * xi
  daxpy(poi[0], std::span<const double>(xi.data(), n), z);

  // H += (1/qv/weight) * xi * vc[1]^T   (packed update via dger)
  {
    auto v1 = std::span<const double>(vc[1].data(), n);
    dger(MatT{}, 1.0 / qv / weight,
         std::span<const double>(xi.data(), n),
         v1,
         H);
  }

  for (int l = 1; l <= right; ++l) {
    dgemv(MatT{}, TR{}, 1.0, P,
          std::span<const double>(xi.data(), n),
          0.0, xi);

    daxpy(poi[static_cast<size_t>(l)],
          std::span<const double>(xi.data(), n),
          z);

    auto v_next = std::span<const double>(vc[static_cast<size_t>(l + 1)].data(), n);
    dger(MatT{}, 1.0 / qv / weight,
         std::span<const double>(xi.data(), n),
         v_next,
         H);
  }

  dscal(1.0 / weight, z);
}

} // namespace marlib
