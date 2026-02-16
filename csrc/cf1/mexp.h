#pragma once

#include <cassert>
#include <vector>

#include "blas1.h"
#include "blas2.h"

namespace phlib {

//====================================================
// mexpv
//   y = ( sum_{k=0..right} poi[k] * P^k x ) / weight
// where multiplication by P is done via dgemv(cf1_matrix,...)
//
// P: size n
// poi: size >= right+1
// x,y,xi: size n (xi is work vector)
//====================================================
template<typename TR>
inline void mexpv(TR,
                  const std::vector<double>& P,
                  const std::vector<double>& poi,
                  int right,
                  double weight,
                  const std::vector<double>& x,
                  std::vector<double>& y,
                  std::vector<double>& xi) {
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
  daxpy(poi[0], xi, y);

  for (size_t k = 1; k <= static_cast<size_t>(right); ++k) {
    dgemv(TR{}, 1.0, P, xi, 0.0, xi);
    daxpy(poi[k], xi, y);
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
template<typename TR>
inline void mexp_conv(TR,
                      const std::vector<double>& P,
                      double qv,
                      const std::vector<double>& poi,
                      int right,
                      double weight,
                      const std::vector<double>& x,
                      const std::vector<double>& y,
                      std::vector<double>& z,
                      std::vector<double>& H,
                      std::vector<double>& xi,
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
  for (size_t i = 0; i <= static_cast<size_t>(right + 1); ++i) {
    assert(vc[i].size() == n);
    dfill(vc[i], 0.0);
  }

  // -------------------------------------------------
  // Backward build: vc[l] for l=1..right+1
  // -------------------------------------------------
  // vc[right+1] += poi[right+1] * y
  {
    auto& v = vc[static_cast<size_t>(right + 1)];
    daxpy(poi[static_cast<size_t>(right + 1)], y, v);
  }

  for (size_t l = static_cast<size_t>(right); l >= 1; --l) {
    const auto& v_next = vc[l + 1];
    auto& v_cur  = vc[l];

    dgemv(notTR{}, 1.0, P, v_next, 0.0, v_cur);
    daxpy(poi[l], y, v_cur);
  }

  // -------------------------------------------------
  // Forward accumulation for z and H
  // -------------------------------------------------
  dcopy(x, xi);
  dfill(z, 0.0);

  // z += poi[0] * xi
  daxpy(poi[0], xi, z);

  // H += (1/qv/weight) * xi * vc[1]^T   (packed update via dger)
  {
    const auto& v1 = vc[1];
    dger(1.0 / qv / weight, xi, v1, H);
  }
  for (size_t l = 1; l <= static_cast<size_t>(right); ++l) {
    dgemv(TR{}, 1.0, P, xi, 0.0, xi);
    daxpy(poi[l], xi, z);
    const auto& v_next = vc[l + 1];
    dger(1.0 / qv / weight, xi, v_next, H);
  }
  dscal(1.0 / weight, z);
}

} // namespace marlib
