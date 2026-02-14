/**
 * @file cf1utils_span.h
 * @brief Utils for CF1 (std::span + vector<vector> workspace version)
 */

#pragma once

#include <span>
#include <vector>
#include <cmath>
#include <limits>
#include <cassert>

#include "blas1_span.h"
#include "blas2_span.h"
#include "mexp_span.h"
#include "poisson.h"   // rightbound, pmf (project-specific)

// #include "debug.h"  // optional

namespace marlib {

//====================================================
// small utils
//====================================================
inline void cf1_swap(int i, int j,
                     std::span<double> alpha,
                     std::span<double> rate) {
  const double w = rate[static_cast<size_t>(j)] / rate[static_cast<size_t>(i)];
  alpha[static_cast<size_t>(i)] += (1.0 - w) * alpha[static_cast<size_t>(j)];
  alpha[static_cast<size_t>(j)] *= w;

  const double tmp = rate[static_cast<size_t>(j)];
  rate[static_cast<size_t>(j)] = rate[static_cast<size_t>(i)];
  rate[static_cast<size_t>(i)] = tmp;
}

inline void cf1_sort(std::span<double> alpha,
                     std::span<double> rate) {
  assert(alpha.size() == rate.size());
  const int n = static_cast<int>(alpha.size());
  for (int i = 0; i < n - 1; ++i) {
    if (rate[static_cast<size_t>(i)] > rate[static_cast<size_t>(i + 1)]) {
      cf1_swap(i, i + 1, alpha, rate);
      int j = i;
      while (j > 0 && rate[static_cast<size_t>(j - 1)] > rate[static_cast<size_t>(j)]) {
        cf1_swap(j - 1, j, alpha, rate);
        --j;
      }
    }
  }
}

//====================================================
// emstep (span inputs + vector<vector> workspaces)
//====================================================
inline double emstep(cf1_matrix, trans,
                     double omega,
                     std::span<const double> alpha,
                     std::span<const double> rate,
                     double& new_omega,
                     std::span<double> new_alpha,
                     std::span<double> new_rate,
                     std::span<const double> time,
                     std::span<const int> num,
                     std::span<const int> type,
                     std::span<const double> P,  // uniformized coeff vector (size n)
                     double qv,
                     std::span<double> prob,     // poisson prob workspace
                     double eps,
                     std::span<double> tmp,      // size n
                     std::span<double> pi2,      // size n
                     double& en0,
                     std::span<double> eb,       // size n
                     std::span<double> eb2,      // size n
                     std::span<double> ey,       // size n
                     std::span<double> en,       // size 2n
                     std::span<double> h0,       // size 2n
                     std::span<double> blf,      // size dsize+2
                     std::span<double> blf2,     // size dsize+2
                     std::vector<std::vector<double>>& vb,   // (dsize+2) x n
                     std::vector<std::vector<double>>& vb2,  // (dsize+2) x n
                     std::span<double> xi,       // size n
                     std::vector<std::vector<double>>& vctmp, // >= (max_right+2) x n (workspace for mexp_conv)
                     std::span<double> lscal) {  // size dsize+1

  const int n = static_cast<int>(alpha.size());
  const int dsize = static_cast<int>(time.size());

  // basic checks
  assert(rate.size() == static_cast<size_t>(n));
  assert(new_alpha.size() == static_cast<size_t>(n));
  assert(new_rate.size() == static_cast<size_t>(n));
  assert(P.size() == static_cast<size_t>(n));

  assert(num.size() == static_cast<size_t>(dsize));
  assert(type.size() == static_cast<size_t>(dsize));

  assert(tmp.size() == static_cast<size_t>(n));
  assert(pi2.size() == static_cast<size_t>(n));
  assert(eb.size() == static_cast<size_t>(n));
  assert(eb2.size() == static_cast<size_t>(n));
  assert(ey.size() == static_cast<size_t>(n));
  assert(xi.size() == static_cast<size_t>(n));

  assert(en.size() == static_cast<size_t>(2 * n));
  assert(h0.size() == static_cast<size_t>(2 * n));

  assert(blf.size() >= static_cast<size_t>(dsize + 2));
  assert(blf2.size() >= static_cast<size_t>(dsize + 2));
  assert(lscal.size() == static_cast<size_t>(dsize + 1));

  // vb/vb2 sizes
  assert(vb.size() >= static_cast<size_t>(dsize + 2));
  assert(vb2.size() >= static_cast<size_t>(dsize + 2));
  for (int k = 0; k <= dsize + 1; ++k) {
    assert(vb[static_cast<size_t>(k)].size() == static_cast<size_t>(n));
    assert(vb2[static_cast<size_t>(k)].size() == static_cast<size_t>(n));
  }

  int right = 0;
  double weight = 1.0;

  lscal[0] = 0.0;

  en0 = 0.0;
  dfill(eb, 0.0);
  dfill(eb2, 0.0);
  dfill(en, 0.0);

  // vb[0] = 1, vb2[0] = 0 with last entry = rate[n-1]
  {
    auto vb0  = std::span<double>(vb[0].data(), n);
    auto vb20 = std::span<double>(vb2[0].data(), n);
    dfill(vb0, 1.0);
    dfill(vb20, 0.0);
    vb20[static_cast<size_t>(n - 1)] = rate[static_cast<size_t>(n - 1)];
  }

  double llf = 0.0;

  // -----------------------
  // backward recursion
  // -----------------------
  for (int k = 1; k <= dsize; ++k) {
    const double t = time[static_cast<size_t>(k - 1)];
    const double x = static_cast<double>(num[static_cast<size_t>(k - 1)]);
    const int    u = type[static_cast<size_t>(k - 1)];

    auto vb_prev  = std::span<const double>(vb[static_cast<size_t>(k - 1)].data(), n);
    auto vb_k     = std::span<double>(vb[static_cast<size_t>(k)].data(), n);

    auto vb2_prev = std::span<const double>(vb2[static_cast<size_t>(k - 1)].data(), n);
    auto vb2_k    = std::span<double>(vb2[static_cast<size_t>(k)].data(), n);

    dcopy(vb_prev, vb_k);

    right  = rightbound(qv * t, eps);
    weight = pmf(qv * t, 0, right, prob); // prob[0..right] filled

    mexpv(cf1_matrix{}, notrans{},
          P, std::span<const double>(prob.data(), prob.size()),
          right, weight,
          std::span<const double>(vb_k.data(), n),
          vb_k,
          xi);

    if (std::fpclassify(x) != FP_ZERO) {
      // tmp = vb[k-1] - vb[k]
      dcopy(vb_prev, tmp);
      daxpy(-1.0, std::span<const double>(vb_k.data(), n), tmp);

      blf[static_cast<size_t>(k)] = ddot(alpha, tmp);
      if (!std::isfinite(blf[static_cast<size_t>(k)])) {
        new_omega = omega;
        dcopy(alpha, new_alpha);
        dcopy(rate, new_rate);
        return std::numeric_limits<double>::quiet_NaN();
      }

      llf += x * std::log(omega * blf[static_cast<size_t>(k)])
          +  x * lscal[static_cast<size_t>(k - 1)]
          -  std::lgamma(x + 1.0);

      en0 += x;
      daxpy(x / blf[static_cast<size_t>(k)], tmp, eb);
    } else {
      blf[static_cast<size_t>(k)] = 1.0; // avoid NaN
    }

    dcopy(vb2_prev, vb2_k);

    mexpv(cf1_matrix{}, notrans{},
          P, std::span<const double>(prob.data(), prob.size()),
          right, weight,
          std::span<const double>(vb2_k.data(), n),
          vb2_k,
          xi);

    if (u == 1) {
      blf2[static_cast<size_t>(k)] = ddot(alpha, std::span<const double>(vb2_k.data(), n));
      if (!std::isfinite(blf2[static_cast<size_t>(k)])) {
        new_omega = omega;
        dcopy(alpha, new_alpha);
        dcopy(rate, new_rate);
        return std::numeric_limits<double>::quiet_NaN();
      }

      llf += std::log(omega * blf2[static_cast<size_t>(k)])
          +  lscal[static_cast<size_t>(k - 1)];

      en0 += 1.0;
      daxpy(1.0 / blf2[static_cast<size_t>(k)],
            std::span<const double>(vb2_k.data(), n),
            eb2);
    }

    // scaling by vb2[k]
    const double scale = dasum(std::span<const double>(vb2_k.data(), n));
    dscal(1.0 / scale, vb_k);
    dscal(1.0 / scale, vb2_k);
    lscal[static_cast<size_t>(k)] = lscal[static_cast<size_t>(k - 1)] + std::log(scale);
  }

  // -----------------------
  // tail part
  // -----------------------
  const double barblf = ddot(alpha, std::span<const double>(vb[static_cast<size_t>(dsize)].data(), n))
                      * std::exp(lscal[static_cast<size_t>(dsize)]);

  llf += -omega * (1.0 - barblf);

  daxpy(omega * std::exp(lscal[static_cast<size_t>(dsize)]),
        std::span<const double>(vb[static_cast<size_t>(dsize)].data(), n),
        eb);

  // -----------------------
  // compute pi2
  // -----------------------
  double cum = 0.0;
  for (int i = 0; i < n - 1; ++i) {
    cum += alpha[static_cast<size_t>(i)];
    pi2[static_cast<size_t>(i)] = cum / rate[static_cast<size_t>(i)];
  }
  pi2[static_cast<size_t>(n - 1)] = 1.0 / rate[static_cast<size_t>(n - 1)];

  // -----------------------
  // sojourn recursion (convolution)
  // -----------------------
  dfill(tmp, 0.0);

  // tmp += (omega*exp(lscal[dsize-1]) - num[dsize-1]/blf[dsize]) * pi2
  daxpy(omega * std::exp(lscal[static_cast<size_t>(dsize - 1)])
          - static_cast<double>(num[static_cast<size_t>(dsize - 1)]) / blf[static_cast<size_t>(dsize)],
        pi2, tmp);

  if (type[static_cast<size_t>(dsize - 1)] == 1) {
    daxpy(1.0 / blf2[static_cast<size_t>(dsize)], alpha, tmp);
  }

  right  = rightbound(qv * time[static_cast<size_t>(dsize - 1)], eps);
  weight = pmf(qv * time[static_cast<size_t>(dsize - 1)], 0, right + 1, prob);

  dfill(h0, 0.0);

  // vctmp workspace must be >= right+2, each size n
  assert(vctmp.size() >= static_cast<size_t>(right + 2));
  for (int i = 0; i <= right + 1; ++i) {
    assert(vctmp[static_cast<size_t>(i)].size() == static_cast<size_t>(n));
  }

  mexp_conv(cf1_matrix{}, trans{},
            P, qv,
            std::span<const double>(prob.data(), prob.size()),
            right, weight,
            tmp,
            std::span<const double>(vb2[static_cast<size_t>(dsize - 1)].data(), n),
            tmp,
            h0,
            xi,
            vctmp);

  daxpy(1.0, h0, en);

  for (int k = dsize - 1; k >= 1; --k) {
    const double t = time[static_cast<size_t>(k - 1)];
    const int    u = type[static_cast<size_t>(k - 1)];

    // tmp scaling: tmp *= exp(lscal[k-1] - lscal[k])
    dscal(std::exp(lscal[static_cast<size_t>(k - 1)] - lscal[static_cast<size_t>(k)]), tmp);

    // tmp += ( num[k]/blf[k+1]*exp(lscal[k-1]-lscal[k]) - num[k-1]/blf[k] ) * pi2
    const double term =
        static_cast<double>(num[static_cast<size_t>(k)]) / blf[static_cast<size_t>(k + 1)]
        * std::exp(lscal[static_cast<size_t>(k - 1)] - lscal[static_cast<size_t>(k)])
        - static_cast<double>(num[static_cast<size_t>(k - 1)]) / blf[static_cast<size_t>(k)];

    daxpy(term, pi2, tmp);

    if (u == 1) {
      daxpy(1.0 / blf2[static_cast<size_t>(k)], alpha, tmp);
    }

    right  = rightbound(qv * t, eps);
    weight = pmf(qv * t, 0, right + 1, prob);

    dfill(h0, 0.0);

    assert(vctmp.size() >= static_cast<size_t>(right + 2));
    mexp_conv(cf1_matrix{}, trans{},
              P, qv,
              std::span<const double>(prob.data(), prob.size()),
              right, weight,
              tmp,
              std::span<const double>(vb2[static_cast<size_t>(k - 1)].data(), n),
              tmp,
              h0,
              xi,
              vctmp);

    daxpy(1.0, h0, en);
  }

  // -----------------------
  // M-step
  // -----------------------
  for (int i = 0; i < n - 1; ++i) {
    ey[static_cast<size_t>(i)] =
        rate[static_cast<size_t>(i)] *
        (en[static_cast<size_t>(2 * i + 1)] + eb[static_cast<size_t>(i + 1)] * pi2[static_cast<size_t>(i)]) /
        (en[static_cast<size_t>(2 * i)]     + eb[static_cast<size_t>(i)]     * pi2[static_cast<size_t>(i)]);
  }

  const double denom_last =
      en[static_cast<size_t>(2 * (n - 1))] + eb[static_cast<size_t>(n - 1)] * pi2[static_cast<size_t>(n - 1)];

  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    eb[static_cast<size_t>(i)] = alpha[static_cast<size_t>(i)] * (eb[static_cast<size_t>(i)] + eb2[static_cast<size_t>(i)]);
    sum += eb[static_cast<size_t>(i)];
  }

  ey[static_cast<size_t>(n - 1)] = sum / denom_last;

  for (int i = 0; i < n; ++i) {
    eb[static_cast<size_t>(i)] /= sum;
  }

  new_omega = en0 + omega * barblf;

  dcopy(eb, new_alpha);
  dcopy(ey, new_rate);

  return llf;
}

//====================================================
// cf1emstep: allocates workspaces internally (vector + vector<vector>)
//====================================================
inline double cf1emstep(double omega,
                        std::span<const double> alpha,
                        std::span<const double> rate,
                        double& new_omega,
                        std::span<double> new_alpha,
                        std::span<double> new_rate,
                        std::span<const double> time,
                        std::span<const int> num,
                        std::span<const int> type,
                        double eps,
                        double ufactor) {
  const int n = static_cast<int>(alpha.size());
  const int dsize = static_cast<int>(time.size());

  assert(rate.size() == static_cast<size_t>(n));
  assert(new_alpha.size() == static_cast<size_t>(n));
  assert(new_rate.size() == static_cast<size_t>(n));
  assert(num.size() == static_cast<size_t>(dsize));
  assert(type.size() == static_cast<size_t>(dsize));
  assert(n > 0);
  assert(dsize >= 1);

  // work vectors
  std::vector<double> tmp_v(static_cast<size_t>(n));
  std::vector<double> pi2_v(static_cast<size_t>(n));
  std::vector<double> P_v(rate.begin(), rate.end()); // P = rate copy

  // uniformize P (in-place)
  double qv = unif(cf1_matrix{}, std::span<double>(P_v), ufactor);

  // tmax
  const double tmax = time[static_cast<size_t>(idamax(time))];
  const int max_right = rightbound(qv * tmax, eps) + 1;

  std::vector<double> prob_v(static_cast<size_t>(max_right + 2)); // generous
  std::vector<double> xi_v(static_cast<size_t>(n));

  double en0 = 0.0;
  std::vector<double> eb_v(static_cast<size_t>(n));
  std::vector<double> eb2_v(static_cast<size_t>(n));
  std::vector<double> en_v(static_cast<size_t>(2 * n));
  std::vector<double> h0_v(static_cast<size_t>(2 * n));
  std::vector<double> ey_v(static_cast<size_t>(n));

  std::vector<double> blf_v(static_cast<size_t>(dsize + 2));
  std::vector<double> blf2_v(static_cast<size_t>(dsize + 2));
  std::vector<double> lscal_v(static_cast<size_t>(dsize + 1));

  // vb/vb2: (dsize+2) x n
  std::vector<std::vector<double>> vb(static_cast<size_t>(dsize + 2),
                                      std::vector<double>(static_cast<size_t>(n)));
  std::vector<std::vector<double>> vb2(static_cast<size_t>(dsize + 2),
                                       std::vector<double>(static_cast<size_t>(n)));

  // vctmp: (max_right+2) x n  (workspace for mexp_conv)
  std::vector<std::vector<double>> vctmp(static_cast<size_t>(max_right + 2),
                                         std::vector<double>(static_cast<size_t>(n)));

  const double llf = emstep(cf1_matrix{}, trans{},
                            omega, alpha, rate,
                            new_omega, new_alpha, new_rate,
                            time, num, type,
                            std::span<const double>(P_v), qv,
                            std::span<double>(prob_v),
                            eps,
                            std::span<double>(tmp_v),
                            std::span<double>(pi2_v),
                            en0,
                            std::span<double>(eb_v),
                            std::span<double>(eb2_v),
                            std::span<double>(ey_v),
                            std::span<double>(en_v),
                            std::span<double>(h0_v),
                            std::span<double>(blf_v),
                            std::span<double>(blf2_v),
                            vb, vb2,
                            std::span<double>(xi_v),
                            vctmp,
                            std::span<double>(lscal_v));

  return llf;
}

} // namespace marlib
