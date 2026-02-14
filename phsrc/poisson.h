/**
 * @file poisson.h
 * @brief Compute p.m.f. of poisson distribution (std::vector version)
 */

#pragma once

#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>

namespace marlib {

/**
 * @brief Compute the right bound of Poisson range for a given error tolerance
 * @param lambda a Poisson rate (mean)
 * @param eps a value of error tolerance
 * @return right bound
 */
int rightbound(double lambda, double eps);

/**
 * @brief Compute poisson probability vector (unnormalized weights)
 *
 * prob[i] (i = 0..n-1) corresponds to k = left + i.
 *
 * This function fills prob with *scaled* weights to avoid numerical underflow:
 *   prob[i] = exp(log P(k) - logP_ref)
 * where logP_ref = max_{k in [left,right]} log P(k).
 *
 * Then,
 *   exact pmf P(k) = prob[i] / weight
 * where
 *   weight = sum_i prob[i].
 *
 * @param lambda Poisson parameter (> = 0)
 * @param left left bound (>= 0)
 * @param right right bound (>= left)
 * @param prob output weights sized (right-left+1)
 * @return weight normalization constant
 */
inline double pmf(double lambda, int left, int right, std::vector<double>& prob) {
  assert(lambda >= 0.0);
  assert(left >= 0);
  assert(right >= left);

  const int n = right - left + 1;
  prob.assign(n, 0.0);

  // Special case: lambda == 0 => P(X=0)=1 else 0
  if (std::fpclassify(lambda) == FP_ZERO) {
    // Only k=0 has mass
    for (int i = 0; i < n; ++i) {
      const int k = left + i;
      prob[i] = (k == 0) ? 1.0 : 0.0;
    }
    // Here "weight" is already the exact normalization
    double weight = 0.0;
    for (double v : prob) weight += v;
    // If the range doesn't include 0, weight==0 (caller should handle)
    return weight;
  }

  // Compute log P(k) = k*log(lambda) - lambda - log(k!)
  // using lgamma(k+1) for log(k!)
  const double loglam = std::log(lambda);

  std::vector<double> logp(n);
  double max_logp = -INFINITY;

  for (int i = 0; i < n; ++i) {
    const int k = left + i;
    // log(k!) = lgamma(k+1)
    const double lp = k * loglam - lambda - std::lgamma(static_cast<double>(k) + 1.0);
    logp[i] = lp;
    if (lp > max_logp) max_logp = lp;
  }

  // Convert to scaled weights
  double weight = 0.0;
  for (int i = 0; i < n; ++i) {
    const double w = std::exp(logp[i] - max_logp);
    prob[i] = w;
    weight += w;
  }

  return weight;
}

} // namespace marlib
