#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

namespace LinkEval {

// --- local helpers (header-only) ---
static inline double norm_pdf_local(double x) {
  static const double INV_SQRT2PI = 0.39894228040143267793994605993438186848;
  return INV_SQRT2PI * std::exp(-0.5 * x * x);
}

static inline double norm_cdf_local(double x) {
  // Phi(x) = 0.5 * erfc(-x/sqrt(2))
  static const double INV_SQRT2 = 0.70710678118654752440084436210484903928;
  return 0.5 * std::erfc(-x * INV_SQRT2);
}

static inline double clip_local(double x, double lo, double hi) {
  return (x < lo) ? lo : ((x > hi) ? hi : x);
}

// Evaluates inverse link mu and derivative dmu/deta
// Supported: logit, probit, cloglog
inline void eval(const std::string& link,
                 const Eigen::VectorXd& eta,
                 Eigen::VectorXd& mu,
                 Eigen::VectorXd& dmu) {
  const int n = static_cast<int>(eta.size());
  mu.resize(n);
  dmu.resize(n);

  if (link == "logit") {
    // stable sigmoid
    for (int i = 0; i < n; ++i) {
      const double e = eta[i];
      if (e >= 0.0) {
        const double t = std::exp(-e);
        const double m = 1.0 / (1.0 + t);
        mu[i] = m;
        dmu[i] = m * (1.0 - m);
      } else {
        const double t = std::exp(e);
        const double m = t / (1.0 + t);
        mu[i] = m;
        dmu[i] = m * (1.0 - m);
      }
    }
    return;
  }

  if (link == "probit") {
    // mu = Phi(eta), dmu = phi(eta)
    for (int i = 0; i < n; ++i) {
      const double e = eta[i];
      mu[i] = norm_cdf_local(e);
      dmu[i] = norm_pdf_local(e);
    }
    return;
  }

  if (link == "cloglog") {
    // mu = 1 - exp(-exp(eta))
    // dmu = exp(eta) * exp(-exp(eta)) = exp(eta - exp(eta))
    //
    // For numerical stability, clamp eta to avoid exp(eta) overflow.
    // Values beyond ~700 overflow in double exp(). 50 already saturates mu≈1.
    const double ETA_CLIP = 50.0;
    for (int i = 0; i < n; ++i) {
      const double e0 = eta[i];
      const double e = clip_local(e0, -ETA_CLIP, ETA_CLIP);
      const double exp_e = std::exp(e);

      // exp(-exp_e) underflows safely to 0 for large exp_e
      const double t = std::exp(-exp_e);
      const double m = 1.0 - t;

      mu[i] = m;
      dmu[i] = exp_e * t;  // = exp(e - exp_e)
    }
    return;
  }

  throw std::invalid_argument("Unsupported link function: " + link);
}

}  // namespace LinkEval