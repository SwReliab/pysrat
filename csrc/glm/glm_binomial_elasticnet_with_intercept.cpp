// glm_binomial_elasticnet_with_intercept.cpp
// ElasticNet Binomial GLM via IRLS (outer) + Coordinate Descent (inner)
// - aggregated binomial: y = success counts, n_trials = totals (can be real -> weights interpretation)
// - intercept is an explicit scalar (NEVER penalized)  << intercept列検出は一切しない >>
// - beta is p-vector; penalty_mask applies only to beta entries
// - standardize mask (0/1, length p):
//     1 -> center+scale column (safe because intercept is explicit)
//     0 -> leave as-is
//
// Return: GLMBinomialENetResult with convergence info (outer/inner iters, max deltas)

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "link_eval.h"

static inline double soft_threshold(double x, double t) {
  if (x > t) return x - t;
  if (x < -t) return x + t;
  return 0.0;
}

struct GLMBinomialENetResult {
  double intercept = 0.0;     // original (unstandardized) intercept
  Eigen::VectorXd beta;       // original (unstandardized) beta (length p)

  bool converged = false;
  int n_outer = 0;            // executed outer IRLS iterations
  int n_inner = 0;            // total executed inner CD sweeps (sum over outer)
  double max_delta = 0.0;     // final outer max(|Δb0|, max|Δbeta|)
  double max_delta_inner = 0.0; // last inner sweep max|Δbeta|
};

// with-intercept ElasticNet Binomial GLM
GLMBinomialENetResult glm_binomial_elasticnet_with_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,          // successes
    const Eigen::VectorXd& n_trials,   // totals
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,   // 0/1 length p
    const Eigen::VectorXi& penalty_mask,  // 0/1 length p (beta only)
    int max_iter,
    double tol,
    const std::string& link,
    double alpha,
    double lambd,
    double ridge,
    double eps_mu,
    double eps_dmu) {

  const int n = static_cast<int>(X.rows());
  const int p = static_cast<int>(X.cols());

  if (y.size() != n || n_trials.size() != n || offset.size() != n) {
    throw std::invalid_argument("y/n_trials/offset length must match X.rows()");
  }
  if (beta0.size() != p) {
    throw std::invalid_argument("beta0 length must match X.cols()");
  }
  if (standardize.size() != p) {
    throw std::invalid_argument("standardize length must match X.cols()");
  }
  if (penalty_mask.size() != p) {
    throw std::invalid_argument("penalty_mask length must match X.cols()");
  }
  if (max_iter <= 0) {
    GLMBinomialENetResult out;
    out.intercept = intercept0;
    out.beta = beta0;
    out.converged = false;
    out.n_outer = 0;
    out.n_inner = 0;
    out.max_delta = 0.0;
    out.max_delta_inner = 0.0;
    return out;
  }
  if (!(alpha >= 0.0 && alpha <= 1.0)) {
    throw std::invalid_argument("alpha must be in [0,1]");
  }
  if (!(lambd >= 0.0)) {
    throw std::invalid_argument("lambd must be >= 0");
  }
  if (!(tol > 0.0) || !std::isfinite(tol)) {
    throw std::invalid_argument("tol must be positive finite");
  }

  const double l1 = alpha * lambd;
  const double l2 = (1.0 - alpha) * lambd + ridge;
  const double tiny_sd = 1e-12;

  auto is_penalized = [&](int j) -> bool {
    return (penalty_mask[j] != 0);
  };

  // ---------- Standardize X according to mask (center+scale) ----------
  Eigen::VectorXd x_mean = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd x_scale = Eigen::VectorXd::Ones(p);
  Eigen::MatrixXd Xs = X;

  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) {
      const double m = X.col(j).mean();
      const Eigen::ArrayXd xc = X.col(j).array() - m;
      const double sd = std::sqrt((xc * xc).mean());
      if (!std::isfinite(sd) || sd < tiny_sd) {
        throw std::invalid_argument("Cannot standardize (near-)constant column j=" + std::to_string(j));
      }
      x_mean[j] = m;
      x_scale[j] = sd;
      Xs.col(j).array() = (Xs.col(j).array() - m) / sd;
    }
  }

  // ---------- Initialize in standardized parameterization ----------
  // beta_s[j] = beta0[j] * sd_j for standardized cols
  // b0_s = intercept0 + sum(mean_j * beta0[j]) over standardized cols
  double b0 = intercept0;
  Eigen::VectorXd beta = beta0;

  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta[j] = beta0[j] * x_scale[j];
  }
  {
    double shift = 0.0;
    for (int j = 0; j < p; ++j) {
      if (standardize[j] != 0) shift += x_mean[j] * beta0[j];
    }
    b0 += shift;
  }

  // IRLS state
  Eigen::VectorXd eta(n), mu(n), dmu(n), W(n), z(n);

  // CD settings
  const int cd_max_iter = 300;
  const double cd_tol = std::max(1e-12, tol * 0.1);

  GLMBinomialENetResult info;
  info.beta = Eigen::VectorXd::Zero(p);

  bool converged = false;
  int total_cd_sweeps = 0;
  double last_outer_delta = std::numeric_limits<double>::infinity();
  double last_inner_delta = std::numeric_limits<double>::infinity();

  for (int outer = 0; outer < max_iter; ++outer) {
    // ---- IRLS: eta/mu/dmu ----
    eta.noalias() = Xs * beta;
    eta.array() += b0;
    eta.array() += offset.array();

    LinkEval::eval(link, eta, mu, dmu);

    for (int i = 0; i < n; ++i) {
      mu[i] = std::min(std::max(mu[i], eps_mu), 1.0 - eps_mu);
      dmu[i] = std::max(dmu[i], eps_dmu);
    }

    // W = n * dmu^2 / (mu(1-mu))
    for (int i = 0; i < n; ++i) {
      double ni = n_trials[i];
      if (!(ni > 0.0) || !std::isfinite(ni)) ni = 0.0;

      const double denom = mu[i] * (1.0 - mu[i]);
      double wi = (ni > 0.0) ? (ni * (dmu[i] * dmu[i]) / denom) : eps_mu;
      if (!(wi > 0.0) || !std::isfinite(wi)) wi = eps_mu;
      if (wi < eps_mu) wi = eps_mu;
      W[i] = wi;
    }

    // z = eta + (y - n*mu)/(n*dmu)  (if n<=0 -> keep z=eta)
    z = eta;
    for (int i = 0; i < n; ++i) {
      double ni = n_trials[i];
      if (!(ni > 0.0) || !std::isfinite(ni)) ni = 0.0;
      if (ni > 0.0) {
        z[i] += (y[i] - ni * mu[i]) / (ni * dmu[i]);
      }
    }

    const Eigen::VectorXd ytilde = z - offset;

    // ---- inner: CD on penalized WLS (b0 explicit, unpenalized) ----
    // r = ytilde - Xs*beta  (exclude b0 first)
    Eigen::VectorXd r = ytilde - (Xs * beta);

    const double sw = W.sum();
    double b0_new = b0;
    if (sw > 0.0 && std::isfinite(sw)) {
      b0_new = (W.array() * r.array()).sum() / sw;
    }
    r.array() -= b0_new; // now r = ytilde - b0_new - Xs*beta

    // aj = sum_i W_i x_ij^2 (+ l2 if penalized)
    Eigen::VectorXd aj(p);
    for (int j = 0; j < p; ++j) {
      double s = 0.0;
      for (int i = 0; i < n; ++i) {
        const double xij = Xs(i, j);
        s += W[i] * xij * xij;
      }
      if (is_penalized(j)) s += l2;
      if (!(s > 0.0) || !std::isfinite(s)) s = 1.0;
      aj[j] = s;
    }

    Eigen::VectorXd beta_new = beta;

    int cd_sweeps_this_outer = 0;
    double cd_last_max_change = std::numeric_limits<double>::infinity();

    for (int cd_it = 0; cd_it < cd_max_iter; ++cd_it) {
      ++cd_sweeps_this_outer;

      // refresh intercept each sweep (cheap and stabilizes)
      if (sw > 0.0 && std::isfinite(sw)) {
        // r currently = ytilde - b0_new - Xs*beta_new
        // => ytilde - Xs*beta_new = r + b0_new
        r.array() += b0_new;
        b0_new = (W.array() * r.array()).sum() / sw;
        r.array() -= b0_new;
      }

      double max_change = 0.0;

      for (int j = 0; j < p; ++j) {
        const double bj_old = beta_new[j];

        // r += x_j * bj_old
        if (bj_old != 0.0) {
          for (int i = 0; i < n; ++i) r[i] += Xs(i, j) * bj_old;
        }

        // rho = x_j^T W r
        double rho = 0.0;
        for (int i = 0; i < n; ++i) {
          rho += W[i] * Xs(i, j) * r[i];
        }

        double bj_new;
        if (is_penalized(j)) {
          bj_new = soft_threshold(rho, l1) / aj[j];
        } else {
          bj_new = rho / aj[j];
        }

        // r -= x_j * bj_new
        if (bj_new != 0.0) {
          for (int i = 0; i < n; ++i) r[i] -= Xs(i, j) * bj_new;
        }

        beta_new[j] = bj_new;

        const double ch = std::abs(bj_new - bj_old);
        if (ch > max_change) max_change = ch;
      }

      cd_last_max_change = max_change;
      if (max_change < cd_tol) break;
    }

    total_cd_sweeps += cd_sweeps_this_outer;

    // ---- outer convergence ----
    double max_abs = std::abs(b0_new - b0);
    for (int j = 0; j < p; ++j) {
      const double d = std::abs(beta_new[j] - beta[j]);
      if (d > max_abs) max_abs = d;
    }

    // update
    b0 = b0_new;
    beta = beta_new;

    last_outer_delta = max_abs;
    last_inner_delta = cd_last_max_change;

    info.n_outer = outer + 1; // executed
    info.n_inner = total_cd_sweeps;
    info.max_delta = last_outer_delta;
    info.max_delta_inner = last_inner_delta;

    if (max_abs < tol) {
      converged = true;
      break;
    }
  }

  // ---------- Unstandardize back to original parameterization ----------
  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta_out[j] = beta[j] / x_scale[j];
  }

  double b0_out = b0;
  {
    double shift = 0.0;
    for (int j = 0; j < p; ++j) {
      if (standardize[j] != 0) shift += x_mean[j] * beta_out[j];
    }
    b0_out -= shift;
  }

  info.intercept = b0_out;
  info.beta = beta_out;
  info.converged = converged;

  return info;
}