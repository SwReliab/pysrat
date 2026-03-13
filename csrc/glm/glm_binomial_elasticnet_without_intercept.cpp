// glm_binomial_elasticnet_without_intercept.cpp
// Binomial GLM without intercept, with per-coordinate Elastic Net penalties
// via IRLS (outer) + Coordinate Descent (inner)
//
// - aggregated binomial: y = success counts, n_trials = totals
// - NO intercept term: eta = Xs * beta + offset
// - standardize mask (0/1, length p):
//     1 -> scale-only (NO centering; centering would change the model without intercept)
//     0 -> leave column as-is
//
// Penalized objective in standardized space:
//   ll(beta)
//   - sum_j lambda_l1[j] * |beta_j|
//   - 0.5 * sum_j lambda_l2[j] * beta_j^2
//
// Return: GLMBinomialENetResult with convergence info.

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
  double intercept = 0.0;       // always 0.0 here
  Eigen::VectorXd beta;         // original (unstandardized) beta (length p)

  bool converged = false;
  int n_outer = 0;              // executed outer IRLS iterations
  int n_inner = 0;              // total executed inner CD sweeps (sum over outer)
  double max_delta = 0.0;       // final outer max|Δbeta|
  double max_delta_inner = 0.0; // last inner sweep max|Δbeta|
};

GLMBinomialENetResult glm_binomial_elasticnet_without_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,          // successes
    const Eigen::VectorXd& n_trials,   // totals
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,   // 0/1 length p (scale-only)
    const Eigen::VectorXd& lambda_l1,     // length p, >= 0
    const Eigen::VectorXd& lambda_l2,     // length p, >= 0
    int max_iter,
    double tol,
    const std::string& link,
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
  if (lambda_l1.size() != p) {
    throw std::invalid_argument("lambda_l1 length must match X.cols()");
  }
  if (lambda_l2.size() != p) {
    throw std::invalid_argument("lambda_l2 length must match X.cols()");
  }
  for (int j = 0; j < p; ++j) {
    if (!(lambda_l1[j] >= 0.0) || !std::isfinite(lambda_l1[j])) {
      throw std::invalid_argument("lambda_l1 must be finite and >= 0");
    }
    if (!(lambda_l2[j] >= 0.0) || !std::isfinite(lambda_l2[j])) {
      throw std::invalid_argument("lambda_l2 must be finite and >= 0");
    }
  }
  if (max_iter <= 0) {
    GLMBinomialENetResult out;
    out.intercept = 0.0;
    out.beta = beta0;
    out.converged = false;
    out.n_outer = 0;
    out.n_inner = 0;
    out.max_delta = 0.0;
    out.max_delta_inner = 0.0;
    return out;
  }
  if (!(tol > 0.0) || !std::isfinite(tol)) {
    throw std::invalid_argument("tol must be positive finite");
  }
  if (!(eps_mu > 0.0) || !std::isfinite(eps_mu)) {
    throw std::invalid_argument("eps_mu must be positive finite");
  }
  if (!(eps_dmu > 0.0) || !std::isfinite(eps_dmu)) {
    throw std::invalid_argument("eps_dmu must be positive finite");
  }

  const double tiny_sd = 1e-12;

  // ---------- Standardize X according to mask (scale-only, NO centering) ----------
  Eigen::VectorXd x_scale = Eigen::VectorXd::Ones(p);
  Eigen::MatrixXd Xs = X;

  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) {
      // scale-only: use RMS around 0 (not centered), so the no-intercept model is preserved
      const Eigen::ArrayXd xc = X.col(j).array();
      const double sd = std::sqrt((xc * xc).mean());
      if (!std::isfinite(sd) || sd < tiny_sd) {
        throw std::invalid_argument(
            "Cannot scale (near-)constant/zero column j=" + std::to_string(j));
      }
      x_scale[j] = sd;
      Xs.col(j).array() /= sd;
    }
  }

  // ---------- Initialize in standardized parameterization ----------
  // If Xs = X / sd, then preserve eta = X * beta0 = (X / sd) * (beta0 * sd)
  Eigen::VectorXd beta = beta0;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta[j] = beta0[j] * x_scale[j];
  }

  // IRLS state
  Eigen::VectorXd eta(n), mu(n), dmu(n), W(n), z(n);

  // CD settings
  const int cd_max_iter = 300;
  const double cd_tol = std::max(1e-12, tol * 0.1);

  // Penalized objective in standardized space (for step-halving)
  auto penalized_obj = [&](const Eigen::VectorXd& beta_try) -> double {
    eta.noalias() = Xs * beta_try;
    eta.array() += offset.array();

    LinkEval::eval(link, eta, mu, dmu);

    double llf = 0.0;
    for (int i = 0; i < n; ++i) {
      double ni = n_trials[i];
      if (!(ni >= 0.0) || !std::isfinite(ni)) {
        throw std::invalid_argument("n_trials must be finite and >= 0");
      }

      const double mui = std::min(std::max(mu[i], eps_mu), 1.0 - eps_mu);
      const double yi = y[i];

      // aggregated binomial log-likelihood up to constants wrt parameters
      llf += yi * std::log(mui) + (ni - yi) * std::log(1.0 - mui);
    }

    double pen = 0.0;
    for (int j = 0; j < p; ++j) {
      const double bj = beta_try[j];
      pen += lambda_l1[j] * std::abs(bj) + 0.5 * lambda_l2[j] * bj * bj;
    }
    return llf - pen;
  };

  GLMBinomialENetResult info;
  info.beta = Eigen::VectorXd::Zero(p);

  bool converged = false;
  int total_cd_sweeps = 0;
  double last_outer_delta = 0.0;
  double last_inner_delta = 0.0;

  double obj_old = penalized_obj(beta);

  for (int outer = 0; outer < max_iter; ++outer) {
    // ---- IRLS: eta / mu / dmu ----
    eta.noalias() = Xs * beta;
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

    // z = eta + (y - n*mu)/(n*dmu), if n<=0 keep z=eta
    z = eta;
    for (int i = 0; i < n; ++i) {
      double ni = n_trials[i];
      if (!(ni > 0.0) || !std::isfinite(ni)) ni = 0.0;
      if (ni > 0.0) {
        z[i] += (y[i] - ni * mu[i]) / (ni * dmu[i]);
      }
    }

    const Eigen::VectorXd ytilde = z - offset;

    // ---- inner: CD on penalized WLS (no intercept) ----
    // residual r = ytilde - Xs * beta
    Eigen::VectorXd r = ytilde - (Xs * beta);

    // u = W .* r
    Eigen::VectorXd u = W.array() * r.array();

    // aj_x = sum_i W_i x_ij^2
    // aj   = aj_x + lambda_l2[j]
    Eigen::VectorXd aj_x(p), aj(p);
    for (int j = 0; j < p; ++j) {
      double s = 0.0;
      for (int i = 0; i < n; ++i) {
        const double xij = Xs(i, j);
        s += W[i] * xij * xij;
      }
      if (!(s >= 0.0) || !std::isfinite(s)) s = 0.0;
      aj_x[j] = s;

      double a = s + lambda_l2[j];
      if (!(a > 0.0) || !std::isfinite(a)) a = 1.0;
      aj[j] = a;
    }

    Eigen::VectorXd beta_new = beta;

    int cd_sweeps_this_outer = 0;
    double cd_last_max_change = std::numeric_limits<double>::infinity();

    for (int cd_it = 0; cd_it < cd_max_iter; ++cd_it) {
      ++cd_sweeps_this_outer;

      double max_change = 0.0;

      for (int j = 0; j < p; ++j) {
        const double bj_old = beta_new[j];

        // With full residual r = ytilde - X beta,
        // rho_j = x_j^T (W .* r) + (x_j^T W x_j) * beta_j
        double xTu = 0.0;
        for (int i = 0; i < n; ++i) {
          xTu += Xs(i, j) * u[i];
        }

        const double rho = xTu + aj_x[j] * bj_old;
        const double bj_new = soft_threshold(rho, lambda_l1[j]) / aj[j];
        const double delta = bj_new - bj_old;

        if (delta != 0.0) {
          for (int i = 0; i < n; ++i) {
            const double xij = Xs(i, j);
            r[i] -= xij * delta;
            u[i] -= W[i] * xij * delta;
          }
          beta_new[j] = bj_new;
        }

        const double ch = std::abs(delta);
        if (ch > max_change) max_change = ch;
      }

      cd_last_max_change = max_change;
      if (max_change < cd_tol) break;
    }

    total_cd_sweeps += cd_sweeps_this_outer;

    // ---- Step-halving on penalized objective (outer stability) ----
    double step = 1.0;
    double obj_new = -INFINITY;
    Eigen::VectorXd beta_acc = beta;

    const int max_halving = 25;
    for (int hs = 0; hs < max_halving; ++hs) {
      Eigen::VectorXd b_try = beta + step * (beta_new - beta);
      obj_new = penalized_obj(b_try);

      if (std::isfinite(obj_new) && obj_new >= obj_old - 1e-12) {
        beta_acc = b_try;
        break;
      }
      step *= 0.5;
    }

    // ---- outer convergence ----
    double max_abs = 0.0;
    for (int j = 0; j < p; ++j) {
      const double d = std::abs(beta_acc[j] - beta[j]);
      if (d > max_abs) max_abs = d;
    }

    beta = beta_acc;

    if (std::isfinite(obj_new)) obj_old = obj_new;

    last_outer_delta = max_abs;
    last_inner_delta = std::isfinite(cd_last_max_change) ? cd_last_max_change : 0.0;

    info.n_outer = outer + 1;
    info.n_inner = total_cd_sweeps;
    info.max_delta = last_outer_delta;
    info.max_delta_inner = last_inner_delta;

    if (max_abs < tol) {
      converged = true;
      break;
    }
  }

  // ---------- Unstandardize back to original beta ----------
  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta_out[j] = beta[j] / x_scale[j];
  }

  info.intercept = 0.0;
  info.beta = beta_out;
  info.converged = converged;
  return info;
}