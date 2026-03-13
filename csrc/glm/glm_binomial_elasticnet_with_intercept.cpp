// glm_binomial_elasticnet_with_intercept.cpp
// Binomial GLM with per-coordinate Elastic Net penalties
// via IRLS (outer) + Coordinate Descent (inner)
//
// - aggregated binomial: y = success counts, n_trials = totals
// - intercept is an explicit scalar (never penalized)
// - beta is p-vector
// - standardize mask (0/1, length p):
//     1 -> center+scale column (safe because intercept is explicit)
//     0 -> leave as-is
//
// Penalized objective in standardized space:
//   ll(b0, beta)
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
  double intercept = 0.0;       // original (unstandardized) intercept
  Eigen::VectorXd beta;         // original (unstandardized) beta (length p)

  bool converged = false;
  int n_outer = 0;              // executed outer IRLS iterations
  int n_inner = 0;              // total executed inner CD sweeps (sum over outer)
  double max_delta = 0.0;       // final outer max(|Δb0|, max|Δbeta|)
  double max_delta_inner = 0.0; // last inner sweep max(|Δb0|, max|Δbeta|)
};

GLMBinomialENetResult glm_binomial_elasticnet_with_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,          // successes
    const Eigen::VectorXd& n_trials,   // totals
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,   // 0/1 length p
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
    out.intercept = intercept0;
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
        throw std::invalid_argument(
            "Cannot standardize (near-)constant column j=" + std::to_string(j));
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
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) b0 += x_mean[j] * beta0[j];
  }

  // IRLS state
  Eigen::VectorXd eta(n), mu(n), dmu(n), W(n), z(n);

  // CD settings
  const int cd_max_iter = 300;
  const double cd_tol = std::max(1e-12, tol * 0.1);

  // Penalized objective in standardized space (for step-halving)
  auto penalized_obj = [&](double b0_try, const Eigen::VectorXd& beta_try) -> double {
    eta.noalias() = Xs * beta_try;
    eta.array() += b0_try;
    eta.array() += offset.array();

    LinkEval::eval(link, eta, mu, dmu);

    double llf = 0.0;
    for (int i = 0; i < n; ++i) {
      double ni = n_trials[i];
      if (!(ni >= 0.0) || !std::isfinite(ni)) {
        throw std::invalid_argument("n_trials must be finite and >= 0");
      }

      double mui = std::min(std::max(mu[i], eps_mu), 1.0 - eps_mu);
      double yi = y[i];

      // aggregated binomial log-likelihood up to constant wrt parameters:
      // y*log(mu) + (n-y)*log(1-mu)
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

  double obj_old = penalized_obj(b0, beta);

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
    // residual r = ytilde - b0 - Xs*beta
    Eigen::VectorXd r = ytilde - (Xs * beta);
    r.array() -= b0;

    // u = W .* r
    Eigen::VectorXd u = W.array() * r.array();

    const double sw = std::max(eps_mu, W.sum());

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

    double b0_new = b0;
    Eigen::VectorXd beta_new = beta;

    int cd_sweeps_this_outer = 0;
    double cd_last_max_change = std::numeric_limits<double>::infinity();

    for (int cd_it = 0; cd_it < cd_max_iter; ++cd_it) {
      ++cd_sweeps_this_outer;

      double max_change = 0.0;

      // --- intercept update (unpenalized) ---
      double delta0 = 0.0;
      for (int i = 0; i < n; ++i) delta0 += u[i];
      delta0 /= sw;

      if (delta0 != 0.0) {
        b0_new += delta0;
        for (int i = 0; i < n; ++i) {
          r[i] -= delta0;
          u[i] -= W[i] * delta0;
        }
      }
      max_change = std::max(max_change, std::abs(delta0));

      // --- beta updates ---
      for (int j = 0; j < p; ++j) {
        const double bj_old = beta_new[j];

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
    const double step_obj_old = obj_old;

    double step = 1.0;
    double obj_new = -INFINITY;
    double b0_acc = b0;
    Eigen::VectorXd beta_acc = beta;

    const int max_halving = 25;
    for (int hs = 0; hs < max_halving; ++hs) {
      const double b0_try = b0 + step * (b0_new - b0);
      Eigen::VectorXd b_try = beta + step * (beta_new - beta);

      obj_new = penalized_obj(b0_try, b_try);

      if (std::isfinite(obj_new) && obj_new >= step_obj_old - 1e-12) {
        b0_acc = b0_try;
        beta_acc = b_try;
        break;
      }
      step *= 0.5;
    }

    // ---- outer convergence ----
    double max_abs = std::abs(b0_acc - b0);
    for (int j = 0; j < p; ++j) {
      const double d = std::abs(beta_acc[j] - beta[j]);
      if (d > max_abs) max_abs = d;
    }

    b0 = b0_acc;
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

  // ---------- Unstandardize back to original parameterization ----------
  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta_out[j] = beta[j] / x_scale[j];
  }

  double b0_out = b0;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) b0_out -= x_mean[j] * beta_out[j];
  }

  info.intercept = b0_out;
  info.beta = beta_out;
  info.converged = converged;

  return info;
}