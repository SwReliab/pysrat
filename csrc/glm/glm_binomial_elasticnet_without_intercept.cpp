// glm_binomial_elasticnet_without_intercept.cpp
//
// Binomial GLM with Elastic Net penalties
// via IRLS (outer) + Coordinate Descent (inner)
//
// Model:
//   y_i ~ Binomial(n_trials_i, mu_i)
//   g(mu_i) = x_i^T beta + offset_i
//
// This file provides overloaded implementations for two L2-penalty forms
// in standardized space:
//
// 1) Per-coordinate L2 penalty:
//      lambda_l2_vec[j] >= 0
//      penalty = sum_j lambda_l1_vec[j] * |beta_j|
//              + 0.5 * sum_j lambda_l2_vec[j] * beta_j^2
//
// 2) Correlated L2 penalty:
//      lambda_l2_mat : p x p matrix
//      penalty = sum_j lambda_l1_vec[j] * |beta_j|
//              + 0.5 * beta^T lambda_l2_mat * beta
//
// Note:
//   This is a WITHOUT-INTERCEPT implementation.
//   Therefore, standardization uses scale-only (no centering).

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
  double intercept = 0.0;   // always 0.0 in without-intercept model
  Eigen::VectorXd beta;

  bool converged = false;
  int n_outer = 0;
  int n_inner = 0;
  double max_delta = 0.0;
  double max_delta_inner = 0.0;
};


// ============================================================================
// overload 1: L2 vector
// ============================================================================

GLMBinomialENetResult glm_binomial_elasticnet_without_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXd& lambda_l1_vec,
    const Eigen::VectorXd& lambda_l2_vec,
    int max_iter,
    double tol,
    const std::string& link,
    double eps_mu,
    double eps_dmu) {

  const int n = static_cast<int>(X.rows());
  const int p = static_cast<int>(X.cols());

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

  const double tiny_sd = 1e-12;

  // --------------------------------------------------------------------------
  // scale-only standardization (NO centering because no intercept)
  // --------------------------------------------------------------------------
  Eigen::VectorXd x_scale = Eigen::VectorXd::Ones(p);
  Eigen::MatrixXd Xs = X;

  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) {
      const double ss = X.col(j).array().square().mean();
      const double sd = std::sqrt(ss);
      if (!std::isfinite(sd) || sd < tiny_sd) {
        throw std::invalid_argument(
            "Cannot standardize (near-)zero column norm j=" + std::to_string(j));
      }
      x_scale[j] = sd;
      Xs.col(j) /= sd;
    }
  }

  // beta_s[j] = beta0[j] * scale_j for standardized cols
  Eigen::VectorXd beta = beta0;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta[j] = beta0[j] * x_scale[j];
  }

  Eigen::VectorXd eta(n), mu(n), dmu(n), W(n), z(n);

  const int cd_max_iter = 300;
  const double cd_tol = std::max(1e-12, tol * 0.1);

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
      llf += y[i] * std::log(mui) + (ni - y[i]) * std::log(1.0 - mui);
    }

    double pen = 0.0;
    for (int j = 0; j < p; ++j) {
      pen += lambda_l1_vec[j] * std::abs(beta_try[j]) +
             0.5 * lambda_l2_vec[j] * beta_try[j] * beta_try[j];
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
    eta.noalias() = Xs * beta;
    eta.array() += offset.array();

    LinkEval::eval(link, eta, mu, dmu);

    for (int i = 0; i < n; ++i) {
      mu[i] = std::min(std::max(mu[i], eps_mu), 1.0 - eps_mu);
      dmu[i] = std::max(dmu[i], eps_dmu);
    }

    for (int i = 0; i < n; ++i) {
      double ni = n_trials[i];
      if (!(ni > 0.0) || !std::isfinite(ni)) ni = 0.0;

      const double denom = mu[i] * (1.0 - mu[i]);
      double wi = (ni > 0.0) ? (ni * (dmu[i] * dmu[i]) / denom) : eps_mu;
      if (!(wi > 0.0) || !std::isfinite(wi)) wi = eps_mu;
      if (wi < eps_mu) wi = eps_mu;
      W[i] = wi;
    }

    z = eta;
    for (int i = 0; i < n; ++i) {
      double ni = n_trials[i];
      if (!(ni > 0.0) || !std::isfinite(ni)) ni = 0.0;
      if (ni > 0.0) {
        z[i] += (y[i] - ni * mu[i]) / (ni * dmu[i]);
      }
    }

    const Eigen::VectorXd ytilde = z - offset;

    // residual r = ytilde - Xs*beta
    Eigen::VectorXd r = ytilde - (Xs * beta);
    Eigen::VectorXd u = W.array() * r.array();

    Eigen::VectorXd aj_x(p), aj(p);
    for (int j = 0; j < p; ++j) {
      double s = 0.0;
      for (int i = 0; i < n; ++i) {
        const double xij = Xs(i, j);
        s += W[i] * xij * xij;
      }
      if (!(s >= 0.0) || !std::isfinite(s)) s = 0.0;
      aj_x[j] = s;

      double a = s + lambda_l2_vec[j];
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

        double xTu = 0.0;
        for (int i = 0; i < n; ++i) {
          xTu += Xs(i, j) * u[i];
        }

        const double rho = xTu + aj_x[j] * bj_old;
        const double bj_new = soft_threshold(rho, lambda_l1_vec[j]) / aj[j];
        const double delta = bj_new - bj_old;

        if (delta != 0.0) {
          for (int i = 0; i < n; ++i) {
            const double xij = Xs(i, j);
            r[i] -= xij * delta;
            u[i] -= W[i] * xij * delta;
          }
          beta_new[j] = bj_new;
        }

        max_change = std::max(max_change, std::abs(delta));
      }

      cd_last_max_change = max_change;
      if (max_change < cd_tol) break;
    }

    total_cd_sweeps += cd_sweeps_this_outer;

    const double step_obj_old = obj_old;
    double step = 1.0;
    double obj_new = -std::numeric_limits<double>::infinity();
    Eigen::VectorXd beta_acc = beta;

    const int max_halving = 25;
    for (int hs = 0; hs < max_halving; ++hs) {
      const Eigen::VectorXd b_try = beta + step * (beta_new - beta);
      obj_new = penalized_obj(b_try);

      if (std::isfinite(obj_new) && obj_new >= step_obj_old - 1e-12) {
        beta_acc = b_try;
        break;
      }
      step *= 0.5;
    }

    double max_abs = 0.0;
    for (int j = 0; j < p; ++j) {
      max_abs = std::max(max_abs, std::abs(beta_acc[j] - beta[j]));
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

  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta_out[j] = beta[j] / x_scale[j];
  }

  info.intercept = 0.0;
  info.beta = beta_out;
  info.converged = converged;

  return info;
}


// ============================================================================
// overload 2: L2 matrix
// ============================================================================

GLMBinomialENetResult glm_binomial_elasticnet_without_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXd& lambda_l1_vec,
    const Eigen::MatrixXd& lambda_l2_mat,
    int max_iter,
    double tol,
    const std::string& link,
    double eps_mu,
    double eps_dmu) {

  const int n = static_cast<int>(X.rows());
  const int p = static_cast<int>(X.cols());

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

  const double tiny_sd = 1e-12;

  // --------------------------------------------------------------------------
  // scale-only standardization (NO centering because no intercept)
  // --------------------------------------------------------------------------
  Eigen::VectorXd x_scale = Eigen::VectorXd::Ones(p);
  Eigen::MatrixXd Xs = X;

  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) {
      const double ss = X.col(j).array().square().mean();
      const double sd = std::sqrt(ss);
      if (!std::isfinite(sd) || sd < tiny_sd) {
        throw std::invalid_argument(
            "Cannot standardize (near-)zero column norm j=" + std::to_string(j));
      }
      x_scale[j] = sd;
      Xs.col(j) /= sd;
    }
  }

  Eigen::VectorXd beta = beta0;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta[j] = beta0[j] * x_scale[j];
  }

  Eigen::VectorXd eta(n), mu(n), dmu(n), W(n), z(n);

  const int cd_max_iter = 300;
  const double cd_tol = std::max(1e-12, tol * 0.1);

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
      llf += y[i] * std::log(mui) + (ni - y[i]) * std::log(1.0 - mui);
    }

    double pen_l1 = 0.0;
    for (int j = 0; j < p; ++j) pen_l1 += lambda_l1_vec[j] * std::abs(beta_try[j]);

    const double pen_l2 = 0.5 * beta_try.dot(lambda_l2_mat * beta_try);
    return llf - pen_l1 - pen_l2;
  };

  GLMBinomialENetResult info;
  info.beta = Eigen::VectorXd::Zero(p);

  bool converged = false;
  int total_cd_sweeps = 0;
  double last_outer_delta = 0.0;
  double last_inner_delta = 0.0;

  double obj_old = penalized_obj(beta);

  for (int outer = 0; outer < max_iter; ++outer) {
    eta.noalias() = Xs * beta;
    eta.array() += offset.array();

    LinkEval::eval(link, eta, mu, dmu);

    for (int i = 0; i < n; ++i) {
      mu[i] = std::min(std::max(mu[i], eps_mu), 1.0 - eps_mu);
      dmu[i] = std::max(dmu[i], eps_dmu);
    }

    for (int i = 0; i < n; ++i) {
      double ni = n_trials[i];
      if (!(ni > 0.0) || !std::isfinite(ni)) ni = 0.0;

      const double denom = mu[i] * (1.0 - mu[i]);
      double wi = (ni > 0.0) ? (ni * (dmu[i] * dmu[i]) / denom) : eps_mu;
      if (!(wi > 0.0) || !std::isfinite(wi)) wi = eps_mu;
      if (wi < eps_mu) wi = eps_mu;
      W[i] = wi;
    }

    z = eta;
    for (int i = 0; i < n; ++i) {
      double ni = n_trials[i];
      if (!(ni > 0.0) || !std::isfinite(ni)) ni = 0.0;
      if (ni > 0.0) {
        z[i] += (y[i] - ni * mu[i]) / (ni * dmu[i]);
      }
    }

    const Eigen::VectorXd ytilde = z - offset;

    Eigen::VectorXd r = ytilde - (Xs * beta);
    Eigen::VectorXd u = W.array() * r.array();

    Eigen::VectorXd aj_x(p), aj(p);
    for (int j = 0; j < p; ++j) {
      double s = 0.0;
      for (int i = 0; i < n; ++i) {
        const double xij = Xs(i, j);
        s += W[i] * xij * xij;
      }
      if (!(s >= 0.0) || !std::isfinite(s)) s = 0.0;
      aj_x[j] = s;

      double a = s + lambda_l2_mat(j, j);
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

        double xTu = 0.0;
        for (int i = 0; i < n; ++i) {
          xTu += Xs(i, j) * u[i];
        }

        double coupling = 0.0;
        for (int k = 0; k < p; ++k) {
          if (k == j) continue;
          coupling += lambda_l2_mat(j, k) * beta_new[k];
        }

        const double rho = xTu + aj_x[j] * bj_old - coupling;
        const double bj_new = soft_threshold(rho, lambda_l1_vec[j]) / aj[j];
        const double delta = bj_new - bj_old;

        if (delta != 0.0) {
          for (int i = 0; i < n; ++i) {
            const double xij = Xs(i, j);
            r[i] -= xij * delta;
            u[i] -= W[i] * xij * delta;
          }
          beta_new[j] = bj_new;
        }

        max_change = std::max(max_change, std::abs(delta));
      }

      cd_last_max_change = max_change;
      if (max_change < cd_tol) break;
    }

    total_cd_sweeps += cd_sweeps_this_outer;

    const double step_obj_old = obj_old;
    double step = 1.0;
    double obj_new = -std::numeric_limits<double>::infinity();
    Eigen::VectorXd beta_acc = beta;

    const int max_halving = 25;
    for (int hs = 0; hs < max_halving; ++hs) {
      const Eigen::VectorXd b_try = beta + step * (beta_new - beta);
      obj_new = penalized_obj(b_try);

      if (std::isfinite(obj_new) && obj_new >= step_obj_old - 1e-12) {
        beta_acc = b_try;
        break;
      }
      step *= 0.5;
    }

    double max_abs = 0.0;
    for (int j = 0; j < p; ++j) {
      max_abs = std::max(max_abs, std::abs(beta_acc[j] - beta[j]));
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

  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta_out[j] = beta[j] / x_scale[j];
  }

  info.intercept = 0.0;
  info.beta = beta_out;
  info.converged = converged;

  return info;
}