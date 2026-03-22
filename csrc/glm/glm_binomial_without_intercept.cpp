// glm_binomial_without_intercept.cpp
//
// Binomial GLM with L2 penalties (ridge-type)
// via IRLS (outer) + penalized weighted least squares solve
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
//      penalty = 0.5 * sum_j lambda_l2_vec[j] * beta_j^2
//
// 2) Correlated L2 penalty:
//      lambda_l2_mat : p x p matrix
//      penalty = 0.5 * beta^T lambda_l2_mat * beta
//
// Notes:
// - this is WITHOUT intercept
// - standardization is scale-only for columns with standardize[j] != 0
// - coefficients are optimized in standardized parameterization and then
//   transformed back to the original scale

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "link_eval.h"

// local clip (keep in this TU)
static inline double clip(double x, double lo, double hi) {
  return std::min(std::max(x, lo), hi);
}

struct GLMBinomialResult {
  double intercept;        // always 0 in this file
  Eigen::VectorXd beta;
  bool converged;
  int n_iter;              // executed outer IRLS iterations
};

// ============================================================================
// overload 1: diagonal / vector L2 penalty
// ============================================================================

GLMBinomialResult glm_binomial_without_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXd& lambda_l2_vec,
    int max_iter,
    double tol,
    const std::string& link,
    double eps_mu,
    double eps_dmu) {

  const int n = static_cast<int>(X.rows());
  const int p = static_cast<int>(X.cols());
  const double tiny_sd = 1e-12;

  if (max_iter <= 0) {
    return {0.0, beta0, false, 0};
  }

  Eigen::VectorXd x_scale = Eigen::VectorXd::Ones(p);
  Eigen::MatrixXd Xs = X;

  // --- scale only (NO centering) ---
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

  // --- transform initial params into standardized parameterization ---
  Eigen::VectorXd beta = beta0;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta[j] = beta0[j] * x_scale[j];
  }

  Eigen::VectorXd eta(n), mu(n), dmu(n), W(n), z(n);

  auto obj_from = [&](const Eigen::VectorXd& beta_try) -> double {
    eta.noalias() = Xs * beta_try;
    eta.array() += offset.array();

    LinkEval::eval(link, eta, mu, dmu);

    double llf = 0.0;
    for (int i = 0; i < n; ++i) {
      const double ni = n_trials[i];
      const double mui = clip(mu[i], eps_mu, 1.0 - eps_mu);
      llf += y[i] * std::log(mui) + (ni - y[i]) * std::log(1.0 - mui);
    }

    double pen = 0.0;
    for (int j = 0; j < p; ++j) {
      pen += 0.5 * lambda_l2_vec[j] * beta_try[j] * beta_try[j];
    }
    return llf - pen;
  };

  bool converged = false;
  int it = 0;
  double obj_old = obj_from(beta);

  for (; it < max_iter; ++it) {
    eta.noalias() = Xs * beta;
    eta.array() += offset.array();

    LinkEval::eval(link, eta, mu, dmu);

    for (int i = 0; i < n; ++i) {
      mu[i] = clip(mu[i], eps_mu, 1.0 - eps_mu);
      dmu[i] = std::max(dmu[i], eps_dmu);
    }

    // W = n * dmu^2 / (mu(1-mu))
    for (int i = 0; i < n; ++i) {
      double ni = n_trials[i];
      if (!(ni > 0.0) || !std::isfinite(ni)) ni = 0.0;

      const double denom = mu[i] * (1.0 - mu[i]);
      double wi = (ni > 0.0) ? (ni * dmu[i] * dmu[i] / denom) : eps_mu;
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

    // weighted least squares target:
    //   z - offset ~= Xs * beta
    const Eigen::VectorXd ytilde = z - offset;
    const Eigen::VectorXd sqrtW = W.array().sqrt();

    Eigen::MatrixXd Xw = Xs;
    for (int i = 0; i < n; ++i) {
      Xw.row(i) *= sqrtW[i];
    }

    const Eigen::VectorXd rhs = (sqrtW.array() * ytilde.array()).matrix();

    // normal equations
    Eigen::MatrixXd A = Xw.transpose() * Xw;
    A.diagonal().array() += lambda_l2_vec.array();

    const Eigen::VectorXd bvec = Xw.transpose() * rhs;

    Eigen::VectorXd beta_full(p);
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if (ldlt.info() == Eigen::Success) {
      beta_full = ldlt.solve(bvec);
    } else {
      beta_full = A.colPivHouseholderQr().solve(bvec);
    }

    // step-halving on penalized objective
    double step = 1.0;
    double obj_new = -INFINITY;
    Eigen::VectorXd beta_new = beta;

    const int max_halving = 25;
    for (int hs = 0; hs < max_halving; ++hs) {
      beta_new = beta + step * (beta_full - beta);
      obj_new = obj_from(beta_new);

      if (std::isfinite(obj_new) && obj_new >= obj_old - 1e-12) {
        break;
      }
      step *= 0.5;
    }

    const double max_diff =
        (beta_new - beta).cwiseAbs().size() > 0
            ? (beta_new - beta).cwiseAbs().maxCoeff()
            : 0.0;

    beta = beta_new;

    if (std::isfinite(obj_new)) obj_old = obj_new;

    if (max_diff < tol) {
      converged = true;
      ++it;
      break;
    }
  }

  // back transform
  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta_out[j] /= x_scale[j];
  }

  return {0.0, beta_out, converged, it};
}


// ============================================================================
// overload 2: full / matrix L2 penalty
// ============================================================================

GLMBinomialResult glm_binomial_without_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::MatrixXd& lambda_l2_mat,
    int max_iter,
    double tol,
    const std::string& link,
    double eps_mu,
    double eps_dmu) {

  const int n = static_cast<int>(X.rows());
  const int p = static_cast<int>(X.cols());
  const double tiny_sd = 1e-12;
  const double sym_tol = 1e-10;

  if (max_iter <= 0) {
    return {0.0, beta0, false, 0};
  }

  Eigen::VectorXd x_scale = Eigen::VectorXd::Ones(p);
  Eigen::MatrixXd Xs = X;

  // --- scale only (NO centering) ---
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

  // --- transform initial params into standardized parameterization ---
  Eigen::VectorXd beta = beta0;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta[j] = beta0[j] * x_scale[j];
  }

  Eigen::VectorXd eta(n), mu(n), dmu(n), W(n), z(n);

  auto obj_from = [&](const Eigen::VectorXd& beta_try) -> double {
    eta.noalias() = Xs * beta_try;
    eta.array() += offset.array();

    LinkEval::eval(link, eta, mu, dmu);

    double llf = 0.0;
    for (int i = 0; i < n; ++i) {
      const double ni = n_trials[i];
      const double mui = clip(mu[i], eps_mu, 1.0 - eps_mu);
      llf += y[i] * std::log(mui) + (ni - y[i]) * std::log(1.0 - mui);
    }

    const double pen = 0.5 * beta_try.dot(lambda_l2_mat * beta_try);
    return llf - pen;
  };

  bool converged = false;
  int it = 0;
  double obj_old = obj_from(beta);

  for (; it < max_iter; ++it) {
    eta.noalias() = Xs * beta;
    eta.array() += offset.array();

    LinkEval::eval(link, eta, mu, dmu);

    for (int i = 0; i < n; ++i) {
      mu[i] = clip(mu[i], eps_mu, 1.0 - eps_mu);
      dmu[i] = std::max(dmu[i], eps_dmu);
    }

    // W = n * dmu^2 / (mu(1-mu))
    for (int i = 0; i < n; ++i) {
      double ni = n_trials[i];
      if (!(ni > 0.0) || !std::isfinite(ni)) ni = 0.0;

      const double denom = mu[i] * (1.0 - mu[i]);
      double wi = (ni > 0.0) ? (ni * dmu[i] * dmu[i] / denom) : eps_mu;
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

    // weighted least squares target:
    //   z - offset ~= Xs * beta
    const Eigen::VectorXd ytilde = z - offset;
    const Eigen::VectorXd sqrtW = W.array().sqrt();

    Eigen::MatrixXd Xw = Xs;
    for (int i = 0; i < n; ++i) {
      Xw.row(i) *= sqrtW[i];
    }

    const Eigen::VectorXd rhs = (sqrtW.array() * ytilde.array()).matrix();

    // normal equations
    Eigen::MatrixXd A = Xw.transpose() * Xw;
    A += lambda_l2_mat;

    const Eigen::VectorXd bvec = Xw.transpose() * rhs;

    Eigen::VectorXd beta_full(p);
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if (ldlt.info() == Eigen::Success) {
      beta_full = ldlt.solve(bvec);
    } else {
      beta_full = A.colPivHouseholderQr().solve(bvec);
    }

    // step-halving on penalized objective
    double step = 1.0;
    double obj_new = -INFINITY;
    Eigen::VectorXd beta_new = beta;

    const int max_halving = 25;
    for (int hs = 0; hs < max_halving; ++hs) {
      beta_new = beta + step * (beta_full - beta);
      obj_new = obj_from(beta_new);

      if (std::isfinite(obj_new) && obj_new >= obj_old - 1e-12) {
        break;
      }
      step *= 0.5;
    }

    const double max_diff =
        (beta_new - beta).cwiseAbs().size() > 0
            ? (beta_new - beta).cwiseAbs().maxCoeff()
            : 0.0;

    beta = beta_new;

    if (std::isfinite(obj_new)) obj_old = obj_new;

    if (max_diff < tol) {
      converged = true;
      ++it;
      break;
    }
  }

  // back transform
  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta_out[j] /= x_scale[j];
  }

  return {0.0, beta_out, converged, it};
}