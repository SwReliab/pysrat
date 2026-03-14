// glm_poisson_without_intercept.cpp
//
// Poisson GLM (log link) via IRLS (outer) + WLS solve (inner)
// WITHOUT intercept
//
// Model:
//   y_i ~ Poisson(mu_i)
//   log(mu_i) = x_i^T beta + offset_i
//
// This file provides overloaded implementations for two L2-penalty forms
// in standardized space:
//
// 1) Per-coordinate L2 penalty:
//      lambda_l2[j] >= 0
//      penalty = 0.5 * sum_j lambda_l2[j] * beta_j^2
//
// 2) Correlated L2 penalty:
//      lambda_l2_mat : p x p symmetric positive semidefinite matrix
//      penalty = 0.5 * beta^T lambda_l2_mat * beta
//
// API:
// - offset is added to the linear predictor
// - standardize[j] = 1 -> scale column j before fitting (NO centering)
// - standardize[j] = 0 -> leave column j as-is
//
// Parameterization:
// - fitting is performed in standardized space
// - returned beta is transformed back to the original scale
//
// Return:
//   GLMPoissonResult { intercept, beta, converged, n_iter }
//
// Notes:
// 1) clip-consistent eta/mu/z: mu = exp(clip(eta)), z uses eta_clip (= log(mu))
// 2) optional step-halving on penalized objective for IRLS stability
// 3) vector-L2 overload uses diagonal ridge-type shrinkage
// 4) matrix-L2 overload supports correlated shrinkage through a symmetric
//    positive semidefinite penalty matrix

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

static inline double clip(double x, double lo, double hi) {
  return std::min(std::max(x, lo), hi);
}

struct GLMPoissonResult {
  double intercept;
  Eigen::VectorXd beta;
  bool converged;
  int n_iter;
};

GLMPoissonResult glm_poisson_without_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXd& lambda_l2_vec,
    int max_iter,
    double tol,
    double eps_mu) {

  const int n = static_cast<int>(X.rows());
  const int p = static_cast<int>(X.cols());

  if (max_iter <= 0) {
    return {0.0, beta0, false, 0};
  }

  const double tiny_sd = 1e-12;

  // --- standardize: scale-only for standardize[j]==1 (NO centering) ---
  Eigen::VectorXd x_scale = Eigen::VectorXd::Ones(p);
  Eigen::MatrixXd Xs = X;

  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) {
      const double sd = std::sqrt((X.col(j).array().square()).mean());
      if (!std::isfinite(sd) || sd < tiny_sd) {
        throw std::invalid_argument(
            "Cannot scale (near-)constant column j=" + std::to_string(j));
      }
      x_scale[j] = sd;
      Xs.col(j).array() /= sd;
    }
  }

  // --- transform initial params into standardized parameterization ---
  // For scaled columns:
  //   beta_s[j] = beta[j] * sd_j
  Eigen::VectorXd beta = beta0;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta[j] = beta0[j] * x_scale[j];
  }

  Eigen::VectorXd eta(n), eta_clip(n), mu(n), W(n), z(n);

  const double eta_hi = 50.0;
  const double eta_lo = -50.0;

  auto obj_from = [&](const Eigen::VectorXd& beta_try) -> double {
    eta.noalias() = Xs * beta_try;
    eta.array() += offset.array();

    double llf = 0.0;
    for (int i = 0; i < n; ++i) {
      const double e = clip(eta[i], eta_lo, eta_hi);
      const double m = std::max(std::exp(e), eps_mu);
      llf += y[i] * e - m - std::lgamma(y[i] + 1.0);
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
    // eta = Xs*beta + offset
    eta.noalias() = Xs * beta;
    eta.array() += offset.array();

    // clip-consistent eta_clip, mu
    for (int i = 0; i < n; ++i) {
      const double e = clip(eta[i], eta_lo, eta_hi);
      eta_clip[i] = e;
      double m = std::exp(e);
      if (!std::isfinite(m) || m < eps_mu) m = eps_mu;
      mu[i] = m;
    }

    // W = mu (Poisson, log-link)
    W = mu;

    // z = eta_clip + (y - mu)/mu
    for (int i = 0; i < n; ++i) {
      const double inv_mu = 1.0 / mu[i];
      z[i] = eta_clip[i] + (y[i] - mu[i]) * inv_mu;
    }

    // Weighted least squares target:
    //   z - offset ~= Xs * beta
    const Eigen::VectorXd ytilde = z - offset;
    const Eigen::VectorXd sqrtW = W.array().sqrt();

    Eigen::MatrixXd Xw = Xs;
    for (int i = 0; i < n; ++i) {
      Xw.row(i) *= sqrtW[i];
    }

    const Eigen::VectorXd rhs =
        (sqrtW.array() * ytilde.array()).matrix();

    // Normal equations:
    //   X^T W X + diag(lambda_l2_vec)
    Eigen::MatrixXd A = Xw.transpose() * Xw;
    A.diagonal().array() += lambda_l2_vec.array();

    const Eigen::VectorXd bvec = Xw.transpose() * rhs;

    Eigen::VectorXd beta_full;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if (ldlt.info() == Eigen::Success) {
      beta_full = ldlt.solve(bvec);
    } else {
      beta_full = A.colPivHouseholderQr().solve(bvec);
    }

    // Step-halving on penalized objective
    double step = 1.0;
    double obj_new = -std::numeric_limits<double>::infinity();
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

    const double max_diff = (beta_new - beta).cwiseAbs().maxCoeff();

    beta = beta_new;

    if (std::isfinite(obj_new)) obj_old = obj_new;

    if (max_diff < tol) {
      converged = true;
      ++it;
      break;
    }
  }

  // --- unscale back to original parameterization ---
  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta_out[j] = beta[j] / x_scale[j];
  }

  return {0.0, beta_out, converged, it};
}

GLMPoissonResult glm_poisson_without_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::MatrixXd& lambda_l2_mat,
    int max_iter,
    double tol,
    double eps_mu) {

  const int n = static_cast<int>(X.rows());
  const int p = static_cast<int>(X.cols());

  if (max_iter <= 0) {
    return {0.0, beta0, false, 0};
  }

  const double tiny_sd = 1e-12;

  // --- standardize: scale-only for standardize[j]==1 (NO centering) ---
  Eigen::VectorXd x_scale = Eigen::VectorXd::Ones(p);
  Eigen::MatrixXd Xs = X;

  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) {
      const double sd = std::sqrt((X.col(j).array().square()).mean());
      if (!std::isfinite(sd) || sd < tiny_sd) {
        throw std::invalid_argument(
            "Cannot scale (near-)constant column j=" + std::to_string(j));
      }
      x_scale[j] = sd;
      Xs.col(j).array() /= sd;
    }
  }

  // --- transform initial params into standardized parameterization ---
  // For scaled columns:
  //   beta_s[j] = beta[j] * sd_j
  Eigen::VectorXd beta = beta0;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta[j] = beta0[j] * x_scale[j];
  }

  Eigen::VectorXd eta(n), eta_clip(n), mu(n), W(n), z(n);

  const double eta_hi = 50.0;
  const double eta_lo = -50.0;

  auto obj_from = [&](const Eigen::VectorXd& beta_try) -> double {
    eta.noalias() = Xs * beta_try;
    eta.array() += offset.array();

    double llf = 0.0;
    for (int i = 0; i < n; ++i) {
      const double e = clip(eta[i], eta_lo, eta_hi);
      const double m = std::max(std::exp(e), eps_mu);
      llf += y[i] * e - m - std::lgamma(y[i] + 1.0);
    }

    const double pen = 0.5 * beta_try.dot(lambda_l2_mat * beta_try);
    return llf - pen;
  };

  bool converged = false;
  int it = 0;

  double obj_old = obj_from(beta);

  for (; it < max_iter; ++it) {
    // eta = Xs*beta + offset
    eta.noalias() = Xs * beta;
    eta.array() += offset.array();

    // clip-consistent eta_clip, mu
    for (int i = 0; i < n; ++i) {
      const double e = clip(eta[i], eta_lo, eta_hi);
      eta_clip[i] = e;
      double m = std::exp(e);
      if (!std::isfinite(m) || m < eps_mu) m = eps_mu;
      mu[i] = m;
    }

    // W = mu (Poisson, log-link)
    W = mu;

    // z = eta_clip + (y - mu)/mu
    for (int i = 0; i < n; ++i) {
      const double inv_mu = 1.0 / mu[i];
      z[i] = eta_clip[i] + (y[i] - mu[i]) * inv_mu;
    }

    // Weighted least squares target:
    //   z - offset ~= Xs * beta
    const Eigen::VectorXd ytilde = z - offset;
    const Eigen::VectorXd sqrtW = W.array().sqrt();

    Eigen::MatrixXd Xw = Xs;
    for (int i = 0; i < n; ++i) {
      Xw.row(i) *= sqrtW[i];
    }

    const Eigen::VectorXd rhs =
        (sqrtW.array() * ytilde.array()).matrix();

    // Normal equations:
    //   X^T W X + lambda_l2_mat
    Eigen::MatrixXd A = Xw.transpose() * Xw;
    A += lambda_l2_mat;

    const Eigen::VectorXd bvec = Xw.transpose() * rhs;

    Eigen::VectorXd beta_full;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if (ldlt.info() == Eigen::Success) {
      beta_full = ldlt.solve(bvec);
    } else {
      beta_full = A.colPivHouseholderQr().solve(bvec);
    }

    // Step-halving on penalized objective
    double step = 1.0;
    double obj_new = -std::numeric_limits<double>::infinity();
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

    const double max_diff = (beta_new - beta).cwiseAbs().maxCoeff();

    beta = beta_new;

    if (std::isfinite(obj_new)) obj_old = obj_new;

    if (max_diff < tol) {
      converged = true;
      ++it;
      break;
    }
  }

  // --- unscale back to original parameterization ---
  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta_out[j] = beta[j] / x_scale[j];
  }

  return {0.0, beta_out, converged, it};
}