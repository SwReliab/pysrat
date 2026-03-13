// glm_binomial_elasticnet.h
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

// Result container for ElasticNet Binomial GLM fits (IRLS outer + CD inner).
// - with_intercept: intercept is estimated (not penalized)
// - without_intercept: intercept is fixed at 0.0
struct GLMBinomialENetResult {
  double intercept = 0.0;       // unstandardized intercept
  Eigen::VectorXd beta;         // unstandardized beta (length p)

  bool converged = false;
  int n_outer = 0;              // executed IRLS iterations
  int n_inner = 0;              // total executed CD sweeps (sum over outer iters)
  double max_delta = 0.0;       // final outer max(|Δintercept|, max|Δbeta|) (no-intercept: max|Δbeta|)
  double max_delta_inner = 0.0; // last inner sweep max|Δintercept|/|Δbeta| or max|Δbeta|
};

// -----------------------------------------------------------------------------
// Core APIs: per-coordinate penalties in standardized space
// -----------------------------------------------------------------------------
//
// Penalty in standardized space:
//   sum_j lambda_l1_vec[j] * |beta_j|
// + 0.5 * sum_j lambda_l2_vec[j] * beta_j^2
//
// These are the actual implementations in .cpp.

GLMBinomialENetResult glm_binomial_elasticnet_with_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,   // 0/1 mask length p (center+scale)
    const Eigen::VectorXd& lambda_l1_vec, // length p, >= 0
    const Eigen::VectorXd& lambda_l2_vec, // length p, >= 0
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15);

GLMBinomialENetResult glm_binomial_elasticnet_without_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,   // 0/1 mask length p (scale-only)
    const Eigen::VectorXd& lambda_l1_vec, // length p, >= 0
    const Eigen::VectorXd& lambda_l2_vec, // length p, >= 0
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15);

// -----------------------------------------------------------------------------
// Backward-compatible wrappers (old API)
// -----------------------------------------------------------------------------
//
// Old API:
//   l1 = alpha * lambd
//   l2 = (1 - alpha) * lambd + ridge
//
// Compatibility behavior:
// - penalty applies only where penalty_mask[j] != 0
// - ridge is also applied only where penalty_mask[j] != 0
//
// Therefore:
//   lambda_l1_vec[j] = penalty_mask[j] ? alpha*lambd : 0
//   lambda_l2_vec[j] = penalty_mask[j] ? ((1-alpha)*lambd + ridge) : 0

inline GLMBinomialENetResult glm_binomial_elasticnet_with_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXi& penalty_mask,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double alpha = 0.5,
    double lambd = 1e-2,
    double ridge = 1e-12,
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15) {

  const int p = static_cast<int>(X.cols());

  if (penalty_mask.size() != p) {
    throw std::invalid_argument("penalty_mask length must match X.cols()");
  }
  if (!(alpha >= 0.0 && alpha <= 1.0)) {
    throw std::invalid_argument("alpha must be in [0,1]");
  }
  if (!(lambd >= 0.0) || !std::isfinite(lambd)) {
    throw std::invalid_argument("lambd must be finite and >= 0");
  }
  if (!(ridge >= 0.0) || !std::isfinite(ridge)) {
    throw std::invalid_argument("ridge must be finite and >= 0");
  }

  const double l1 = alpha * lambd;
  const double l2 = (1.0 - alpha) * lambd + ridge;

  Eigen::VectorXd lambda_l1_vec(p);
  Eigen::VectorXd lambda_l2_vec(p);

  for (int j = 0; j < p; ++j) {
    const bool penalized = (penalty_mask[j] != 0);
    lambda_l1_vec[j] = penalized ? l1 : 0.0;
    lambda_l2_vec[j] = penalized ? l2 : 0.0;
  }

  return glm_binomial_elasticnet_with_intercept_core(
      X, y, n_trials, offset,
      intercept0, beta0, standardize,
      lambda_l1_vec, lambda_l2_vec,
      max_iter, tol, link, eps_mu, eps_dmu);
}

inline GLMBinomialENetResult glm_binomial_elasticnet_without_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXi& penalty_mask,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double alpha = 0.5,
    double lambd = 1e-2,
    double ridge = 1e-12,
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15) {

  const int p = static_cast<int>(X.cols());

  if (penalty_mask.size() != p) {
    throw std::invalid_argument("penalty_mask length must match X.cols()");
  }
  if (!(alpha >= 0.0 && alpha <= 1.0)) {
    throw std::invalid_argument("alpha must be in [0,1]");
  }
  if (!(lambd >= 0.0) || !std::isfinite(lambd)) {
    throw std::invalid_argument("lambd must be finite and >= 0");
  }
  if (!(ridge >= 0.0) || !std::isfinite(ridge)) {
    throw std::invalid_argument("ridge must be finite and >= 0");
  }

  const double l1 = alpha * lambd;
  const double l2 = (1.0 - alpha) * lambd + ridge;

  Eigen::VectorXd lambda_l1_vec(p);
  Eigen::VectorXd lambda_l2_vec(p);

  for (int j = 0; j < p; ++j) {
    const bool penalized = (penalty_mask[j] != 0);
    lambda_l1_vec[j] = penalized ? l1 : 0.0;
    lambda_l2_vec[j] = penalized ? l2 : 0.0;
  }

  return glm_binomial_elasticnet_without_intercept_core(
      X, y, n_trials, offset,
      beta0, standardize,
      lambda_l1_vec, lambda_l2_vec,
      max_iter, tol, link, eps_mu, eps_dmu);
}