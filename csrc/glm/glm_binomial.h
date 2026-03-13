// glm_binomial.h
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

#include "glm_binomial_elasticnet.h"

// Result record used by low-level binomial routines that optionally return
// intercept + coefficients + status.
struct GLMBinomialResult {
  double intercept;        // 0.0 for *_without_intercept
  Eigen::VectorXd beta;
  bool converged;
  int n_iter;
};

// -----------------------------------------------------------------------------
// Core APIs: per-coordinate L2 penalties in standardized space
// -----------------------------------------------------------------------------
//
// Penalty in standardized space:
//   0.5 * sum_j lambda_l2_vec[j] * beta_j^2
//
// These are the actual implementations in .cpp.

GLMBinomialResult glm_binomial_with_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXd& lambda_l2_vec,  // length p, >= 0
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-12,
    double eps_dmu = 1e-15);

GLMBinomialResult glm_binomial_without_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXd& lambda_l2_vec,  // length p, >= 0
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-12,
    double eps_dmu = 1e-15);

// -----------------------------------------------------------------------------
// Backward-compatible wrappers: scalar ridge only
// -----------------------------------------------------------------------------
//
// Existing API compatibility:
// - ridge is applied equally to all coefficients in standardized space
// - no L1 term

inline GLMBinomialResult glm_binomial_with_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double ridge = 1e-12,
    double eps_mu = 1e-12,
    double eps_dmu = 1e-15) {

  const int p = static_cast<int>(X.cols());

  if (!(ridge >= 0.0) || !std::isfinite(ridge)) {
    throw std::invalid_argument("ridge must be finite and >= 0");
  }

  const Eigen::VectorXd lambda_l2_vec = Eigen::VectorXd::Constant(p, ridge);

  return glm_binomial_with_intercept_core(
      X, y, n_trials, offset,
      intercept0, beta0, standardize,
      lambda_l2_vec,
      max_iter, tol, link, eps_mu, eps_dmu);
}

inline GLMBinomialResult glm_binomial_without_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double ridge = 1e-12,
    double eps_mu = 1e-12,
    double eps_dmu = 1e-15) {

  const int p = static_cast<int>(X.cols());

  if (!(ridge >= 0.0) || !std::isfinite(ridge)) {
    throw std::invalid_argument("ridge must be finite and >= 0");
  }

  const Eigen::VectorXd lambda_l2_vec = Eigen::VectorXd::Constant(p, ridge);

  return glm_binomial_without_intercept_core(
      X, y, n_trials, offset,
      beta0, standardize,
      lambda_l2_vec,
      max_iter, tol, link, eps_mu, eps_dmu);
}

// -----------------------------------------------------------------------------
// Wrappers for ElasticNet(alpha = 0) routing
// -----------------------------------------------------------------------------
//
// Old elasticnet API at alpha = 0:
//   L1 = 0
//   L2 = lambd + ridge
//
// If penalty_mask[j] == 0, the coefficient is fully unpenalized,
// including the stabilization ridge (to match the old elasticnet behavior).
//
// These wrappers return GLMBinomialENetResult directly so callers (including
// pybind) do not need to manually convert GLMBinomialResult -> GLMBinomialENetResult.

inline GLMBinomialENetResult glm_binomial_with_intercept_from_elasticnet_alpha0(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXi& penalty_mask,   // 0/1 length p
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double lambd = 1e-2,
    double ridge = 1e-12,
    double eps_mu = 1e-12,
    double eps_dmu = 1e-15) {

  const int p = static_cast<int>(X.cols());

  if (penalty_mask.size() != p) {
    throw std::invalid_argument("penalty_mask length must match X.cols()");
  }
  if (!(lambd >= 0.0) || !std::isfinite(lambd)) {
    throw std::invalid_argument("lambd must be finite and >= 0");
  }
  if (!(ridge >= 0.0) || !std::isfinite(ridge)) {
    throw std::invalid_argument("ridge must be finite and >= 0");
  }

  const double l2 = lambd + ridge;

  Eigen::VectorXd lambda_l2_vec(p);
  for (int j = 0; j < p; ++j) {
    lambda_l2_vec[j] = (penalty_mask[j] != 0) ? l2 : 0.0;
  }

  const GLMBinomialResult r = glm_binomial_with_intercept_core(
      X, y, n_trials, offset,
      intercept0, beta0, standardize,
      lambda_l2_vec,
      max_iter, tol, link, eps_mu, eps_dmu);

  GLMBinomialENetResult out;
  out.intercept = r.intercept;
  out.beta = r.beta;
  out.converged = r.converged;
  out.n_outer = r.n_iter;
  out.n_inner = 0;
  out.max_delta = 0.0;
  out.max_delta_inner = 0.0;
  return out;
}

inline GLMBinomialENetResult glm_binomial_without_intercept_from_elasticnet_alpha0(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXi& penalty_mask,   // 0/1 length p
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double lambd = 1e-2,
    double ridge = 1e-12,
    double eps_mu = 1e-12,
    double eps_dmu = 1e-15) {

  const int p = static_cast<int>(X.cols());

  if (penalty_mask.size() != p) {
    throw std::invalid_argument("penalty_mask length must match X.cols()");
  }
  if (!(lambd >= 0.0) || !std::isfinite(lambd)) {
    throw std::invalid_argument("lambd must be finite and >= 0");
  }
  if (!(ridge >= 0.0) || !std::isfinite(ridge)) {
    throw std::invalid_argument("ridge must be finite and >= 0");
  }

  const double l2 = lambd + ridge;

  Eigen::VectorXd lambda_l2_vec(p);
  for (int j = 0; j < p; ++j) {
    lambda_l2_vec[j] = (penalty_mask[j] != 0) ? l2 : 0.0;
  }

  const GLMBinomialResult r = glm_binomial_without_intercept_core(
      X, y, n_trials, offset,
      beta0, standardize,
      lambda_l2_vec,
      max_iter, tol, link, eps_mu, eps_dmu);

  GLMBinomialENetResult out;
  out.intercept = r.intercept;
  out.beta = r.beta;
  out.converged = r.converged;
  out.n_outer = r.n_iter;
  out.n_inner = 0;
  out.max_delta = 0.0;
  out.max_delta_inner = 0.0;
  return out;
}

// -----------------------------------------------------------------------------
// Legacy glm_binomial(...) convenience API
// -----------------------------------------------------------------------------
//
// This legacy function historically returned only beta and assumed the caller
// handled any intercept convention externally. It is retained here as a thin
// wrapper around the no-intercept solver.

inline Eigen::VectorXd glm_binomial(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double ridge = 1e-12,
    double eps_mu = 1e-12,
    double eps_dmu = 1e-15) {

  return glm_binomial_without_intercept(
      X, y, n_trials, offset,
      beta0, standardize,
      max_iter, tol, link, ridge, eps_mu, eps_dmu).beta;
}