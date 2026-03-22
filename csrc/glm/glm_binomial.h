// glm_binomial.h
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

// -----------------------------------------------------------------------------
// Result containers
// -----------------------------------------------------------------------------

struct GLMBinomialResult {
  double intercept;        // 0.0 for without-intercept fits
  Eigen::VectorXd beta;    // length p
  bool converged;
  int n_iter;
};

struct GLMBinomialENetResult {
  double intercept = 0.0;       // unstandardized intercept (0.0 for without-intercept)
  Eigen::VectorXd beta;         // unstandardized beta (length p)

  bool converged = false;
  int n_outer = 0;              // executed IRLS iterations
  int n_inner = 0;              // total executed CD sweeps
  double max_delta = 0.0;       // last outer max change
  double max_delta_inner = 0.0; // last inner sweep max change
};

// -----------------------------------------------------------------------------
// Internal core declarations
// -----------------------------------------------------------------------------
//
// glm cores:
//   L2 only
//
// elasticnet cores:
//   L1 + L2
//
// Overloads exist for:
//   - vector L2
//   - matrix L2
//
// All penalties are interpreted in standardized space.

// ----- glm core: with intercept -----
GLMBinomialResult glm_binomial_with_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXd& lambda_l2_vec,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15);

GLMBinomialResult glm_binomial_with_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::MatrixXd& lambda_l2_mat,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15);

// ----- glm core: without intercept -----
GLMBinomialResult glm_binomial_without_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXd& lambda_l2_vec,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15);

GLMBinomialResult glm_binomial_without_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::MatrixXd& lambda_l2_mat,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15);

// ----- elasticnet core: with intercept -----
GLMBinomialENetResult glm_binomial_elasticnet_with_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXd& lambda_l1_vec,
    const Eigen::VectorXd& lambda_l2_vec,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15);

GLMBinomialENetResult glm_binomial_elasticnet_with_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXd& lambda_l1_vec,
    const Eigen::MatrixXd& lambda_l2_mat,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15);

// ----- elasticnet core: without intercept -----
GLMBinomialENetResult glm_binomial_elasticnet_without_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXd& lambda_l1_vec,
    const Eigen::VectorXd& lambda_l2_vec,
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
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXd& lambda_l1_vec,
    const Eigen::MatrixXd& lambda_l2_mat,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15);

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

inline void glm_binomial_validate_penalty_factor(
    const Eigen::VectorXd& penalty_factor,
    int p) {
  if (penalty_factor.size() != p) {
    throw std::invalid_argument("penalty_factor length must match X.cols()");
  }
  if (!penalty_factor.allFinite()) {
    throw std::invalid_argument("penalty_factor must be finite");
  }
  for (int j = 0; j < p; ++j) {
    if (penalty_factor[j] < 0.0) {
      throw std::invalid_argument("penalty_factor must be >= 0");
    }
  }
}

inline void glm_binomial_validate_lambda_l2_mat(
    const Eigen::MatrixXd& lambda_l2_mat,
    int p) {
  if (lambda_l2_mat.rows() != p || lambda_l2_mat.cols() != p) {
    throw std::invalid_argument("lambda_l2_mat must be p x p");
  }
  if (!lambda_l2_mat.allFinite()) {
    throw std::invalid_argument("lambda_l2_mat must be finite");
  }
}

inline Eigen::VectorXd glm_binomial_default_penalty_factor(int p) {
  return Eigen::VectorXd::Ones(p);
}

inline GLMBinomialENetResult glm_binomial_as_enet_result(
    const GLMBinomialResult& r) {
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

inline Eigen::VectorXd glm_binomial_identity_l2_from_penalty_factor(
    double lambda,
    const Eigen::VectorXd& penalty_factor) {
  if (!(lambda >= 0.0) || !std::isfinite(lambda)) {
    throw std::invalid_argument("lambda must be finite and >= 0");
  }
  return lambda * penalty_factor;
}

inline Eigen::MatrixXd glm_binomial_correlated_l2_from_penalty_factor(
    double lambda,
    const Eigen::MatrixXd& lambda_l2_mat,
    const Eigen::VectorXd& penalty_factor) {
  const int p = static_cast<int>(penalty_factor.size());
  glm_binomial_validate_lambda_l2_mat(lambda_l2_mat, p);

  if (!(lambda >= 0.0) || !std::isfinite(lambda)) {
    throw std::invalid_argument("lambda must be finite and >= 0");
  }

  const Eigen::VectorXd sqrt_pf = penalty_factor.array().sqrt();
  const Eigen::MatrixXd D = sqrt_pf.asDiagonal();
  return lambda * (D * lambda_l2_mat * D);
}

// -----------------------------------------------------------------------------
// Public API 1/4:
// glm_binomial_identity
// -----------------------------------------------------------------------------
//
// L2 penalty in standardized space:
//
//   0.5 * lambda * sum_j penalty_factor[j] * beta_j^2
//
// fit_intercept:
//   true  -> with-intercept core
//   false -> without-intercept core

inline GLMBinomialResult glm_binomial_identity(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    bool fit_intercept,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    double lambda,
    const Eigen::VectorXd& penalty_factor,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15) {

  const int p = static_cast<int>(X.cols());
  glm_binomial_validate_penalty_factor(penalty_factor, p);

  const Eigen::VectorXd lambda_l2_vec =
      glm_binomial_identity_l2_from_penalty_factor(lambda, penalty_factor);

  if (fit_intercept) {
    return glm_binomial_with_intercept_core(
        X, y, n_trials, offset, intercept0, beta0, standardize,
        lambda_l2_vec, max_iter, tol, link, eps_mu, eps_dmu);
  } else {
    return glm_binomial_without_intercept_core(
        X, y, n_trials, offset, beta0, standardize,
        lambda_l2_vec, max_iter, tol, link, eps_mu, eps_dmu);
  }
}

inline GLMBinomialResult glm_binomial_identity(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    bool fit_intercept,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    double lambda,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15) {

  const int p = static_cast<int>(X.cols());
  return glm_binomial_identity(
      X, y, n_trials, offset, fit_intercept, intercept0, beta0, standardize,
      lambda, glm_binomial_default_penalty_factor(p),
      max_iter, tol, link, eps_mu, eps_dmu);
}

// -----------------------------------------------------------------------------
// Public API 2/4:
// glm_binomial_correlated
// -----------------------------------------------------------------------------
//
// L2 penalty in standardized space:
//
//   0.5 * beta^T [ lambda * D^(1/2) * lambda_l2_mat * D^(1/2) ] beta
//
// where D = diag(penalty_factor)

inline GLMBinomialResult glm_binomial_correlated(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    bool fit_intercept,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::MatrixXd& lambda_l2_mat,
    double lambda,
    const Eigen::VectorXd& penalty_factor,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15) {

  const int p = static_cast<int>(X.cols());
  glm_binomial_validate_penalty_factor(penalty_factor, p);

  const Eigen::MatrixXd lambda_l2_eff =
      glm_binomial_correlated_l2_from_penalty_factor(
          lambda, lambda_l2_mat, penalty_factor);

  if (fit_intercept) {
    return glm_binomial_with_intercept_core(
        X, y, n_trials, offset, intercept0, beta0, standardize,
        lambda_l2_eff, max_iter, tol, link, eps_mu, eps_dmu);
  } else {
    return glm_binomial_without_intercept_core(
        X, y, n_trials, offset, beta0, standardize,
        lambda_l2_eff, max_iter, tol, link, eps_mu, eps_dmu);
  }
}

inline GLMBinomialResult glm_binomial_correlated(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    bool fit_intercept,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::MatrixXd& lambda_l2_mat,
    double lambda,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15) {

  const int p = static_cast<int>(X.cols());
  return glm_binomial_correlated(
      X, y, n_trials, offset, fit_intercept, intercept0, beta0, standardize,
      lambda_l2_mat, lambda, glm_binomial_default_penalty_factor(p),
      max_iter, tol, link, eps_mu, eps_dmu);
}

// -----------------------------------------------------------------------------
// Public API 3/4:
// glmnet_binomial_identity
// -----------------------------------------------------------------------------
//
// L1 penalty:
//   sum_j alpha * lambda * penalty_factor[j] * |beta_j|
//
// L2 penalty:
//   0.5 * sum_j (1-alpha) * lambda * penalty_factor[j] * beta_j^2
//
// If alpha == 0, route internally to glm_binomial_identity.

inline GLMBinomialENetResult glmnet_binomial_identity(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    bool fit_intercept,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    double alpha,
    double lambda,
    const Eigen::VectorXd& penalty_factor,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15) {

  const int p = static_cast<int>(X.cols());
  glm_binomial_validate_penalty_factor(penalty_factor, p);

  if (!(alpha >= 0.0 && alpha <= 1.0) || !std::isfinite(alpha)) {
    throw std::invalid_argument("alpha must be finite and in [0,1]");
  }
  if (!(lambda >= 0.0) || !std::isfinite(lambda)) {
    throw std::invalid_argument("lambda must be finite and >= 0");
  }

  if (alpha == 0.0) {
    return glm_binomial_as_enet_result(
        glm_binomial_identity(
            X, y, n_trials, offset, fit_intercept, intercept0, beta0, standardize,
            lambda, penalty_factor, max_iter, tol, link, eps_mu, eps_dmu));
  }

  Eigen::VectorXd lambda_l1_vec(p);
  Eigen::VectorXd lambda_l2_vec(p);

  for (int j = 0; j < p; ++j) {
    lambda_l1_vec[j] = alpha * lambda * penalty_factor[j];
    lambda_l2_vec[j] = (1.0 - alpha) * lambda * penalty_factor[j];
  }

  if (fit_intercept) {
    return glm_binomial_elasticnet_with_intercept_core(
        X, y, n_trials, offset, intercept0, beta0, standardize,
        lambda_l1_vec, lambda_l2_vec,
        max_iter, tol, link, eps_mu, eps_dmu);
  } else {
    return glm_binomial_elasticnet_without_intercept_core(
        X, y, n_trials, offset, beta0, standardize,
        lambda_l1_vec, lambda_l2_vec,
        max_iter, tol, link, eps_mu, eps_dmu);
  }
}

inline GLMBinomialENetResult glmnet_binomial_identity(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    bool fit_intercept,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    double alpha,
    double lambda,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15) {

  const int p = static_cast<int>(X.cols());
  return glmnet_binomial_identity(
      X, y, n_trials, offset, fit_intercept, intercept0, beta0, standardize,
      alpha, lambda, glm_binomial_default_penalty_factor(p),
      max_iter, tol, link, eps_mu, eps_dmu);
}

// -----------------------------------------------------------------------------
// Public API 4/4:
// glmnet_binomial_correlated
// -----------------------------------------------------------------------------
//
// L1 penalty:
//   sum_j alpha * lambda * penalty_factor[j] * |beta_j|
//
// L2 penalty:
//   0.5 * beta^T [ (1-alpha) * lambda * D^(1/2) * lambda_l2_mat * D^(1/2) ] beta
//
// If alpha == 0, route internally to glm_binomial_correlated.

inline GLMBinomialENetResult glmnet_binomial_correlated(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    bool fit_intercept,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::MatrixXd& lambda_l2_mat,
    double alpha,
    double lambda,
    const Eigen::VectorXd& penalty_factor,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15) {

  const int p = static_cast<int>(X.cols());
  glm_binomial_validate_penalty_factor(penalty_factor, p);
  glm_binomial_validate_lambda_l2_mat(lambda_l2_mat, p);

  if (!(alpha >= 0.0 && alpha <= 1.0) || !std::isfinite(alpha)) {
    throw std::invalid_argument("alpha must be finite and in [0,1]");
  }
  if (!(lambda >= 0.0) || !std::isfinite(lambda)) {
    throw std::invalid_argument("lambda must be finite and >= 0");
  }

  if (alpha == 0.0) {
    return glm_binomial_as_enet_result(
        glm_binomial_correlated(
            X, y, n_trials, offset, fit_intercept, intercept0, beta0, standardize,
            lambda_l2_mat, lambda, penalty_factor, max_iter, tol, link, eps_mu, eps_dmu));
  }

  Eigen::VectorXd lambda_l1_vec(p);
  for (int j = 0; j < p; ++j) {
    lambda_l1_vec[j] = alpha * lambda * penalty_factor[j];
  }

  const Eigen::MatrixXd lambda_l2_eff =
      glm_binomial_correlated_l2_from_penalty_factor(
          (1.0 - alpha) * lambda, lambda_l2_mat, penalty_factor);

  if (fit_intercept) {
    return glm_binomial_elasticnet_with_intercept_core(
        X, y, n_trials, offset, intercept0, beta0, standardize,
        lambda_l1_vec, lambda_l2_eff,
        max_iter, tol, link, eps_mu, eps_dmu);
  } else {
    return glm_binomial_elasticnet_without_intercept_core(
        X, y, n_trials, offset, beta0, standardize,
        lambda_l1_vec, lambda_l2_eff,
        max_iter, tol, link, eps_mu, eps_dmu);
  }
}

inline GLMBinomialENetResult glmnet_binomial_correlated(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    bool fit_intercept,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::MatrixXd& lambda_l2_mat,
    double alpha,
    double lambda,
    int max_iter = 25,
    double tol = 1e-8,
    const std::string& link = "logit",
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15) {

  const int p = static_cast<int>(X.cols());
  return glmnet_binomial_correlated(
      X, y, n_trials, offset, fit_intercept, intercept0, beta0, standardize,
      lambda_l2_mat, alpha, lambda,
      glm_binomial_default_penalty_factor(p),
      max_iter, tol, link, eps_mu, eps_dmu);
}