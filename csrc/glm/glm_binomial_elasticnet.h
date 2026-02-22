// glm_binomial_elasticnet.h
#pragma once

#include <Eigen/Dense>
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
  double max_delta_inner = 0.0; // last inner sweep max|Δbeta|
};

// ElasticNet Binomial GLM with explicit intercept (never penalized).
// Model:
//   eta = intercept + X*beta + offset
//   mu  = linkinv(eta)
// Aggregated binomial uses y as successes, n_trials as totals.
GLMBinomialENetResult glm_binomial_elasticnet_with_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,   // 0/1 mask length p (with_intercept: center+scale)
    const Eigen::VectorXi& penalty_mask,  // 0/1 mask length p (applies to beta only)
    int max_iter,
    double tol,
    const std::string& link,
    double alpha,
    double lambd,
    double ridge,
    double eps_mu,
    double eps_dmu);

// ElasticNet Binomial GLM without intercept.
// Model:
//   eta = X*beta + offset
//   mu  = linkinv(eta)
GLMBinomialENetResult glm_binomial_elasticnet_without_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,   // 0/1 mask length p (without_intercept: scale-only)
    const Eigen::VectorXi& penalty_mask,  // 0/1 mask length p
    int max_iter,
    double tol,
    const std::string& link,
    double alpha,
    double lambd,
    double ridge,
    double eps_mu,
    double eps_dmu);