// glm_poisson_elasticnet.h
#pragma once

#include <Eigen/Dense>
#include <string>

// Result struct for Poisson ElasticNet GLM (IRLS outer + CD inner)
struct GLMPoissonENetResult {
  double intercept = 0.0;       // unstandardized intercept (0.0 for without-intercept)
  Eigen::VectorXd beta;         // unstandardized beta (length p)

  bool converged = false;
  int n_outer = 0;              // executed IRLS iterations
  int n_inner = 0;              // total executed CD sweeps (sum over outer iters)
  double max_delta = 0.0;       // last outer max change
  double max_delta_inner = 0.0; // last inner sweep max change
};

// Poisson ElasticNet GLM (log link), WITH intercept (explicit; never penalized).
//
// Model: eta = intercept + X*beta + offset,  mu = exp(eta)
//
// standardize (0/1 length p):
//   1 -> center+scale column j
//   0 -> leave as-is
//
// penalty_mask (0/1 length p): applies to beta only
//   1 -> penalize beta_j
//   0 -> no penalty on beta_j
GLMPoissonENetResult glm_poisson_elasticnet_with_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXi& penalty_mask,
    int max_iter = 25,
    double tol = 1e-8,
    double alpha = 0.5,
    double lambd = 1e-2,
    double ridge = 1e-12,
    double eps_mu = 1e-15);

// Poisson ElasticNet GLM (log link), WITHOUT intercept.
//
// Model: eta = X*beta + offset,  mu = exp(eta)
//
// standardize (0/1 length p):
//   1 -> scale-only (no centering)
//   0 -> leave as-is
//
// penalty_mask (0/1 length p): applies to beta only
GLMPoissonENetResult glm_poisson_elasticnet_without_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXi& penalty_mask,
    int max_iter = 25,
    double tol = 1e-8,
    double alpha = 0.5,
    double lambd = 1e-2,
    double ridge = 1e-12,
    double eps_mu = 1e-15);