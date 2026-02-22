// glm_poisson.h
#pragma once

#include <Eigen/Dense>
#include <string>

// Result container for Poisson GLM fits
struct GLMPoissonResult {
  double intercept;        // 0.0 for *_without_intercept
  Eigen::VectorXd beta;    // length p
  bool converged;
  int n_iter;
};

// Poisson GLM with log link via IRLS
// eta = intercept + X * beta + offset
// standardize mask (0/1, length p):
//   - with_intercept: 1 -> center+scale, 0 -> leave as-is
//   - without_intercept: 1 -> scale only (NO centering), 0 -> leave as-is
GLMPoissonResult glm_poisson_with_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    int max_iter = 25,
    double tol = 1e-8,
    double ridge = 1e-12,
    double eps_mu = 1e-15);

GLMPoissonResult glm_poisson_without_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    int max_iter = 25,
    double tol = 1e-8,
    double ridge = 1e-12,
    double eps_mu = 1e-15);