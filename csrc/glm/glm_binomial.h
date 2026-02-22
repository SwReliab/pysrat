// glm_binomial.h
#pragma once

#include <Eigen/Dense>
#include <string>

// Result record used by low-level binomial routines that optionally return
// intercept + coefficients + status.
struct GLMBinomialResult {
    double intercept;
    Eigen::VectorXd beta;
    bool converged;
    int n_iter;
};

// Low-level variants that expose intercept handling. These are used by the
// pybind wrappers when callers want access to the intercept separately.
GLMBinomialResult glm_binomial_with_intercept(
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
    double eps_dmu = 1e-15);

GLMBinomialResult glm_binomial_without_intercept(
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
    double eps_dmu = 1e-15);

// Binomial GLM via IRLS (aggregated binomial)
// Assumptions:
//  - y is success counts in [0, n_trials]
//  - offset is additive to linear predictor (often 0)
//  - standardize is 0/1 mask of length p:
//      1: center+scale column
//      0: leave column as-is (e.g., intercept)
// Notes:
//  - ridge is a tiny diagonal add for numerical stability (not a penalty model).
Eigen::VectorXd glm_binomial(
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
    double eps_dmu = 1e-15);

