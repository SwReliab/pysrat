// glm_poisson.cpp
// Poisson GLM (log link) via IRLS (outer) + WLS solve (inner)
// with per-coordinate L2 penalties.
//
// - offset is added to linear predictor: eta = intercept + X*beta + offset
// - with_intercept: standardize[j]==1 -> center+scale; standardize[j]==0 -> leave column as-is
// - without_intercept: standardize[j]==1 -> scale only (NO centering); standardize[j]==0 -> leave column as-is
//
// Penalty in standardized space:
//   0.5 * sum_j lambda_l2_vec[j] * beta_j^2
//
// Return: GLMPoissonResult { intercept, beta, converged, n_iter }

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
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

GLMPoissonResult glm_poisson_with_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    const Eigen::VectorXd& lambda_l2_vec,
    int max_iter,
    double tol,
    double eps_mu) {

  const int n = static_cast<int>(X.rows());
  const int p = static_cast<int>(X.cols());

  if (y.size() != n || offset.size() != n) {
    throw std::invalid_argument("y/offset length must match X.rows()");
  }
  if (beta0.size() != p) {
    throw std::invalid_argument("beta0 length must match X.cols()");
  }
  if (standardize.size() != p) {
    throw std::invalid_argument("standardize length must match X.cols()");
  }
  if (lambda_l2_vec.size() != p) {
    throw std::invalid_argument("lambda_l2_vec length must match X.cols()");
  }
  for (int j = 0; j < p; ++j) {
    if (!(lambda_l2_vec[j] >= 0.0) || !std::isfinite(lambda_l2_vec[j])) {
      throw std::invalid_argument("lambda_l2_vec must be finite and >= 0");
    }
  }
  if (max_iter <= 0) {
    return {intercept0, beta0, false, 0};
  }

  const double tiny_sd = 1e-12;

  // --- standardize: center+scale for standardize[j]==1 ---
  Eigen::VectorXd x_mean = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd x_scale = Eigen::VectorXd::Ones(p);
  Eigen::MatrixXd Xs = X;

  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) {
      const double m = X.col(j).mean();
      const Eigen::ArrayXd xc = X.col(j).array() - m;
      const double sd = std::sqrt((xc * xc).mean());
      if (!std::isfinite(sd) || sd < tiny_sd) {
        throw std::invalid_argument(
            "Cannot standardize (near-)constant column j=" + std::to_string(j));
      }
      x_mean[j] = m;
      x_scale[j] = sd;
      Xs.col(j).array() = (Xs.col(j).array() - m) / sd;
    }
  }

  // --- transform initial params into standardized parameterization ---
  // For standardized columns:
  //   X_j * beta_j = ((X_j - mean_j)/sd_j) * (beta_j * sd_j) + mean_j * beta_j
  // so
  //   beta_s[j] = beta[j] * sd_j
  //   b0_s = b0 + sum_j mean_j * beta_j
  double b0 = intercept0;
  Eigen::VectorXd beta = beta0;

  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta[j] = beta0[j] * x_scale[j];
  }
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) b0 += x_mean[j] * beta0[j];
  }

  Eigen::VectorXd eta(n), eta_clip(n), mu(n), W(n), z(n);

  const double eta_hi = 50.0;
  const double eta_lo = -50.0;

  auto obj_from = [&](double b0_try, const Eigen::VectorXd& beta_try) -> double {
    eta.noalias() = Xs * beta_try;
    eta.array() += b0_try;
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

  double obj_old = obj_from(b0, beta);

  for (; it < max_iter; ++it) {
    // eta = b0 + Xs*beta + offset
    eta.noalias() = Xs * beta;
    eta.array() += b0;
    eta.array() += offset.array();

    // clip-consistent eta_clip, mu
    for (int i = 0; i < n; ++i) {
      const double e = clip(eta[i], eta_lo, eta_hi);
      eta_clip[i] = e;
      double m = std::exp(e);
      if (!std::isfinite(m) || m < eps_mu) m = eps_mu;
      mu[i] = m;
    }

    // W = mu  (Poisson, log-link)
    W = mu;

    // z = eta_clip + (y - mu)/mu
    for (int i = 0; i < n; ++i) {
      const double inv_mu = 1.0 / mu[i];
      z[i] = eta_clip[i] + (y[i] - mu[i]) * inv_mu;
    }

    // Weighted least squares target for linear predictor:
    //   z - offset ~= b0 + Xs * beta
    const Eigen::VectorXd ytilde = z - offset;
    const Eigen::VectorXd sqrtW = W.array().sqrt();

    // Build weighted design with explicit intercept column:
    //   Xtilde = [1, Xs]
    Eigen::MatrixXd Xaug_w(n, p + 1);
    for (int i = 0; i < n; ++i) {
      Xaug_w(i, 0) = sqrtW[i];
    }
    Xaug_w.block(0, 1, n, p) = Xs;
    for (int i = 0; i < n; ++i) {
      Xaug_w.row(i).segment(1, p) *= sqrtW[i];
    }

    const Eigen::VectorXd rhs =
        (sqrtW.array() * ytilde.array()).matrix();

    // Normal equations:
    //   [1 X]^T W [1 X] + diag(0, lambda_l2_vec)
    Eigen::MatrixXd A = Xaug_w.transpose() * Xaug_w;
    A.diagonal().segment(1, p).array() += lambda_l2_vec.array();

    const Eigen::VectorXd bvec = Xaug_w.transpose() * rhs;

    Eigen::VectorXd theta_full(p + 1);
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if (ldlt.info() == Eigen::Success) {
      theta_full = ldlt.solve(bvec);
    } else {
      theta_full = A.colPivHouseholderQr().solve(bvec);
    }

    const double b0_full = theta_full[0];
    const Eigen::VectorXd beta_full = theta_full.tail(p);

    // Step-halving on penalized objective
    double step = 1.0;
    double obj_new = -INFINITY;
    Eigen::VectorXd beta_new = beta;
    double b0_new = b0;

    const int max_halving = 25;
    for (int hs = 0; hs < max_halving; ++hs) {
      beta_new = beta + step * (beta_full - beta);
      b0_new   = b0   + step * (b0_full   - b0);

      obj_new = obj_from(b0_new, beta_new);

      if (std::isfinite(obj_new) && obj_new >= obj_old - 1e-12) {
        break;
      }
      step *= 0.5;
    }

    const double max_diff_beta = (beta_new - beta).cwiseAbs().maxCoeff();
    const double max_diff = std::max(max_diff_beta, std::abs(b0_new - b0));

    beta = beta_new;
    b0 = b0_new;

    if (std::isfinite(obj_new)) obj_old = obj_new;

    if (max_diff < tol) {
      converged = true;
      ++it;
      break;
    }
  }

  // --- unstandardize back to original parameterization ---
  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta_out[j] = beta[j] / x_scale[j];
  }

  double b0_out = b0;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) b0_out -= x_mean[j] * beta_out[j];
  }

  return {b0_out, beta_out, converged, it};
}

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

  if (y.size() != n || offset.size() != n) {
    throw std::invalid_argument("y/offset length must match X.rows()");
  }
  if (beta0.size() != p) {
    throw std::invalid_argument("beta0 length must match X.cols()");
  }
  if (standardize.size() != p) {
    throw std::invalid_argument("standardize length must match X.cols()");
  }
  if (lambda_l2_vec.size() != p) {
    throw std::invalid_argument("lambda_l2_vec length must match X.cols()");
  }
  for (int j = 0; j < p; ++j) {
    if (!(lambda_l2_vec[j] >= 0.0) || !std::isfinite(lambda_l2_vec[j])) {
      throw std::invalid_argument("lambda_l2_vec must be finite and >= 0");
    }
  }
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

  // beta0 -> standardized parameterization
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

    for (int i = 0; i < n; ++i) {
      const double e = clip(eta[i], eta_lo, eta_hi);
      eta_clip[i] = e;
      double m = std::exp(e);
      if (!std::isfinite(m) || m < eps_mu) m = eps_mu;
      mu[i] = m;
    }

    // W = mu
    W = mu;

    // z = eta_clip + (y - mu)/mu
    for (int i = 0; i < n; ++i) {
      const double inv_mu = 1.0 / mu[i];
      z[i] = eta_clip[i] + (y[i] - mu[i]) * inv_mu;
    }

    // Weighted least squares target:
    //   z - offset ~= Xs * beta
    const Eigen::VectorXd sqrtW = W.array().sqrt();

    Eigen::MatrixXd Xw = Xs;
    for (int i = 0; i < n; ++i) Xw.row(i) *= sqrtW[i];

    const Eigen::VectorXd rhs =
        (sqrtW.array() * (z.array() - offset.array())).matrix();

    Eigen::MatrixXd A = Xw.transpose() * Xw;
    A.diagonal().array() += lambda_l2_vec.array();
    const Eigen::VectorXd bvec = Xw.transpose() * rhs;

    Eigen::VectorXd beta_full;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if (ldlt.info() == Eigen::Success) beta_full = ldlt.solve(bvec);
    else beta_full = A.colPivHouseholderQr().solve(bvec);

    // Step-halving
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

    const double max_diff = (beta_new - beta).cwiseAbs().maxCoeff();

    beta = beta_new;
    if (std::isfinite(obj_new)) obj_old = obj_new;

    if (max_diff < tol) {
      converged = true;
      ++it;
      break;
    }
  }

  // --- unscale back ---
  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta_out[j] = beta[j] / x_scale[j];
  }

  return {0.0, beta_out, converged, it};
}