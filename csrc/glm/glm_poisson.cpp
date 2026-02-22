// glm_poisson.cpp
// Poisson GLM (log link) via IRLS (outer) + WLS solve (inner)
// - offset is added to linear predictor: eta = intercept + X*beta + offset
// - with_intercept: standardize[j]==1 -> center+scale; standardize[j]==0 -> leave column as-is
// - without_intercept: standardize[j]==1 -> scale only (NO centering); standardize[j]==0 -> leave column as-is
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

GLMPoissonResult glm_poisson_with_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    int max_iter,
    double tol,
    double ridge,
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
        throw std::invalid_argument("Cannot standardize (near-)constant column j=" + std::to_string(j));
      }
      x_mean[j] = m;
      x_scale[j] = sd;
      Xs.col(j).array() = (Xs.col(j).array() - m) / sd;
    }
  }

  // --- transform initial params into standardized parameterization ---
  // Xs uses: (X - mean)/sd for standardized cols
  // so (X*beta + b0) = (Xs*beta_s + b0_s)
  // with:
  //   beta_s[j] = beta[j] * sd_j (for standardized cols)
  //   b0_s = b0 + sum(mean_j * beta[j]) over standardized cols
  double b0 = intercept0;
  Eigen::VectorXd beta = beta0;

  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta[j] = beta0[j] * x_scale[j];
  }
  {
    double shift = 0.0;
    for (int j = 0; j < p; ++j) {
      if (standardize[j] != 0) shift += x_mean[j] * beta0[j];
    }
    b0 += shift;
  }

  Eigen::VectorXd eta(n), eta_clip(n), mu(n), W(n), z(n);

  // exp overflow guard: exp(30) ~ 1e13, exp(50) ~ 5e21
  // Choose conservatively; 30 is often enough, but keep your original 50 if needed.
  const double eta_hi = 50.0;
  const double eta_lo = -50.0;

  auto llf_from = [&](double b0_try, const Eigen::VectorXd& beta_try) -> double {
    // Poisson log-likelihood (up to constant -lgamma(y+1), which we include)
    // Use eta_clip consistently with mu to avoid mismatched objective under clipping.
    eta.noalias() = Xs * beta_try;
    eta.array() += b0_try;
    eta.array() += offset.array();

    double llf = 0.0;
    for (int i = 0; i < n; ++i) {
      const double e = clip(eta[i], eta_lo, eta_hi);
      const double m = std::max(std::exp(e), eps_mu);
      // y*eta - mu - log(y!)
      llf += y[i] * e - m - std::lgamma(y[i] + 1.0);
    }
    return llf;
  };

  bool converged = false;
  int it = 0;

  // initial llf (for step-halving)
  double llf_old = llf_from(b0, beta);

  for (; it < max_iter; ++it) {
    // eta = b0 + Xs*beta + offset
    eta.noalias() = Xs * beta;
    eta.array() += b0;
    eta.array() += offset.array();

    // eta_clip and mu must be consistent
    for (int i = 0; i < n; ++i) {
      const double e = clip(eta[i], eta_lo, eta_hi);
      eta_clip[i] = e;
      double m = std::exp(e);
      if (!std::isfinite(m) || m < eps_mu) m = eps_mu;
      mu[i] = m;
    }

    // W = mu  (Poisson, log-link)
    W = mu;

    // z = eta_clip + (y - mu)/mu  (IMPORTANT: use eta_clip, not raw eta)
    for (int i = 0; i < n; ++i) {
      const double inv_mu = 1.0 / mu[i];
      z[i] = eta_clip[i] + (y[i] - mu[i]) * inv_mu;
    }

    // Weighted least squares for beta given current z
    const Eigen::VectorXd sqrtW = W.array().sqrt();

    Eigen::MatrixXd Xw = Xs;
    for (int i = 0; i < n; ++i) Xw.row(i) *= sqrtW[i];

    // rhs corresponds to (z - offset - b0) * sqrtW
    const Eigen::VectorXd rhs = (sqrtW.array() * (z.array() - offset.array() - b0)).matrix();

    Eigen::MatrixXd A = Xw.transpose() * Xw;
    A.diagonal().array() += ridge;
    const Eigen::VectorXd bvec = Xw.transpose() * rhs;

    Eigen::VectorXd beta_full;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if (ldlt.info() == Eigen::Success) beta_full = ldlt.solve(bvec);
    else beta_full = A.colPivHouseholderQr().solve(bvec);

    // Update intercept in closed form (weighted mean residual)
    double b0_full = b0;
    const double sw = W.sum();
    if (sw > 0.0 && std::isfinite(sw)) {
      // r = z - offset - Xs*beta_full
      const Eigen::VectorXd r = (z - offset) - (Xs * beta_full);
      b0_full = (W.array() * r.array()).sum() / sw;
    }

    // --- step-halving: accept only if llf improves (or stays)
    // This is important for stability on small n / near-separation cases.
    double step = 1.0;
    double llf_new = -INFINITY;
    Eigen::VectorXd beta_new = beta;
    double b0_new = b0;

    const int max_halving = 25;
    for (int hs = 0; hs < max_halving; ++hs) {
      beta_new = beta + step * (beta_full - beta);
      b0_new   = b0   + step * (b0_full   - b0);

      llf_new = llf_from(b0_new, beta_new);

      if (std::isfinite(llf_new) && llf_new >= llf_old - 1e-12) {
        break; // accept
      }
      step *= 0.5;
    }

    // If still not finite, just take the (damped) step we have; caller can see non-convergence.
    const double max_diff_beta = (beta_new - beta).cwiseAbs().maxCoeff();
    const double max_diff = std::max(max_diff_beta, std::abs(b0_new - b0));

    beta = beta_new;
    b0 = b0_new;

    if (std::isfinite(llf_new)) llf_old = llf_new;

    if (max_diff < tol) {
      converged = true;
      ++it; // count this iteration as executed
      break;
    }
  }

  // --- unstandardize back to original parameterization ---
  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta_out[j] = beta[j] / x_scale[j];
  }
  double b0_out = b0;
  {
    double shift = 0.0;
    for (int j = 0; j < p; ++j) {
      if (standardize[j] != 0) shift += x_mean[j] * beta_out[j];
    }
    b0_out -= shift;
  }

  return {b0_out, beta_out, converged, it};
}

GLMPoissonResult glm_poisson_without_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    int max_iter,
    double tol,
    double ridge,
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
        throw std::invalid_argument("Cannot scale (near-)constant column j=" + std::to_string(j));
      }
      x_scale[j] = sd;
      Xs.col(j).array() /= sd;
    }
  }

  // beta0 -> standardized parameterization (since Xs = X / sd, beta_s = beta * sd)
  Eigen::VectorXd beta = beta0;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta[j] = beta0[j] * x_scale[j];
  }

  Eigen::VectorXd eta(n), eta_clip(n), mu(n), W(n), z(n);

  const double eta_hi = 50.0;
  const double eta_lo = -50.0;

  auto llf_from = [&](const Eigen::VectorXd& beta_try) -> double {
    // Poisson log-likelihood (including -lgamma(y+1))
    // Use clipping consistently to match the IRLS linearization under clipping.
    eta.noalias() = Xs * beta_try;
    eta.array() += offset.array();

    double llf = 0.0;
    for (int i = 0; i < n; ++i) {
      const double e = clip(eta[i], eta_lo, eta_hi);
      const double m = std::max(std::exp(e), eps_mu);
      llf += y[i] * e - m - std::lgamma(y[i] + 1.0);
    }
    return llf;
  };

  bool converged = false;
  int it = 0;

  double llf_old = llf_from(beta);

  for (; it < max_iter; ++it) {
    // eta = Xs*beta + offset (no intercept)
    eta.noalias() = Xs * beta;
    eta.array() += offset.array();

    // eta_clip and mu must be consistent
    for (int i = 0; i < n; ++i) {
      const double e = clip(eta[i], eta_lo, eta_hi);
      eta_clip[i] = e;
      double m = std::exp(e);
      if (!std::isfinite(m) || m < eps_mu) m = eps_mu;
      mu[i] = m;
    }

    // W = mu (Poisson log-link)
    W = mu;

    // z = eta_clip + (y - mu)/mu   IMPORTANT: use eta_clip
    for (int i = 0; i < n; ++i) {
      const double inv_mu = 1.0 / mu[i];
      z[i] = eta_clip[i] + (y[i] - mu[i]) * inv_mu;
    }

    // Weighted least squares for beta given current z:
    // minimize Σ W_i * (z_i - (Xs*beta + offset)_i)^2
    const Eigen::VectorXd sqrtW = W.array().sqrt();

    Eigen::MatrixXd Xw = Xs;
    for (int i = 0; i < n; ++i) Xw.row(i) *= sqrtW[i];

    const Eigen::VectorXd rhs = (sqrtW.array() * (z.array() - offset.array())).matrix();

    Eigen::MatrixXd A = Xw.transpose() * Xw;
    A.diagonal().array() += ridge;
    const Eigen::VectorXd bvec = Xw.transpose() * rhs;

    Eigen::VectorXd beta_full;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if (ldlt.info() == Eigen::Success) beta_full = ldlt.solve(bvec);
    else beta_full = A.colPivHouseholderQr().solve(bvec);

    // --- step-halving: accept only if llf improves (or stays)
    double step = 1.0;
    double llf_new = -INFINITY;
    Eigen::VectorXd beta_new = beta;

    const int max_halving = 25;
    for (int hs = 0; hs < max_halving; ++hs) {
      beta_new = beta + step * (beta_full - beta);
      llf_new = llf_from(beta_new);

      if (std::isfinite(llf_new) && llf_new >= llf_old - 1e-12) {
        break; // accept
      }
      step *= 0.5;
    }

    const double max_diff = (beta_new - beta).cwiseAbs().maxCoeff();

    beta = beta_new;
    if (std::isfinite(llf_new)) llf_old = llf_new;

    if (max_diff < tol) {
      converged = true;
      ++it;
      break;
    }
  }

  // unscale back
  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta_out[j] = beta[j] / x_scale[j];
  }

  return {0.0, beta_out, converged, it};
}
