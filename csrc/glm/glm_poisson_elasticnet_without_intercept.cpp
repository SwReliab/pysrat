// glm_poisson_elasticnet_without_intercept.cpp
//
// Poisson GLM without intercept, with per-coordinate Elastic Net penalties
// via IRLS (outer) + Coordinate Descent (inner)
//
// Model:
//   y_i ~ Poisson(mu_i)
//   log(mu_i) = x_i^T beta + offset_i
//
// API:
// - standardize[j] = 1 -> scale-only column j before fitting
// - standardize[j] = 0 -> leave column j as-is
// - lambda_l1[j] >= 0 : L1 penalty for beta_j in standardized space
// - lambda_l2[j] >= 0 : L2 penalty for beta_j in standardized space
// - intercept is fixed at 0.0
//
// Penalized objective in standardized space:
//   ll(beta)
//   - sum_j lambda_l1[j] * |beta_j|
//   - 0.5 * sum_j lambda_l2[j] * beta_j^2
//
// Notes:
// 1) clip-consistent eta/mu/z: mu = exp(clip(eta)), z uses eta_clip (= log(mu))
// 2) optional step-halving on penalized objective for IRLS stability
// 3) correct n_outer tracking
// 4) faster inner CD via delta update and cached u = W .* r
//
// Returns: GLMPoissonENetResult with convergence diagnostics.

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

static inline double clip(double x, double lo, double hi) {
  return std::min(std::max(x, lo), hi);
}

static inline double soft_threshold(double x, double t) {
  if (x > t) return x - t;
  if (x < -t) return x + t;
  return 0.0;
}

struct GLMPoissonENetResult {
  double intercept = 0.0;       // always 0.0 for without-intercept
  Eigen::VectorXd beta;         // unstandardized beta (length p)

  bool converged = false;
  int n_outer = 0;              // executed IRLS iterations
  int n_inner = 0;              // total executed CD sweeps (sum over outer iters)
  double max_delta = 0.0;       // final outer max(max|Δbeta|)
  double max_delta_inner = 0.0; // last inner sweep max|Δbeta|
};

GLMPoissonENetResult glm_poisson_elasticnet_without_intercept_core(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,   // 0/1 length p (scale-only)
    const Eigen::VectorXd& lambda_l1,     // length p, >= 0
    const Eigen::VectorXd& lambda_l2,     // length p, >= 0
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
  if (lambda_l1.size() != p) {
    throw std::invalid_argument("lambda_l1 length must match X.cols()");
  }
  if (lambda_l2.size() != p) {
    throw std::invalid_argument("lambda_l2 length must match X.cols()");
  }
  for (int j = 0; j < p; ++j) {
    if (!(lambda_l1[j] >= 0.0) || !std::isfinite(lambda_l1[j])) {
      throw std::invalid_argument("lambda_l1 must be finite and >= 0");
    }
    if (!(lambda_l2[j] >= 0.0) || !std::isfinite(lambda_l2[j])) {
      throw std::invalid_argument("lambda_l2 must be finite and >= 0");
    }
  }
  if (max_iter <= 0) {
    GLMPoissonENetResult out;
    out.intercept = 0.0;
    out.beta = beta0;
    out.converged = false;
    out.n_outer = 0;
    out.n_inner = 0;
    out.max_delta = 0.0;
    out.max_delta_inner = 0.0;
    return out;
  }
  if (!(tol > 0.0) || !std::isfinite(tol)) {
    throw std::invalid_argument("tol must be positive finite");
  }
  if (!(eps_mu > 0.0) || !std::isfinite(eps_mu)) {
    throw std::invalid_argument("eps_mu must be positive finite");
  }

  const double tiny_sd = 1e-12;

  // -------- Standardize X -> Xs (scale-only where standardize[j]==1) --------
  // No centering because there is no intercept term.
  Eigen::VectorXd x_scale(p);
  x_scale.setOnes();

  Eigen::MatrixXd Xs = X;

  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) {
      // Scale-only: use SD around the mean, but do NOT center Xs.
      const double m = X.col(j).mean();
      const Eigen::ArrayXd xc = X.col(j).array() - m;
      const double var = (xc * xc).mean();
      const double sd = std::sqrt(var);

      if (!std::isfinite(sd) || sd < tiny_sd) {
        throw std::invalid_argument(
            "Cannot scale (near-)constant column j=" + std::to_string(j));
      }
      x_scale[j] = sd;
      Xs.col(j).array() /= sd;
    }
  }

  // -------- Initialize in standardized parameterization --------
  // X*beta == Xs*beta_s  (since Xs = X / sd)
  // => beta_s[j] = beta[j] * sd
  Eigen::VectorXd beta = beta0;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta[j] = beta0[j] * x_scale[j];
  }

  // IRLS state
  Eigen::VectorXd eta(n), eta_clip(n), mu(n), W(n), z(n);

  // exp overflow guard (must match z / objective)
  const double eta_hi = 50.0;
  const double eta_lo = -50.0;

  // Coordinate descent settings
  const int cd_max_iter = 300;
  const double cd_tol = std::max(1e-12, tol * 0.1);

  // Penalized objective in standardized space (for step-halving)
  auto penalized_obj = [&](const Eigen::VectorXd& beta_try) -> double {
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
      const double bj = beta_try[j];
      pen += lambda_l1[j] * std::abs(bj) + 0.5 * lambda_l2[j] * bj * bj;
    }
    return llf - pen;
  };

  bool converged = false;
  int outer_exec = 0;
  int total_cd_sweeps = 0;
  double last_outer_delta = 0.0;
  double last_inner_delta = 0.0;

  double obj_old = penalized_obj(beta);

  for (int outer = 0; outer < max_iter; ++outer) {
    ++outer_exec;

    // ---- IRLS: eta, mu ----
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
      z[i] = eta_clip[i] + (y[i] - mu[i]) / mu[i];
    }

    // Working response:
    // ytilde = z - offset  and model: ytilde ~= Xs * beta
    const Eigen::VectorXd ytilde = z - offset;

    // ---- inner: coordinate descent on penalized WLS (no intercept) ----
    // residual r = ytilde - Xs * beta
    Eigen::VectorXd r = ytilde - (Xs * beta);

    // u = W .* r
    Eigen::VectorXd u = W.array() * r.array();

    // Precompute:
    //   aj_x[j] = sum_i W_i x_ij^2
    //   aj[j]   = aj_x[j] + lambda_l2[j]
    Eigen::VectorXd aj_x(p), aj(p);
    for (int j = 0; j < p; ++j) {
      double s = 0.0;
      for (int i = 0; i < n; ++i) {
        const double xij = Xs(i, j);
        s += W[i] * xij * xij;
      }
      if (!(s >= 0.0) || !std::isfinite(s)) s = 0.0;
      aj_x[j] = s;

      double a = s + lambda_l2[j];
      if (!(a > 0.0) || !std::isfinite(a)) a = 1.0;
      aj[j] = a;
    }

    Eigen::VectorXd beta_new = beta;

    int cd_sweeps_this_outer = 0;
    double cd_last_max_change = std::numeric_limits<double>::infinity();

    for (int cd_it = 0; cd_it < cd_max_iter; ++cd_it) {
      ++cd_sweeps_this_outer;

      double max_change = 0.0;

      for (int j = 0; j < p; ++j) {
        const double bj_old = beta_new[j];

        // With full residual r = ytilde - X beta,
        // rho_j = x_j^T (W .* r) + (x_j^T W x_j) * beta_j
        double xTu = 0.0;
        for (int i = 0; i < n; ++i) {
          xTu += Xs(i, j) * u[i];
        }

        const double rho = xTu + aj_x[j] * bj_old;
        const double bj_new = soft_threshold(rho, lambda_l1[j]) / aj[j];
        const double delta = bj_new - bj_old;

        if (delta != 0.0) {
          for (int i = 0; i < n; ++i) {
            const double xij = Xs(i, j);
            r[i] -= xij * delta;
            u[i] -= W[i] * xij * delta;
          }
          beta_new[j] = bj_new;
        }

        const double ch = std::abs(delta);
        if (ch > max_change) max_change = ch;
      }

      cd_last_max_change = max_change;
      if (max_change < cd_tol) break;
    }

    total_cd_sweeps += cd_sweeps_this_outer;

    // ---- Step-halving on penalized objective (outer stability) ----
    double step = 1.0;
    double obj_new = -INFINITY;
    Eigen::VectorXd beta_acc = beta;

    const int max_halving = 25;
    for (int hs = 0; hs < max_halving; ++hs) {
      Eigen::VectorXd b_try = beta + step * (beta_new - beta);
      obj_new = penalized_obj(b_try);

      if (std::isfinite(obj_new) && obj_new >= obj_old - 1e-12) {
        beta_acc = b_try;
        break;
      }
      step *= 0.5;
    }

    // ---- outer convergence ----
    double max_abs = 0.0;
    for (int j = 0; j < p; ++j) {
      const double d = std::abs(beta_acc[j] - beta[j]);
      if (d > max_abs) max_abs = d;
    }

    beta = beta_acc;

    if (std::isfinite(obj_new)) obj_old = obj_new;

    last_outer_delta = max_abs;
    last_inner_delta = std::isfinite(cd_last_max_change) ? cd_last_max_change : 0.0;

    if (max_abs < tol) {
      converged = true;
      break;
    }
  }

  // -------- Unstandardize back to original space --------
  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) beta_out[j] = beta[j] / x_scale[j];
  }

  GLMPoissonENetResult out;
  out.intercept = 0.0;
  out.beta = beta_out;
  out.converged = converged;
  out.n_outer = outer_exec;
  out.n_inner = total_cd_sweeps;
  out.max_delta = last_outer_delta;
  out.max_delta_inner = last_inner_delta;
  return out;
}