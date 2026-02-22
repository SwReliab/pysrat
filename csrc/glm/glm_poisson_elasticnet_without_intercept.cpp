// glm_poisson_elasticnet_without_intercept.cpp
//
// ElasticNet Poisson GLM via IRLS (outer) + Coordinate Descent (inner)
// - y: counts (>=0; double accepted)
// - log link: eta = X*beta + offset
// - offset: typically log-exposure
// - standardize mask (0/1, length p):
//     1 -> scale only (NO centering; because no intercept)
//     0 -> leave as-is
// - penalty_mask (0/1, length p):
//     1 -> penalize beta_j, 0 -> do not penalize beta_j
// - intercept is fixed at 0.0 (returned as 0.0)
//
// Penalty in standardized space:
//   L1 = alpha*lambd
//   L2 = (1-alpha)*lambd + ridge
//
// Fixes vs previous version:
//   1) clip-consistent eta/mu/z: mu = exp(clip(eta)), z uses eta_clip (log(mu))
//   2) optional step-halving on penalized objective for IRLS stability
//   3) correct n_outer tracking
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

GLMPoissonENetResult glm_poisson_elasticnet_without_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,   // 0/1 length p (scale-only)
    const Eigen::VectorXi& penalty_mask,  // 0/1 length p (beta only)
    int max_iter,
    double tol,
    double alpha,
    double lambd,
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
  if (penalty_mask.size() != p) {
    throw std::invalid_argument("penalty_mask length must match X.cols()");
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
  if (!(alpha >= 0.0 && alpha <= 1.0)) {
    throw std::invalid_argument("alpha must be in [0,1]");
  }
  if (!(lambd >= 0.0) || !std::isfinite(lambd)) {
    throw std::invalid_argument("lambd must be finite and >= 0");
  }
  if (!(tol > 0.0) || !std::isfinite(tol)) {
    throw std::invalid_argument("tol must be positive finite");
  }
  if (!(eps_mu > 0.0) || !std::isfinite(eps_mu)) {
    throw std::invalid_argument("eps_mu must be positive finite");
  }

  const double l1 = alpha * lambd;
  const double l2 = (1.0 - alpha) * lambd + ridge;
  const double tiny_sd = 1e-12;

  auto is_penalized = [&](int j) -> bool { return (penalty_mask[j] != 0); };

  // -------- Standardize X -> Xs (scale-only where standardize[j]==1) --------
  // No centering because there is no intercept term.
  Eigen::VectorXd x_scale(p);
  x_scale.setOnes();

  Eigen::MatrixXd Xs = X;

  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) {
      // Scale-only: choose a stable scale. Use SD around the mean (but do NOT center Xs).
      const double m = X.col(j).mean();
      const Eigen::ArrayXd xc = X.col(j).array() - m;
      const double var = (xc * xc).mean();
      const double sd = std::sqrt(var);

      if (!std::isfinite(sd) || sd < tiny_sd) {
        throw std::invalid_argument("Cannot scale (near-)constant column j=" + std::to_string(j));
      }
      x_scale[j] = sd;
      Xs.col(j).array() /= sd; // scale only
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

    double l1sum = 0.0;
    double l2sum = 0.0;
    for (int j = 0; j < p; ++j) {
      if (is_penalized(j)) {
        const double bj = beta_try[j];
        l1sum += std::abs(bj);
        l2sum += bj * bj;
      }
    }
    return llf - l1 * l1sum - 0.5 * l2 * l2sum;
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

    // clip-consistent eta_clip, mu
    for (int i = 0; i < n; ++i) {
      const double e = clip(eta[i], eta_lo, eta_hi);
      eta_clip[i] = e;
      double m = std::exp(e);
      if (!std::isfinite(m) || m < eps_mu) m = eps_mu;
      mu[i] = m;
    }

    // W = mu
    W = mu;

    // z = eta_clip + (y - mu)/mu   IMPORTANT: use eta_clip
    for (int i = 0; i < n; ++i) {
      z[i] = eta_clip[i] + (y[i] - mu[i]) / mu[i];
    }

    // Working response:
    // ytilde = z - offset  and model: ytilde ~= Xs*beta
    const Eigen::VectorXd ytilde = z - offset;

    // ---- inner: coordinate descent on penalized WLS (no intercept) ----
    // residual r = ytilde - Xs*beta
    Eigen::VectorXd r = ytilde - (Xs * beta);

    // Precompute aj = sum_i W_i x_ij^2 (+ l2 if penalized)
    Eigen::VectorXd aj(p);
    for (int j = 0; j < p; ++j) {
      double s = 0.0;
      for (int i = 0; i < n; ++i) {
        const double xij = Xs(i, j);
        s += W[i] * xij * xij;
      }
      if (is_penalized(j)) s += l2;
      if (!(s > 0.0) || !std::isfinite(s)) s = 1.0;
      aj[j] = s;
    }

    Eigen::VectorXd beta_new = beta;

    int cd_sweeps_this_outer = 0;
    double cd_last_max_change = std::numeric_limits<double>::infinity();

    for (int cd_it = 0; cd_it < cd_max_iter; ++cd_it) {
      ++cd_sweeps_this_outer;

      double max_change = 0.0;

      for (int j = 0; j < p; ++j) {
        const double bj_old = beta_new[j];

        // r += x_j * bj_old
        if (bj_old != 0.0) {
          for (int i = 0; i < n; ++i) r[i] += Xs(i, j) * bj_old;
        }

        // rho = x_j^T W r
        double rho = 0.0;
        for (int i = 0; i < n; ++i) {
          rho += W[i] * Xs(i, j) * r[i];
        }

        double bj_new;
        if (is_penalized(j)) {
          bj_new = soft_threshold(rho, l1) / aj[j];
        } else {
          bj_new = rho / aj[j];
        }

        // r -= x_j * bj_new
        if (bj_new != 0.0) {
          for (int i = 0; i < n; ++i) r[i] -= Xs(i, j) * bj_new;
        }

        beta_new[j] = bj_new;

        const double ch = std::abs(bj_new - bj_old);
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
    double max_abs = (beta_acc - beta).cwiseAbs().maxCoeff();

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