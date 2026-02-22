#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "link_eval.h"

// local clip (keep in this TU)
static inline double clip(double x, double lo, double hi) {
  return std::min(std::max(x, lo), hi);
}

struct GLMBinomialResult {
  double intercept;        // always 0 in the case of 'without_intercept'
  Eigen::VectorXd beta;
  bool converged;
  int n_iter;
};

GLMBinomialResult glm_binomial_with_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    double intercept0,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    int max_iter,
    double tol,
    const std::string& link,
    double ridge,
    double eps_mu,
    double eps_dmu) {

  const int n = X.rows();
  const int p = X.cols();
  const double tiny_sd = 1e-12;

  Eigen::VectorXd x_mean = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd x_scale = Eigen::VectorXd::Ones(p);
  Eigen::MatrixXd Xs = X;

  // --- center + scale ---
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) {
      const double m = X.col(j).mean();
      const Eigen::ArrayXd xc = (X.col(j).array() - m);
      const double sd = std::sqrt((xc * xc).mean());
      if (sd < tiny_sd) throw std::invalid_argument("constant column");
      x_mean[j] = m;
      x_scale[j] = sd;
      Xs.col(j).array() = (Xs.col(j).array() - m) / sd;
    }
  }

  double b0 = intercept0;
  Eigen::VectorXd beta = beta0;

  for (int j = 0; j < p; ++j)
    if (standardize[j] != 0)
      beta[j] *= x_scale[j];

  double shift = 0.0;
  for (int j = 0; j < p; ++j)
    if (standardize[j] != 0)
      shift += x_mean[j] * beta0[j];
  b0 += shift;

  Eigen::VectorXd eta(n), mu(n), dmu(n), W(n), z(n);

  bool converged = false;
  int it = 0;

  for (; it < max_iter; ++it) {

    eta = Xs * beta + offset;
    eta.array() += b0;

    LinkEval::eval(link, eta, mu, dmu);

    for (int i = 0; i < n; ++i) {
      mu[i] = clip(mu[i], eps_mu, 1.0 - eps_mu);
      dmu[i] = std::max(dmu[i], eps_dmu);
    }

    for (int i = 0; i < n; ++i) {
      const double ni = std::max(n_trials[i], 0.0);
      const double denom = mu[i] * (1.0 - mu[i]);
      W[i] = std::max(ni * dmu[i] * dmu[i] / denom, eps_mu);
    }

    z = eta;
    for (int i = 0; i < n; ++i) {
      if (n_trials[i] > 0.0)
        z[i] += (y[i] - n_trials[i] * mu[i]) /
                (n_trials[i] * dmu[i]);
    }

    Eigen::VectorXd sqrtW = W.array().sqrt();
    Eigen::MatrixXd Xw = Xs;
    for (int i = 0; i < n; ++i)
      Xw.row(i) *= sqrtW[i];

    Eigen::VectorXd rhs =
      (sqrtW.array() * ((z - offset).array() - b0)).matrix();

    Eigen::MatrixXd A = Xw.transpose() * Xw;
    A.diagonal().array() += ridge;

    Eigen::VectorXd beta_new =
        A.ldlt().solve(Xw.transpose() * rhs);

    double sw = W.sum();
    double b0_new = b0;
    if (sw > 0.0) {
      Eigen::VectorXd r = z - offset - Xs * beta_new;
      b0_new = (W.array() * r.array()).sum() / sw;
    }

    double diff =
        std::max((beta_new - beta).cwiseAbs().maxCoeff(),
                 std::abs(b0_new - b0));

    beta = beta_new;
    b0 = b0_new;

    if (diff < tol) {
      converged = true;
      ++it;
      break;
    }
  }

  // back transform
  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j)
    if (standardize[j] != 0)
      beta_out[j] /= x_scale[j];

  double b0_out = b0;
  double corr = 0.0;
  for (int j = 0; j < p; ++j)
    if (standardize[j] != 0)
      corr += x_mean[j] * beta_out[j];
  b0_out -= corr;

  return {b0_out, beta_out, converged, it};
}

GLMBinomialResult glm_binomial_without_intercept(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    const Eigen::VectorXi& standardize,
    int max_iter,
    double tol,
    const std::string& link,
    double ridge,
    double eps_mu,
    double eps_dmu) {

  const int n = X.rows();
  const int p = X.cols();
  const double tiny_sd = 1e-12;

  Eigen::VectorXd x_scale = Eigen::VectorXd::Ones(p);
  Eigen::MatrixXd Xs = X;

  // scale only (no centering)
  for (int j = 0; j < p; ++j) {
    if (standardize[j] != 0) {
      const double sd =
          std::sqrt((X.col(j).array().square()).mean());
      if (sd < tiny_sd)
        throw std::invalid_argument("constant column");
      x_scale[j] = sd;
      Xs.col(j) /= sd;
    }
  }

  Eigen::VectorXd beta = beta0;
  for (int j = 0; j < p; ++j)
    if (standardize[j] != 0)
      beta[j] *= x_scale[j];

  Eigen::VectorXd eta(n), mu(n), dmu(n), W(n), z(n);

  bool converged = false;
  int it = 0;

  for (; it < max_iter; ++it) {

    eta = Xs * beta + offset;

    LinkEval::eval(link, eta, mu, dmu);

    for (int i = 0; i < n; ++i) {
      mu[i] = clip(mu[i], eps_mu, 1.0 - eps_mu);
      dmu[i] = std::max(dmu[i], eps_dmu);
    }

    for (int i = 0; i < n; ++i) {
      const double ni = std::max(n_trials[i], 0.0);
      const double denom = mu[i] * (1.0 - mu[i]);
      W[i] = std::max(ni * dmu[i] * dmu[i] / denom, eps_mu);
    }

    z = eta;
    for (int i = 0; i < n; ++i) {
      if (n_trials[i] > 0.0)
        z[i] += (y[i] - n_trials[i] * mu[i]) /
                (n_trials[i] * dmu[i]);
    }

    Eigen::VectorXd sqrtW = W.array().sqrt();
    Eigen::MatrixXd Xw = Xs;
    for (int i = 0; i < n; ++i)
      Xw.row(i) *= sqrtW[i];

    Eigen::VectorXd rhs =
        (sqrtW.array() * (z - offset).array()).matrix();

    Eigen::MatrixXd A = Xw.transpose() * Xw;
    A.diagonal().array() += ridge;

    Eigen::VectorXd beta_new =
        A.ldlt().solve(Xw.transpose() * rhs);

    double diff =
        (beta_new - beta).cwiseAbs().maxCoeff();

    beta = beta_new;

    if (diff < tol) {
      converged = true;
      ++it;
      break;
    }
  }

  Eigen::VectorXd beta_out = beta;
  for (int j = 0; j < p; ++j)
    if (standardize[j] != 0)
      beta_out[j] /= x_scale[j];

  return {0.0, beta_out, converged, it};
}

// // glm_binomial (success counts y, aggregated binomial with n_trials)
// Eigen::VectorXd glm_binomial(
//     const Eigen::MatrixXd& X,
//     const Eigen::VectorXd& y,
//     const Eigen::VectorXd& n_trials,
//     const Eigen::VectorXd& offset,
//     const Eigen::VectorXd& beta0,
//     const Eigen::VectorXi& standardize,
//     int max_iter,
//     double tol,
//     const std::string& link,
//     double ridge,
//     double eps_mu,
//     double eps_dmu) {

//   const int n = static_cast<int>(X.rows());
//   const int p = static_cast<int>(X.cols());
//   if (y.size() != n || n_trials.size() != n || offset.size() != n) {
//     throw std::invalid_argument("y/n_trials/offset length must match X.rows()");
//   }
//   if (beta0.size() != p) {
//     throw std::invalid_argument("beta0 length must match X.cols()");
//   }
//   if (standardize.size() != p) {
//     throw std::invalid_argument("standardize length must match X.cols()");
//   }
//   if (max_iter <= 0) return beta0;

//   const double tiny_sd = 1e-12;

//   Eigen::VectorXd x_mean(p);
//   Eigen::VectorXd x_scale(p);
//   x_mean.setZero();
//   x_scale.setOnes();

//   Eigen::MatrixXd Xs = X;

//   for (int j = 0; j < p; ++j) {
//     if (standardize[j] != 0) {
//       const double m = X.col(j).mean();
//       const Eigen::ArrayXd xc = (X.col(j).array() - m);
//       const double var = (xc * xc).mean();
//       const double sd = std::sqrt(var);

//       if (!std::isfinite(sd) || sd < tiny_sd) {
//         throw std::invalid_argument("Cannot standardize (near-)constant column j=" + std::to_string(j));
//       }
//       x_mean[j] = m;
//       x_scale[j] = sd;
//       Xs.col(j).array() = (Xs.col(j).array() - m) / sd;
//     }
//   }

//   int intercept_col = -1;
//   for (int j = 0; j < p; ++j) {
//     if (standardize[j] == 0) {
//       const double mn = X.col(j).minCoeff();
//       const double mx = X.col(j).maxCoeff();
//       if (std::abs(mx - mn) < tiny_sd) { intercept_col = j; break; }
//     }
//   }
//   if (intercept_col < 0) {
//     for (int j = 0; j < p; ++j) {
//       if (standardize[j] == 0) { intercept_col = j; break; }
//     }
//   }

//   Eigen::VectorXd beta_s = beta0;
//   for (int j = 0; j < p; ++j) {
//     if (standardize[j] != 0) beta_s[j] = beta0[j] * x_scale[j];
//   }
//   if (intercept_col >= 0) {
//     double shift = 0.0;
//     for (int j = 0; j < p; ++j) {
//       if (standardize[j] != 0) shift += x_mean[j] * beta0[j];
//     }
//     beta_s[intercept_col] = beta0[intercept_col] + shift;
//   }

//   Eigen::VectorXd beta = beta_s;
//   Eigen::VectorXd eta(n), mu(n), dmu(n), W(n), z(n);

//   for (int it = 0; it < max_iter; ++it) {
//     eta.noalias() = Xs * beta + offset;

//     LinkEval::eval(link, eta, mu, dmu);

//     for (int i = 0; i < n; ++i) {
//       mu[i]  = clip(mu[i], eps_mu, 1.0 - eps_mu);
//       dmu[i] = std::max(dmu[i], eps_dmu);
//     }

//     for (int i = 0; i < n; ++i) {
//       double ni = n_trials[i];
//       if (!(ni > 0.0) || !std::isfinite(ni)) ni = 0.0;

//       const double denom = mu[i] * (1.0 - mu[i]);
//       W[i] = (ni > 0.0) ? (ni * (dmu[i] * dmu[i]) / denom) : eps_mu;

//       if (!(W[i] > 0.0) || !std::isfinite(W[i])) W[i] = eps_mu;
//       if (W[i] < eps_mu) W[i] = eps_mu;
//     }

//     z = eta;
//     for (int i = 0; i < n; ++i) {
//       double ni = n_trials[i];
//       if (!(ni > 0.0) || !std::isfinite(ni)) ni = 0.0;
//       if (ni > 0.0) {
//         z[i] += (y[i] - ni * mu[i]) / (ni * dmu[i]);
//       }
//     }

//     const Eigen::VectorXd sqrtW = W.array().sqrt();
//     Eigen::MatrixXd Xw = Xs;
//     for (int i = 0; i < n; ++i) Xw.row(i) *= sqrtW[i];
//     const Eigen::VectorXd rhs = (sqrtW.array() * (z - offset).array()).matrix();

//     Eigen::MatrixXd A = Xw.transpose() * Xw;
//     A.diagonal().array() += ridge;
//     const Eigen::VectorXd b = Xw.transpose() * rhs;

//     Eigen::VectorXd beta_new;
//     Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
//     if (ldlt.info() == Eigen::Success) beta_new = ldlt.solve(b);
//     else beta_new = A.colPivHouseholderQr().solve(b);

//     double max_abs = 0.0;
//     for (int j = 0; j < p; ++j) {
//       const double d = std::abs(beta_new[j] - beta[j]);
//       if (d > max_abs) max_abs = d;
//     }
//     beta = beta_new;
//     if (max_abs < tol) break;
//   }

//   Eigen::VectorXd beta_out = beta;
//   for (int j = 0; j < p; ++j) {
//     if (standardize[j] != 0) beta_out[j] = beta[j] / x_scale[j];
//   }
//   if (intercept_col >= 0) {
//     double shift = 0.0;
//     for (int j = 0; j < p; ++j) {
//       if (standardize[j] != 0) shift += x_mean[j] * beta_out[j];
//     }
//     beta_out[intercept_col] = beta[intercept_col] - shift;
//   }

//   return beta_out;
// }