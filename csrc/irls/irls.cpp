#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

static inline double clip(double x, double lo, double hi) {
  return (x < lo) ? lo : ((x > hi) ? hi : x);
}

static inline double norm_pdf(double x) {
  static const double INV_SQRT2PI = 0.39894228040143267793994605993438186848;
  return INV_SQRT2PI * std::exp(-0.5 * x * x);
}

static inline double norm_cdf(double x) {
  // Phi(x) = 0.5 * erfc(-x/sqrt(2))
  static const double INV_SQRT2 = 0.70710678118654752440084436210484903928;
  return 0.5 * std::erfc(-x * INV_SQRT2);
}

struct LinkEval {
  // returns mu and dmu/deta for each eta
  static void eval(
      const std::string& link,
      const Eigen::VectorXd& eta,
      Eigen::VectorXd& mu,
      Eigen::VectorXd& dmu) {

    const int n = static_cast<int>(eta.size());
    mu.resize(n);
    dmu.resize(n);

    if (link == "logit") {
      for (int i = 0; i < n; ++i) {
        const double e = eta[i];
        const double m = 1.0 / (1.0 + std::exp(-e));
        mu[i] = m;
        dmu[i] = m * (1.0 - m);
      }
      return;
    }

    if (link == "probit") {
      for (int i = 0; i < n; ++i) {
        const double e = eta[i];
        mu[i] = norm_cdf(e);
        dmu[i] = norm_pdf(e);
      }
      return;
    }

    if (link == "cloglog") {
      for (int i = 0; i < n; ++i) {
        const double e = eta[i];
        const double exp_e = std::exp(e);
        // mu = 1 - exp(-exp(eta))
        const double m = 1.0 - std::exp(-exp_e);
        mu[i] = m;
        // dmu = exp(eta) * exp(-exp(eta)) = exp(eta - exp(eta))
        dmu[i] = exp_e * std::exp(-exp_e);
      }
      return;
    }

    throw std::invalid_argument("Unknown link: " + link);
  }
};

Eigen::VectorXd irls_binomial_offset_eigen(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& n_trials,
    const Eigen::VectorXd& offset,
    const Eigen::VectorXd& beta0,
    int max_iter,
    double tol,
    const std::string& link,
    double ridge = 1e-12,
    double eps_mu = 1e-15,
    double eps_dmu = 1e-15) {

  const int n = static_cast<int>(X.rows());
  const int p = static_cast<int>(X.cols());
  if (y.size() != n || n_trials.size() != n || offset.size() != n) {
    throw std::invalid_argument("y/n/offset length must match X.rows()");
  }
  if (beta0.size() != p) {
    throw std::invalid_argument("beta0 length must match X.cols()");
  }
  if (max_iter <= 0) {
    return beta0;
  }

  Eigen::VectorXd beta = beta0;
  Eigen::VectorXd eta(n), mu(n), dmu(n), W(n), z(n);

  for (int it = 0; it < max_iter; ++it) {
    eta.noalias() = X * beta + offset;

    LinkEval::eval(link, eta, mu, dmu);

    // clip
    for (int i = 0; i < n; ++i) {
      mu[i] = clip(mu[i], eps_mu, 1.0 - eps_mu);
      dmu[i] = std::max(dmu[i], eps_dmu);
    }

    // W = n * (dmu^2) / (mu(1-mu))
    for (int i = 0; i < n; ++i) {
      const double denom = mu[i] * (1.0 - mu[i]);
      W[i] = n_trials[i] * (dmu[i] * dmu[i]) / denom;
      if (!(W[i] > 0.0) || std::isnan(W[i]) || std::isinf(W[i])) {
        W[i] = eps_mu;
      }
      if (W[i] < eps_mu) W[i] = eps_mu;
    }

    // z = eta + (y - mu) / dmu
    z = eta;
    for (int i = 0; i < n; ++i) {
      z[i] += (y[i] - mu[i]) / dmu[i];
    }

    // Build normal equations:
    // A = X^T W X, b = X^T W (z - offset)
    // Efficiently: scale rows of X by sqrt(W)
    Eigen::VectorXd sqrtW = W.array().sqrt();
    Eigen::MatrixXd Xw = X;
    for (int i = 0; i < n; ++i) {
      Xw.row(i) *= sqrtW[i];
    }
    Eigen::VectorXd rhs = (sqrtW.array() * (z - offset).array()).matrix();

    Eigen::MatrixXd A = Xw.transpose() * Xw;
    A.diagonal().array() += ridge;
    Eigen::VectorXd b = Xw.transpose() * rhs;

    // Solve
    Eigen::VectorXd beta_new;
    // Prefer LDLT for symmetric (A is SPD-ish with ridge)
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if (ldlt.info() == Eigen::Success) {
      beta_new = ldlt.solve(b);
    } else {
      // fallback
      beta_new = A.colPivHouseholderQr().solve(b);
    }

    // check convergence
    double max_abs = 0.0;
    for (int j = 0; j < p; ++j) {
      const double d = std::abs(beta_new[j] - beta[j]);
      if (d > max_abs) max_abs = d;
    }
    beta = beta_new;
    if (max_abs < tol) break;
  }

  return beta;
}

// ---- pybind wrapper (numpy <-> Eigen) --------------------------------

py::array_t<double> irls_binomial_offset_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> n_trials,
    py::array_t<double, py::array::c_style | py::array::forcecast> offset,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta0,
    int max_iter,
    double tol,
    const std::string& link,
    double ridge = 1e-12) {

  auto Xb = X.request();
  if (Xb.ndim != 2) throw std::invalid_argument("X must be 2D");
  const int n = static_cast<int>(Xb.shape[0]);
  const int p = static_cast<int>(Xb.shape[1]);

  auto yb = y.request();
  auto nb = n_trials.request();
  auto ob = offset.request();
  auto bb = beta0.request();
  if (yb.ndim != 1 || nb.ndim != 1 || ob.ndim != 1 || bb.ndim != 1)
    throw std::invalid_argument("y/n/offset/beta0 must be 1D");
  if (static_cast<int>(yb.shape[0]) != n ||
      static_cast<int>(nb.shape[0]) != n ||
      static_cast<int>(ob.shape[0]) != n)
    throw std::invalid_argument("y/n/offset length must match X.rows()");
  if (static_cast<int>(bb.shape[0]) != p)
    throw std::invalid_argument("beta0 length must match X.cols()");

  // Map to Eigen
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_e(
      static_cast<double*>(Xb.ptr), n, p);
  Eigen::Map<const Eigen::VectorXd> y_e(static_cast<double*>(yb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> n_e(static_cast<double*>(nb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> off_e(static_cast<double*>(ob.ptr), n);
  Eigen::Map<const Eigen::VectorXd> b0_e(static_cast<double*>(bb.ptr), p);

  Eigen::VectorXd out = irls_binomial_offset_eigen(
      X_e, y_e, n_e, off_e, b0_e, max_iter, tol, link, ridge);

  py::array_t<double> ret({p});
  auto rb = ret.request();
  std::memcpy(rb.ptr, out.data(), sizeof(double) * static_cast<size_t>(p));
  return ret;
}

void bind_dglm_irls(py::module_& m) {
  m.def(
      "irls_binomial_offset",
      &irls_binomial_offset_py,
      py::arg("X"),
      py::arg("y"),
      py::arg("n_trials"),
      py::arg("offset"),
      py::arg("beta0"),
      py::arg("max_iter"),
      py::arg("tol"),
      py::arg("link"),
      py::arg("ridge") = 1e-12,
      R"pbdoc(
IRLS for aggregated binomial GLM with offset.
X: (n,p), y: (n,) success proportion, n_trials: (n,), offset: (n,), beta0: (p,)
link: "logit" | "probit" | "cloglog"
returns beta (p,)
)pbdoc");
}

PYBIND11_MODULE(_irls, m) {
  m.doc() = "pysrat IRLS helpers";
  bind_dglm_irls(m);
}