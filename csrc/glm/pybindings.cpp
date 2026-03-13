#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cfenv>
#pragma STDC FENV_ACCESS ON

#include "glm_binomial.h"
#include "glm_poisson.h"
#include "glm_binomial_elasticnet.h"
#include "glm_poisson_elasticnet.h"

namespace py = pybind11;

static Eigen::VectorXi default_standardize_mask(int p) {
  Eigen::VectorXi m(p);
  m.setOnes();
  return m;
}

static Eigen::VectorXi parse_standardize_mask(py::object standardize_obj, int p) {
  Eigen::VectorXi stdmask;

  if (standardize_obj.is_none()) {
    stdmask = default_standardize_mask(p);
    return stdmask;
  }

  py::array_t<int, py::array::c_style | py::array::forcecast> standardize =
      standardize_obj.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
  auto sb = standardize.request();
  if (sb.ndim != 1) throw std::invalid_argument("standardize must be 1D");
  if (static_cast<int>(sb.shape[0]) != p)
    throw std::invalid_argument("standardize length must match X.cols()");

  Eigen::Map<const Eigen::VectorXi> std_e(static_cast<const int*>(sb.ptr), p);
  stdmask = std_e;
  return stdmask;
}

static Eigen::VectorXi default_penalty_mask(int p) {
  Eigen::VectorXi m(p);
  if (p <= 0) return m;
  m.setOnes();
  return m;
}

static Eigen::VectorXi parse_penalty_mask(py::object penalty_obj, int p) {
  Eigen::VectorXi penmask;

  if (penalty_obj.is_none()) {
    penmask = default_penalty_mask(p);
    return penmask;
  }

  py::array_t<int, py::array::c_style | py::array::forcecast> penalty =
      penalty_obj.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
  auto pb = penalty.request();

  if (pb.ndim != 1) throw std::invalid_argument("penalty must be 1D");
  if (static_cast<int>(pb.shape[0]) != p)
    throw std::invalid_argument("penalty length must match X.cols()");

  Eigen::Map<const Eigen::VectorXi> p_e(static_cast<const int*>(pb.ptr), p);
  penmask = p_e;
  return penmask;
}

GLMBinomialResult glm_binomial_with_intercept_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> n_trials,
    py::array_t<double, py::array::c_style | py::array::forcecast> offset,
    double intercept0,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta0,
    py::object standardize_obj,
    int max_iter,
    double tol,
    const std::string& link,
    double ridge,
    double eps_mu,
    double eps_dmu) {

  std::feclearexcept(FE_ALL_EXCEPT);

  auto Xb = X.request();
  if (Xb.ndim != 2) throw std::invalid_argument("X must be 2D");
  const int n = static_cast<int>(Xb.shape[0]);
  const int p = static_cast<int>(Xb.shape[1]);

  auto yb = y.request();
  auto nb = n_trials.request();
  auto ob = offset.request();
  auto bb = beta0.request();

  if (yb.ndim != 1 || nb.ndim != 1 || ob.ndim != 1 || bb.ndim != 1)
    throw std::invalid_argument("y/n_trials/offset/beta0 must be 1D");
  if (static_cast<int>(yb.shape[0]) != n ||
      static_cast<int>(nb.shape[0]) != n ||
      static_cast<int>(ob.shape[0]) != n)
    throw std::invalid_argument("y/n_trials/offset length must match X.rows()");
  if (static_cast<int>(bb.shape[0]) != p)
    throw std::invalid_argument("beta0 length must match X.cols()");

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_e(
      static_cast<const double*>(Xb.ptr), n, p);
  Eigen::Map<const Eigen::VectorXd> y_e(static_cast<const double*>(yb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> n_e(static_cast<const double*>(nb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> off_e(static_cast<const double*>(ob.ptr), n);
  Eigen::Map<const Eigen::VectorXd> b0_e(static_cast<const double*>(bb.ptr), p);

  const Eigen::VectorXi stdmask = parse_standardize_mask(standardize_obj, p);

  return glm_binomial_with_intercept(
      X_e, y_e, n_e, off_e,
      intercept0, b0_e, stdmask,
      max_iter, tol, link, ridge, eps_mu, eps_dmu);
}

GLMBinomialResult glm_binomial_without_intercept_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> n_trials,
    py::array_t<double, py::array::c_style | py::array::forcecast> offset,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta0,
    py::object standardize_obj,
    int max_iter,
    double tol,
    const std::string& link,
    double ridge,
    double eps_mu,
    double eps_dmu) {

  std::feclearexcept(FE_ALL_EXCEPT);

  auto Xb = X.request();
  if (Xb.ndim != 2) throw std::invalid_argument("X must be 2D");
  const int n = static_cast<int>(Xb.shape[0]);
  const int p = static_cast<int>(Xb.shape[1]);

  auto yb = y.request();
  auto nb = n_trials.request();
  auto ob = offset.request();
  auto bb = beta0.request();

  if (yb.ndim != 1 || nb.ndim != 1 || ob.ndim != 1 || bb.ndim != 1)
    throw std::invalid_argument("y/n_trials/offset/beta0 must be 1D");
  if (static_cast<int>(yb.shape[0]) != n ||
      static_cast<int>(nb.shape[0]) != n ||
      static_cast<int>(ob.shape[0]) != n)
    throw std::invalid_argument("y/n_trials/offset length must match X.rows()");
  if (static_cast<int>(bb.shape[0]) != p)
    throw std::invalid_argument("beta0 length must match X.cols()");

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_e(
      static_cast<const double*>(Xb.ptr), n, p);
  Eigen::Map<const Eigen::VectorXd> y_e(static_cast<const double*>(yb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> n_e(static_cast<const double*>(nb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> off_e(static_cast<const double*>(ob.ptr), n);
  Eigen::Map<const Eigen::VectorXd> b0_e(static_cast<const double*>(bb.ptr), p);

  const Eigen::VectorXi stdmask = parse_standardize_mask(standardize_obj, p);

  return glm_binomial_without_intercept(
      X_e, y_e, n_e, off_e,
      b0_e, stdmask,
      max_iter, tol, link, ridge, eps_mu, eps_dmu);
}

// ---- with intercept ----
GLMPoissonResult glm_poisson_with_intercept_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> offset,
    double intercept0,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta0,
    py::object standardize_obj,
    int max_iter,
    double tol,
    double ridge,
    double eps_mu) {

  std::feclearexcept(FE_ALL_EXCEPT);

  auto Xb = X.request();
  if (Xb.ndim != 2) throw std::invalid_argument("X must be 2D");
  const int n = static_cast<int>(Xb.shape[0]);
  const int p = static_cast<int>(Xb.shape[1]);

  auto yb = y.request();
  auto ob = offset.request();
  auto bb = beta0.request();

  if (yb.ndim != 1 || ob.ndim != 1 || bb.ndim != 1)
    throw std::invalid_argument("y/offset/beta0 must be 1D");
  if (static_cast<int>(yb.shape[0]) != n ||
      static_cast<int>(ob.shape[0]) != n)
    throw std::invalid_argument("y/offset length must match X.rows()");
  if (static_cast<int>(bb.shape[0]) != p)
    throw std::invalid_argument("beta0 length must match X.cols()");

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_e(
      static_cast<const double*>(Xb.ptr), n, p);
  Eigen::Map<const Eigen::VectorXd> y_e(static_cast<const double*>(yb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> off_e(static_cast<const double*>(ob.ptr), n);
  Eigen::Map<const Eigen::VectorXd> b0_e(static_cast<const double*>(bb.ptr), p);

  const Eigen::VectorXi stdmask = parse_standardize_mask(standardize_obj, p);

  return glm_poisson_with_intercept(
      X_e, y_e, off_e,
      intercept0, b0_e, stdmask,
      max_iter, tol, ridge, eps_mu);
}

// ---- without intercept ----
GLMPoissonResult glm_poisson_without_intercept_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> offset,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta0,
    py::object standardize_obj,
    int max_iter,
    double tol,
    double ridge,
    double eps_mu) {

  std::feclearexcept(FE_ALL_EXCEPT);

  auto Xb = X.request();
  if (Xb.ndim != 2) throw std::invalid_argument("X must be 2D");
  const int n = static_cast<int>(Xb.shape[0]);
  const int p = static_cast<int>(Xb.shape[1]);

  auto yb = y.request();
  auto ob = offset.request();
  auto bb = beta0.request();

  if (yb.ndim != 1 || ob.ndim != 1 || bb.ndim != 1)
    throw std::invalid_argument("y/offset/beta0 must be 1D");
  if (static_cast<int>(yb.shape[0]) != n ||
      static_cast<int>(ob.shape[0]) != n)
    throw std::invalid_argument("y/offset length must match X.rows()");
  if (static_cast<int>(bb.shape[0]) != p)
    throw std::invalid_argument("beta0 length must match X.cols()");

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_e(
      static_cast<const double*>(Xb.ptr), n, p);
  Eigen::Map<const Eigen::VectorXd> y_e(static_cast<const double*>(yb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> off_e(static_cast<const double*>(ob.ptr), n);
  Eigen::Map<const Eigen::VectorXd> b0_e(static_cast<const double*>(bb.ptr), p);

  const Eigen::VectorXi stdmask = parse_standardize_mask(standardize_obj, p);

  return glm_poisson_without_intercept(
      X_e, y_e, off_e,
      b0_e, stdmask,
      max_iter, tol, ridge, eps_mu);
}

GLMBinomialENetResult glm_binomial_elasticnet_with_intercept_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> n_trials,
    py::array_t<double, py::array::c_style | py::array::forcecast> offset,
    double intercept0,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta0,
    py::object standardize_obj,
    py::object penalty_obj,
    int max_iter,
    double tol,
    const std::string& link,
    double alpha,
    double lambd,
    double ridge,
    double eps_mu,
    double eps_dmu) {

  std::feclearexcept(FE_ALL_EXCEPT);

  auto Xb = X.request();
  if (Xb.ndim != 2) throw std::invalid_argument("X must be 2D");
  const int n = static_cast<int>(Xb.shape[0]);
  const int p = static_cast<int>(Xb.shape[1]);

  auto yb = y.request();
  auto nb = n_trials.request();
  auto ob = offset.request();
  auto bb = beta0.request();

  if (yb.ndim != 1 || nb.ndim != 1 || ob.ndim != 1 || bb.ndim != 1)
    throw std::invalid_argument("y/n_trials/offset/beta0 must be 1D");
  if (static_cast<int>(yb.shape[0]) != n ||
      static_cast<int>(nb.shape[0]) != n ||
      static_cast<int>(ob.shape[0]) != n)
    throw std::invalid_argument("y/n_trials/offset length must match X.rows()");
  if (static_cast<int>(bb.shape[0]) != p)
    throw std::invalid_argument("beta0 length must match X.cols()");

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_e(
      static_cast<const double*>(Xb.ptr), n, p);
  Eigen::Map<const Eigen::VectorXd> y_e(static_cast<const double*>(yb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> n_e(static_cast<const double*>(nb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> off_e(static_cast<const double*>(ob.ptr), n);
  Eigen::Map<const Eigen::VectorXd> b0_e(static_cast<const double*>(bb.ptr), p);

  const Eigen::VectorXi stdmask = parse_standardize_mask(standardize_obj, p);
  const Eigen::VectorXi penmask = parse_penalty_mask(penalty_obj, p);

  if (alpha == 0.0) {
    return glm_binomial_with_intercept_from_elasticnet_alpha0(
        X_e, y_e, n_e, off_e,
        intercept0, b0_e,
        stdmask, penmask,
        max_iter, tol, link, lambd, ridge, eps_mu, eps_dmu);
  }

  return glm_binomial_elasticnet_with_intercept(
      X_e, y_e, n_e, off_e,
      intercept0, b0_e, stdmask, penmask,
      max_iter, tol, link,
      alpha, lambd, ridge, eps_mu, eps_dmu);
}

GLMBinomialENetResult glm_binomial_elasticnet_without_intercept_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> n_trials,
    py::array_t<double, py::array::c_style | py::array::forcecast> offset,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta0,
    py::object standardize_obj,
    py::object penalty_obj,
    int max_iter,
    double tol,
    const std::string& link,
    double alpha,
    double lambd,
    double ridge,
    double eps_mu,
    double eps_dmu) {

  std::feclearexcept(FE_ALL_EXCEPT);

  auto Xb = X.request();
  if (Xb.ndim != 2) throw std::invalid_argument("X must be 2D");
  const int n = static_cast<int>(Xb.shape[0]);
  const int p = static_cast<int>(Xb.shape[1]);

  auto yb = y.request();
  auto nb = n_trials.request();
  auto ob = offset.request();
  auto bb = beta0.request();

  if (yb.ndim != 1 || nb.ndim != 1 || ob.ndim != 1 || bb.ndim != 1)
    throw std::invalid_argument("y/n_trials/offset/beta0 must be 1D");
  if (static_cast<int>(yb.shape[0]) != n ||
      static_cast<int>(nb.shape[0]) != n ||
      static_cast<int>(ob.shape[0]) != n)
    throw std::invalid_argument("y/n_trials/offset length must match X.rows()");
  if (static_cast<int>(bb.shape[0]) != p)
    throw std::invalid_argument("beta0 length must match X.cols()");

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_e(
      static_cast<const double*>(Xb.ptr), n, p);
  Eigen::Map<const Eigen::VectorXd> y_e(static_cast<const double*>(yb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> n_e(static_cast<const double*>(nb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> off_e(static_cast<const double*>(ob.ptr), n);
  Eigen::Map<const Eigen::VectorXd> b0_e(static_cast<const double*>(bb.ptr), p);

  const Eigen::VectorXi stdmask = parse_standardize_mask(standardize_obj, p);
  const Eigen::VectorXi penmask = parse_penalty_mask(penalty_obj, p);

  if (alpha == 0.0) {
    return glm_binomial_without_intercept_from_elasticnet_alpha0(
        X_e, y_e, n_e, off_e,
        b0_e,
        stdmask, penmask,
        max_iter, tol, link, lambd, ridge, eps_mu, eps_dmu);
  }

  return glm_binomial_elasticnet_without_intercept(
      X_e, y_e, n_e, off_e,
      b0_e,
      stdmask, penmask,
      max_iter, tol, link,
      alpha, lambd, ridge, eps_mu, eps_dmu);
}

// ------------------------- with intercept -------------------------
GLMPoissonENetResult glm_poisson_elasticnet_with_intercept_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> offset,
    double intercept0,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta0,
    py::object standardize_obj,
    py::object penalty_obj,
    int max_iter,
    double tol,
    double alpha,
    double lambd,
    double ridge,
    double eps_mu) {

  std::feclearexcept(FE_ALL_EXCEPT);

  auto Xb = X.request();
  if (Xb.ndim != 2) throw std::invalid_argument("X must be 2D");
  const int n = static_cast<int>(Xb.shape[0]);
  const int p = static_cast<int>(Xb.shape[1]);

  auto yb = y.request();
  auto ob = offset.request();
  auto bb = beta0.request();

  if (yb.ndim != 1 || ob.ndim != 1 || bb.ndim != 1)
    throw std::invalid_argument("y/offset/beta0 must be 1D");
  if (static_cast<int>(yb.shape[0]) != n ||
      static_cast<int>(ob.shape[0]) != n)
    throw std::invalid_argument("y/offset length must match X.rows()");
  if (static_cast<int>(bb.shape[0]) != p)
    throw std::invalid_argument("beta0 length must match X.cols()");

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_e(
      static_cast<const double*>(Xb.ptr), n, p);
  Eigen::Map<const Eigen::VectorXd> y_e(static_cast<const double*>(yb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> off_e(static_cast<const double*>(ob.ptr), n);
  Eigen::Map<const Eigen::VectorXd> b0_e(static_cast<const double*>(bb.ptr), p);

  const Eigen::VectorXi stdmask = parse_standardize_mask(standardize_obj, p);
  const Eigen::VectorXi penmask = parse_penalty_mask(penalty_obj, p);

  if (alpha == 0.0) {
    return glm_poisson_with_intercept_from_elasticnet_alpha0(
        X_e, y_e, off_e,
        intercept0, b0_e,
        stdmask, penmask,
        max_iter, tol, lambd, ridge, eps_mu);
  }

  return glm_poisson_elasticnet_with_intercept(
      X_e, y_e, off_e,
      intercept0, b0_e,
      stdmask, penmask,
      max_iter, tol, alpha, lambd, ridge, eps_mu);
}

// ------------------------- without intercept -------------------------
GLMPoissonENetResult glm_poisson_elasticnet_without_intercept_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> offset,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta0,
    py::object standardize_obj,
    py::object penalty_obj,
    int max_iter,
    double tol,
    double alpha,
    double lambd,
    double ridge,
    double eps_mu) {

  std::feclearexcept(FE_ALL_EXCEPT);

  auto Xb = X.request();
  if (Xb.ndim != 2) throw std::invalid_argument("X must be 2D");
  const int n = static_cast<int>(Xb.shape[0]);
  const int p = static_cast<int>(Xb.shape[1]);

  auto yb = y.request();
  auto ob = offset.request();
  auto bb = beta0.request();

  if (yb.ndim != 1 || ob.ndim != 1 || bb.ndim != 1)
    throw std::invalid_argument("y/offset/beta0 must be 1D");
  if (static_cast<int>(yb.shape[0]) != n ||
      static_cast<int>(ob.shape[0]) != n)
    throw std::invalid_argument("y/offset length must match X.rows()");
  if (static_cast<int>(bb.shape[0]) != p)
    throw std::invalid_argument("beta0 length must match X.cols()");

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_e(
      static_cast<const double*>(Xb.ptr), n, p);
  Eigen::Map<const Eigen::VectorXd> y_e(static_cast<const double*>(yb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> off_e(static_cast<const double*>(ob.ptr), n);
  Eigen::Map<const Eigen::VectorXd> b0_e(static_cast<const double*>(bb.ptr), p);

  const Eigen::VectorXi stdmask = parse_standardize_mask(standardize_obj, p);
  const Eigen::VectorXi penmask = parse_penalty_mask(penalty_obj, p);

  if (alpha == 0.0) {
    return glm_poisson_without_intercept_from_elasticnet_alpha0(
        X_e, y_e, off_e,
        b0_e,
        stdmask, penmask,
        max_iter, tol, lambd, ridge, eps_mu);
  }

  return glm_poisson_elasticnet_without_intercept(
      X_e, y_e, off_e,
      b0_e,
      stdmask, penmask,
      max_iter, tol, alpha, lambd, ridge, eps_mu);
}

void bind_glm(py::module_& m) {

  py::class_<GLMBinomialResult>(m, "GLMBinomialResult")
      .def_readonly("intercept", &GLMBinomialResult::intercept)
      .def_readonly("beta", &GLMBinomialResult::beta)
      .def_readonly("converged", &GLMBinomialResult::converged)
      .def_readonly("n_iter", &GLMBinomialResult::n_iter);

  m.def("glm_binomial_with_intercept", &glm_binomial_with_intercept_py,
        py::arg("X"), py::arg("y"), py::arg("n_trials"), py::arg("offset"),
        py::arg("intercept0"), py::arg("beta0"),
        py::arg("standardize") = py::none(),
        py::arg("max_iter") = 25, py::arg("tol") = 1e-8,
        py::arg("link") = "logit",
        py::arg("ridge") = 1e-12, py::arg("eps_mu") = 1e-15, py::arg("eps_dmu") = 1e-15);

  m.def("glm_binomial_without_intercept", &glm_binomial_without_intercept_py,
        py::arg("X"), py::arg("y"), py::arg("n_trials"), py::arg("offset"),
        py::arg("beta0"),
        py::arg("standardize") = py::none(),
        py::arg("max_iter") = 25, py::arg("tol") = 1e-8,
        py::arg("link") = "logit",
        py::arg("ridge") = 1e-12, py::arg("eps_mu") = 1e-15, py::arg("eps_dmu") = 1e-15);

  py::class_<GLMPoissonResult>(m, "GLMPoissonResult")
      .def_readonly("intercept", &GLMPoissonResult::intercept)
      .def_readonly("beta", &GLMPoissonResult::beta)
      .def_readonly("converged", &GLMPoissonResult::converged)
      .def_readonly("n_iter", &GLMPoissonResult::n_iter);

  m.def("glm_poisson_with_intercept", &glm_poisson_with_intercept_py,
        py::arg("X"), py::arg("y"), py::arg("offset"),
        py::arg("intercept0"), py::arg("beta0"),
        py::arg("standardize") = py::none(),
        py::arg("max_iter") = 25, py::arg("tol") = 1e-8,
        py::arg("ridge") = 1e-12, py::arg("eps_mu") = 1e-15);

  m.def("glm_poisson_without_intercept", &glm_poisson_without_intercept_py,
        py::arg("X"), py::arg("y"), py::arg("offset"),
        py::arg("beta0"),
        py::arg("standardize") = py::none(),
        py::arg("max_iter") = 25, py::arg("tol") = 1e-8,
        py::arg("ridge") = 1e-12, py::arg("eps_mu") = 1e-15);

  py::class_<GLMBinomialENetResult>(m, "GLMBinomialENetResult")
      .def_readonly("intercept", &GLMBinomialENetResult::intercept)
      .def_readonly("beta", &GLMBinomialENetResult::beta)
      .def_readonly("converged", &GLMBinomialENetResult::converged)
      .def_readonly("n_outer", &GLMBinomialENetResult::n_outer)
      .def_readonly("n_inner", &GLMBinomialENetResult::n_inner)
      .def_readonly("max_delta", &GLMBinomialENetResult::max_delta)
      .def_readonly("max_delta_inner", &GLMBinomialENetResult::max_delta_inner);

  m.def("glm_binomial_elasticnet_with_intercept", &glm_binomial_elasticnet_with_intercept_py,
        py::arg("X"), py::arg("y"), py::arg("n_trials"), py::arg("offset"),
        py::arg("intercept0"), py::arg("beta0"),
        py::arg("standardize") = py::none(),
        py::arg("penalty") = py::none(),
        py::arg("max_iter") = 25, py::arg("tol") = 1e-8,
        py::arg("link") = "logit",
        py::arg("alpha") = 0.5, py::arg("lambd") = 1e-2,
        py::arg("ridge") = 1e-12, py::arg("eps_mu") = 1e-15, py::arg("eps_dmu") = 1e-15);

  m.def("glm_binomial_elasticnet_without_intercept",
        &glm_binomial_elasticnet_without_intercept_py,
        py::arg("X"), py::arg("y"), py::arg("n_trials"), py::arg("offset"),
        py::arg("beta0"),
        py::arg("standardize") = py::none(),
        py::arg("penalty") = py::none(),
        py::arg("max_iter") = 25,
        py::arg("tol") = 1e-8,
        py::arg("link") = "logit",
        py::arg("alpha") = 0.5,
        py::arg("lambd") = 1e-2,
        py::arg("ridge") = 1e-12,
        py::arg("eps_mu") = 1e-15,
        py::arg("eps_dmu") = 1e-15);

  py::class_<GLMPoissonENetResult>(m, "GLMPoissonENetResult")
      .def_readonly("intercept", &GLMPoissonENetResult::intercept)
      .def_readonly("beta", &GLMPoissonENetResult::beta)
      .def_readonly("converged", &GLMPoissonENetResult::converged)
      .def_readonly("n_outer", &GLMPoissonENetResult::n_outer)
      .def_readonly("n_inner", &GLMPoissonENetResult::n_inner)
      .def_readonly("max_delta", &GLMPoissonENetResult::max_delta)
      .def_readonly("max_delta_inner", &GLMPoissonENetResult::max_delta_inner);

  m.def("glm_poisson_elasticnet_with_intercept",
        &glm_poisson_elasticnet_with_intercept_py,
        py::arg("X"),
        py::arg("y"),
        py::arg("offset"),
        py::arg("intercept0"),
        py::arg("beta0"),
        py::arg("standardize") = py::none(),
        py::arg("penalty") = py::none(),
        py::arg("max_iter") = 25,
        py::arg("tol") = 1e-8,
        py::arg("alpha") = 0.5,
        py::arg("lambd") = 1e-2,
        py::arg("ridge") = 1e-12,
        py::arg("eps_mu") = 1e-15);

  m.def("glm_poisson_elasticnet_without_intercept",
        &glm_poisson_elasticnet_without_intercept_py,
        py::arg("X"),
        py::arg("y"),
        py::arg("offset"),
        py::arg("beta0"),
        py::arg("standardize") = py::none(),
        py::arg("penalty") = py::none(),
        py::arg("max_iter") = 25,
        py::arg("tol") = 1e-8,
        py::arg("alpha") = 0.5,
        py::arg("lambd") = 1e-2,
        py::arg("ridge") = 1e-12,
        py::arg("eps_mu") = 1e-15);
}

PYBIND11_MODULE(_glm, m) {
  m.doc() = "pysrat GLM helpers";
  bind_glm(m);
}