#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <cfenv>
#include <stdexcept>
#include <string>

#include "bind_common.h"
#include "bind_glm.h"
#include "glm_binomial.h"

#pragma STDC FENV_ACCESS ON

namespace py = pybind11;

static GLMBinomialResult glm_binomial_with_intercept_py(
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

  const Eigen::VectorXi stdmask = glm_pybind::parse_standardize_mask(standardize_obj, p);

  return glm_binomial_with_intercept(
      X_e, y_e, n_e, off_e,
      intercept0, b0_e, stdmask,
      max_iter, tol, link, ridge, eps_mu, eps_dmu);
}

static GLMBinomialResult glm_binomial_without_intercept_py(
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

  const Eigen::VectorXi stdmask = glm_pybind::parse_standardize_mask(standardize_obj, p);

  return glm_binomial_without_intercept(
      X_e, y_e, n_e, off_e,
      b0_e, stdmask,
      max_iter, tol, link, ridge, eps_mu, eps_dmu);
}

void bind_glm_binomial(py::module_& m) {
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
}
