#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <cfenv>
#include <stdexcept>

#include "bind_common.h"
#include "bind_glm.h"
#include "glm_poisson.h"

#pragma STDC FENV_ACCESS ON

namespace py = pybind11;

static Eigen::VectorXd parse_penalty_factor(py::object penalty_factor_obj, int p) {
  if (penalty_factor_obj.is_none()) {
    return Eigen::VectorXd::Ones(p);
  }

  py::array_t<double, py::array::c_style | py::array::forcecast> penalty_factor =
      penalty_factor_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
  auto pfb = penalty_factor.request();

  if (pfb.ndim != 1) {
    throw std::invalid_argument("penalty_factor must be 1D");
  }
  if (static_cast<int>(pfb.shape[0]) != p) {
    throw std::invalid_argument("penalty_factor length must match X.cols()");
  }

  Eigen::Map<const Eigen::VectorXd> pf_e(static_cast<const double*>(pfb.ptr), p);
  return pf_e;
}

static GLMPoissonResult glm_poisson_identity_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> offset,
    bool fit_intercept,
    double intercept0,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta0,
    py::object standardize_obj,
    double lambda,
    py::object penalty_factor_obj,
    int max_iter,
    double tol,
    double eps_mu) {

  std::feclearexcept(FE_ALL_EXCEPT);

  auto Xb = X.request();
  if (Xb.ndim != 2) throw std::invalid_argument("X must be 2D");
  const int n = static_cast<int>(Xb.shape[0]);
  const int p = static_cast<int>(Xb.shape[1]);

  auto yb = y.request();
  auto ob = offset.request();
  auto bb = beta0.request();

  if (yb.ndim != 1 || ob.ndim != 1 || bb.ndim != 1) {
    throw std::invalid_argument("y/offset/beta0 must be 1D");
  }
  if (static_cast<int>(yb.shape[0]) != n ||
      static_cast<int>(ob.shape[0]) != n) {
    throw std::invalid_argument("y/offset length must match X.rows()");
  }
  if (static_cast<int>(bb.shape[0]) != p) {
    throw std::invalid_argument("beta0 length must match X.cols()");
  }

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_e(
      static_cast<const double*>(Xb.ptr), n, p);
  Eigen::Map<const Eigen::VectorXd> y_e(static_cast<const double*>(yb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> off_e(static_cast<const double*>(ob.ptr), n);
  Eigen::Map<const Eigen::VectorXd> b0_e(static_cast<const double*>(bb.ptr), p);

  const Eigen::VectorXi stdmask =
      glm_pybind::parse_standardize_mask(standardize_obj, p);
  const Eigen::VectorXd penalty_factor_e =
      parse_penalty_factor(penalty_factor_obj, p);

  return glm_poisson_identity(
      X_e, y_e, off_e,
      fit_intercept, intercept0, b0_e, stdmask,
      lambda, penalty_factor_e,
      max_iter, tol, eps_mu);
}

static GLMPoissonResult glm_poisson_correlated_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> offset,
    bool fit_intercept,
    double intercept0,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta0,
    py::object standardize_obj,
    py::array_t<double, py::array::c_style | py::array::forcecast> lambda_l2_mat,
    double lambda,
    py::object penalty_factor_obj,
    int max_iter,
    double tol,
    double eps_mu) {

  std::feclearexcept(FE_ALL_EXCEPT);

  auto Xb = X.request();
  if (Xb.ndim != 2) throw std::invalid_argument("X must be 2D");
  const int n = static_cast<int>(Xb.shape[0]);
  const int p = static_cast<int>(Xb.shape[1]);

  auto yb = y.request();
  auto ob = offset.request();
  auto bb = beta0.request();
  auto lb = lambda_l2_mat.request();

  if (yb.ndim != 1 || ob.ndim != 1 || bb.ndim != 1) {
    throw std::invalid_argument("y/offset/beta0 must be 1D");
  }
  if (static_cast<int>(yb.shape[0]) != n ||
      static_cast<int>(ob.shape[0]) != n) {
    throw std::invalid_argument("y/offset length must match X.rows()");
  }
  if (static_cast<int>(bb.shape[0]) != p) {
    throw std::invalid_argument("beta0 length must match X.cols()");
  }
  if (lb.ndim != 2) {
    throw std::invalid_argument("lambda_l2_mat must be 2D");
  }
  if (static_cast<int>(lb.shape[0]) != p ||
      static_cast<int>(lb.shape[1]) != p) {
    throw std::invalid_argument("lambda_l2_mat must be p x p");
  }

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_e(
      static_cast<const double*>(Xb.ptr), n, p);
  Eigen::Map<const Eigen::VectorXd> y_e(static_cast<const double*>(yb.ptr), n);
  Eigen::Map<const Eigen::VectorXd> off_e(static_cast<const double*>(ob.ptr), n);
  Eigen::Map<const Eigen::VectorXd> b0_e(static_cast<const double*>(bb.ptr), p);
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lambda_l2_mat_e(
      static_cast<const double*>(lb.ptr), p, p);

  const Eigen::VectorXi stdmask =
      glm_pybind::parse_standardize_mask(standardize_obj, p);
  const Eigen::VectorXd penalty_factor_e =
      parse_penalty_factor(penalty_factor_obj, p);

  return glm_poisson_correlated(
      X_e, y_e, off_e,
      fit_intercept, intercept0, b0_e, stdmask,
      lambda_l2_mat_e, lambda, penalty_factor_e,
      max_iter, tol, eps_mu);
}

void bind_glm_poisson(py::module_& m) {
  py::class_<GLMPoissonResult>(m, "GLMPoissonResult")
      .def_readonly("intercept", &GLMPoissonResult::intercept)
      .def_readonly("beta", &GLMPoissonResult::beta)
      .def_readonly("converged", &GLMPoissonResult::converged)
      .def_readonly("n_iter", &GLMPoissonResult::n_iter);

  m.def("glm_poisson_identity", &glm_poisson_identity_py,
        py::arg("X"),
        py::arg("y"),
        py::arg("offset"),
        py::arg("fit_intercept") = true,
        py::arg("intercept0") = 0.0,
        py::arg("beta0"),
        py::arg("standardize") = py::none(),
        py::arg("lambda") = 1.0,
        py::arg("penalty_factor") = py::none(),
        py::arg("max_iter") = 25,
        py::arg("tol") = 1e-8,
        py::arg("eps_mu") = 1e-15);

  m.def("glm_poisson_correlated", &glm_poisson_correlated_py,
        py::arg("X"),
        py::arg("y"),
        py::arg("offset"),
        py::arg("fit_intercept") = true,
        py::arg("intercept0") = 0.0,
        py::arg("beta0"),
        py::arg("standardize") = py::none(),
        py::arg("lambda_l2_mat"),
        py::arg("lambda") = 1.0,
        py::arg("penalty_factor") = py::none(),
        py::arg("max_iter") = 25,
        py::arg("tol") = 1e-8,
        py::arg("eps_mu") = 1e-15);
}