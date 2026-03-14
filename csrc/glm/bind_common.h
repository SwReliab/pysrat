#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <stdexcept>

namespace py = pybind11;

namespace glm_pybind {

inline Eigen::VectorXi default_standardize_mask(int p) {
  Eigen::VectorXi m(p);
  m.setOnes();
  return m;
}

inline Eigen::VectorXi parse_standardize_mask(py::object standardize_obj, int p) {
  if (standardize_obj.is_none()) {
    return default_standardize_mask(p);
  }

  py::array_t<int, py::array::c_style | py::array::forcecast> standardize =
      standardize_obj.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
  auto sb = standardize.request();
  if (sb.ndim != 1) {
    throw std::invalid_argument("standardize must be 1D");
  }
  if (static_cast<int>(sb.shape[0]) != p) {
    throw std::invalid_argument("standardize length must match X.cols()");
  }

  Eigen::Map<const Eigen::VectorXi> std_e(static_cast<const int*>(sb.ptr), p);
  return std_e;
}

inline Eigen::VectorXi default_penalty_mask(int p) {
  Eigen::VectorXi m(p);
  if (p <= 0) {
    return m;
  }
  m.setOnes();
  return m;
}

inline Eigen::VectorXi parse_penalty_mask(py::object penalty_obj, int p) {
  if (penalty_obj.is_none()) {
    return default_penalty_mask(p);
  }

  py::array_t<int, py::array::c_style | py::array::forcecast> penalty =
      penalty_obj.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
  auto pb = penalty.request();
  if (pb.ndim != 1) {
    throw std::invalid_argument("penalty must be 1D");
  }
  if (static_cast<int>(pb.shape[0]) != p) {
    throw std::invalid_argument("penalty length must match X.cols()");
  }

  Eigen::Map<const Eigen::VectorXi> pen_e(static_cast<const int*>(pb.ptr), p);
  return pen_e;
}

}  // namespace glm_pybind
