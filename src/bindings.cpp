#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <numeric>

namespace py = pybind11;

py::dict em_exp_emstep(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data
);

py::dict em_tnorm_emstep(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data
);

py::dict em_pareto_emstep(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data
);

double sum_array(py::array_t<double, py::array::c_style | py::array::forcecast> x) {
  auto buf = x.request();
  auto ptr = static_cast<const double*>(buf.ptr);
  return std::accumulate(ptr, ptr + buf.size, 0.0);
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "pysrat core bindings";

  m.def(
    "em_exp_emstep",
    &em_exp_emstep,
    py::arg("params"),
    py::arg("data"),
    "EM M-step for exponential SRGM (ported from Rcpp)"
  );

  m.def(
    "em_tnorm_emstep",
    &em_tnorm_emstep,
    py::arg("params"),
    py::arg("data"),
    "EM step for truncated normal NHPP SRM (returns dict: param/pdiff/llf/total)."
  );

  m.def(
    "em_pareto_emstep",
    &em_pareto_emstep,
    py::arg("params"),
    py::arg("data"),
    "EM step for Pareto(type2/Lomax) NHPP SRM (returns dict: param/pdiff/llf/total)."
  );

  m.def(
    "sum",
    &sum_array,
    py::arg("x"),
    "Sum of a 1-D array of doubles."
  );
}
