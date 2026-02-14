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

py::dict em_lnorm_emstep(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data
);

py::dict em_tlogis_emstep_mo(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data
);

py::dict em_tlogis_estep(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data
);

double em_tlogis_pllf(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data,
  double w0,
  double w1
);

py::dict em_llogis_emstep(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data
);

py::dict em_llogis_estep(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data
);

double em_llogis_pllf(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data,
  double w1
);

py::dict em_gamma_emstep(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data,
  int divide,
  double eps
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
    "em_lnorm_emstep",
    &em_lnorm_emstep,
    py::arg("params"),
    py::arg("data"),
    "EM step for lognormal NHPP SRM (returns dict: param/pdiff/llf/total)."
  );

  m.def(
    "em_tlogis_emstep_mo",
    &em_tlogis_emstep_mo,
    py::arg("params"),
    py::arg("data"),
    "EM step (Marshall-Olkin type) for truncated logistic SRM (dict: param/pdiff/llf/total)."
  );

  m.def(
    "em_tlogis_estep",
    &em_tlogis_estep,
    py::arg("params"),
    py::arg("data"),
    "E-step for truncated logistic SRM (dict: llf/omega/w0/w1/total)."
  );

  m.def(
    "em_tlogis_pllf",
    &em_tlogis_pllf,
    py::arg("params"),
    py::arg("data"),
    py::arg("w0"),
    py::arg("w1"),
    "Partial log-likelihood for truncated logistic SRM (params=[loc,scale])."
  );

  m.def(
    "em_llogis_emstep",
    &em_llogis_emstep,
    py::arg("params"),
    py::arg("data"),
    "EM step for log-logistic NHPP SRM (dict: param/pdiff/llf/total/residual)."
  );

  m.def(
    "em_llogis_estep",
    &em_llogis_estep,
    py::arg("params"),
    py::arg("data"),
    "E-step for log-logistic NHPP SRM (dict: llf/omega/w1/total)."
  );

  m.def(
    "em_llogis_pllf",
    &em_llogis_pllf,
    py::arg("params"),
    py::arg("data"),
    py::arg("w1"),
    "Partial log-likelihood for log-logistic NHPP SRM (params=[loc,scale])."
  );

  m.def(
    "em_gamma_emstep",
    &em_gamma_emstep,
    py::arg("params"),
    py::arg("data"),
    py::arg("divide") = 15,
    py::arg("eps") = 1.0e-10,
    "EM step for gamma NHPP SRM (returns dict: param/pdiff/llf/total)."
  );

  m.def(
    "sum",
    &sum_array,
    py::arg("x"),
    "Sum of a 1-D array of doubles."
  );
}
