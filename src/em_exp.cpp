#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <stdexcept>

extern "C" {
#include "srat_core.h"
}

namespace py = pybind11;

namespace expsrm {
  inline double func_barFi(double t, double rate) {
    return std::exp(-rate * t);
  }

  inline double func_barH1i(double t, double rate) {
    return (t + 1.0 / rate) * std::exp(-rate * t);
  }
}

// em_exp_emstep(params, data)
// params: 1-D numpy array length 2 [omega, rate]
// data: dict with keys: "len", "time", "fault", "type"
//   - len: int
//   - time: 1-D array float64, length len (intervals)
//   - fault: 1-D array float64, length len (counts)
//   - type: 1-D array int32/int64, length len (0/1)
py::dict em_exp_emstep(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data
) {
  auto pbuf = params.request();
  if (pbuf.ndim != 1 || pbuf.shape[0] < 2) {
    throw std::runtime_error("params must be a 1-D array with at least 2 elements [omega, rate].");
  }
  const double* p = static_cast<double*>(pbuf.ptr);
  const double omega = p[0];
  const double rate  = p[1];

  if (!data.contains("len") || !data.contains("time") || !data.contains("fault") || !data.contains("type")) {
    throw std::runtime_error("data must contain keys: 'len', 'time', 'fault', 'type'.");
  }

  const int dsize = py::cast<int>(data["len"]);

  auto time  = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(data["time"]);
  auto num   = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(data["fault"]);
  auto type  = py::cast<py::array_t<long long, py::array::c_style | py::array::forcecast>>(data["type"]);

  auto tbuf = time.request();
  auto nbuf = num.request();
  auto ybuf = type.request();

  if (tbuf.ndim != 1 || nbuf.ndim != 1 || ybuf.ndim != 1) {
    throw std::runtime_error("time/fault/type must be 1-D arrays.");
  }
  if (dsize != static_cast<int>(tbuf.shape[0]) || dsize != static_cast<int>(nbuf.shape[0]) || dsize != static_cast<int>(ybuf.shape[0])) {
    throw std::runtime_error("Invalid data: len does not match array lengths.");
  }

  const double* time_ptr = static_cast<double*>(tbuf.ptr);
  const double* num_ptr  = static_cast<double*>(nbuf.ptr);
  const long long* type_ptr = static_cast<long long*>(ybuf.ptr);

  double nn = 0.0;
  double en1 = 0.0;
  double en2 = 0.0;
  double llf = 0.0;

  double t = 0.0;
  double prev_barFi  = 1.0;
  double prev_barH1i = 1.0 / rate;

  for (int i = 0; i < dsize; i++) {
    t += time_ptr[i];

    const double barFi  = expsrm::func_barFi(t, rate);
    const double barH1i = expsrm::func_barH1i(t, rate);

    const double x = num_ptr[i];
    if (x != 0.0) {
      const double tmp1 = prev_barFi  - barFi;
      const double tmp2 = prev_barH1i - barH1i;

      nn  += x;
      en1 += x;
      en2 += x * tmp2 / tmp1;

      llf += x * std::log(tmp1) - std::lgamma(x + 1.0);
    }

    if (type_ptr[i] == 1) {
      nn  += 1.0;
      en1 += 1.0;
      en2 += t;
      llf += std::log(rate) - rate * t;
    }

    prev_barFi  = barFi;
    prev_barH1i = barH1i;
  }

  llf += nn * std::log(omega) - omega * (1.0 - prev_barFi);
  en1 += omega * prev_barFi;
  en2 += omega * prev_barH1i;

  const double new_omega = en1;
  const double new_rate  = en1 / en2;
  const double total     = en1;

  py::array_t<double> param_arr(2);
  py::array_t<double> pdiff_arr(2);

  auto param_mut = param_arr.mutable_unchecked<1>();
  auto pdiff_mut = pdiff_arr.mutable_unchecked<1>();
  param_mut(0) = new_omega;
  param_mut(1) = new_rate;
  pdiff_mut(0) = new_omega - omega;
  pdiff_mut(1) = new_rate  - rate;

  py::dict out;
  out["param"] = param_arr;
  out["pdiff"] = pdiff_arr;
  out["llf"]   = llf;
  out["total"] = total;
  return out;
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "pysrat core bindings";

  m.def(
    "sum",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> x) {
      auto buf = x.request();
      if (buf.ndim != 1) throw std::runtime_error("x must be 1-D");
      auto* ptr = static_cast<double*>(buf.ptr);
      int n = static_cast<int>(buf.shape[0]);
      return srat_sum(ptr, n);
    },
    py::arg("x"),
    "Sum a 1-D array using the SRAT C core"
  );

  m.def(
    "em_exp_emstep",
    &em_exp_emstep,
    py::arg("params"),
    py::arg("data"),
    "EM M-step for exponential SRGM (ported from Rcpp)"
  );
}
