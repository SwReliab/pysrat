#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "gumbel.h"

namespace py = pybind11;

py::dict em_lxvmax_estep(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data
) {
  if (params.ndim() != 1 || params.shape(0) < 3) {
    throw std::invalid_argument("params must be 1-D array length>=3: [omega, loc, scale].");
  }
  auto p = params.unchecked<1>();
  const double omega = p(0);
  const double loc = p(1);
  const double scale = p(2);

  if (!(omega > 0.0) || !std::isfinite(omega)) {
    throw std::invalid_argument("omega must be >0.");
  }
  if (!(scale > 0.0) || !std::isfinite(scale)) {
    throw std::invalid_argument("scale must be >0.");
  }

  if (!data.contains("len") || !data.contains("time") || !data.contains("fault") || !data.contains("type")) {
    throw std::invalid_argument("data must contain keys: 'len','time','fault','type'.");
  }
  const int dsize = py::cast<int>(data["len"]);

  auto time = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(data["time"]);
  auto num = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(data["fault"]);
  auto type = py::cast<py::array_t<long long, py::array::c_style | py::array::forcecast>>(data["type"]);

  if (time.ndim() != 1 || num.ndim() != 1 || type.ndim() != 1) {
    throw std::invalid_argument("time/fault/type must be 1-D arrays.");
  }
  if (dsize != static_cast<int>(time.shape(0)) ||
      dsize != static_cast<int>(num.shape(0)) ||
      dsize != static_cast<int>(type.shape(0))) {
    throw std::invalid_argument("Invalid data: len mismatch.");
  }

  auto time_u = time.unchecked<1>();
  auto num_u = num.unchecked<1>();
  auto type_u = type.unchecked<1>();

  double nn = 0.0;
  double llf = 0.0;
  double t = 0.0;
  double prev_Fi = 0.0;

  for (int i = 0; i < dsize; i++) {
    t += time_u(i);
    if (!(t > 0.0)) {
      throw std::invalid_argument("Invalid data: cumulative time must be >0 for log(t).");
    }
    const double logt = std::log(t);
    const double Fi = Revd::pgumbel(logt, loc, scale, true, false);

    const double x = num_u(i);
    if (x != 0.0) {
      nn += x;
      llf += x * std::log(Fi - prev_Fi) - std::lgamma(x + 1.0);
    }
    if (type_u(i) == 1) {
      nn += 1.0;
      llf += Revd::dgumbel(logt, loc, scale, true) - logt;
    }
    prev_Fi = Fi;
  }

  llf += nn * std::log(omega) - omega * prev_Fi;
  const double w1 = omega * (1.0 - prev_Fi);

  py::dict out;
  out["llf"] = llf;
  out["omega"] = nn + w1;
  out["w1"] = w1;
  out["total"] = nn + w1;
  return out;
}

double em_lxvmax_pllf(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data,
  double w1
) {
  if (params.ndim() != 1 || params.shape(0) < 2) {
    throw std::invalid_argument("params must be 1-D array length>=2: [loc, scale].");
  }
  auto p = params.unchecked<1>();
  const double loc = p(0);
  const double scale = p(1);

  if (!(scale > 0.0) || !std::isfinite(scale)) {
    throw std::invalid_argument("scale must be >0.");
  }

  if (!data.contains("len") || !data.contains("time") || !data.contains("fault") || !data.contains("type")) {
    throw std::invalid_argument("data must contain keys: 'len','time','fault','type'.");
  }
  const int dsize = py::cast<int>(data["len"]);

  auto time = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(data["time"]);
  auto num = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(data["fault"]);
  auto type = py::cast<py::array_t<long long, py::array::forcecast | py::array::c_style>>(data["type"]);

  if (time.ndim() != 1 || num.ndim() != 1 || type.ndim() != 1) {
    throw std::invalid_argument("time/fault/type must be 1-D arrays.");
  }
  if (dsize != static_cast<int>(time.shape(0)) ||
      dsize != static_cast<int>(num.shape(0)) ||
      dsize != static_cast<int>(type.shape(0))) {
    throw std::invalid_argument("Invalid data: len mismatch.");
  }

  auto time_u = time.unchecked<1>();
  auto num_u = num.unchecked<1>();
  auto type_u = type.unchecked<1>();

  double llf = 0.0;
  double prev_Fi = 0.0;
  double t = 0.0;

  for (int i = 0; i < dsize; i++) {
    t += time_u(i);
    if (!(t > 0.0)) {
      throw std::invalid_argument("Invalid data: cumulative time must be >0 for log(t).");
    }
    const double logt = std::log(t);
    const double Fi = Revd::pgumbel(logt, loc, scale, true, false);

    if (num_u(i) != 0.0) {
      llf += num_u(i) * std::log(Fi - prev_Fi);
    }
    if (type_u(i) == 1) {
      llf += Revd::dgumbel(logt, loc, scale, true);
    }
    prev_Fi = Fi;
  }

  llf += w1 * std::log(1.0 - prev_Fi);
  return llf;
}
