#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "numlib.h"
#include "inv_log_psi.h"

namespace py = pybind11;

// data: {"len": int, "time": np.ndarray[float], "fault": np.ndarray[float], "type": np.ndarray[int]}
// params: np.ndarray[float] length>=3 [omega, shape, scale]
// return: {"param": np.ndarray, "pdiff": np.ndarray, "llf": float, "total": float}
py::dict em_pareto_emstep(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data
) {
  if (params.ndim() != 1 || params.shape(0) < 3) {
    throw std::invalid_argument(
      "params must be 1-D array with length >= 3: [omega, shape, scale]."
    );
  }
  auto p = params.unchecked<1>();
  const double omega = p(0);
  const double shape = p(1);
  const double scale = p(2);

  if (!(omega > 0.0) || !std::isfinite(omega)) {
    throw std::invalid_argument("omega must be positive and finite.");
  }
  if (!(shape > 0.0) || !std::isfinite(shape)) {
    throw std::invalid_argument("shape must be positive and finite.");
  }
  if (!(scale > 0.0) || !std::isfinite(scale)) {
    throw std::invalid_argument("scale must be positive and finite.");
  }

  if (!data.contains("len") || !data.contains("time") || !data.contains("fault") || !data.contains("type")) {
    throw std::invalid_argument("data must contain keys: 'len', 'time', 'fault', 'type'.");
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
    throw std::invalid_argument("Invalid data: len != lengths of time/fault/type.");
  }
  if (dsize <= 0) {
    throw std::invalid_argument("Invalid data: len must be > 0.");
  }

  auto time_u = time.unchecked<1>();
  auto num_u = num.unchecked<1>();
  auto type_u = type.unchecked<1>();

  double t = time_u(0);
  double x = num_u(0);

  double en1 = 0.0;
  double en2 = 0.0;
  double en3 = 0.0;
  double llf = 0.0;

  double g00 = 1.0;
  double g01 = shape / scale;
  double g02 = psi(shape) - std::log(scale);

  double g10 = std::pow(scale / (scale + t), shape);
  double g11 = shape / (scale + t) * g10;
  double g12 = (psi(shape) - std::log(scale + t)) * g10;

  if (x != 0.0) {
    const double tmp1 = g00 - g10;
    const double tmp2 = g01 - g11;
    const double tmp3 = g02 - g12;

    if (!(tmp1 > 0.0) || !std::isfinite(tmp1)) {
      llf = std::numeric_limits<double>::quiet_NaN();
    } else {
      en1 += x;
      en2 += x * tmp2 / tmp1;
      en3 += x * tmp3 / tmp1;
      llf += x * std::log(tmp1) - std::lgamma(x + 1.0);
    }
  }

  if (type_u(0) == 1) {
    en1 += 1.0;
    en2 += (shape + 1.0) / (scale + t);
    en3 += psi(shape + 1.0) - std::log(scale + t);
    llf += std::log(shape) + shape * std::log(scale) - (shape + 1.0) * std::log(scale + t);
  }

  for (int j = 1; j < dsize; j++) {
    x = num_u(j);

    if (time_u(j) != 0.0) {
      t += time_u(j);
      g00 = g10;
      g01 = g11;
      g02 = g12;

      g10 = std::pow(scale / (scale + t), shape);
      g11 = shape / (scale + t) * g10;
      g12 = (psi(shape) - std::log(scale + t)) * g10;
    }

    if (x != 0.0) {
      const double tmp1 = g00 - g10;
      const double tmp2 = g01 - g11;
      const double tmp3 = g02 - g12;

      if (!(tmp1 > 0.0) || !std::isfinite(tmp1)) {
        llf = std::numeric_limits<double>::quiet_NaN();
      } else {
        en1 += x;
        en2 += x * tmp2 / tmp1;
        en3 += x * tmp3 / tmp1;
        llf += x * std::log(tmp1) - std::lgamma(x + 1.0);
      }
    }

    if (type_u(j) == 1) {
      en1 += 1.0;
      en2 += (shape + 1.0) / (scale + t);
      en3 += psi(shape + 1.0) - std::log(scale + t);
      llf += std::log(shape) + shape * std::log(scale) - (shape + 1.0) * std::log(scale + t);
    }
  }

  llf += std::log(omega) * en1;
  en1 += omega * g10;
  en2 += omega * g11;
  en3 += omega * g12;
  llf += -omega * (1.0 - g10);

  const double total = en1;
  const double new_omega = en1;

  const double ratio = en2 / en1;
  if (!(ratio > 0.0) || !std::isfinite(ratio)) {
    throw std::runtime_error("Numerical error: en2/en1 is non-positive or non-finite.");
  }

  const double new_shape = inv_log_psi(std::log(ratio) - en3 / en1);
  if (!(new_shape > 0.0) || !std::isfinite(new_shape)) {
    throw std::runtime_error("Numerical error: new_shape is non-positive or non-finite.");
  }

  const double new_scale = new_shape * en1 / en2;
  if (!(new_scale > 0.0) || !std::isfinite(new_scale)) {
    throw std::runtime_error("Numerical error: new_scale is non-positive or non-finite.");
  }

  py::dict out;

  auto param_arr = py::array_t<double>(3);
  auto param_mut = param_arr.mutable_unchecked<1>();
  param_mut(0) = new_omega;
  param_mut(1) = new_shape;
  param_mut(2) = new_scale;
  out["param"] = std::move(param_arr);

  auto pdiff_arr = py::array_t<double>(3);
  auto pdiff_mut = pdiff_arr.mutable_unchecked<1>();
  pdiff_mut(0) = new_omega - omega;
  pdiff_mut(1) = new_shape - shape;
  pdiff_mut(2) = new_scale - scale;
  out["pdiff"] = std::move(pdiff_arr);

  out["llf"] = llf;
  out["total"] = total;
  return out;
}
