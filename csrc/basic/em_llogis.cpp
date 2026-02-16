#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "logistic.h"

namespace py = pybind11;

py::dict em_llogis_emstep(
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
  if (dsize <= 0) {
    throw std::invalid_argument("Invalid data: len must be >0.");
  }

  auto time_u = time.unchecked<1>();
  auto num_u = num.unchecked<1>();
  auto type_u = type.unchecked<1>();

  double t = time_u(0);
  if (!(t > 0.0)) {
    throw std::invalid_argument("Invalid data: cumulative time must be >0 for log-logistic (t>0).");
  }

  double x = num_u(0);

  double logt = std::log(t);
  double y = std::exp((logt - loc) / scale);

  double en1 = 0.0;
  double en2 = 0.0;
  double en3 = 0.0;
  double llf = 0.0;

  double g00 = 1.0;
  double g01 = 0.5;
  double g02 = 1.0;

  double g10 = 1.0 / (1.0 + y);
  double g11 = 1.0 / (2.0 * (1.0 + y) * (1.0 + y));
  double g12 = (1.0 + (1.0 + std::log(y)) * y) / ((1.0 + y) * (1.0 + y));

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
    en2 += 1.0 / (1.0 + y);
    en3 += (y - 1.0) * std::log(y) / (1.0 + y);
    llf += logistic_logpdf(logt, loc, scale) - logt;
  }

  for (int j = 1; j < dsize; j++) {
    x = num_u(j);

    if (time_u(j) != 0.0) {
      t += time_u(j);
      if (!(t > 0.0)) {
        throw std::invalid_argument("Invalid data: cumulative time must remain >0 for log-logistic (t>0).");
      }
      logt = std::log(t);
      y = std::exp((logt - loc) / scale);

      g00 = g10;
      g01 = g11;
      g02 = g12;

      g10 = 1.0 / (1.0 + y);
      g11 = 1.0 / (2.0 * (1.0 + y) * (1.0 + y));
      g12 = (1.0 + (1.0 + std::log(y)) * y) / ((1.0 + y) * (1.0 + y));
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
      en2 += 1.0 / (1.0 + y);
      en3 += (y - 1.0) * std::log(y) / (1.0 + y);
      llf += logistic_logpdf(logt, loc, scale) - logt;
    }
  }

  llf += std::log(omega) * en1;
  en1 += omega * g10;
  en2 += omega * g11;
  en3 += omega * g12;
  llf += -omega * (1.0 - g10);

  const double total = en1;
  const double new_omega = en1;

  const double new_scale = scale * en3 / en1;
  const double new_loc = loc + new_scale * (std::log(en1 / 2.0) - std::log(en2));

  py::dict out;

  auto param_arr = py::array_t<double>(3);
  auto param_mut = param_arr.mutable_unchecked<1>();
  param_mut(0) = new_omega;
  param_mut(1) = new_loc;
  param_mut(2) = new_scale;
  out["param"] = std::move(param_arr);

  auto pdiff_arr = py::array_t<double>(3);
  auto pdiff_mut = pdiff_arr.mutable_unchecked<1>();
  pdiff_mut(0) = new_omega - omega;
  pdiff_mut(1) = new_loc - loc;
  pdiff_mut(2) = new_scale - scale;
  out["pdiff"] = std::move(pdiff_arr);

  out["llf"] = llf;
  out["total"] = total;
  out["residual"] = omega * g10;
  return out;
}

py::dict em_llogis_estep(
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
      throw std::invalid_argument("Invalid data: cumulative time must be >0 for log-logistic (t>0).");
    }
    const double logt = std::log(t);
    const double Fi = logistic_cdf(logt, loc, scale);

    if (num_u(i) != 0.0) {
      nn += num_u(i);
      llf += num_u(i) * std::log(Fi - prev_Fi) - std::lgamma(num_u(i) + 1.0);
    }
    if (type_u(i) == 1) {
      nn += 1.0;
      llf += logistic_logpdf(logt, loc, scale) - logt;
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

double em_llogis_pllf(
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

  double llf = 0.0;
  double prev_Fi = 0.0;
  double t = 0.0;

  for (int i = 0; i < dsize; i++) {
    t += time_u(i);
    if (!(t > 0.0)) {
      throw std::invalid_argument("Invalid data: cumulative time must be >0 for log-logistic (t>0).");
    }
    const double logt = std::log(t);
    const double Fi = logistic_cdf(logt, loc, scale);

    if (num_u(i) != 0.0) {
      llf += num_u(i) * std::log(Fi - prev_Fi);
    }
    if (type_u(i) == 1) {
      llf += logistic_logpdf(logt, loc, scale);
    }
    prev_Fi = Fi;
  }

  llf += w1 * std::log(1.0 - prev_Fi);
  return llf;
}
