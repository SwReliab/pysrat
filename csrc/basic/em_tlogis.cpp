#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "logistic.h"

namespace py = pybind11;

namespace tlogis_mo {

inline double func_Fi(double t, double p, double b) {
  const double e = std::exp(-b * t);
  return p * (1.0 - e) / (p + (1.0 - p) * e);
}

inline double func_h1i(double t, double p, double b) {
  const double e = std::exp(-b * t);
  return 2.0 * (1.0 - p) * (1.0 - e) / (p + (1.0 - p) * e);
}

inline double func_h2i(double t, double p, double b) {
  const double e = std::exp(-b * t);
  return 2.0 * (1.0 - p) * (1.0 - (1.0 + b * t) * e) / (p + (1.0 - p) * e) / b;
}

inline double func_H1i(double t, double p, double b) {
  const double e = std::exp(-b * t);
  const double denom = p + (1.0 - p) * e;
  return p * (1.0 - e) / (denom * denom);
}

inline double func_H2i(double t, double p, double b) {
  const double e = std::exp(-b * t);
  const double denom = p + (1.0 - p) * e;
  return p * (1.0 - (1.0 + b * t) * e) / (denom * denom) / b;
}

} // namespace tlogis_mo

py::dict em_tlogis_emstep_mo(
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

  const double p_mo = 1.0 / (1.0 + std::exp(loc / scale));
  const double b = 1.0 / scale;

  double en1 = 0.0;
  double en2 = 0.0;
  double en3 = 0.0;
  double llf = 0.0;
  double t = 0.0;
  double prev_Fi = 0.0;
  double prev_H1i = 0.0;
  double prev_H2i = 0.0;

  for (int i = 0; i < dsize; i++) {
    t += time_u(i);

    const double Fi = tlogis_mo::func_Fi(t, p_mo, b);
    const double H1i = tlogis_mo::func_H1i(t, p_mo, b);
    const double H2i = tlogis_mo::func_H2i(t, p_mo, b);

    const double x = num_u(i);
    if (x != 0.0) {
      const double tmp1 = Fi - prev_Fi;
      const double tmp2 = H1i - prev_H1i;
      const double tmp3 = H2i - prev_H2i;

      if (!(tmp1 > 0.0) || !std::isfinite(tmp1)) {
        llf = std::numeric_limits<double>::quiet_NaN();
      } else {
        en1 += x;
        en2 += x * tmp2 / tmp1;
        en3 += x * tmp3 / tmp1;
        llf += x * std::log(tmp1) - std::lgamma(x + 1.0);
      }
    }

    if (type_u(i) == 1) {
      en1 += 1.0;
      en2 += 1.0 + tlogis_mo::func_h1i(t, p_mo, b);
      en3 += 1.0 + tlogis_mo::func_h2i(t, p_mo, b);
      llf += logistic_logpdf(t, loc, scale) - logistic_logsf(0.0, loc, scale);
    }

    prev_Fi = Fi;
    prev_H1i = H1i;
    prev_H2i = H2i;
  }

  llf += en1 * std::log(omega) - omega * prev_Fi;

  en1 += omega * (1.0 - prev_Fi);
  en2 += omega * (1.0 / p_mo - prev_H1i);
  en3 += omega * (1.0 / (p_mo * b) - prev_H2i);

  const double new_p = en1 / en2;
  const double new_b = en2 / en3;

  const double total = en1;
  const double new_omega = en1;
  const double new_loc = std::log(1.0 / new_p - 1.0) / new_b;
  const double new_scale = 1.0 / new_b;

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
  return out;
}

py::dict em_tlogis_estep(
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

  const double F0 = logistic_cdf(0.0, loc, scale);
  const double barF0 = logistic_sf(0.0, loc, scale);
  const double log_barF0 = logistic_logsf(0.0, loc, scale);

  double nn = 0.0;
  double llf = 0.0;
  double t = 0.0;

  double prev_Fi = F0;

  for (int i = 0; i < dsize; i++) {
    t += time_u(i);
    const double Fi = logistic_cdf(t, loc, scale);

    const double x = num_u(i);
    if (x != 0.0) {
      nn += x;
      const double diff = Fi - prev_Fi;
      llf += x * (std::log(diff) - log_barF0) - std::lgamma(x + 1.0);
    }
    if (type_u(i) == 1) {
      nn += 1.0;
      llf += logistic_logpdf(t, loc, scale) - log_barF0;
    }

    prev_Fi = Fi;
  }

  llf += nn * std::log(omega) - omega * (prev_Fi - F0) / barF0;

  const double w1 = omega * (1.0 - prev_Fi) / barF0;
  const double w0 = (nn + w1) * F0 / barF0;

  py::dict out;
  out["llf"] = llf;
  out["omega"] = nn + w1;
  out["w0"] = w0;
  out["w1"] = w1;
  out["total"] = nn + w1;
  return out;
}

double em_tlogis_pllf(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data,
  double w0,
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

  double llf = w0 * logistic_logcdf(0.0, loc, scale);

  double prev = logistic_cdf(0.0, loc, scale);
  double t = 0.0;

  for (int i = 0; i < dsize; i++) {
    t += time_u(i);
    const double cur = logistic_cdf(t, loc, scale);

    if (num_u(i) != 0.0) {
      llf += num_u(i) * std::log(cur - prev);
    }
    if (type_u(i) == 1) {
      llf += logistic_logpdf(t, loc, scale);
    }
    prev = cur;
  }

  llf += w1 * logistic_logsf(t, loc, scale);
  return llf;
}
