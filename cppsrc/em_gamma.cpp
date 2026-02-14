#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "numlib.h"
#include "gauss_inte.h"
#include "inv_log_psi.h"

namespace py = pybind11;

#define ZERO 1.0e-10

template <typename T>
static std::string to_string_local(const T& v) {
  std::ostringstream oss;
  oss << v;
  return oss.str();
}

static double em_gamma_int(
  double t0,
  double t1,
  double shape,
  double rate,
  int n,
  std::vector<double>& x,
  std::vector<double>& w,
  std::vector<double>& fx,
  std::vector<double>& fv
) {
  const double c = gauss_inte_fx(n, x.data(), t0, t1, fx.data());
  for (int i = 0; i < n; i++) {
    const double t = fx[i];
    const double y = rate * t;
    const double log_pdf =
      std::log(rate) + (shape - 1.0) * std::log(y) - y - std::lgamma(shape);

    fv[i] = std::log(t) * std::exp(log_pdf);
  }
  return gauss_inte_fv(n, w.data(), c, fv.data());
}

py::dict em_gamma_emstep(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data,
  int divide = 15,
  double eps = 1.0e-10
) {
  if (params.ndim() != 1 || params.shape(0) < 3) {
    throw std::invalid_argument("params must be 1-D array length>=3: [omega, shape, rate].");
  }
  auto p = params.unchecked<1>();
  const double omega = p(0);
  const double shape = p(1);
  const double rate = p(2);

  if (!(omega > 0.0) || !std::isfinite(omega)) {
    throw std::invalid_argument("omega must be > 0 and finite.");
  }
  if (!(shape > 0.0) || !std::isfinite(shape)) {
    throw std::invalid_argument("shape must be > 0 and finite.");
  }
  if (!(rate > 0.0) || !std::isfinite(rate)) {
    throw std::invalid_argument("rate must be > 0 and finite.");
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
    throw std::invalid_argument("Invalid data: len must be > 0.");
  }
  if (divide <= 0) {
    throw std::invalid_argument("divide must be positive.");
  }

  const int n = divide;
  std::vector<double> x(n), w(n), fx(n), fv(n);

  gauss_inte_w(n, x.data(), w.data(), eps);

  auto time_u = time.unchecked<1>();
  auto num_u = num.unchecked<1>();
  auto type_u = type.unchecked<1>();

  const double a0 = std::lgamma(shape);
  const double a1 = std::lgamma(shape + 1.0);

  double en1 = 0.0;
  double en2 = 0.0;
  double en3 = 0.0;
  double llf = 0.0;

  double t0 = 0.0;
  double t1 = time_u(0);
  double x1 = num_u(0);

  double gam10 = 1.0;
  double gam11 = 1.0;
  double gam20 = q_gamma(shape, rate * t1, a0);
  double gam21 = q_gamma(shape + 1., rate * t1, a1);

  double tmp1;
  double tmp2;
  double tmp3;
  double tmp4;

  tmp3 = em_gamma_int(ZERO, t1, shape, rate, n, x, w, fx, fv);
  tmp4 = tmp3;

  if (x1 != 0.0) {
    tmp1 = gam10 - gam20;
    tmp2 = (shape / rate) * (gam11 - gam21);

    if (!(tmp1 > 0.0) || !std::isfinite(tmp1)) {
      llf = std::numeric_limits<double>::quiet_NaN();
    } else {
      en1 += x1;
      en2 += x1 * tmp2 / tmp1;
      en3 += x1 * tmp3 / tmp1;
      llf += x1 * std::log(tmp1) - std::lgamma(x1 + 1.0);
    }
  }

  if (type_u(0) == 1) {
    if (!(t1 > 0.0)) {
      throw std::invalid_argument("Invalid data: t must be >0 for log(t).");
    }
    en1 += 1.0;
    en2 += t1;
    en3 += std::log(t1);
    llf += shape * std::log(rate) + (shape - 1.0) * std::log(t1) - rate * t1 - std::lgamma(shape);
  }

  for (int j = 1; j < dsize; j++) {
    x1 = num_u(j);

    if (time_u(j) != 0.0) {
      t0 = t1;
      t1 = t0 + time_u(j);

      gam10 = gam20;
      gam11 = gam21;
      gam20 = q_gamma(shape, rate * t1, a0);
      gam21 = q_gamma(shape + 1., rate * t1, a1);

      tmp3 = em_gamma_int(t0, t1, shape, rate, n, x, w, fx, fv);
      tmp4 += tmp3;
    }

    if (x1 != 0.0) {
      tmp1 = gam10 - gam20;
      tmp2 = (shape / rate) * (gam11 - gam21);

      if (!(tmp1 > 0.0) || !std::isfinite(tmp1)) {
        llf = std::numeric_limits<double>::quiet_NaN();
      } else {
        en1 += x1;
        en2 += x1 * tmp2 / tmp1;
        en3 += x1 * tmp3 / tmp1;
        llf += x1 * std::log(tmp1) - std::lgamma(x1 + 1.0);
      }
    }

    if (type_u(j) == 1) {
      if (!(t1 > 0.0)) {
        throw std::invalid_argument("Invalid data: t must be >0 for log(t).");
      }
      en1 += 1.0;
      en2 += t1;
      en3 += std::log(t1);
      llf += shape * std::log(rate) + (shape - 1.0) * std::log(t1) - rate * t1 - std::lgamma(shape);
    }
  }

  llf += std::log(omega) * en1;
  en1 += omega * gam20;
  en2 += omega * (shape / rate) * gam21;
  en3 += omega * (psi(shape) - std::log(rate) - tmp4);
  llf += -omega * (1.0 - gam20);

  const double total = en1;
  const double new_omega = en1;
  const double new_shape = inv_log_psi(std::log(en2 / en1) - en3 / en1);
  const double new_rate = new_shape * en1 / en2;

  py::dict out;

  auto param_arr = py::array_t<double>(3);
  auto param_mut = param_arr.mutable_unchecked<1>();
  param_mut(0) = new_omega;
  param_mut(1) = new_shape;
  param_mut(2) = new_rate;
  out["param"] = std::move(param_arr);

  auto pdiff_arr = py::array_t<double>(3);
  auto pdiff_mut = pdiff_arr.mutable_unchecked<1>();
  pdiff_mut(0) = new_omega - omega;
  pdiff_mut(1) = new_shape - shape;
  pdiff_mut(2) = new_rate - rate;
  out["pdiff"] = std::move(pdiff_arr);

  out["llf"] = llf;
  out["total"] = total;
  return out;
}
