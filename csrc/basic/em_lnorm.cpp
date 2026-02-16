#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "norm.h"

namespace py = pybind11;

py::dict em_lnorm_emstep(
  py::array_t<double, py::array::c_style | py::array::forcecast> params,
  py::dict data
) {
  if (params.ndim() != 1 || params.shape(0) < 3) {
    throw std::invalid_argument("params must be 1-D array with length >= 3: [omega, mu, sig].");
  }
  auto p = params.unchecked<1>();
  const double omega = p(0);
  const double mu = p(1);
  const double sig = p(2);

  if (!(omega > 0.0) || !std::isfinite(omega)) {
    throw std::invalid_argument("omega must be positive and finite.");
  }
  if (!(sig > 0.0) || !std::isfinite(sig)) {
    throw std::invalid_argument("sig must be positive and finite.");
  }
  if (!std::isfinite(mu)) {
    throw std::invalid_argument("mu must be finite.");
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
  if (!(t > 0.0)) {
    throw std::invalid_argument("Invalid data: cumulative time must be > 0 for lognormal (t>0).");
  }

  double x = num_u(0);

  double logt = std::log(t);
  double y = (logt - mu) / sig;

  double en1 = 0.0;
  double en2 = 0.0;
  double en3 = 0.0;
  double llf = 0.0;

  double tmp_pdf = norm_phi(y);

  double g00 = 1.0;
  double g01 = mu;
  double g02 = sig * sig + mu * mu;

  double g10 = norm_Q(y);
  double g11 = sig * tmp_pdf + mu * g10;
  double g12 = (sig * logt + mu * sig) * tmp_pdf + (sig * sig + mu * mu) * g10;

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
    en2 += logt;
    en3 += logt * logt;
    llf += norm_logpdf(logt, mu, sig) - logt;
  }

  for (int j = 1; j < dsize; j++) {
    x = num_u(j);

    if (time_u(j) != 0.0) {
      t += time_u(j);
      if (!(t > 0.0)) {
        throw std::invalid_argument("Invalid data: cumulative time must remain > 0 for lognormal (t>0).");
      }

      logt = std::log(t);
      y = (logt - mu) / sig;
      tmp_pdf = norm_phi(y);

      g00 = g10;
      g01 = g11;
      g02 = g12;

      g10 = norm_Q(y);
      g11 = sig * tmp_pdf + mu * g10;
      g12 = (sig * logt + mu * sig) * tmp_pdf + (sig * sig + mu * mu) * g10;
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
      en2 += logt;
      en3 += logt * logt;
      llf += norm_logpdf(logt, mu, sig) - logt;
    }
  }

  llf += std::log(omega) * en1;
  en1 += omega * g10;
  en2 += omega * g11;
  en3 += omega * g12;
  llf += -omega * (1.0 - g10);

  const double total = en1;
  const double new_omega = en1;
  const double new_mu = en2 / en1;

  const double var = en3 / en1 - new_mu * new_mu;
  const double new_sig = (var > 0.0) ? std::sqrt(var) : 0.0;

  py::dict out;

  auto param_arr = py::array_t<double>(3);
  auto param_mut = param_arr.mutable_unchecked<1>();
  param_mut(0) = new_omega;
  param_mut(1) = new_mu;
  param_mut(2) = new_sig;
  out["param"] = std::move(param_arr);

  auto pdiff_arr = py::array_t<double>(3);
  auto pdiff_mut = pdiff_arr.mutable_unchecked<1>();
  pdiff_mut(0) = new_omega - omega;
  pdiff_mut(1) = new_mu - mu;
  pdiff_mut(2) = new_sig - sig;
  out["pdiff"] = std::move(pdiff_arr);

  out["llf"] = llf;
  out["total"] = total;
  return out;
}
