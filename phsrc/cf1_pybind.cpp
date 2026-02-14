// cf1_pybind.cpp
//
// pybind11 bindings for CF1 (std::span backend + vector<vector<double>> workspaces)
//
// Assumptions / dependencies:
//   - blas1_span.h, blas2_span.h, mexp_span.h, cf1utils_span.h exist and are consistent
//   - poisson.h provides:
//        int rightbound(double lambda, double eps);
//        double pmf(double lambda, int left, int right, std::span<double> prob);
//     (pmf fills prob[0..right] at least; returns "weight" like your original code)
//
// Build:
//   - C++20
//   - link with pybind11 and NumPy
//
// Exposed functions:
//   - cf1pdf(dx, alpha, rate, eps=1e-8, ufactor=1.01, log=False)
//   - cf1cdf(dx, alpha, rate, eps=1e-8, ufactor=1.01, lower=True, log=False)
//   - cf1sample(n, alpha, rate, seed=None)
//   - cf1reform(alpha, rate) -> {"alpha":..., "rate":...}
//   - cf1sojourn(alpha, rate, f, b, t, eps=1e-8, ufactor=1.01) -> H (len 2n)
//   - em_cf1_emstep(params:dict, data:dict, eps=1e-8, ufactor=1.01) -> dict
//
// Notes:
//   - All numpy arrays must be 1D contiguous; we forcecast/copy if needed.
//   - num/type are expected integer arrays; accepted dtypes: int32/int64 (forcecast).
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <string>
#include <optional>
#include <stdexcept>
#include <span>
#include <algorithm>

#include "blas1_span.h"
#include "blas2_span.h"
#include "mexp_span.h"
#include "cf1utils_span.h"
#include "poisson.h"

namespace py = pybind11;
using namespace marlib;

// -----------------------------
// Span helpers for numpy 1D arrays
// -----------------------------
static inline std::span<const double>
 as_span_const_1d(py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
  auto buf = a.request();
  if (buf.ndim != 1) throw std::invalid_argument("expected 1D numpy array (double)");
  return {static_cast<const double*>(buf.ptr), static_cast<size_t>(buf.shape[0])};
}

static inline std::span<double>
as_span_mut_1d(py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
  auto buf = a.request();
  if (buf.ndim != 1) throw std::invalid_argument("expected 1D numpy array (double)");
  return {static_cast<double*>(buf.ptr), static_cast<size_t>(buf.shape[0])};
}

template <class IntT>
static inline std::span<const IntT>
 as_span_const_1d_int(py::array_t<IntT, py::array::c_style | py::array::forcecast>& a) {
  auto buf = a.request();
  if (buf.ndim != 1) throw std::invalid_argument("expected 1D numpy array (int)");
  return {static_cast<const IntT*>(buf.ptr), static_cast<size_t>(buf.shape[0])};
}

static inline py::array_t<double> make_double_array(size_t n) {
  return py::array_t<double>(static_cast<py::ssize_t>(n));
}

// -----------------------------
// cf1pdf
// -----------------------------
py::array_t<double>
cf1pdf(py::array_t<double, py::array::c_style | py::array::forcecast> dx,
       py::array_t<double, py::array::c_style | py::array::forcecast> alpha,
       py::array_t<double, py::array::c_style | py::array::forcecast> rate,
       double eps = 1.0e-8,
       double ufactor = 1.01,
       bool log = false) {

  auto dxs = as_span_const_1d(dx);
  auto as  = as_span_const_1d(alpha);
  auto rs  = as_span_const_1d(rate);

  if (as.size() == 0) throw std::invalid_argument("alpha must be non-empty");
  if (rs.size() != as.size()) throw std::invalid_argument("alpha and rate must have same length");
  if (dxs.size() == 0) return make_double_array(0);

  const int k = static_cast<int>(dxs.size());
  const int n = static_cast<int>(as.size());

  const double tmax = dxs[static_cast<size_t>(idamax(dxs))];

  // P = rate copy (mutable), uniformize
  std::vector<double> P(rs.begin(), rs.end());
  const double qv = unif(cf1_matrix{}, std::span<double>(P), ufactor);

  // prob buffer sized by worst-case rightbound(qv*tmax, eps)
  const int prob_len = rightbound(qv * tmax, eps) + 1;
  std::vector<double> prob(static_cast<size_t>(prob_len));

  // tmp is state vector (starts from alpha, updated each step), xi work
  std::vector<double> tmp(as.begin(), as.end());
  std::vector<double> xi(static_cast<size_t>(n));

  py::array_t<double> out = make_double_array(dxs.size());
  auto outs = as_span_mut_1d(out);

  for (int i = 0; i < k; ++i) {
    const double t = dxs[static_cast<size_t>(i)];
    const int right = rightbound(qv * t, eps);
    const double weight = pmf(qv * t, 0, right, std::span<double>(prob));

    mexpv(cf1_matrix{}, trans{},
          std::span<const double>(P),
          std::span<const double>(prob.data(), prob.size()),
          right, weight,
          std::span<const double>(tmp),
          std::span<double>(tmp),
          std::span<double>(xi));

    outs[static_cast<size_t>(i)] = rs[static_cast<size_t>(n - 1)] * tmp[static_cast<size_t>(n - 1)];
  }

  if (log) {
    for (size_t i = 0; i < outs.size(); ++i) outs[i] = std::log(outs[i]);
  }

  return out;
}

// -----------------------------
// cf1cdf
// -----------------------------
py::array_t<double>
cf1cdf(py::array_t<double, py::array::c_style | py::array::forcecast> dx,
       py::array_t<double, py::array::c_style | py::array::forcecast> alpha,
       py::array_t<double, py::array::c_style | py::array::forcecast> rate,
       double eps = 1.0e-8,
       double ufactor = 1.01,
       bool lower = true,
       bool log = false) {

  auto dxs = as_span_const_1d(dx);
  auto as  = as_span_const_1d(alpha);
  auto rs  = as_span_const_1d(rate);

  if (as.size() == 0) throw std::invalid_argument("alpha must be non-empty");
  if (rs.size() != as.size()) throw std::invalid_argument("alpha and rate must have same length");
  if (dxs.size() == 0) return make_double_array(0);

  const int k = static_cast<int>(dxs.size());
  const int n = static_cast<int>(as.size());

  const double tmax = dxs[static_cast<size_t>(idamax(dxs))];

  std::vector<double> P(rs.begin(), rs.end());
  const double qv = unif(cf1_matrix{}, std::span<double>(P), ufactor);

  const int prob_len = rightbound(qv * tmax, eps) + 1;
  std::vector<double> prob(static_cast<size_t>(prob_len));

  std::vector<double> tmp(as.begin(), as.end());
  std::vector<double> xi(static_cast<size_t>(n));

  py::array_t<double> out = make_double_array(dxs.size());
  auto outs = as_span_mut_1d(out);

  for (int i = 0; i < k; ++i) {
    const double t = dxs[static_cast<size_t>(i)];
    const int right = rightbound(qv * t, eps);
    const double weight = pmf(qv * t, 0, right, std::span<double>(prob));

    mexpv(cf1_matrix{}, trans{},
          std::span<const double>(P),
          std::span<const double>(prob.data(), prob.size()),
          right, weight,
          std::span<const double>(tmp),
          std::span<double>(tmp),
          std::span<double>(xi));

    outs[static_cast<size_t>(i)] = dasum(std::span<const double>(tmp));
  }

  // Match original Rcpp behavior:
  //   result = dasum(tmp)
  //   if lower==true => 1-result
  //   if log==true => log(...)
  if (!lower && !log) {
    // keep outs
  } else if (lower && !log) {
    for (size_t i = 0; i < outs.size(); ++i) outs[i] = 1.0 - outs[i];
  } else if (!lower && log) {
    for (size_t i = 0; i < outs.size(); ++i) outs[i] = std::log(outs[i]);
  } else { // lower && log
    for (size_t i = 0; i < outs.size(); ++i) outs[i] = std::log(1.0 - outs[i]);
  }

  return out;
}

// -----------------------------
// cf1sample
// -----------------------------
py::array_t<double>
cf1sample(int n,
          py::array_t<double, py::array::c_style | py::array::forcecast> alpha,
          py::array_t<double, py::array::c_style | py::array::forcecast> rate,
          std::optional<std::uint64_t> seed = std::nullopt) {

  auto as = as_span_const_1d(alpha);
  auto rs = as_span_const_1d(rate);

  if (n < 0) throw std::invalid_argument("n must be non-negative");
  if (as.size() == 0) throw std::invalid_argument("alpha must be non-empty");
  if (rs.size() != as.size()) throw std::invalid_argument("alpha and rate must have same length");

  py::array_t<double> out = make_double_array(static_cast<size_t>(n));
  auto outs = as_span_mut_1d(out);
  dfill(outs, 0.0);

  std::mt19937_64 rng(seed ? *seed : std::random_device{}());
  std::uniform_real_distribution<double> unif01(0.0, 1.0);

  int y = 0;
  double remain = 1.0;

  for (size_t l = 0; l < as.size(); ++l) {
    const int trials = n - y;
    if (trials <= 0) break;

    const double p = (remain > 0.0) ? (as[l] / remain) : 0.0;
    const double p_clamped = std::min(1.0, std::max(0.0, p));

    std::binomial_distribution<int> binom(trials, p_clamped);
    const int add = binom(rng);
    y += add;

    remain -= as[l];

    const double lam = rs[l];
    for (int i = 0; i < y; ++i) {
      const double u = unif01(rng);
      outs[static_cast<size_t>(i)] += -std::log(u) / lam;
    }
  }

  return out;
}

// -----------------------------
// cf1reform
// -----------------------------
py::dict
cf1reform(py::array_t<double, py::array::c_style | py::array::forcecast> alpha,
          py::array_t<double, py::array::c_style | py::array::forcecast> rate) {

  auto as = as_span_const_1d(alpha);
  auto rs = as_span_const_1d(rate);
  if (as.size() == 0) throw std::invalid_argument("alpha must be non-empty");
  if (rs.size() != as.size()) throw std::invalid_argument("alpha and rate must have same length");

  py::array_t<double> a = make_double_array(as.size());
  py::array_t<double> r = make_double_array(rs.size());
  auto am = as_span_mut_1d(a);
  auto rm = as_span_mut_1d(r);

  dcopy(as, am);
  dcopy(rs, rm);
  cf1_sort(am, rm);

  py::dict d;
  d["alpha"] = a;
  d["rate"]  = r;
  return d;
}

// -----------------------------
// cf1sojourn
// -----------------------------
py::array_t<double>
cf1sojourn(py::array_t<double, py::array::c_style | py::array::forcecast> alpha,
           py::array_t<double, py::array::c_style | py::array::forcecast> rate,
           py::array_t<double, py::array::c_style | py::array::forcecast> f,
           py::array_t<double, py::array::c_style | py::array::forcecast> b,
           double t,
           double eps = 1.0e-8,
           double ufactor = 1.01) {

  auto as = as_span_const_1d(alpha);
  auto rs = as_span_const_1d(rate);
  auto fs = as_span_const_1d(f);
  auto bs = as_span_const_1d(b);

  const int n = static_cast<int>(as.size());
  if (n == 0) throw std::invalid_argument("alpha must be non-empty");
  if (rs.size() != as.size()) throw std::invalid_argument("alpha and rate must have same length");
  if (fs.size() != as.size() || bs.size() != as.size()) {
    throw std::invalid_argument("f and b must have same length as alpha");
  }

  // P = rate copy (mutable)
  std::vector<double> P(rs.begin(), rs.end());
  const double qv = unif(cf1_matrix{}, std::span<double>(P), ufactor);

  const int right = rightbound(qv * t, eps);

  std::vector<double> prob(static_cast<size_t>(right + 1));
  const double weight = pmf(qv * t, 0, right + 1, std::span<double>(prob));

  py::array_t<double> H = make_double_array(static_cast<size_t>(2 * n));
  auto Hs = as_span_mut_1d(H);
  dfill(Hs, 0.0);

  std::vector<double> f2(fs.begin(), fs.end());
  std::vector<double> xi(static_cast<size_t>(n));

  // vc workspace: (right+2) x n
  std::vector<std::vector<double>> vc(static_cast<size_t>(right + 2),
                                      std::vector<double>(static_cast<size_t>(n)));

  mexp_conv(cf1_matrix{}, trans{},
            std::span<const double>(P), qv,
            std::span<const double>(prob.data(), prob.size()),
            right, weight,
            std::span<const double>(f2),
            bs,
            std::span<double>(f2),
            Hs,
            std::span<double>(xi),
            vc);

  return H;
}

// -----------------------------
// em_cf1_emstep
// -----------------------------
// params: {"omega": float, "alpha": np.ndarray[double], "rate": np.ndarray[double]}
// data:   {"time": np.ndarray[double], "fault": np.ndarray[int], "type": np.ndarray[int]}
py::dict
em_cf1_emstep(py::dict params,
              py::dict data,
              double eps = 1.0e-8,
              double ufactor = 1.01) {

  if (!params.contains("omega") || !params.contains("alpha") || !params.contains("rate")) {
    throw std::invalid_argument("params must contain keys: 'omega', 'alpha', 'rate'");
  }
  if (!data.contains("time") || !data.contains("fault") || !data.contains("type")) {
    throw std::invalid_argument("data must contain keys: 'time', 'fault', 'type'");
  }

  const double omega = py::float_(params["omega"]);

  auto alpha = params["alpha"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
  auto rate  = params["rate"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
  auto time  = data["time"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();

  // Accept int32/int64; cast to int64 first, then to vector<int>
  auto num64  = data["fault"].cast<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>>();
  auto type64 = data["type"].cast<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>>();

  auto as = as_span_const_1d(alpha);
  auto rs = as_span_const_1d(rate);
  auto ts = as_span_const_1d(time);
  auto ns64 = as_span_const_1d_int<std::int64_t>(num64);
  auto us64 = as_span_const_1d_int<std::int64_t>(type64);

  if (as.size() == 0) throw std::invalid_argument("alpha must be non-empty");
  if (rs.size() != as.size()) throw std::invalid_argument("alpha and rate must have same length");
  if (ts.size() == 0) throw std::invalid_argument("time must be non-empty");
  if (ns64.size() != ts.size() || us64.size() != ts.size()) {
    throw std::invalid_argument("time, fault, type must have same length");
  }

  // convert to vector<int>
  std::vector<int> num(static_cast<size_t>(ns64.size()));
  std::vector<int> typ(static_cast<size_t>(us64.size()));
  for (size_t i = 0; i < ns64.size(); ++i) num[i] = static_cast<int>(ns64[i]);
  for (size_t i = 0; i < us64.size(); ++i) typ[i] = static_cast<int>(us64[i]);

  double new_omega = 0.0;

  py::array_t<double> new_alpha = make_double_array(as.size());
  py::array_t<double> new_rate  = make_double_array(as.size());

  const double llf = marlib::cf1emstep(
      omega,
      as, rs,
      new_omega,
      as_span_mut_1d(new_alpha),
      as_span_mut_1d(new_rate),
      ts,
      std::span<const int>(num.data(), num.size()),
      std::span<const int>(typ.data(), typ.size()),
      eps, ufactor
  );

  // sort new params
  cf1_sort(as_span_mut_1d(new_alpha), as_span_mut_1d(new_rate));

  // pdiff arrays
  py::array_t<double> diff_alpha = make_double_array(as.size());
  py::array_t<double> diff_rate  = make_double_array(as.size());
  auto da = as_span_mut_1d(diff_alpha);
  auto dr = as_span_mut_1d(diff_rate);

  auto na = as_span_const_1d(new_alpha);
  auto nr = as_span_const_1d(new_rate);

  for (size_t i = 0; i < as.size(); ++i) da[i] = na[i] - as[i];
  for (size_t i = 0; i < rs.size(); ++i) dr[i] = nr[i] - rs[i];

  py::dict param;
  param["omega"] = new_omega;
  param["alpha"] = new_alpha;
  param["rate"]  = new_rate;

  py::dict pdiff;
  pdiff["omega"] = new_omega - omega;
  pdiff["alpha"] = diff_alpha;
  pdiff["rate"]  = diff_rate;

  py::dict out;
  out["param"] = param;
  out["pdiff"] = pdiff;
  out["llf"]   = llf;
  out["total"] = new_omega;

  return out;
}

// -----------------------------
// PYBIND11 module
// -----------------------------
PYBIND11_MODULE(marlib_cf1, m) {
  m.doc() = "CF1 utilities (pybind11 + std::span backend + vector<vector> workspaces)";

  m.def("cf1pdf", &cf1pdf,
        py::arg("dx"),
        py::arg("alpha"),
        py::arg("rate"),
        py::arg("eps") = 1.0e-8,
        py::arg("ufactor") = 1.01,
        py::arg("log") = false);

  m.def("cf1cdf", &cf1cdf,
        py::arg("dx"),
        py::arg("alpha"),
        py::arg("rate"),
        py::arg("eps") = 1.0e-8,
        py::arg("ufactor") = 1.01,
        py::arg("lower") = true,
        py::arg("log") = false);

  m.def("cf1sample", &cf1sample,
        py::arg("n"),
        py::arg("alpha"),
        py::arg("rate"),
        py::arg("seed") = std::nullopt);

  m.def("cf1reform", &cf1reform,
        py::arg("alpha"),
        py::arg("rate"));

  m.def("cf1sojourn", &cf1sojourn,
        py::arg("alpha"),
        py::arg("rate"),
        py::arg("f"),
        py::arg("b"),
        py::arg("t"),
        py::arg("eps") = 1.0e-8,
        py::arg("ufactor") = 1.01);

  m.def("em_cf1_emstep", &em_cf1_emstep,
        py::arg("params"),
        py::arg("data"),
        py::arg("eps") = 1.0e-8,
        py::arg("ufactor") = 1.01);
}
