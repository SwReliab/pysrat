#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <stdexcept>
#include <limits>

namespace py = pybind11;

namespace tnorm {

constexpr double kSqrt2 = 1.4142135623730950488016887242096980785696718753769;
constexpr double kInvSqrt2Pi = 0.3989422804014326779399460599343818684758586311649;

inline double phi(double z) {
    return kInvSqrt2Pi * std::exp(-0.5 * z * z);
}

inline double Q(double z) {
    return 0.5 * std::erfc(z / kSqrt2);
}

inline double func_barFi(double t, double mu, double sig) {
    const double z = (t - mu) / sig;
    return Q(z);
}

inline double func_barPhi1i(double t, double mu, double sig) {
    const double z = (t - mu) / sig;
    return sig * phi(z) + mu * Q(z);
}

inline double func_barPhi2i(double t, double mu, double sig) {
    const double z = (t - mu) / sig;
    return (sig * t + sig * mu) * phi(z) + (sig * sig + mu * mu) * Q(z);
}

inline double log_norm_pdf(double t, double mu, double sig) {
    const double z = (t - mu) / sig;
    return -std::log(sig) - 0.5 * std::log(2.0 * M_PI) - 0.5 * z * z;
}

} // namespace tnorm

py::dict em_tnorm_emstep(
    py::array_t<double, py::array::c_style | py::array::forcecast> params,
    py::dict data
) {
    auto p = params.unchecked<1>();
    if (p.shape(0) < 3) {
        throw std::invalid_argument("params must have length >= 3: [omega, mu, sig].");
    }
    const double omega = p(0);
    const double mu    = p(1);
    const double sig   = p(2);

    if (!(sig > 0.0) || !std::isfinite(sig)) {
        throw std::invalid_argument("sig must be positive and finite.");
    }

    if (!data.contains("len") || !data.contains("time") || !data.contains("fault") || !data.contains("type")) {
        throw std::invalid_argument("data must contain keys: 'len', 'time', 'fault', 'type'.");
    }

    const int dsize = py::cast<int>(data["len"]);

    auto time = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(data["time"]);
    auto num  = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(data["fault"]);
    auto type = py::cast<py::array_t<long long, py::array::c_style | py::array::forcecast>>(data["type"]);

    if (time.ndim() != 1 || num.ndim() != 1 || type.ndim() != 1) {
        throw std::invalid_argument("time/fault/type must be 1-D arrays.");
    }
    if (dsize != static_cast<int>(time.shape(0)) || dsize != static_cast<int>(num.shape(0)) || dsize != static_cast<int>(type.shape(0))) {
        throw std::invalid_argument("Invalid data: len != lengths of time/fault/type.");
    }

    auto time_u = time.unchecked<1>();
    auto num_u  = num.unchecked<1>();
    auto type_u = type.unchecked<1>();

    const double barF0    = tnorm::func_barFi(0.0, mu, sig);
    const double barPhi10 = tnorm::func_barPhi1i(0.0, mu, sig);
    const double barPhi20 = tnorm::func_barPhi2i(0.0, mu, sig);

    if (!(barF0 > 0.0) || !std::isfinite(barF0)) {
        throw std::runtime_error("barF0 is non-positive or non-finite (numerical issue).\n");
    }

    double nn  = 0.0;
    double en1 = 0.0;
    double en2 = 0.0;
    double en3 = 0.0;
    double llf = 0.0;

    double t = 0.0;
    double prev_barFi    = barF0;
    double prev_barPhi1i = barPhi10;
    double prev_barPhi2i = barPhi20;

    for (int i = 0; i < dsize; i++) {
        t += time_u(i);

        const double barFi    = tnorm::func_barFi(t, mu, sig);
        const double barPhi1i = tnorm::func_barPhi1i(t, mu, sig);
        const double barPhi2i = tnorm::func_barPhi2i(t, mu, sig);

        const double x = num_u(i);
        if (x != 0.0) {
            const double tmp1 = prev_barFi - barFi;
            const double tmp2 = prev_barPhi1i - barPhi1i;
            const double tmp3 = prev_barPhi2i - barPhi2i;

            if (!(tmp1 > 0.0) || !std::isfinite(tmp1)) {
                llf = std::numeric_limits<double>::quiet_NaN();
            } else {
                nn  += x;
                en1 += x;
                en2 += x * tmp2 / tmp1;
                en3 += x * tmp3 / tmp1;
                llf += x * std::log(tmp1) - std::lgamma(x + 1.0);
            }
        }

        if (type_u(i) == 1) {
            nn  += 1.0;
            en1 += 1.0;
            en2 += t;
            en3 += t * t;
            llf += tnorm::log_norm_pdf(t, mu, sig);
        }

        prev_barFi    = barFi;
        prev_barPhi1i = barPhi1i;
        prev_barPhi2i = barPhi2i;
    }

    llf += nn * (std::log(omega) - std::log(barF0)) - omega * (barF0 - prev_barFi) / barF0;

    en1 += omega * prev_barFi / barF0;
    en2 += omega * prev_barPhi1i / barF0;
    en3 += omega * prev_barPhi2i / barF0;

    const double en1dash = en1 + en1 * (1.0 - barF0) / barF0;
    en2 += en1 * (mu - barPhi10) / barF0;
    en3 += en1 * (sig * sig + mu * mu - barPhi20) / barF0;

    const double new_mu = en2 / en1dash;
    const double var = en3 / en1dash - new_mu * new_mu;
    const double new_sig = (var > 0.0) ? std::sqrt(var) : 0.0;

    const double new_omega = en1;
    const double total = en1;

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

    out["llf"]   = llf;
    out["total"] = total;
    return out;
}
