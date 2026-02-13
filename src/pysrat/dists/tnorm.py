from __future__ import annotations

import math
import numpy as np


_SQRT2 = np.sqrt(2.0)
_LOG_SQRT2PI = 0.5 * np.log(2.0 * np.pi)


def _erfc(x):
    vec = np.vectorize(math.erfc, otypes=[float])
    return vec(x)


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def _norm_logpdf(x, mean, sd):
    z = (x - mean) / sd
    return -np.log(sd) - _LOG_SQRT2PI - 0.5 * z * z


def _norm_pdf(x, mean, sd):
    return np.exp(_norm_logpdf(x, mean, sd))


def _norm_sf(x, mean, sd):
    """Survival function P(X > x) for Normal(mean, sd)."""
    z = (x - mean) / sd
    return 0.5 * _erfc(z / _SQRT2)


def _norm_cdf(x, mean, sd):
    return 1.0 - _norm_sf(x, mean, sd)


# ---- Inverse normal CDF (Acklam approximation) ----
# Reference: Peter J. Acklam's approximation (commonly used; good accuracy for double)
def _norm_ppf(p):
    p = _as_float_array(p)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("p must be in (0,1) for norm ppf.")

    # Coefficients in rational approximations
    a = np.array([
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00
    ])
    b = np.array([
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01
    ])
    c = np.array([
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00
    ])
    d = np.array([
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00
    ])

    plow = 0.02425
    phigh = 1.0 - plow

    x = np.empty_like(p)

    # lower region
    m = p < plow
    if np.any(m):
        q = np.sqrt(-2.0 * np.log(p[m]))
        x[m] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        x[m] = -x[m]

    # central region
    m = (p >= plow) & (p <= phigh)
    if np.any(m):
        q = p[m] - 0.5
        r = q*q
        x[m] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)

    # upper region
    m = p > phigh
    if np.any(m):
        q = np.sqrt(-2.0 * np.log(1.0 - p[m]))
        x[m] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

    return x


def _norm_isf(p):
    """Inverse survival: x s.t. P(X > x) = p for standard normal."""
    return _norm_ppf(1.0 - _as_float_array(p))


# -------------------------
# Truncated normal at 0 (left-truncated)
# -------------------------

def dtnorm(x, mean=0.0, sd=1.0, log: bool = False):
    """
    Density of Normal(mean, sd) truncated to [0, +inf).
    """
    x = _as_float_array(x)
    mean = float(mean)
    sd = float(sd)
    if sd <= 0:
        raise ValueError("sd must be > 0.")

    denom = _norm_sf(0.0, mean, sd)
    logv = _norm_logpdf(x, mean, sd) - np.log(denom)

    logv = np.where(x >= 0.0, logv, -np.inf)
    return logv if log else np.exp(logv)


def ptnorm(q, mean=0.0, sd=1.0, lower_tail: bool = True, log_p: bool = False):
    """
    CDF / upper-tail probability for Normal(mean, sd) truncated to [0, +inf).
    """
    q = _as_float_array(q)
    mean = float(mean)
    sd = float(sd)
    if sd <= 0:
        raise ValueError("sd must be > 0.")

    denom = _norm_sf(0.0, mean, sd)
    upper = _norm_sf(q, mean, sd) / denom

    if lower_tail:
        v = 1.0 - upper
        v = np.where(q < 0.0, 0.0, v)
    else:
        v = upper
        v = np.where(q < 0.0, 1.0, v)

    if log_p:
        return np.log(v)
    return v


def qtnorm(p, mean=0.0, sd=1.0, lower_tail: bool = True, log_p: bool = False):
    """
    Quantile for Normal(mean, sd) truncated to [0, +inf).
    """
    p = _as_float_array(p)
    mean = float(mean)
    sd = float(sd)
    if sd <= 0:
        raise ValueError("sd must be > 0.")

    if log_p:
        p = np.exp(p)

    if not lower_tail:
        p = 1.0 - p

    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must be in [0,1].")

    pdash = (1.0 - p) * _norm_sf(0.0, mean, sd)

    out = np.empty_like(pdash)
    m = (pdash > 0.0) & (pdash < 1.0)
    out[m] = mean + sd * _norm_isf(pdash[m])

    out = np.where(pdash <= 0.0, np.inf, out)
    out = np.where(pdash >= 1.0, -np.inf, out)

    out = np.maximum(out, 0.0)
    return out


def rtnorm(n: int, mean=0.0, sd=1.0, rng: np.random.Generator | None = None):
    """
    Random variates from Normal(mean, sd) truncated to [0, +inf).
    """
    if n < 0:
        raise ValueError("n must be >= 0.")
    rng = rng if rng is not None else np.random.default_rng()
    u = rng.uniform(0.0, 1.0, size=int(n))
    return qtnorm(u, mean=mean, sd=sd, lower_tail=True, log_p=False)
