from __future__ import annotations

import numpy as np
from scipy.stats import expon


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def dexp(x, rate: float = 1.0, log: bool = False):
    """
    Density of Exponential(rate).
    """
    x = _as_float_array(x)
    rate = float(rate)
    if rate <= 0:
        raise ValueError("rate must be > 0.")
    scale = 1.0 / rate
    if log:
        return expon.logpdf(x, scale=scale)
    return expon.pdf(x, scale=scale)


def pexp(q, rate: float = 1.0, lower_tail: bool = True, log_p: bool = False):
    """
    CDF / upper-tail probability for Exponential(rate).
    """
    q = _as_float_array(q)
    rate = float(rate)
    if rate <= 0:
        raise ValueError("rate must be > 0.")
    scale = 1.0 / rate
    if lower_tail:
        v = expon.cdf(q, scale=scale)
    else:
        v = expon.sf(q, scale=scale)
    if log_p:
        return np.log(v)
    return v


def qexp(p, rate: float = 1.0, lower_tail: bool = True, log_p: bool = False):
    """
    Quantile for Exponential(rate).
    """
    p = _as_float_array(p)
    rate = float(rate)
    if rate <= 0:
        raise ValueError("rate must be > 0.")
    if log_p:
        p = np.exp(p)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must be in [0,1].")
    scale = 1.0 / rate
    if lower_tail:
        return expon.ppf(p, scale=scale)
    return expon.isf(p, scale=scale)


def rexp(n: int, rate: float = 1.0, rng: np.random.Generator | None = None):
    """
    Random variates from Exponential(rate).
    """
    if n < 0:
        raise ValueError("n must be >= 0.")
    rate = float(rate)
    if rate <= 0:
        raise ValueError("rate must be > 0.")
    rng = rng if rng is not None else np.random.default_rng()
    scale = 1.0 / rate
    return expon.rvs(scale=scale, size=int(n), random_state=rng)
