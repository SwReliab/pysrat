"""Gamma distribution helpers."""
from __future__ import annotations

import numpy as np
from scipy.stats import gamma as _gamma

__all__ = ["dgamma", "pgamma", "qgamma", "rgamma"]


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def _dist(shape: float, rate: float):
    return _gamma(a=shape, scale=1.0 / rate)


def dgamma(x, shape: float = 1.0, rate: float = 1.0, log: bool = False):
    """Density of Gamma(shape, rate)."""
    x = _as_float_array(x)
    shape = float(shape)
    rate = float(rate)
    if shape <= 0:
        raise ValueError("shape must be > 0.")
    if rate <= 0:
        raise ValueError("rate must be > 0.")
    dist = _dist(shape, rate)
    if log:
        out = dist.logpdf(x)
        return np.where(x <= 0.0, -np.inf, out)
    out = dist.pdf(x)
    return np.where(x <= 0.0, 0.0, out)


def pgamma(q, shape: float = 1.0, rate: float = 1.0, lower_tail: bool = True, log_p: bool = False):
    """CDF / upper-tail probability for Gamma(shape, rate)."""
    q = _as_float_array(q)
    shape = float(shape)
    rate = float(rate)
    if shape <= 0:
        raise ValueError("shape must be > 0.")
    if rate <= 0:
        raise ValueError("rate must be > 0.")
    dist = _dist(shape, rate)
    if lower_tail:
        v = dist.cdf(q)
        v = np.where(q <= 0.0, 0.0, v)
    else:
        v = dist.sf(q)
        v = np.where(q <= 0.0, 1.0, v)
    if log_p:
        return np.log(v)
    return v


def qgamma(p, shape: float = 1.0, rate: float = 1.0, lower_tail: bool = True, log_p: bool = False):
    """Quantile for Gamma(shape, rate)."""
    p = _as_float_array(p)
    shape = float(shape)
    rate = float(rate)
    if shape <= 0:
        raise ValueError("shape must be > 0.")
    if rate <= 0:
        raise ValueError("rate must be > 0.")
    if log_p:
        p = np.exp(p)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must be in [0,1].")
    dist = _dist(shape, rate)
    if lower_tail:
        return dist.ppf(p)
    return dist.isf(p)


def rgamma(n: int, shape: float = 1.0, rate: float = 1.0, rng: np.random.Generator | None = None):
    """Random variates from Gamma(shape, rate)."""
    if n < 0:
        raise ValueError("n must be >= 0.")
    shape = float(shape)
    rate = float(rate)
    if shape <= 0:
        raise ValueError("shape must be > 0.")
    if rate <= 0:
        raise ValueError("rate must be > 0.")
    rng = rng if rng is not None else np.random.default_rng()
    dist = _dist(shape, rate)
    return dist.rvs(size=int(n), random_state=rng)
