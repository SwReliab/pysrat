"""Truncated logistic distribution at 0."""
from __future__ import annotations

import numpy as np
from scipy.stats import logistic

__all__ = ["dtlogis", "ptlogis", "qtlogis", "rtlogis"]


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def _dist(location: float, scale: float):
    return logistic(loc=location, scale=scale)


def dtlogis(x, location: float = 0.0, scale: float = 1.0, log: bool = False):
    """Density of logistic distribution truncated to [0, +inf)."""
    x = _as_float_array(x)
    location = float(location)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    dist = _dist(location, scale)
    denom = dist.sf(0.0)
    if log:
        out = dist.logpdf(x) - np.log(denom)
        return np.where(x < 0.0, -np.inf, out)
    out = dist.pdf(x) / denom
    return np.where(x < 0.0, 0.0, out)


def ptlogis(q, location: float = 0.0, scale: float = 1.0, lower_tail: bool = True, log_p: bool = False):
    """CDF / upper-tail probability for truncated logistic distribution."""
    q = _as_float_array(q)
    location = float(location)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    dist = _dist(location, scale)
    denom = dist.sf(0.0)
    upper = dist.sf(q) / denom
    if lower_tail:
        v = 1.0 - upper
        v = np.where(q < 0.0, 0.0, v)
    else:
        v = upper
        v = np.where(q < 0.0, 1.0, v)
    if log_p:
        return np.log(v)
    return v


def qtlogis(p, location: float = 0.0, scale: float = 1.0, lower_tail: bool = True, log_p: bool = False):
    """Quantile for truncated logistic distribution."""
    p = _as_float_array(p)
    location = float(location)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1.0 - p
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must be in [0,1].")
    dist = _dist(location, scale)
    pdash = (1.0 - p) * dist.sf(0.0)
    out = dist.isf(pdash)
    return np.maximum(out, 0.0)


def rtlogis(n: int, location: float = 0.0, scale: float = 1.0, rng: np.random.Generator | None = None):
    """Random variates from truncated logistic distribution."""
    if n < 0:
        raise ValueError("n must be >= 0.")
    location = float(location)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    rng = rng if rng is not None else np.random.default_rng()
    u = rng.uniform(0.0, 1.0, size=int(n))
    return qtlogis(u, location=location, scale=scale, lower_tail=True, log_p=False)
