"""Gumbel (max/min) distribution helpers."""
from __future__ import annotations

import numpy as np
from scipy.stats import gumbel_r, gumbel_l

__all__ = ["dgumbel", "pgumbel", "qgumbel", "rgumbel"]


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def _dist(min: bool, loc: float, scale: float):
    return gumbel_l(loc=loc, scale=scale) if min else gumbel_r(loc=loc, scale=scale)


def dgumbel(x, loc: float = 0.0, scale: float = 1.0, log: bool = False, min: bool = False):
    """Density of Gumbel distribution (max/min)."""
    x = _as_float_array(x)
    loc = float(loc)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    dist = _dist(min, loc, scale)
    if log:
        return dist.logpdf(x)
    return dist.pdf(x)


def pgumbel(q, loc: float = 0.0, scale: float = 1.0, lower: bool = True, log: bool = False, min: bool = False):
    """CDF / upper-tail probability for Gumbel distribution (max/min)."""
    q = _as_float_array(q)
    loc = float(loc)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    dist = _dist(min, loc, scale)
    if lower:
        v = dist.cdf(q)
        if log:
            return dist.logcdf(q)
        return v
    v = dist.sf(q)
    if log:
        return dist.logsf(q)
    return v


def qgumbel(p, loc: float = 0.0, scale: float = 1.0, lower: bool = True, log: bool = False, min: bool = False):
    """Quantile for Gumbel distribution (max/min)."""
    p = _as_float_array(p)
    loc = float(loc)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    if log:
        p = np.exp(p)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must be in [0,1].")
    dist = _dist(min, loc, scale)
    if lower:
        return dist.ppf(p)
    return dist.isf(p)


def rgumbel(n: int, loc: float = 0.0, scale: float = 1.0, min: bool = False, rng: np.random.Generator | None = None):
    """Random variates from Gumbel distribution (max/min)."""
    if n < 0:
        raise ValueError("n must be >= 0.")
    loc = float(loc)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    rng = rng if rng is not None else np.random.default_rng()
    dist = _dist(min, loc, scale)
    return dist.rvs(size=int(n), random_state=rng)
