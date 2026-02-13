"""Truncated Gumbel (max/min) distribution at 0."""
from __future__ import annotations

import numpy as np
from scipy.stats import gumbel_r, gumbel_l

__all__ = ["dtgumbel", "ptgumbel", "qtgumbel", "rtgumbel"]


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def _dist(min: bool, loc: float, scale: float):
    return gumbel_l(loc=loc, scale=scale) if min else gumbel_r(loc=loc, scale=scale)


def dtgumbel(x, loc: float = 0.0, scale: float = 1.0, log: bool = False, min: bool = False):
    """Density of Gumbel distribution truncated to [0, +inf) (max/min)."""
    x = _as_float_array(x)
    loc = float(loc)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    dist = _dist(min, loc, scale)
    denom = dist.sf(0.0)
    if log:
        out = dist.logpdf(x) - np.log(denom)
        return np.where(x < 0.0, -np.inf, out)
    out = dist.pdf(x) / denom
    return np.where(x < 0.0, 0.0, out)


def ptgumbel(q, loc: float = 0.0, scale: float = 1.0, lower_tail: bool = True, log_p: bool = False, min: bool = False):
    """CDF / upper-tail probability for truncated Gumbel distribution (max/min)."""
    q = _as_float_array(q)
    loc = float(loc)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    dist = _dist(min, loc, scale)
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


def qtgumbel(p, loc: float = 0.0, scale: float = 1.0, lower_tail: bool = True, log_p: bool = False, min: bool = False):
    """Quantile for truncated Gumbel distribution (max/min)."""
    p = _as_float_array(p)
    loc = float(loc)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1.0 - p
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must be in [0,1].")
    dist = _dist(min, loc, scale)
    pdash = (1.0 - p) * dist.sf(0.0)
    out = dist.isf(pdash)
    return np.maximum(out, 0.0)


def rtgumbel(n: int, loc: float = 0.0, scale: float = 1.0, min: bool = False, rng: np.random.Generator | None = None):
    """Random variates from truncated Gumbel distribution (max/min)."""
    if n < 0:
        raise ValueError("n must be >= 0.")
    loc = float(loc)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    rng = rng if rng is not None else np.random.default_rng()
    u = rng.uniform(0.0, 1.0, size=int(n))
    return qtgumbel(u, loc=loc, scale=scale, lower_tail=True, log_p=False, min=min)
