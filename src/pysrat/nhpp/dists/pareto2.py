"""Pareto type II (Lomax) distribution helpers."""
from __future__ import annotations

import numpy as np
from scipy.stats import lomax

__all__ = ["dpareto2", "ppareto2", "qpareto2", "rpareto2"]


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def _dist(shape: float, scale: float):
    return lomax(c=shape, scale=scale, loc=0.0)


def dpareto2(x, shape: float = 1.0, scale: float = 1.0, log: bool = False):
    """Density of Pareto type II (Lomax)."""
    x = _as_float_array(x)
    shape = float(shape)
    scale = float(scale)
    if shape <= 0 or scale <= 0:
        raise ValueError("shape and scale must be > 0.")
    dist = _dist(shape, scale)
    if log:
        out = dist.logpdf(x)
        return np.where(x < 0.0, -np.inf, out)
    out = dist.pdf(x)
    return np.where(x < 0.0, 0.0, out)


def ppareto2(q, shape: float = 1.0, scale: float = 1.0, lower_tail: bool = True, log_p: bool = False):
    """CDF / upper-tail probability for Pareto type II (Lomax)."""
    q = _as_float_array(q)
    shape = float(shape)
    scale = float(scale)
    if shape <= 0 or scale <= 0:
        raise ValueError("shape and scale must be > 0.")
    dist = _dist(shape, scale)
    if lower_tail:
        v = dist.cdf(q)
        v = np.where(q < 0.0, 0.0, v)
        if log_p:
            return np.log(v)
        return v
    v = dist.sf(q)
    v = np.where(q < 0.0, 1.0, v)
    if log_p:
        return np.log(v)
    return v


def qpareto2(p, shape: float = 1.0, scale: float = 1.0, lower_tail: bool = True, log_p: bool = False):
    """Quantile for Pareto type II (Lomax)."""
    p = _as_float_array(p)
    shape = float(shape)
    scale = float(scale)
    if shape <= 0 or scale <= 0:
        raise ValueError("shape and scale must be > 0.")
    if log_p:
        p = np.exp(p)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must be in [0,1].")
    dist = _dist(shape, scale)
    if lower_tail:
        return dist.ppf(p)
    return dist.isf(p)


def rpareto2(n: int, shape: float = 1.0, scale: float = 1.0, rng: np.random.Generator | None = None):
    """Random variates from Pareto type II (Lomax)."""
    if n < 0:
        raise ValueError("n must be >= 0.")
    shape = float(shape)
    scale = float(scale)
    if shape <= 0 or scale <= 0:
        raise ValueError("shape and scale must be > 0.")
    rng = rng if rng is not None else np.random.default_rng()
    dist = _dist(shape, scale)
    return dist.rvs(size=int(n), random_state=rng)
