"""Log-logistic distribution helpers."""
from __future__ import annotations

import numpy as np
from scipy.stats import logistic

__all__ = ["dllogis", "pllogis", "qllogis", "rllogis"]


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def _dist(locationlog: float, scalelog: float):
    return logistic(loc=locationlog, scale=scalelog)


def dllogis(x, locationlog: float = 0.0, scalelog: float = 1.0, log: bool = False):
    """Density of log-logistic distribution."""
    x = _as_float_array(x)
    locationlog = float(locationlog)
    scalelog = float(scalelog)
    if scalelog <= 0:
        raise ValueError("scalelog must be > 0.")
    dist = _dist(locationlog, scalelog)
    logx = np.full_like(x, -np.inf, dtype=float)
    pos = x > 0.0
    logx[pos] = np.log(x[pos])
    if log:
        out = dist.logpdf(logx) - logx
        return np.where(x <= 0.0, -np.inf, out)
    out = dist.pdf(logx) / x
    return np.where(x <= 0.0, 0.0, out)


def pllogis(q, locationlog: float = 0.0, scalelog: float = 1.0, lower_tail: bool = True, log_p: bool = False):
    """CDF / upper-tail probability for log-logistic distribution."""
    q = _as_float_array(q)
    locationlog = float(locationlog)
    scalelog = float(scalelog)
    if scalelog <= 0:
        raise ValueError("scalelog must be > 0.")
    dist = _dist(locationlog, scalelog)
    logq = np.full_like(q, -np.inf, dtype=float)
    pos = q > 0.0
    logq[pos] = np.log(q[pos])
    if lower_tail:
        v = dist.cdf(logq)
        v = np.where(q <= 0.0, 0.0, v)
        if log_p:
            return np.log(v)
        return v
    v = dist.sf(logq)
    v = np.where(q <= 0.0, 1.0, v)
    if log_p:
        return np.log(v)
    return v


def qllogis(p, locationlog: float = 0.0, scalelog: float = 1.0, lower_tail: bool = True, log_p: bool = False):
    """Quantile for log-logistic distribution."""
    p = _as_float_array(p)
    locationlog = float(locationlog)
    scalelog = float(scalelog)
    if scalelog <= 0:
        raise ValueError("scalelog must be > 0.")
    if log_p:
        p = np.exp(p)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must be in [0,1].")
    dist = _dist(locationlog, scalelog)
    if lower_tail:
        return np.exp(dist.ppf(p))
    return np.exp(dist.isf(p))


def rllogis(n: int, locationlog: float = 0.0, scalelog: float = 1.0, rng: np.random.Generator | None = None):
    """Random variates from log-logistic distribution."""
    if n < 0:
        raise ValueError("n must be >= 0.")
    locationlog = float(locationlog)
    scalelog = float(scalelog)
    if scalelog <= 0:
        raise ValueError("scalelog must be > 0.")
    rng = rng if rng is not None else np.random.default_rng()
    dist = _dist(locationlog, scalelog)
    return np.exp(dist.rvs(size=int(n), random_state=rng))
