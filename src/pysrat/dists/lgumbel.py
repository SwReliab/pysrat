"""Log-Gumbel (Frechet/Weibull as min variant) distribution helpers."""
from __future__ import annotations

import numpy as np
from scipy.stats import gumbel_r, gumbel_l

__all__ = ["dlgumbel", "plgumbel", "qlgumbel", "rlgumbel"]


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def _dist(min: bool, loclog: float, scalelog: float):
    return gumbel_l(loc=-loclog, scale=scalelog) if min else gumbel_r(loc=loclog, scale=scalelog)


def dlgumbel(x, loclog: float = 0.0, scalelog: float = 1.0, log: bool = False, min: bool = False):
    """Density of log-Gumbel distribution (max/min)."""
    x = _as_float_array(x)
    loclog = float(loclog)
    scalelog = float(scalelog)
    if scalelog <= 0:
        raise ValueError("scalelog must be > 0.")
    dist = _dist(min, loclog, scalelog)
    pos = x > 0.0
    if log:
        out = np.full_like(x, -np.inf, dtype=float)
        logx = np.log(x[pos])
        out[pos] = dist.logpdf(logx) - logx
        return out
    out = np.zeros_like(x, dtype=float)
    logx = np.log(x[pos])
    out[pos] = dist.pdf(logx) / x[pos]
    return out


def plgumbel(q, loclog: float = 0.0, scalelog: float = 1.0, lower_tail: bool = True, log_p: bool = False, min: bool = False):
    """CDF / upper-tail probability for log-Gumbel distribution (max/min)."""
    q = _as_float_array(q)
    loclog = float(loclog)
    scalelog = float(scalelog)
    if scalelog <= 0:
        raise ValueError("scalelog must be > 0.")
    dist = _dist(min, loclog, scalelog)
    pos = q > 0.0
    if lower_tail:
        v = np.zeros_like(q, dtype=float)
        if np.any(pos):
            v[pos] = dist.cdf(np.log(q[pos]))
        if log_p:
            return np.log(v)
        return v
    v = np.ones_like(q, dtype=float)
    if np.any(pos):
        v[pos] = dist.sf(np.log(q[pos]))
    if log_p:
        return np.log(v)
    return v


def qlgumbel(p, loclog: float = 0.0, scalelog: float = 1.0, lower_tail: bool = True, log_p: bool = False, min: bool = False):
    """Quantile for log-Gumbel distribution (max/min)."""
    p = _as_float_array(p)
    loclog = float(loclog)
    scalelog = float(scalelog)
    if scalelog <= 0:
        raise ValueError("scalelog must be > 0.")
    if log_p:
        p = np.exp(p)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must be in [0,1].")
    dist = _dist(min, loclog, scalelog)
    if lower_tail:
        return np.exp(dist.ppf(p))
    return np.exp(dist.isf(p))


def rlgumbel(n: int, loclog: float = 0.0, scalelog: float = 1.0, min: bool = False, rng: np.random.Generator | None = None):
    """Random variates from log-Gumbel distribution (max/min)."""
    if n < 0:
        raise ValueError("n must be >= 0.")
    loclog = float(loclog)
    scalelog = float(scalelog)
    if scalelog <= 0:
        raise ValueError("scalelog must be > 0.")
    rng = rng if rng is not None else np.random.default_rng()
    dist = _dist(min, loclog, scalelog)
    return np.exp(dist.rvs(size=int(n), random_state=rng))
