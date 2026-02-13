"""Log-normal distribution helpers."""
from __future__ import annotations

import numpy as np
from scipy.stats import lognorm

__all__ = ["dlnorm", "plnorm", "qlnorm", "rlnorm"]


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def _dist(meanlog: float, sdlog: float):
    return lognorm(s=sdlog, scale=np.exp(meanlog))


def dlnorm(x, meanlog: float = 0.0, sdlog: float = 1.0, log: bool = False):
    """Density of log-normal distribution."""
    x = _as_float_array(x)
    meanlog = float(meanlog)
    sdlog = float(sdlog)
    if sdlog <= 0:
        raise ValueError("sdlog must be > 0.")
    dist = _dist(meanlog, sdlog)
    if log:
        out = dist.logpdf(x)
        return np.where(x <= 0.0, -np.inf, out)
    out = dist.pdf(x)
    return np.where(x <= 0.0, 0.0, out)


def plnorm(q, meanlog: float = 0.0, sdlog: float = 1.0, lower_tail: bool = True, log_p: bool = False):
    """CDF / upper-tail probability for log-normal distribution."""
    q = _as_float_array(q)
    meanlog = float(meanlog)
    sdlog = float(sdlog)
    if sdlog <= 0:
        raise ValueError("sdlog must be > 0.")
    dist = _dist(meanlog, sdlog)
    if lower_tail:
        v = dist.cdf(q)
        v = np.where(q <= 0.0, 0.0, v)
        if log_p:
            return np.log(v)
        return v
    v = dist.sf(q)
    v = np.where(q <= 0.0, 1.0, v)
    if log_p:
        return np.log(v)
    return v


def qlnorm(p, meanlog: float = 0.0, sdlog: float = 1.0, lower_tail: bool = True, log_p: bool = False):
    """Quantile for log-normal distribution."""
    p = _as_float_array(p)
    meanlog = float(meanlog)
    sdlog = float(sdlog)
    if sdlog <= 0:
        raise ValueError("sdlog must be > 0.")
    if log_p:
        p = np.exp(p)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must be in [0,1].")
    dist = _dist(meanlog, sdlog)
    if lower_tail:
        return dist.ppf(p)
    return dist.isf(p)


def rlnorm(n: int, meanlog: float = 0.0, sdlog: float = 1.0, rng: np.random.Generator | None = None):
    """Random variates from log-normal distribution."""
    if n < 0:
        raise ValueError("n must be >= 0.")
    meanlog = float(meanlog)
    sdlog = float(sdlog)
    if sdlog <= 0:
        raise ValueError("sdlog must be > 0.")
    rng = rng if rng is not None else np.random.default_rng()
    dist = _dist(meanlog, sdlog)
    return dist.rvs(size=int(n), random_state=rng)
