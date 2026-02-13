from __future__ import annotations

import numpy as np
from scipy.stats import truncnorm


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def _truncnorm_dist(mean, sd):
    a = (0.0 - mean) / sd
    b = np.inf
    return truncnorm(a=a, b=b, loc=mean, scale=sd)


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

    dist = _truncnorm_dist(mean, sd)
    if log:
        return dist.logpdf(x)
    return dist.pdf(x)


def ptnorm(q, mean=0.0, sd=1.0, lower_tail: bool = True, log_p: bool = False):
    """
    CDF / upper-tail probability for Normal(mean, sd) truncated to [0, +inf).
    """
    q = _as_float_array(q)
    mean = float(mean)
    sd = float(sd)
    if sd <= 0:
        raise ValueError("sd must be > 0.")

    dist = _truncnorm_dist(mean, sd)
    if lower_tail:
        v = dist.cdf(q)
    else:
        v = dist.sf(q)
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

    dist = _truncnorm_dist(mean, sd)
    if lower_tail:
        return dist.ppf(p)
    return dist.isf(p)


def rtnorm(n: int, mean=0.0, sd=1.0, rng: np.random.Generator | None = None):
    """
    Random variates from Normal(mean, sd) truncated to [0, +inf).
    """
    if n < 0:
        raise ValueError("n must be >= 0.")
    rng = rng if rng is not None else np.random.default_rng()
    dist = _truncnorm_dist(float(mean), float(sd))
    return dist.rvs(size=int(n), random_state=rng)
