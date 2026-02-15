from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from .. import marlib_cf1 as _cf1
from ..data import NHPPData
from ..models.cf1 import CanonicalPhaseTypeNHPP

__all__ = [
    "dcf1",
    "pcf1",
    "qcf1",
    "rcf1",
    "cf1_params_power",
    "cf1_params_linear",
    "cf1_params_init",
    "cf1llf",
]


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def _restore_order(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    out = np.empty_like(values)
    out[order] = values
    return out


def dcf1(x, alpha=1.0, rate=1.0, log: bool = False, eps: float = 1.0e-8, unif_factor: float = 1.01):
    x = _as_float_array(x)
    if x.size == 0:
        return x
    s = np.argsort(x)
    dx = np.diff(np.concatenate([[0.0], x[s]]))
    res = _cf1.cf1pdf(dx, _as_float_array(alpha), _as_float_array(rate), eps, unif_factor, log)
    return _restore_order(np.asarray(res, dtype=float), s)


def pcf1(
    q,
    alpha=1.0,
    rate=1.0,
    lower_tail: bool = True,
    log_p: bool = False,
    eps: float = 1.0e-8,
    unif_factor: float = 1.01,
):
    q = _as_float_array(q)
    if q.size == 0:
        return q
    s = np.argsort(q)
    dx = np.diff(np.concatenate([[0.0], q[s]]))
    res = _cf1.cf1cdf(dx, _as_float_array(alpha), _as_float_array(rate), eps, unif_factor, lower_tail, log_p)
    return _restore_order(np.asarray(res, dtype=float), s)


def qcf1(p, alpha=1.0, rate=1.0, lower_tail: bool = True, log_p: bool = False):
    raise NotImplementedError("qcf1 is not implemented")


def rcf1(n: int, alpha=1.0, rate=1.0, scramble: bool = True):
    if n < 0:
        raise ValueError("n must be >= 0")
    res = _cf1.cf1sample(int(n), _as_float_array(alpha), _as_float_array(rate))
    res = np.asarray(res, dtype=float)
    if scramble:
        rng = np.random.default_rng()
        return rng.permutation(res)
    return res


def cf1_params_power(n: int, scale: float, shape: float) -> dict:
    n = int(n)
    rate = np.zeros(n, dtype=float)
    p = np.exp(1.0 / (n - 1) * np.log(shape))
    total = 1.0
    tmp = 1.0
    for i in range(1, n):
        tmp = tmp * (i + 1) / (i * p)
        total += tmp
    base = total / (n * scale)
    tmp = base
    for i in range(n):
        rate[i] = tmp
        tmp *= p
    return {"alpha": np.full(n, 1.0 / n, dtype=float), "rate": rate}


def cf1_params_linear(n: int, scale: float, shape: float) -> dict:
    n = int(n)
    rate = np.zeros(n, dtype=float)
    al = (shape - 1.0) / (n - 1)
    total = 1.0
    for i in range(1, n + 1):
        total += (i + 1) / (al * i + 1.0)
    base = total / (n * scale)
    for i in range(1, n + 1):
        rate[i - 1] = base * (al + i + 1.0)
    return {"alpha": np.full(n, 1.0 / n, dtype=float), "rate": rate}


def cf1_params_init(
    n: int,
    data: NHPPData,
    shape_init=(1, 4, 16, 64, 256, 1024),
    scale_init=(0.5, 1.0, 2.0),
    maxiter_init: int = 5,
    verbose: bool = False,
) -> dict | None:
    if verbose:
        print("Initializing CF1 ...")

    m = float(data.mean)
    max_llf = -np.inf
    best = None

    model = CanonicalPhaseTypeNHPP(n)

    for fn in (cf1_params_power, cf1_params_linear):
        for x in scale_init:
            for s in shape_init:
                param = {"omega": float(data.total), **fn(n, scale=m * float(x), shape=float(s))}
                try:
                    res_param = param
                    for _ in range(int(maxiter_init)):
                        params_vec = np.concatenate(
                            [
                                [float(res_param["omega"])],
                                np.asarray(res_param["alpha"], dtype=float),
                                np.asarray(res_param["rate"], dtype=float),
                            ]
                        )
                        step = model.em_step(params_vec, data)
                        new_params = np.asarray(step["param"], dtype=float)
                        res_param = {
                            "omega": float(new_params[0]),
                            "alpha": np.asarray(new_params[1 : 1 + n], dtype=float),
                            "rate": np.asarray(new_params[1 + n : 1 + 2 * n], dtype=float),
                        }
                    llf = float(step["llf"])
                except Exception:
                    if verbose:
                        print("-", end="")
                    continue

                if np.isfinite(llf):
                    if llf > max_llf:
                        max_llf = llf
                        best = res_param
                        if verbose:
                            print("o", end="")
                    else:
                        if verbose:
                            print("x", end="")
                else:
                    if verbose:
                        print("-", end="")
        if verbose:
            print()

    return best


def cf1llf(data: NHPPData, omega: float, alpha, rate) -> float:
    omega = float(omega)
    alpha = _as_float_array(alpha)
    rate = _as_float_array(rate)

    te = float(np.sum(data.time))
    llf = -omega * float(pcf1(te, alpha=alpha, rate=rate, lower_tail=True))

    tt = np.asarray(data.time, dtype=float)[np.asarray(data.type, dtype=int) == 1]
    if tt.size != 0:
        llf += np.sum(np.log(omega * dcf1(np.cumsum(tt), alpha=alpha, rate=rate)))

    gt = np.asarray(data.time, dtype=float)[np.asarray(data.type, dtype=int) == 0]
    if gt.size != 0:
        barp = -np.diff(pcf1(np.concatenate([[0.0], np.cumsum(gt)]),
                             alpha=alpha,
                             rate=rate,
                             lower_tail=False))
        faults = np.asarray(data.fault, dtype=float)
        i = faults != 0
        if np.any(i):
            llf += np.sum(faults[i] * np.log(omega * barp[i]) - gammaln(faults[i] + 1.0))

    return float(llf)
