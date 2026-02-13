from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np

from .nhpp import NHPPModel
from .data import NHPPData


@dataclass
class EmFitResult:
    initial: np.ndarray
    srm: NHPPModel
    llf: float
    df: int
    convergence: bool
    iter: int
    aerror: float
    rerror: float


def _comp_error_llf(res0: dict, res1: dict) -> np.ndarray:
    """Rの comp_error() 互換: [abs_error, rel_error, signed_diff]."""
    sdiff = float(res1["llf"] - res0["llf"])
    aerror = abs(sdiff)
    denom = abs(float(res0["llf"]))
    rerror = aerror / denom if denom > 0 else np.inf
    return np.array([aerror, rerror, sdiff], dtype=float)


def emfit(
    srm: NHPPModel,
    data: NHPPData,
    initialize: bool = True,
    maxiter: int = 2000,
    reltol: float = 1.0e-6,
    abstol: float = 1.0e-3,
    trace: bool = False,
    printsteps: int = 50,
    **kwargs,
) -> EmFitResult:
    """
    Python版 emfit: EMでMLE推定（R版のemfitに準拠）

    kwargs は model.em_step(params, data, **kwargs) に渡す。
    """
    if initialize:
        init = srm.init_params(data)
        srm.set_params(np.asarray(init, dtype=float))

    iter_ = 1
    conv = False
    initial_param = np.asarray(srm._get_params(), dtype=float).copy()

    res0 = {"param": np.asarray(initial_param, dtype=float), "llf": -np.inf}
    res1 = res0
    error = np.array([np.inf, np.inf, np.inf], dtype=float)

    while True:
        res1 = srm.em_step(res0["param"], data, **kwargs)

        res1 = {
            "param": np.asarray(res1["param"], dtype=float),
            "llf": float(res1["llf"]),
            "total": float(res1.get("total", np.nan)),
            "pdiff": np.asarray(res1.get("pdiff", np.asarray(res1["param"], dtype=float) - res0["param"]), dtype=float),
        }

        error = _comp_error_llf(res0, res1)

        if trace and (iter_ % printsteps == 0):
            print(f"llf={res1['llf']} ({error[2]:.6e}) params=({res1['param']})")

        if not np.isfinite(res1["llf"]):
            warnings.warn(f"LLF becomes +-Inf, NaN or NA: {srm.name} {iter_}")
            res1 = res0
            break

        if error[2] < 0:
            warnings.warn(f"LLF decreases: {srm.name} {iter_} {error[2]:.6e}")

        if (error[0] < abstol) and (error[1] < reltol):
            conv = True
            srm.set_params(res1["param"])
            break

        if iter_ >= maxiter:
            warnings.warn("Did not converge to MLE by max iteration.")
            srm.set_params(res1["param"])
            break

        iter_ += 1
        res0 = res1

    srm.set_data(data)
    srm._fitted = True

    return EmFitResult(
        initial=initial_param,
        srm=srm,
        llf=float(res1["llf"]),
        df=int(srm.df),
        convergence=bool(conv),
        iter=int(iter_),
        aerror=float(error[0]),
        rerror=float(error[1]),
    )
