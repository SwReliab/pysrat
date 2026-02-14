from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None

from .data import NHPPData
from .base import NHPPModel

__all__ = []


@dataclass
class EmFitResult:
    initial: np.ndarray
    llf: float
    convergence: bool
    iter: int
    aerror: float
    rerror: float


def _comp_error_llf(res0: dict, res1: dict) -> np.ndarray:
    sdiff = float(res1["llf"] - res0["llf"])
    aerror = abs(sdiff)
    denom = abs(float(res0["llf"]))
    rerror = aerror / denom if denom > 0 else np.inf
    return np.array([aerror, rerror, sdiff], dtype=float)


def emfit_internal(
    model: NHPPModel,
    data: NHPPData,
    *,
    initialize: bool = True,
    maxiter: int = 2000,
    reltol: float = 1.0e-6,
    abstol: float = 1.0e-3,
    trace: bool = False,
    verbose: bool = False,
    printsteps: int = 50,
    **kwargs,
) -> EmFitResult:
    if initialize:
        init = model.init_params(data)
        model._set_fitted_params(np.asarray(init, dtype=float))

    iter_ = 1
    conv = False
    initial_param = np.asarray(model._get_fitted_params(), dtype=float).copy()

    res0 = {"param": initial_param, "llf": -np.inf}
    res1 = res0
    error = np.array([np.inf, np.inf, np.inf], dtype=float)

    pbar = None
    if verbose:
        if _tqdm is None:
            warnings.warn("tqdm is not installed; set verbose=False or install tqdm to see a progress bar.")
        else:
            pbar = _tqdm(total=maxiter, desc=f"{model.name} fit", leave=False)

    while True:
        step = model.em_step(res0["param"], data, **kwargs)
        res1 = {
            "param": np.asarray(step["param"], dtype=float),
            "llf": float(step["llf"]),
            "pdiff": np.asarray(step.get("pdiff", np.asarray(step["param"], float) - res0["param"]), dtype=float),
        }
        error = _comp_error_llf(res0, res1)

        if pbar is not None:
            pbar.update(1)

        if trace and (iter_ % printsteps == 0):
            print(f"llf={res1['llf']} ({error[2]:.6e}) params=({res1['param']})")

        if not np.isfinite(res1["llf"]):
            warnings.warn(f"LLF becomes +-Inf, NaN or NA: {model.name} {iter_}")
            res1 = res0
            break

        if error[2] < 0:
            warnings.warn(f"LLF decreases: {model.name} {iter_} {error[2]:.6e}")

        if (error[0] < abstol) and (error[1] < reltol):
            conv = True
            model._set_fitted_params(res1["param"])
            break

        if iter_ >= maxiter:
            warnings.warn("Did not converge to MLE by max iteration.")
            model._set_fitted_params(res1["param"])
            break

        iter_ += 1
        res0 = res1

    if pbar is not None:
        pbar.n = min(iter_, maxiter)
        pbar.refresh()
        pbar.close()

    model.set_data(data)
    return EmFitResult(
        initial=initial_param,
        llf=float(res1["llf"]),
        convergence=bool(conv),
        iter=int(iter_),
        aerror=float(error[0]),
        rerror=float(error[1]),
    )
