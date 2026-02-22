from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
import warnings
import numpy as np

from ..data import NHPPData
from ._base import NHPPModel


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
    printsteps: int = 50,
    debug_dir: str | None = None,
    strict: bool = False,
    callback: Optional[Callable[..., None]] = None,
    **kwargs,
) -> EmFitResult:
    """
    EM fitting routine with user callback.

    callback signature (recommended):
        callback(
            *,
            event: str,
            model: NHPPModel,
            iter: int,
            llf: float,
            aerror: float,
            rerror: float,
            sdiff: float,
            param: np.ndarray,
            pdiff: np.ndarray,
            converged: bool,
        ) -> None
    """
    import json
    import os

    def _dump_state(tag: str, param, llf_val):
        if debug_dir is None:
            return
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, f"{model.name}_{tag}.json")
        payload = {
            "tag": tag,
            "param": np.asarray(param, dtype=float).tolist(),
            "llf": float(llf_val),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _safe_callback(**info: Any):
        if callback is None:
            return
        try:
            callback(**info)
        except Exception as exc:  # callback must never break fitting
            warnings.warn(f"Exception in callback: {exc}")

    # ---- initialize ----
    if initialize:
        init = model.init_params(data)
        model._set_fitted_params(np.asarray(init, dtype=float))

    iter_ = 1
    conv = False

    initial_param = np.asarray(model._get_fitted_params(), dtype=float).copy()
    res0 = {"param": initial_param, "llf": -np.inf}
    res1 = res0
    error = np.array([np.inf, np.inf, np.inf], dtype=float)

    # initial callback
    _safe_callback(
        event="start",
        model=model,
        iter=0,
        llf=float(res0["llf"]),
        aerror=float(error[0]),
        rerror=float(error[1]),
        sdiff=float(error[2]),
        param=np.asarray(res0["param"], dtype=float).copy(),
        pdiff=np.zeros_like(np.asarray(res0["param"], dtype=float)),
        converged=False,
    )

    # ---- loop ----
    while True:
        try:
            step = model.em_step(res0["param"], data, **kwargs)
        except Exception as exc:
            _dump_state(f"exception_iter{iter_}", res0["param"], res0["llf"])
            _safe_callback(
                event="exception",
                model=model,
                iter=iter_,
                llf=float(res0["llf"]),
                aerror=float(error[0]),
                rerror=float(error[1]),
                sdiff=float(error[2]),
                param=np.asarray(res0["param"], dtype=float).copy(),
                pdiff=np.zeros_like(np.asarray(res0["param"], dtype=float)),
                converged=False,
                exception=exc,
            )
            if strict:
                raise
            warnings.warn(f"Exception in EM step: {exc}")
            break

        new_param = np.asarray(step["param"], dtype=float)
        new_llf = float(step["llf"])

        pdiff = np.asarray(step.get("pdiff", new_param - np.asarray(res0["param"], dtype=float)), dtype=float)

        res1 = {
            "param": new_param,
            "llf": new_llf,
            "pdiff": pdiff,
        }

        # ---- sanity checks ----
        if not np.all(np.isfinite(new_param)):
            _dump_state(f"param_nan_iter{iter_}", new_param, new_llf)
            _safe_callback(
                event="param_nan",
                model=model,
                iter=iter_,
                llf=float(new_llf),
                aerror=float(error[0]),
                rerror=float(error[1]),
                sdiff=float(error[2]),
                param=new_param.copy(),
                pdiff=pdiff.copy(),
                converged=False,
            )
            if strict:
                raise FloatingPointError("Non-finite parameter in EM")
            warnings.warn(f"Non-finite parameter detected at iter {iter_}")
            break

        if not np.isfinite(new_llf):
            _dump_state(f"llf_nan_iter{iter_}", new_param, new_llf)
            _safe_callback(
                event="llf_nan",
                model=model,
                iter=iter_,
                llf=float(new_llf),
                aerror=float(error[0]),
                rerror=float(error[1]),
                sdiff=float(error[2]),
                param=new_param.copy(),
                pdiff=pdiff.copy(),
                converged=False,
            )
            if strict:
                raise FloatingPointError("LLF became NaN/Inf")
            warnings.warn(f"LLF becomes NaN/Inf at iter {iter_}")
            break

        # ---- errors ----
        error = _comp_error_llf(res0, res1)

        # optional trace print
        if trace and (iter_ % printsteps == 0):
            print(f"llf={res1['llf']} ({error[2]:.6e}) params=({res1['param']})")

        if error[2] < 0:
            warnings.warn(f"LLF decreases: {model.name} {iter_} {error[2]:.6e}")

        # ---- callback (per-iter) ----
        _safe_callback(
            event="iter",
            model=model,
            iter=iter_,
            llf=float(res1["llf"]),
            aerror=float(error[0]),
            rerror=float(error[1]),
            sdiff=float(error[2]),
            param=np.asarray(res1["param"], dtype=float).copy(),
            pdiff=np.asarray(res1["pdiff"], dtype=float).copy(),
            converged=False,
        )

        # ---- convergence ----
        if (error[0] < abstol) and (error[1] < reltol):
            conv = True
            model._set_fitted_params(res1["param"])
            _safe_callback(
                event="converged",
                model=model,
                iter=iter_,
                llf=float(res1["llf"]),
                aerror=float(error[0]),
                rerror=float(error[1]),
                sdiff=float(error[2]),
                param=np.asarray(res1["param"], dtype=float).copy(),
                pdiff=np.asarray(res1["pdiff"], dtype=float).copy(),
                converged=True,
            )
            break

        if iter_ >= maxiter:
            warnings.warn("Did not converge to MLE by max iteration.")
            model._set_fitted_params(res1["param"])
            _safe_callback(
                event="maxiter",
                model=model,
                iter=iter_,
                llf=float(res1["llf"]),
                aerror=float(error[0]),
                rerror=float(error[1]),
                sdiff=float(error[2]),
                param=np.asarray(res1["param"], dtype=float).copy(),
                pdiff=np.asarray(res1["pdiff"], dtype=float).copy(),
                converged=False,
            )
            break

        iter_ += 1
        res0 = res1

    model.set_data(data)

    # final callback
    _safe_callback(
        event="done",
        model=model,
        iter=int(iter_),
        llf=float(res1["llf"]),
        aerror=float(error[0]),
        rerror=float(error[1]),
        sdiff=float(error[2]),
        param=np.asarray(res1["param"], dtype=float).copy(),
        pdiff=np.asarray(res1.get("pdiff", np.zeros_like(res1["param"])), dtype=float).copy(),
        converged=bool(conv),
    )

    return EmFitResult(
        initial=initial_param,
        llf=float(res1["llf"]),
        convergence=bool(conv),
        iter=int(iter_),
        aerror=float(error[0]),
        rerror=float(error[1]),
    )