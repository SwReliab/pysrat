from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .data import NHPPData
from .nhpp import NHPPModel


@dataclass
class FitEMResult:
    name: str
    params: np.ndarray
    llf: float
    total: float
    converged: bool
    n_iter: int
    history: list  # list[dict]


def fit_em(
    model: NHPPModel,
    data: NHPPData,
    init: np.ndarray | None = None,
    max_iter: int = 200,
    tol_param: float = 1e-8,
    tol_llf: float = 1e-8,
) -> FitEMResult:
    if init is None:
        params = model.init_params(data)
    else:
        params = np.asarray(init, dtype=float)

    history: list[dict] = []
    prev_llf = None
    converged = False
    last = None

    for it in range(1, max_iter + 1):
        res = model.em_step(params, data)

        new_params = np.asarray(res["param"], dtype=float)
        pdiff = np.asarray(res.get("pdiff", new_params - params), dtype=float)
        llf = float(res["llf"])
        total = float(res.get("total", np.nan))

        history.append(
            {
                "iter": it,
                "params": new_params.copy(),
                "pdiff": pdiff.copy(),
                "llf": llf,
                "total": total,
            }
        )

        param_norm = float(np.linalg.norm(pdiff, ord=2))
        llf_diff = None if prev_llf is None else abs(llf - prev_llf)

        if prev_llf is not None:
            if (param_norm <= tol_param) and (llf_diff <= tol_llf):
                converged = True
                params = new_params
                last = (llf, total)
                break

        prev_llf = llf
        params = new_params
        last = (llf, total)

    model.set_params(params)
    model._fitted = True

    llf, total = last if last is not None else (float("nan"), float("nan"))
    return FitEMResult(
        name=model.name,
        params=params,
        llf=llf,
        total=total,
        converged=converged,
        n_iter=len(history),
        history=history,
    )


def compare(models: list[NHPPModel], data: NHPPData, *, criterion: str = "AIC"):
    fitted = [m.fit(data) for m in models]
    crit = criterion.upper()
    if crit != "AIC":
        raise ValueError(f"Unsupported criterion: {criterion}")
    best = min(fitted, key=lambda m: m.aic_)
    return fitted, best
