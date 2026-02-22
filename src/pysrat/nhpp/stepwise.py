from __future__ import annotations

from typing import Literal

import numpy as np

from ..data import DMetricsData


def stepwise(
    model,
    *,
    data=None,
    direction: Literal["both", "forward", "backward"] = "both",
    max_steps: int = 100,
    keep_intercept: bool = True,
    allow_empty: bool = False,   # ★追加：空モデルを許すか
    verbose: bool = False,
    **fit_kwargs,
):
    """AIC-based stepwise selection for dGLM models (NO intercept column in X)."""

    if data is None:
        data = getattr(model, "data_", None)
        if data is None:
            raise ValueError("No data provided and model.data_ is not set.")

    X = np.asarray(getattr(data, "metrics"), dtype=float)
    n, p = X.shape

    names = list(getattr(data, "metrics_name", [f"m{i+1}" for i in range(p)]))
    if len(names) != p:
        names = [f"m{i+1}" for i in range(p)]

    offset = np.asarray(getattr(data, "offset"), dtype=float)
    fault = np.asarray(getattr(data, "fault"), dtype=float)
    total = getattr(data, "total", None)

    all_idxs = list(range(p))

    def _build_data(cols: list[int]) -> DMetricsData:
        cols = list(cols)
        if (not allow_empty) and len(cols) == 0:
            raise ValueError("Empty model is disabled (allow_empty=False).")
        metrics_sub = X[:, cols]
        names_sub = [names[i] for i in cols]
        return DMetricsData(
            metrics=metrics_sub,
            offset=offset,
            fault=fault,
            metrics_name=names_sub,
            total=total,
        )

    def _fit_cols(cols: list[int]):
        new_model = model.__class__(**model.get_params())
        if hasattr(new_model, "has_intercept"):
            new_model.has_intercept = bool(keep_intercept)
        dsub = _build_data(cols)
        new_model.fit(dsub, **fit_kwargs)
        return new_model

    # ---- initialize ----
    if direction == "backward":
        best_cols = list(all_idxs)
        if (not allow_empty) and len(best_cols) == 0:
            raise ValueError("No predictors available (p=0) and allow_empty=False.")
    else:
        best_cols = [] if allow_empty else ([0] if p > 0 else [])
        if (not allow_empty) and len(best_cols) == 0:
            raise ValueError("No predictors available (p=0) and allow_empty=False.")

    best_model = None
    best_aic = float("inf")

    # fit initial
    m0 = _fit_cols(best_cols)
    best_model = m0
    best_aic = float(m0.aic_)

    steps = 0
    improved = True

    while improved and steps < max_steps:
        steps += 1
        improved = False

        # forward
        if direction in ("forward", "both"):
            candidates = [i for i in all_idxs if i not in best_cols]
            best_add = None
            best_add_aic = best_aic

            for c in candidates:
                cols_try = list(best_cols) + [c]
                try:
                    m = _fit_cols(cols_try)
                    aic = float(m.aic_)
                    if aic + 1e-12 < best_add_aic:
                        best_add_aic = aic
                        best_add = (c, m)
                except Exception:
                    continue

            if best_add is not None and best_add_aic + 1e-12 < best_aic:
                idx, new_m = best_add
                best_cols = list(best_cols) + [idx]
                best_model = new_m
                best_aic = best_add_aic
                improved = True
                if verbose:
                    print(f"step {steps} add {names[idx]} -> AIC={best_aic:.4f}")

        # backward
        if direction in ("backward", "both") and len(best_cols) > (0 if allow_empty else 1):
            best_rem = None
            best_rem_aic = best_aic

            for c in list(best_cols):
                cols_try = [i for i in best_cols if i != c]
                if (not allow_empty) and len(cols_try) == 0:
                    continue
                try:
                    m = _fit_cols(cols_try)
                    aic = float(m.aic_)
                    if aic + 1e-12 < best_rem_aic:
                        best_rem_aic = aic
                        best_rem = (c, m)
                except Exception:
                    continue

            if best_rem is not None and best_rem_aic + 1e-12 < best_aic:
                idx, new_m = best_rem
                best_cols = [i for i in best_cols if i != idx]
                best_model = new_m
                best_aic = best_rem_aic
                improved = True
                if verbose:
                    print(f"step {steps} remove {names[idx]} -> AIC={best_aic:.4f}")

    return best_model