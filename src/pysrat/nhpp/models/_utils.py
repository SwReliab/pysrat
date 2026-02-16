from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def optimize_params(
    pllf_fn,
    data_dict: dict,
    x0: np.ndarray,
    *,
    w0=None,
    w1=None,
    method: str = "L-BFGS-B",
    bounds=None,
    options: dict | None = None,
) -> np.ndarray:
    def obj(x):
        loc, scale = float(x[0]), float(x[1])
        if scale <= 0:
            return np.inf
        params = np.asarray([loc, scale], dtype=float)
        if w0 is None:
            return -float(pllf_fn(params, data_dict, float(w1)))
        return -float(pllf_fn(params, data_dict, float(w0), float(w1)))

    if method.upper() == "L-BFGS-B" and bounds is None:
        bounds = [(None, None), (1.0e-12, None)]
    res = minimize(
        obj,
        x0=np.asarray(x0, dtype=float),
        method=method,
        bounds=bounds,
        options=options,
    )
    if not res.success or not np.all(np.isfinite(res.x)):
        return np.asarray(x0, dtype=float)
    return np.asarray(res.x, dtype=float)
