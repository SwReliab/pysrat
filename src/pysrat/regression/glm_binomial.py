from __future__ import annotations
from typing import Optional, TypedDict
import numpy as np

from .. import _glm as _cglm

class GLMBinomialFit(TypedDict):
    intercept: float
    beta: np.ndarray
    converged: bool
    n_iter: int

def glm_binomial(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    *,
    intercept0: float = 0.0,
    beta0: Optional[np.ndarray] = None,
    fit_intercept: bool = True,
    link: str = "logit",
    max_iter: int = 25,
    tol: float = 1e-8,
    y_is_proportion: bool = False,
    standardize: Optional[np.ndarray] = None,
    ridge: float = 1e-12,
    eps_mu: float = 1e-15,
    eps_dmu: float = 1e-15,
) -> GLMBinomialFit:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_obs, p = X.shape

    if n_trials is None:
        n_trials = np.ones(n_obs, dtype=np.float64)
    else:
        n_trials = np.asarray(n_trials, dtype=np.float64)
        if n_trials.shape != (n_obs,):
            raise ValueError("n_trials length must match X.rows()")

    if offset is None:
        offset = np.zeros(n_obs, dtype=np.float64)
    else:
        offset = np.asarray(offset, dtype=np.float64)
        if offset.shape != (n_obs,):
            raise ValueError("offset length must match X.rows()")

    if beta0 is None:
        beta0 = np.zeros(p, dtype=np.float64)
    else:
        beta0 = np.asarray(beta0, dtype=np.float64)
        if beta0.shape != (p,):
            raise ValueError("beta0 length must match X.cols()")

    if y_is_proportion:
        y_call = y * n_trials
    else:
        y_call = y

    std_arg = None if standardize is None else np.asarray(standardize, dtype=np.int32)

    if fit_intercept:
        res = _cglm.glm_binomial_with_intercept(
            X, y_call, n_trials, offset,
            float(intercept0), beta0, std_arg,
            int(max_iter), float(tol), link,
            float(ridge), float(eps_mu), float(eps_dmu),
        )
    else:
        res = _cglm.glm_binomial_without_intercept(
            X, y_call, n_trials, offset,
            beta0, std_arg,
            int(max_iter), float(tol), link,
            float(ridge), float(eps_mu), float(eps_dmu),
        )

    return {
        "intercept": float(res.intercept),
        "beta": np.asarray(res.beta, dtype=np.float64),
        "converged": bool(res.converged),
        "n_iter": int(res.n_iter),
    }
