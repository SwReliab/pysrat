from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np

from .. import _glm as _cglm

class GLMPoissonFit(TypedDict):
    intercept: float
    beta: np.ndarray
    converged: bool
    n_iter: int


def glm_poisson(
    X: np.ndarray,
    y: np.ndarray,
    offset: Optional[np.ndarray] = None,
    *,
    intercept0: float = 0.0,
    beta0: Optional[np.ndarray] = None,
    fit_intercept: bool = True,
    max_iter: int = 25,
    tol: float = 1e-8,
    standardize: Optional[np.ndarray] = None,
    ridge: float = 1e-12,
    eps_mu: float = 1e-15,
) -> GLMPoissonFit:
    """Poisson GLM (log link) via C++ IRLS.

    Model:
        eta = intercept + X @ beta + offset
        mu  = exp(eta)

    Notes
    -----
    - `offset` is typically log-exposure. If None, uses zeros.
    - If `fit_intercept` is True, the intercept is estimated (separately from X).
      X must NOT contain an all-ones intercept column.
    - `standardize` is a 0/1 mask of length p. If None, defaults to all-ones.
      * with_intercept: 1 -> center+scale
      * without_intercept: 1 -> scale only (no centering)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n_obs, p = X.shape

    if y.shape != (n_obs,):
        raise ValueError("y length must match X.rows()")

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

    std_arg = None if standardize is None else np.asarray(standardize, dtype=np.int32)

    if fit_intercept:
        res = _cglm.glm_poisson_with_intercept(
            X, y, offset,
            float(intercept0), beta0, std_arg,
            int(max_iter), float(tol),
            float(ridge), float(eps_mu),
        )
    else:
        res = _cglm.glm_poisson_without_intercept(
            X, y, offset,
            beta0, std_arg,
            int(max_iter), float(tol),
            float(ridge), float(eps_mu),
        )

    return {
        "intercept": float(res.intercept),
        "beta": np.asarray(res.beta, dtype=np.float64),
        "converged": bool(res.converged),
        "n_iter": int(res.n_iter),
    }