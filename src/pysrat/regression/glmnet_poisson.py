# src/pysrat/regression/glm_poisson_elasticnet.py
from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np

from .. import _glm as _cglm


class GLMPoissonENetFit(TypedDict):
    intercept: float
    beta: np.ndarray
    converged: bool
    n_iter: int
    n_inner: int
    max_delta: float
    max_delta_inner: float


def glmnet_poisson(
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
    penalty_factor: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    lambda_: float = 1.0,
    lambda_l2_mat: Optional[np.ndarray] = None,
    eps_mu: float = 1e-15,
) -> GLMPoissonENetFit:
    """ElasticNet Poisson GLM (log link) via C++.

    Uses IRLS on the outer loop and coordinate descent on the inner loop
    when L1 is present.

    Model (with intercept):
        eta = intercept + X @ beta + offset
        mu  = exp(eta)

    Model (without intercept):
        eta = X @ beta + offset
        mu  = exp(eta)

    Notes
    -----
    - `y` is count data and is passed as float64 to C++.
    - `offset` is typically log-exposure. If None, uses zeros.
    - `standardize` is a 0/1 mask of length p:
        * fit_intercept=True  -> 1 means center+scale
        * fit_intercept=False -> 1 means scale-only (no centering)
      If None, defaults to all-ones in the C++ layer.
    - `penalty_factor` is a nonnegative vector of length p controlling
      per-coefficient penalty strength. If None, defaults to all-ones.
    - If `lambda_l2_mat` is None, identity-L2 penalty is used.
    - If `lambda_l2_mat` is provided, correlated L2 penalty is used.
    - If `alpha == 0`, the implementation routes internally to the GLM
      (L2-only) solver.
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
    if std_arg is not None and std_arg.shape != (p,):
        raise ValueError("standardize length must match X.cols()")

    pf_arg = None if penalty_factor is None else np.asarray(penalty_factor, dtype=np.float64)
    if pf_arg is not None and pf_arg.shape != (p,):
        raise ValueError("penalty_factor length must match X.cols()")

    if lambda_l2_mat is None:
        res = _cglm.glmnet_poisson_identity(
            X,
            y,
            offset,
            bool(fit_intercept),
            float(intercept0),
            beta0,
            std_arg,
            float(alpha),
            float(lambda_),
            pf_arg,
            int(max_iter),
            float(tol),
            float(eps_mu),
        )
    else:
        lambda_l2_mat = np.asarray(lambda_l2_mat, dtype=np.float64)
        if lambda_l2_mat.shape != (p, p):
            raise ValueError("lambda_l2_mat must have shape (p, p)")

        res = _cglm.glmnet_poisson_correlated(
            X,
            y,
            offset,
            bool(fit_intercept),
            float(intercept0),
            beta0,
            std_arg,
            lambda_l2_mat,
            float(alpha),
            float(lambda_),
            pf_arg,
            int(max_iter),
            float(tol),
            float(eps_mu),
        )

    return {
        "intercept": float(res.intercept),
        "beta": np.asarray(res.beta, dtype=np.float64),
        "converged": bool(res.converged),
        "n_iter": int(res.n_outer),
        "n_inner": int(res.n_inner),
        "max_delta": float(res.max_delta),
        "max_delta_inner": float(res.max_delta_inner),
    }