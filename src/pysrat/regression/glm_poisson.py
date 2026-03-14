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
    lambda_: float = 1.0,
    penalty_factor: Optional[np.ndarray] = None,
    lambda_l2_mat: Optional[np.ndarray] = None,
    eps_mu: float = 1e-15,
) -> GLMPoissonFit:
    """Poisson GLM (log link) via C++ IRLS.

    Model:
        eta = intercept + X @ beta + offset
        mu  = exp(eta)

    Parameters
    ----------
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,)
    offset : ndarray, shape (n,), optional
        Typically log-exposure. If None, uses zeros.
    intercept0 : float
        Initial intercept value. Ignored when fit_intercept=False.
    beta0 : ndarray, shape (p,), optional
        Initial coefficient vector. If None, uses zeros.
    fit_intercept : bool
        If True, estimate intercept separately from X.
    max_iter : int
    tol : float
    standardize : ndarray, shape (p,), optional
        0/1 mask. If None, defaults to all-ones in the C++ layer.
        - fit_intercept=True  -> 1 means center+scale
        - fit_intercept=False -> 1 means scale only
    lambda_ : float
        Overall L2 penalty scale.
    penalty_factor : ndarray, shape (p,), optional
        Per-coefficient penalty weights. If None, uses ones.
    lambda_l2_mat : ndarray, shape (p, p), optional
        If None, uses identity-L2 penalty.
        If provided, uses correlated L2 penalty matrix.
    eps_mu : float

    Returns
    -------
    dict with keys:
        intercept, beta, converged, n_iter
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
        res = _cglm.glm_poisson_identity(
            X,
            y,
            offset,
            bool(fit_intercept),
            float(intercept0),
            beta0,
            std_arg,
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

        res = _cglm.glm_poisson_correlated(
            X,
            y,
            offset,
            bool(fit_intercept),
            float(intercept0),
            beta0,
            std_arg,
            lambda_l2_mat,
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
        "n_iter": int(res.n_iter),
    }