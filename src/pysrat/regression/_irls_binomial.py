"""Binomial GLM via IRLS in pure Python, for user-supplied link functions.

The C++ IRLS (``csrc/glm/glm_binomial_with_intercept.cpp``) depends on the link
function through a single call, ``LinkEval::eval(link, eta) -> (mu, dmu)``.
That layer only accepts the built-in names ``logit``, ``probit`` and
``cloglog``, so a user-defined link cannot reach it.

This module is a faithful port of the same algorithm with that one call
replaced by a caller-supplied ``link_eval``.  It is used only when
:func:`pysrat.regression.glm_binomial` is given a non-string link; built-in
links keep going through C++ unchanged.

Anything that changes the fitted optimum is kept identical to the C++ version:
standardization, the penalized objective, the IRLS weights and working
variate, step-halving on the objective, and the convergence criterion.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

LinkEval = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]

_TINY_SD = 1e-12
_MAX_HALVING = 25


def irls_binomial(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: np.ndarray,
    offset: np.ndarray,
    *,
    link_eval: LinkEval,
    domain: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    intercept0: float = 0.0,
    beta0: Optional[np.ndarray] = None,
    fit_intercept: bool = True,
    standardize: Optional[np.ndarray] = None,
    max_iter: int = 25,
    tol: float = 1e-8,
    lambda_: float = 0.0,
    penalty_factor: Optional[np.ndarray] = None,
    eps_mu: float = 1e-15,
    eps_dmu: float = 1e-15,
) -> dict:
    """Fit an aggregated binomial GLM by IRLS.

    Parameters
    ----------
    y : ndarray
        Success **counts** (not proportions).
    link_eval : callable
        ``eta -> (mu, dmu)``, where ``dmu`` is ``d mu / d eta``.
    domain : callable, optional
        ``eta -> bool ndarray`` marking where the link is defined. A step that
        leaves the domain gets an objective of ``-inf``, so step-halving
        rejects it. Pass ``None``, or return ``None`` from the callable, for a
        link defined on the whole line.
    standardize : ndarray of int, optional
        0/1 mask per column. ``None`` means all ones, as in the C++ layer.

    Returns
    -------
    dict with keys ``intercept``, ``beta``, ``converged``, ``n_iter``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_trials = np.asarray(n_trials, dtype=float)
    offset = np.asarray(offset, dtype=float)
    n, p = X.shape

    beta0 = np.zeros(p) if beta0 is None else np.asarray(beta0, dtype=float)
    if max_iter <= 0:
        return {"intercept": float(intercept0), "beta": beta0.copy(),
                "converged": False, "n_iter": 0}

    std = (np.ones(p, dtype=int) if standardize is None
           else np.asarray(standardize, dtype=int))
    pf = (np.ones(p) if penalty_factor is None
          else np.asarray(penalty_factor, dtype=float))
    lam_vec = float(lambda_) * pf

    # --- center + scale ---
    x_mean, x_scale, Xs = np.zeros(p), np.ones(p), X.copy()
    for j in range(p):
        if std[j] != 0:
            m = float(X[:, j].mean())
            sd = float(np.sqrt(np.mean((X[:, j] - m) ** 2)))
            if not np.isfinite(sd) or sd < _TINY_SD:
                raise ValueError(
                    f"Cannot standardize (near-)constant column j={j}")
            x_mean[j], x_scale[j] = m, sd
            Xs[:, j] = (X[:, j] - m) / sd

    # --- initial parameters in the standardized parameterization ---
    beta = np.where(std != 0, beta0 * x_scale, beta0).astype(float)
    b0 = float(intercept0) + float(np.sum(np.where(std != 0, x_mean * beta0, 0.0)))

    def objective(b0_try: float, beta_try: np.ndarray) -> float:
        eta = Xs @ beta_try + b0_try + offset
        if domain is not None:
            mask = domain(eta)          # None は「全域で定義される」の意
            if mask is not None and not bool(np.all(mask)):
                return -np.inf
        mu, _ = link_eval(eta)
        mu = np.clip(mu, eps_mu, 1.0 - eps_mu)
        llf = float(np.sum(y * np.log(mu) + (n_trials - y) * np.log(1.0 - mu)))
        return llf - 0.5 * float(np.sum(lam_vec * beta_try**2))

    converged, it = False, 0
    obj_old = objective(b0, beta)

    for it in range(max_iter):
        eta = Xs @ beta + b0 + offset
        mu, dmu = link_eval(eta)
        mu = np.clip(mu, eps_mu, 1.0 - eps_mu)
        dmu = np.maximum(dmu, eps_dmu)

        pos = (n_trials > 0.0) & np.isfinite(n_trials)
        nt = np.where(pos, n_trials, 0.0)

        # W = n dmu^2 / (mu (1 - mu)); z = eta + (y - n mu) / (n dmu)
        with np.errstate(divide="ignore", invalid="ignore"):
            W = nt * dmu**2 / (mu * (1.0 - mu))
        W = np.where(pos & np.isfinite(W) & (W > 0.0), W, eps_mu)
        W = np.maximum(W, eps_mu)
        z = np.where(pos, eta + (y - nt * mu) / np.where(pos, nt * dmu, 1.0), eta)

        # weighted least squares: z - offset ~ intercept + Xs beta
        sqrtW = np.sqrt(W)
        Xaug = np.hstack([np.ones((n, 1)), Xs]) if fit_intercept else Xs
        Xaug_w = Xaug * sqrtW[:, None]
        rhs = sqrtW * (z - offset)

        A = Xaug_w.T @ Xaug_w
        if fit_intercept:                      # the intercept is unpenalized
            A[np.arange(1, p + 1), np.arange(1, p + 1)] += lam_vec
        else:
            A[np.arange(p), np.arange(p)] += lam_vec
        bvec = Xaug_w.T @ rhs

        try:
            theta_full = np.linalg.solve(A, bvec)
        except np.linalg.LinAlgError:
            theta_full = np.linalg.lstsq(A, bvec, rcond=None)[0]
        if not np.all(np.isfinite(theta_full)):
            break

        if fit_intercept:
            b0_full, beta_full = float(theta_full[0]), theta_full[1:]
        else:
            b0_full, beta_full = b0, theta_full

        # --- step-halving on the penalized objective ---
        step, obj_new = 1.0, -np.inf
        b0_new, beta_new = b0, beta
        for _ in range(_MAX_HALVING):
            b0_new = b0 + step * (b0_full - b0)
            beta_new = beta + step * (beta_full - beta)
            obj_new = objective(b0_new, beta_new)
            if np.isfinite(obj_new) and obj_new >= obj_old - 1e-12:
                break
            step *= 0.5

        max_diff = max(
            float(np.max(np.abs(beta_new - beta))) if p > 0 else 0.0,
            abs(b0_new - b0),
        )
        beta, b0 = beta_new, b0_new
        if np.isfinite(obj_new):
            obj_old = obj_new
        if max_diff < tol:
            converged, it = True, it + 1
            break
    else:
        it = max_iter

    # --- back transform ---
    beta_out = np.where(std != 0, beta / x_scale, beta)
    b0_out = b0 - float(np.sum(np.where(std != 0, x_mean * beta_out, 0.0)))

    return {"intercept": float(b0_out), "beta": beta_out,
            "converged": bool(converged), "n_iter": int(it)}
