from __future__ import annotations

from typing import Callable, Optional, Tuple, TypedDict, Union

import numpy as np

from .. import _glm as _cglm
from ._binomial_validation import prepare_binomial_response
from ._irls_binomial import irls_binomial

#: A link is either a built-in name handled by the C++ layer, or a tuple
#: ``(linkinv, dmu)`` -- optionally ``(linkinv, dmu, domain)`` -- of callables
#: evaluated in Python.
LinkArg = Union[str, Tuple[Callable, ...]]


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
    link: LinkArg = "logit",
    max_iter: int = 25,
    tol: float = 1e-8,
    y_is_proportion: bool = False,
    standardize: Optional[np.ndarray] = None,
    lambda_: float = 1.0,
    penalty_factor: Optional[np.ndarray] = None,
    lambda_l2_mat: Optional[np.ndarray] = None,
    eps_mu: float = 1e-15,
    eps_dmu: float = 1e-15,
) -> GLMBinomialFit:
    """Binomial GLM via C++ IRLS.

    Model
    -----
    Aggregated binomial model:
        y_i ~ Binomial(n_trials_i, mu_i)
        g(mu_i) = intercept + X @ beta + offset

    Parameters
    ----------
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,)
        Either success counts or proportions depending on y_is_proportion.
    n_trials : ndarray, shape (n,), optional
        Number of trials for each observation. If None, uses ones.
    offset : ndarray, shape (n,), optional
        Additive offset on the linear predictor. If None, uses zeros.
    intercept0 : float
        Initial intercept value. Ignored when fit_intercept=False.
    beta0 : ndarray, shape (p,), optional
        Initial coefficient vector. If None, uses zeros.
    fit_intercept : bool
        If True, estimate intercept separately from X.
    link : str or (callable, callable)
        Either a link name passed to the C++ layer ("logit", "probit",
        "cloglog"), or a pair ``(linkinv, dmu)`` of callables defining a
        custom link. ``linkinv(eta)`` returns mu and ``dmu(eta, mu)`` returns
        d mu / d eta. A third element ``domain(eta)`` may mark where the link
        is defined, for links with a restricted domain. A custom link is fitted by the Python IRLS in
        :mod:`pysrat.regression._irls_binomial`, which implements the same
        algorithm as the C++ layer.
    max_iter : int
    tol : float
    y_is_proportion : bool
        If True, interpret y as proportions and internally convert to
        success counts by y * n_trials.
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
        Lower/upper clipping level for mu.
    eps_dmu : float
        Lower clipping level for dmu/deta in IRLS.

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

    y_call = prepare_binomial_response(
        y,
        n_trials,
        y_is_proportion=y_is_proportion,
    )

    std_arg = None if standardize is None else np.asarray(standardize, dtype=np.int32)
    if std_arg is not None and std_arg.shape != (p,):
        raise ValueError("standardize length must match X.cols()")

    pf_arg = None if penalty_factor is None else np.asarray(penalty_factor, dtype=np.float64)
    if pf_arg is not None and pf_arg.shape != (p,):
        raise ValueError("penalty_factor length must match X.cols()")

    if not isinstance(link, str):
        # custom link -> Python IRLS (the C++ layer only knows the built-in names)
        if lambda_l2_mat is not None:
            raise NotImplementedError(
                "lambda_l2_mat is not supported for custom link functions")
        if len(link) == 2:
            (linkinv, dmu_fn), domain = link, None
        else:
            linkinv, dmu_fn, domain = link

        def _link_eval(eta):
            mu = np.asarray(linkinv(eta), dtype=np.float64)
            return mu, np.asarray(dmu_fn(eta, mu), dtype=np.float64)

        return irls_binomial(
            X, y_call, n_trials, offset,
            link_eval=_link_eval,
            domain=domain,
            intercept0=float(intercept0),
            beta0=beta0,
            fit_intercept=bool(fit_intercept),
            standardize=std_arg,
            max_iter=int(max_iter),
            tol=float(tol),
            lambda_=float(lambda_),
            penalty_factor=pf_arg,
            eps_mu=float(eps_mu),
            eps_dmu=float(eps_dmu),
        )

    if lambda_l2_mat is None:
        res = _cglm.glm_binomial_identity(
            X,
            y_call,
            n_trials,
            offset,
            bool(fit_intercept),
            float(intercept0),
            beta0,
            std_arg,
            float(lambda_),
            pf_arg,
            int(max_iter),
            float(tol),
            str(link),
            float(eps_mu),
            float(eps_dmu),
        )
    else:
        lambda_l2_mat = np.asarray(lambda_l2_mat, dtype=np.float64)
        if lambda_l2_mat.shape != (p, p):
            raise ValueError("lambda_l2_mat must have shape (p, p)")

        res = _cglm.glm_binomial_correlated(
            X,
            y_call,
            n_trials,
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
            str(link),
            float(eps_mu),
            float(eps_dmu),
        )

    return {
        "intercept": float(res.intercept),
        "beta": np.asarray(res.beta, dtype=np.float64),
        "converged": bool(res.converged),
        "n_iter": int(res.n_iter),
    }