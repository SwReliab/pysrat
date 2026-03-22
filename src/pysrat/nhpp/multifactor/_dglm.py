from __future__ import annotations

from abc import abstractmethod
import math
from typing import Literal, Optional

import numpy as np
from scipy.special import ndtr

from ...regression.glm_binomial import glm_binomial
from .._base import NHPPModel
from ...data import NHPPData


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * x * x)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return ndtr(x)


class DynamicGLMBase(NHPPModel):
    """
    NHPP-based SRM with d-metrics (ported from the R6 dGLM).

    params_:
        omega, (intercept), beta_1, ..., beta_p

    Notes
    -----
    - mvf(t) interprets t as an index (1..n), consistent with the original R code.
    - This implementation treats `metrics` as X (WITHOUT an intercept column).
      If `has_intercept=True`, the intercept is stored/estimated separately.
    """
    has_intercept: bool = True

    def __init__(self, has_intercept: Optional[bool] = None):
        super().__init__(
            has_intercept=has_intercept
            if has_intercept is not None
            else getattr(self, "has_intercept", True),
        )
        self.has_intercept = bool(
            has_intercept
            if has_intercept is not None
            else getattr(self, "has_intercept", True)
        )

    def param_names(self) -> list[str]:
        # If data is set, use metrics_name if available
        if self.data_ is not None and hasattr(self.data_, "metrics"):
            X = np.asarray(getattr(self.data_, "metrics"))
            p = int(X.shape[1])

            metrics_name = getattr(self.data_, "metrics_name", None)
            if metrics_name is not None and len(metrics_name) == p:
                beta_names = list(metrics_name)
            else:
                beta_names = [f"b{i}" for i in range(p)]

            if self.has_intercept:
                return ["omega", "intercept", *beta_names]
            return ["omega", *beta_names]

        # fallback based on params_ length
        if self.params_ is not None:
            k = int(len(self.params_) - 1)  # excluding omega
            if self.has_intercept:
                # k = 1 + p
                p = max(k - 1, 0)
                return ["omega", "intercept", *[f"b{i}" for i in range(p)]]
            else:
                p = max(k, 0)
                return ["omega", *[f"b{i}" for i in range(p)]]

        return ["omega"]

    def init_params(self, data: NHPPData) -> np.ndarray:
        X = np.asarray(getattr(data, "metrics"), dtype=float)
        fault = np.asarray(getattr(data, "fault"), dtype=float)

        total_attr = getattr(data, "total", None)
        total = float(np.sum(fault)) if total_attr is None else float(total_attr)

        p = int(X.shape[1])
        # omega + (intercept?) + p betas
        k = 1 + (1 if self.has_intercept else 0) + p
        params0 = np.zeros(k, dtype=float)
        params0[0] = total + 1.0
        self.df = int(params0.size)
        return params0

    def omega_(self) -> float:
        return float(self.params_[0])

    def coefficients_(self) -> np.ndarray:
        # includes intercept (if present) + betas
        return np.asarray(self.params_[1:], dtype=float)

    def mvf(self, t):
        data = self.data_
        if data is None:
            raise RuntimeError("data_ is not set. Call fit() or set_data().")

        X = np.asarray(getattr(data, "metrics"), dtype=float)
        offset = np.asarray(getattr(data, "offset"), dtype=float)

        params = np.asarray(self._get_fitted_params(), dtype=float)
        omega = float(params[0])

        if self.has_intercept:
            intercept = float(params[1])
            beta = params[2:]
        else:
            intercept = 0.0
            beta = params[1:]

        eta_all = intercept + X @ beta + offset
        mu_all = self._linkinv(eta_all)

        t = np.asarray(t, dtype=int)
        out = np.empty_like(t, dtype=float)
        for i, t0 in enumerate(t):
            if t0 <= 0:
                out[i] = 0.0
            else:
                mu = mu_all[:t0]
                out[i] = omega * (1.0 - float(np.prod(1.0 - mu)))
        return out

    def intensity(self, t):
        m = self.mvf(t)
        if np.size(m) == 0:
            return m
        return np.concatenate([[m[0]], np.diff(m)])

    def em_step(self, params: np.ndarray, data: NHPPData, **kwargs) -> dict:
        params = np.asarray(params, dtype=float)
        omega = float(params[0])

        if self.has_intercept:
            intercept = float(params[1])
            beta = np.asarray(params[2:], dtype=float)
        else:
            intercept = 0.0
            beta = np.asarray(params[1:], dtype=float)

        X = np.asarray(getattr(data, "metrics"), dtype=float)
        offset = np.asarray(getattr(data, "offset"), dtype=float)
        fault = np.asarray(getattr(data, "fault"), dtype=float)

        # E-step
        eta = intercept + X @ beta + offset
        mu = np.clip(self._linkinv(eta), 1e-15, 1.0 - 1e-15)

        residual = omega * float(np.prod(1.0 - mu))
        total = float(np.sum(fault) + residual)

        rfault = total - np.cumsum(fault)
        n_trials = np.maximum(fault + rfault, 1e-12)
        y = fault / n_trials

        # M-step: aggregated binomial IRLS (y is proportion -> convert inside glm_binomial)
        # Keep default behavior effectively unpenalized, while accepting both
        # new and legacy regularization keyword names from fit(...).
        glm_lambda = kwargs.get("glm_lambda", kwargs.get("lambd", 0.0))
        glm_penalty_factor = kwargs.get("glm_penalty_factor", kwargs.get("penalty", None))
        glm_lambda_l2_mat = kwargs.get("glm_lambda_l2_mat", kwargs.get("l2matrix", None))
        glm_standardize = kwargs.get("glm_standardize", kwargs.get("standardize", None))

        fit = glm_binomial(
            X=X,
            y=y,
            n_trials=n_trials,
            offset=offset,
            intercept0=intercept,
            beta0=beta,
            fit_intercept=self.has_intercept,
            link=self.link,
            max_iter=int(kwargs.get("max_glm_iter", 50)),
            tol=float(kwargs.get("glm_tol", 1e-9)),
            y_is_proportion=True,
            standardize=glm_standardize,
            lambda_=float(glm_lambda),
            penalty_factor=glm_penalty_factor,
            lambda_l2_mat=glm_lambda_l2_mat,
            eps_mu=float(kwargs.get("glm_eps_mu", 1e-15)),
            eps_dmu=float(kwargs.get("glm_eps_dmu", 1e-15)),
        )

        if self.has_intercept:
            newparams = np.concatenate([[total], [float(fit["intercept"])], np.asarray(fit["beta"], dtype=float)])
        else:
            newparams = np.concatenate([[total], np.asarray(fit["beta"], dtype=float)])

        pdiff = np.abs(params - newparams)
        llf = self._llf(data=data, fault=fault, omega=omega, mu=mu)

        return {"param": newparams, "pdiff": pdiff, "llf": llf, "total": total}

    # --- link functions ----------------------------------------------------
    @abstractmethod
    def _linkinv(self, eta: np.ndarray) -> np.ndarray:
        ...

    # --- likelihood --------------------------------------------------------

    def _llf(self, *, data: NHPPData, fault: np.ndarray, omega: float, mu: np.ndarray) -> float:
        mu = np.clip(mu, 1e-15, 1.0 - 1e-15)

        rfault = float(np.sum(fault)) - np.cumsum(fault)
        mask_n = fault != 0
        mask_r = rfault != 0

        term1 = float(np.sum(fault[mask_n] * np.log(mu[mask_n])))
        term2 = float(np.sum(rfault[mask_r] * np.log(1.0 - mu[mask_r])))

        term3 = -float(np.sum([math.lgamma(x + 1.0) for x in fault]))
        term4 = float(np.sum(fault)) * math.log(float(omega))
        term5 = -float(omega) * (1.0 - float(np.prod(1.0 - mu)))

        return term1 + term2 + term3 + term4 + term5
