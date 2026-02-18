from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.special import ndtr

from ... import _irls
from ..base import NHPPModel
from ..data import NHPPData


# --- Data contract ---------------------------------------------------------
# dGLM expects NHPPData-like object with:
#   - metrics: (n, p) float array
#   - offset: (n,) float array
#   - fault:  (n,) float array (counts)
#   - nmetrics: int (p)  [optional; derived from metrics.shape[1] if missing]
#   - total: float       [optional; if missing -> sum(fault)]
#
# If your NHPPData does not have these, define a dedicated data class
# (e.g., DMetricsData) and pass it to fit(); emfit_internal must not assume NHPPData fields.


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * x * x)

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return ndtr(x)

class dGLMBase(NHPPModel):
    """
    NHPP-based SRM with d-metrics (ported from the R6 dGLM).

    Parameters (params_):
        omega, beta_1, ..., beta_p

    Notes:
        - mvf(t) interprets t as an index (1..n), consistent with the original R code:
          metrics[1:t0] / offset[1:t0].
        - intensity(t) is discrete: mvf differences.
    """

    name: str = "dGLM"
    link: Literal["logit", "probit", "cloglog"] = "logit"

    def param_names(self) -> list[str]:
        # If data is set, prefer that p
        if self.data_ is not None and hasattr(self.data_, "metrics"):
            p = int(np.asarray(self.data_.metrics).shape[1])
            metrics_name = getattr(self.data_, "metrics_name", None)
            if metrics_name:
                names = list(metrics_name)
                if len(names) == p - 1:
                    names = ["intercept"] + names
                if len(names) == p:
                    return ["omega"] + names
            return ["omega"] + [f"b{i}" for i in range(p)]
        if self.params_ is not None:
            return ["omega"] + [f"b{i}" for i in range(len(self.params_) - 1)]
        return ["omega"]

    def init_params(self, data: NHPPData) -> np.ndarray:
        X = np.asarray(getattr(data, "metrics"), dtype=float)
        fault = np.asarray(getattr(data, "fault"), dtype=float)

        total_attr = getattr(data, "total", None)
        total = float(np.sum(fault)) if total_attr is None else float(total_attr)

        p_attr = getattr(data, "nmetrics", None)
        p = int(X.shape[1]) if p_attr is None else int(p_attr)

        params0 = np.zeros(1 + p, dtype=float)
        params0[0] = total + 1.0
        self.df = int(params0.size)
        return params0

    def omega_(self) -> float:
        return float(self.params_[0])

    def coefficients_(self) -> np.ndarray:
        return self.params_[1:]

    def mvf(self, t):
        data = self.data_
        if data is None:
            raise RuntimeError("data_ is not set. Call fit() or set_data().")

        X = np.asarray(getattr(data, "metrics"), dtype=float)
        offset = np.asarray(getattr(data, "offset"), dtype=float)

        params = self._get_fitted_params()
        omega = float(params[0])
        beta = np.asarray(params[1:], dtype=float)

        t = np.asarray(t, dtype=int)

        eta_all = X @ beta + offset
        mu_all = self._linkinv(eta_all)

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
        if m.size == 0:
            return m
        return np.concatenate([[m[0]], np.diff(m)])

    def em_step(self, params: np.ndarray, data: NHPPData, **kwargs) -> dict:
        params = np.asarray(params, dtype=float)
        omega = float(params[0])
        beta = np.asarray(params[1:], dtype=float)

        X = np.asarray(getattr(data, "metrics"), dtype=float)
        offset = np.asarray(getattr(data, "offset"), dtype=float)
        fault = np.asarray(getattr(data, "fault"), dtype=float)

        # E-step: current mu, residual, total
        eta = X @ beta + offset
        mu = np.clip(self._linkinv(eta), 1e-15, 1.0 - 1e-15)

        residual = omega * float(np.prod(1.0 - mu))
        total = float(np.sum(fault) + residual)

        rfault = total - np.cumsum(fault)
        n_trials = np.maximum(fault + rfault, 1e-12)
        y = fault / n_trials

        # M-step: IRLS for aggregated binomial with offset
        beta_new = self._irls_binomial_offset(
            X=X,
            y=y,
            n=n_trials,
            offset=offset,
            beta0=beta,
            max_iter=int(kwargs.get("max_irls_iter", 50)),
            tol=float(kwargs.get("irls_tol", 1e-9)),
        )

        newparams = np.concatenate([[total], beta_new])
        pdiff = np.abs(params - newparams)
        llf = self._llf(data=data, fault=fault, omega=omega, mu=mu)

        return {"param": newparams, "pdiff": pdiff, "llf": llf, "total": total}

    # --- link functions ----------------------------------------------------

    def _linkinv(self, eta: np.ndarray) -> np.ndarray:
        if self.link == "logit":
            return 1.0 / (1.0 + np.exp(-eta))
        if self.link == "probit":
            return _norm_cdf(eta)
        if self.link == "cloglog":
            exp_eta = np.exp(eta)
            return 1.0 - np.exp(-exp_eta)
        raise ValueError(f"Unknown link: {self.link}")

    def _dmu_deta(self, eta: np.ndarray, mu: np.ndarray) -> np.ndarray:
        if self.link == "logit":
            return mu * (1.0 - mu)
        if self.link == "probit":
            return _norm_pdf(eta)
        if self.link == "cloglog":
            exp_eta = np.exp(eta)
            return exp_eta * np.exp(-exp_eta)
        raise ValueError(f"Unknown link: {self.link}")

    # --- likelihood (as in the R code) ------------------------------------

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

    # --- IRLS --------------------------------------------------------------

    def _irls_binomial_offset(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        n: np.ndarray,
        offset: np.ndarray,
        beta0: np.ndarray,
        max_iter: int,
        tol: float,
    ) -> np.ndarray:
        return _irls.irls_binomial_offset(
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(n, dtype=float),
            np.asarray(offset, dtype=float),
            np.asarray(beta0, dtype=float),
            int(max_iter),
            float(tol),
            self.link,
        )


class dGLMLogit(dGLMBase):
    name = "dGLM.logit"
    link = "logit"


class dGLMProbit(dGLMBase):
    name = "dGLM.probit"
    link = "probit"


class dGLMCloglog(dGLMBase):
    name = "dGLM.cloglog"
    link = "cloglog"