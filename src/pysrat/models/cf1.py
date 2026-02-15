from __future__ import annotations

import numpy as np

from ..base import NHPPModel
from ..data import NHPPData
from .. import marlib_cf1 as _cf1
from ..dists.cf1 import dcf1, pcf1, cf1_params_init, cf1llf


class CanonicalPhaseTypeNHPP(NHPPModel):
    name = "cf1"

    def __init__(
        self,
        n: int,
        *,
        omega0: float = 1.0,
        alpha0: np.ndarray | None = None,
        rate0: np.ndarray | None = None,
    ):
        if n <= 0:
            raise ValueError("n must be positive")
        if alpha0 is None:
            alpha0 = np.full(n, 1.0 / n, dtype=float)
        if rate0 is None:
            rate0 = np.ones(n, dtype=float)
        alpha0 = np.asarray(alpha0, dtype=float)
        rate0 = np.asarray(rate0, dtype=float)
        if alpha0.shape[0] != n or rate0.shape[0] != n:
            raise ValueError("alpha0 and rate0 must have length n")

        self.n = int(n)
        self.df = int(2 * n)
        self.name = f"cph{n}"

        super().__init__(n=int(n), omega0=float(omega0), alpha0=alpha0, rate0=rate0)
        self._set_fitted_params(
            np.concatenate([[float(omega0)], alpha0.astype(float), rate0.astype(float)])
        )

    def param_names(self) -> list[str]:
        alpha_names = [f"alpha{i + 1}" for i in range(self.n)]
        rate_names = [f"rate{i + 1}" for i in range(self.n)]
        return ["omega", *alpha_names, *rate_names]

    def init_params(self, data: NHPPData) -> np.ndarray:
        params = cf1_params_init(self.n, data)
        if params is None:
            omega = float(data.total)
            alpha = np.full(self.n, 1.0 / self.n, dtype=float)
            if data.mean > 0:
                rate = np.full(self.n, 1.0 / data.mean, dtype=float)
            else:
                rate = np.ones(self.n, dtype=float)
        else:
            omega = float(params["omega"])
            alpha = np.asarray(params["alpha"], dtype=float)
            rate = np.asarray(params["rate"], dtype=float)
        return np.concatenate([[omega], alpha, rate])

    def em_step(self, params: np.ndarray, data: NHPPData, **kwargs) -> dict:
        params = np.asarray(params, dtype=float)
        if params.size != 1 + 2 * self.n:
            raise ValueError("invalid parameter length")

        omega = float(params[0])
        alpha = np.asarray(params[1 : 1 + self.n], dtype=float)
        rate = np.asarray(params[1 + self.n : 1 + 2 * self.n], dtype=float)

        pdata = {
            "time": np.asarray(data.time, dtype=float),
            "fault": np.asarray(data.fault, dtype=np.int64),
            "type": np.asarray(data.type, dtype=np.int64),
        }
        pparams = {
            "omega": omega,
            "alpha": alpha,
            "rate": rate,
        }

        res = _cf1.em_cf1_emstep(
            pparams,
            pdata,
            kwargs.get("eps", 1.0e-8),
            kwargs.get("ufactor", 1.01),
        )

        new_omega = float(res["param"]["omega"])
        new_alpha = np.asarray(res["param"]["alpha"], dtype=float)
        new_rate = np.asarray(res["param"]["rate"], dtype=float)

        param = np.concatenate([[new_omega], new_alpha, new_rate])

        if "pdiff" in res:
            pd_omega = float(res["pdiff"]["omega"])
            pd_alpha = np.asarray(res["pdiff"]["alpha"], dtype=float)
            pd_rate = np.asarray(res["pdiff"]["rate"], dtype=float)
            pdiff = np.concatenate([[pd_omega], pd_alpha, pd_rate])
        else:
            pdiff = param - params

        return {
            "param": param,
            "pdiff": pdiff,
            "llf": float(res["llf"]),
            "total": float(res["total"]),
        }

    def omega_(self) -> float:
        return float(self.params_[0])

    def alpha_(self) -> np.ndarray:
        return np.asarray(self.params_[1 : 1 + self.n], dtype=float)

    def rate_(self) -> np.ndarray:
        return np.asarray(self.params_[1 + self.n : 1 + 2 * self.n], dtype=float)

    def mvf(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega_() * pcf1(t, alpha=self.alpha_(), rate=self.rate_(), lower_tail=True, log_p=False)

    def intensity(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega_() * dcf1(t, alpha=self.alpha_(), rate=self.rate_(), log=False)

    def llf(self, data: NHPPData) -> float:
        return cf1llf(data, self.omega_(), self.alpha_(), self.rate_())
