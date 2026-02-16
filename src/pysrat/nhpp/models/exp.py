from __future__ import annotations

import numpy as np

from ..dists.exp import dexp, pexp

from ..base import NHPPModel
from ..data import NHPPData
from ... import _core


class ExponentialNHPP(NHPPModel):
    name = "exp"
    df = 2

    def __init__(self, *, omega0: float = 1.0, rate0: float = 1.0):
        super().__init__(omega0=float(omega0), rate0=float(rate0))
        self._set_fitted_params(np.array([omega0, rate0], dtype=float))

    def param_names(self) -> list[str]:
        return ["omega", "rate"]

    def init_params(self, data: NHPPData) -> np.ndarray:
        return np.array([data.total, 1.0 / data.mean], dtype=float)

    def em_step(self, params: np.ndarray, data: NHPPData, **kwargs) -> dict:
        return _core.em_exp_emstep(np.asarray(params, dtype=float), data.to_core_dict())

    def omega_(self) -> float:
        return float(self.params_[0])

    def rate_(self) -> float:
        return float(self.params_[1])

    def mvf(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega_() * pexp(t, rate=self.rate_(), lower_tail=True, log_p=False)

    def dmvf(self, t):
        t = np.asarray(t, dtype=float)
        t2 = np.concatenate([[0.0], t])
        return np.diff(self.mvf(t2))

    def intensity(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega_() * dexp(t, rate=self.rate_(), log=False)


