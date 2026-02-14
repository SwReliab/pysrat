from __future__ import annotations

import numpy as np

from ..base import NHPPModel
from ..data import NHPPData
from ..dists.gamma import dgamma, pgamma
from .. import _core


class GammaNHPP(NHPPModel):
    name = "gamma"
    df = 3

    def __init__(self, *, omega0: float = 1.0, shape0: float = 1.0, rate0: float = 1.0):
        super().__init__(omega0=float(omega0), shape0=float(shape0), rate0=float(rate0))
        self._set_fitted_params(np.array([omega0, shape0, rate0], dtype=float))

    def param_names(self) -> list[str]:
        return ["omega", "shape", "rate"]

    def init_params(self, data: NHPPData) -> np.ndarray:
        return np.array([data.total, 1.0, 1.0 / data.mean], dtype=float)

    def em_step(self, params: np.ndarray, data: NHPPData, **kwargs) -> dict:
        divide = int(kwargs.get("divide", 15))
        eps = float(kwargs.get("eps", 1.0e-10))
        return _core.em_gamma_emstep(np.asarray(params, dtype=float), data.to_core_dict(), divide, eps)

    def omega_(self) -> float:
        return float(self.params_[0])

    def shape_(self) -> float:
        return float(self.params_[1])

    def rate_(self) -> float:
        return float(self.params_[2])

    def mvf(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega_() * pgamma(t, shape=self.shape_(), rate=self.rate_(), lower_tail=True)

    def intensity(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega_() * dgamma(t, shape=self.shape_(), rate=self.rate_(), log=False)


