from __future__ import annotations

import numpy as np

from ..nhpp import NHPPModel
from ..data import NHPPData
from .. import _core


class ExponentialNHPP(NHPPModel):
    name = "exp"
    df = 2

    def __init__(self, omega: float = 1.0, rate: float = 1.0):
        super().__init__(params=np.array([omega, rate], dtype=float))

    @property
    def omega(self) -> float:
        return float(self.params[0])

    @property
    def rate_param(self) -> float:
        return float(self.params[1])

    def param_names(self) -> list[str]:
        return ["omega", "rate"]

    def mvf(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega * (1.0 - np.exp(-self.rate_param * t))

    def intensity(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega * self.rate_param * np.exp(-self.rate_param * t)

    def init_params(self, data: NHPPData) -> np.ndarray:
        return np.array([data.total, 1.0 / data.mean], dtype=float)

    def em_step(self, params: np.ndarray, data: NHPPData, **kwargs) -> dict:
        return _core.em_exp_emstep(np.asarray(params, dtype=float), data.to_core_dict())


ExpSRM = ExponentialNHPP
