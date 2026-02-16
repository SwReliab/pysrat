from __future__ import annotations

import numpy as np

from ..base import NHPPModel
from ..data.nhpp import NHPPData
from ..dists.pareto2 import dpareto2, ppareto2
from .. import _core


class Pareto2NHPP(NHPPModel):
    name = "pareto"
    df = 3

    def __init__(self, *, omega0: float = 1.0, shape0: float = 1.0, scale0: float = 1.0):
        super().__init__(omega0=float(omega0), shape0=float(shape0), scale0=float(scale0))
        self._set_fitted_params(np.array([omega0, shape0, scale0], dtype=float))

    def param_names(self) -> list[str]:
        return ["omega", "shape", "scale"]

    def init_params(self, data: NHPPData) -> np.ndarray:
        return np.array([1.0, 1.0, 1.0], dtype=float)

    def em_step(self, params: np.ndarray, data: NHPPData, **kwargs) -> dict:
        return _core.em_pareto_emstep(np.asarray(params, dtype=float), data.to_core_dict())

    def omega_(self) -> float:
        return float(self.params_[0])

    def shape_(self) -> float:
        return float(self.params_[1])

    def scale_(self) -> float:
        return float(self.params_[2])

    def mvf(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega_() * ppareto2(t, shape=self.shape_(), scale=self.scale_(), lower_tail=True)

    def dmvf(self, t):
        t = np.asarray(t, dtype=float)
        t2 = np.concatenate([[0.0], t])
        return np.diff(self.mvf(t2))

    def intensity(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega_() * dpareto2(t, shape=self.shape_(), scale=self.scale_(), log=False)


