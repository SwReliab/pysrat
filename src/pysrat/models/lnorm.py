from __future__ import annotations

import numpy as np

from ..base import NHPPModel
from ..data.nhpp import NHPPData
from ..dists.lnorm import dlnorm, plnorm
from .. import _core


class LogNormalNHPP(NHPPModel):
    name = "lnorm"
    df = 3

    def __init__(self, *, omega0: float = 1.0, meanlog0: float = 0.0, sdlog0: float = 1.0):
        super().__init__(omega0=float(omega0), meanlog0=float(meanlog0), sdlog0=float(sdlog0))
        self._set_fitted_params(np.array([omega0, meanlog0, sdlog0], dtype=float))

    def param_names(self) -> list[str]:
        return ["omega", "meanlog", "sdlog"]

    def init_params(self, data: NHPPData) -> np.ndarray:
        return np.array([1.0, 1.0, max(np.log(data.mean), 1.0)], dtype=float)

    def em_step(self, params: np.ndarray, data: NHPPData, **kwargs) -> dict:
        return _core.em_lnorm_emstep(np.asarray(params, dtype=float), data.to_core_dict())

    def omega_(self) -> float:
        return float(self.params_[0])

    def meanlog_(self) -> float:
        return float(self.params_[1])

    def sdlog_(self) -> float:
        return float(self.params_[2])

    def mvf(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega_() * plnorm(t, meanlog=self.meanlog_(), sdlog=self.sdlog_(), lower_tail=True)

    def intensity(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega_() * dlnorm(t, meanlog=self.meanlog_(), sdlog=self.sdlog_(), log=False)


