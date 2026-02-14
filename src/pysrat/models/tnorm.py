from __future__ import annotations

import numpy as np

from ..base import NHPPModel
from ..data import NHPPData
from ..dists.tnorm import ptnorm, dtnorm
from .. import _core


class TruncatedNormalNHPP(NHPPModel):
    name = "tnorm"
    df = 3

    def __init__(self, *, omega0: float = 1.0, mu0: float = 0.0, sig0: float = 1.0):
        super().__init__(omega0=float(omega0), mu0=float(mu0), sig0=float(sig0))
        self._set_fitted_params(np.array([omega0, mu0, sig0], dtype=float))

    def param_names(self) -> list[str]:
        return ["omega", "mu", "sig"]

    def init_params(self, data: NHPPData) -> np.ndarray:
        return np.array([1.0, 0.0, data.mean], dtype=float)

    def em_step(self, params: np.ndarray, data: NHPPData, **kwargs) -> dict:
        return _core.em_tnorm_emstep(np.asarray(params, dtype=float), data.to_core_dict())

    def omega_(self) -> float:
        return float(self.params_[0])

    def mu_(self) -> float:
        return float(self.params_[1])

    def sig_(self) -> float:
        return float(self.params_[2])

    def mvf(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega_() * ptnorm(t, mean=self.mu_(), sd=self.sig_(), lower_tail=True)

    def intensity(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega_() * dtnorm(t, mean=self.mu_(), sd=self.sig_(), log=False)


