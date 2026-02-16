from __future__ import annotations

import numpy as np

from ..base import NHPPModel
from ..data import NHPPData
from ..dists.llogis import dllogis, pllogis
from ... import _core
from ._utils import optimize_params


class LogLogisticNHPP(NHPPModel):
    name = "llogis"
    df = 3

    def __init__(self, *, omega0: float = 1.0, locationlog0: float = 0.0, scalelog0: float = 1.0):
        super().__init__(omega0=float(omega0), locationlog0=float(locationlog0), scalelog0=float(scalelog0))
        self._set_fitted_params(np.array([omega0, locationlog0, scalelog0], dtype=float))

    def param_names(self) -> list[str]:
        return ["omega", "locationlog", "scalelog"]

    def init_params(self, data: NHPPData) -> np.ndarray:
        return np.array([1.0, 1.0, max(np.log(data.mean), 1.0)], dtype=float)

    def em_step(self, params: np.ndarray, data: NHPPData, **kwargs) -> dict:
        core = data.to_core_dict()
        eres = _core.em_llogis_estep(np.asarray(params, dtype=float), core)
        method = kwargs.get("method", "L-BFGS-B")
        options = kwargs.get("options")
        loc_scale = optimize_params(
            _core.em_llogis_pllf,
            core,
            np.asarray(params[1:3], dtype=float),
            w1=float(eres["w1"]),
            method=method,
            options=options,
        )
        new_param = np.array([float(eres["omega"]), loc_scale[0], loc_scale[1]], dtype=float)
        return {
            "param": new_param,
            "pdiff": new_param - np.asarray(params, dtype=float),
            "llf": float(eres["llf"]),
            "total": float(eres["total"]),
        }

    def omega_(self) -> float:
        return float(self.params_[0])

    def locationlog_(self) -> float:
        return float(self.params_[1])

    def scalelog_(self) -> float:
        return float(self.params_[2])

    def mvf(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega_() * pllogis(t, locationlog=self.locationlog_(), scalelog=self.scalelog_(), lower_tail=True)

    def intensity(self, t):
        t = np.asarray(t, dtype=float)
        return self.omega_() * dllogis(t, locationlog=self.locationlog_(), scalelog=self.scalelog_(), log=False)


