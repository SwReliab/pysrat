from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from .data import NHPPData


@dataclass
class NHPPModel(ABC):
    name: str = "nhpp"
    df: int = 0

    def __init__(self, params: np.ndarray | None = None):
        self._params = None if params is None else np.asarray(params, dtype=float)
        self._fitted = False
        self.data_ = None

    @property
    def params(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        return self._params

    def _get_params(self) -> np.ndarray | None:
        return self._params

    def set_params(self, params: np.ndarray) -> None:
        self._params = np.asarray(params, dtype=float)

    def set_data(self, data: NHPPData) -> None:
        self.data_ = data

    def param_names(self) -> list[str]:
        if self._params is None:
            return ["omega"]
        return ["omega"] + [f"p{i}" for i in range(1, len(self._params))]

    def fit(
        self,
        data: NHPPData,
        *,
        maxiter: int = 10000,
        reltol: float = 1.0e-8,
        abstol: float = 1.0e-12,
        trace: bool = False,
        printsteps: int = 50,
        **kwargs,
    ) -> "NHPPModel":
        from .emfit import emfit

        res = emfit(
            self,
            data,
            initialize=True,
            maxiter=maxiter,
            reltol=reltol,
            abstol=abstol,
            trace=trace,
            printsteps=printsteps,
            **kwargs,
        )

        self._fitted = True
        self.llf_ = float(res.llf)
        self.aic_ = float(-2.0 * res.llf + 2.0 * res.df)
        self.n_iter_ = int(res.iter)
        self.convergence_ = bool(res.convergence)
        return self

    @abstractmethod
    def mvf(self, t):
        ...

    @abstractmethod
    def intensity(self, t):
        ...

    def dmvf(self, t):
        return self.intensity(t)

    def rate(self, t):
        return self.intensity(t)

    @abstractmethod
    def em_step(self, params: np.ndarray, data: NHPPData, **kwargs) -> dict:
        """Return dict: param(np.ndarray), pdiff(np.ndarray), llf(float), total(float), ..."""
        ...

    @abstractmethod
    def init_params(self, data: NHPPData) -> np.ndarray:
        ...
