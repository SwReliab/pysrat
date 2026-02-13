from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from .data import NHPPData


class NHPPModel(ABC):
    name: str
    df: int

    def __init__(self, **hyperparams):
        self._hyperparams = dict(hyperparams)
        self._fitted = False
        self.params_: np.ndarray | None = None
        self.llf_: float | None = None
        self.aic_: float | None = None
        self.converged_: bool | None = None
        self.n_iter_: int | None = None
        self.initial_params_: np.ndarray | None = None
        self.aerror_: float | None = None
        self.rerror_: float | None = None
        self.data_: NHPPData | None = None

    def get_params(self, deep: bool = True) -> dict:
        return dict(self._hyperparams)

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self._hyperparams:
                raise ValueError(f"Invalid parameter {k} for estimator {self.__class__.__name__}.")
            self._hyperparams[k] = v
        return self

    def _get_fitted_params(self) -> np.ndarray:
        if self.params_ is None:
            raise RuntimeError("Fitted parameters are not set.")
        return self.params_

    def _set_fitted_params(self, params: np.ndarray):
        self.params_ = np.asarray(params, dtype=float)

    def set_data(self, data: NHPPData):
        self.data_ = data

    @property
    def params(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        return self.params_

    @property
    def params_dict_(self) -> dict:
        if not self._fitted or self.params_ is None:
            raise RuntimeError("Model is not fitted.")
        names = self.param_names()
        vals = np.asarray(self.params_, dtype=float)
        if len(names) != len(vals):
            names = [f"p{i}" for i in range(len(vals))]
        return {k: float(v) for k, v in zip(names, vals)}

    def fit(
        self,
        data: NHPPData,
        *,
        initialize: bool = True,
        maxiter: int = 2000,
        reltol: float = 1.0e-6,
        abstol: float = 1.0e-3,
        trace: bool = False,
        printsteps: int = 50,
        **kwargs,
    ):
        from ._em import emfit_internal

        r = emfit_internal(
            self,
            data,
            initialize=initialize,
            maxiter=maxiter,
            reltol=reltol,
            abstol=abstol,
            trace=trace,
            printsteps=printsteps,
            **kwargs,
        )

        self._fitted = True
        self.initial_params_ = r.initial
        self.llf_ = r.llf
        self.converged_ = r.convergence
        self.n_iter_ = r.iter
        self.aerror_ = r.aerror
        self.rerror_ = r.rerror
        self.aic_ = float(-2.0 * self.llf_ + 2.0 * self.df)
        return self

    @abstractmethod
    def param_names(self) -> list[str]:
        ...

    @abstractmethod
    def init_params(self, data: NHPPData) -> np.ndarray:
        ...

    @abstractmethod
    def em_step(self, params: np.ndarray, data: NHPPData, **kwargs) -> dict:
        ...

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
