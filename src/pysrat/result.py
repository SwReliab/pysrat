from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .nhpp import NHPPModel


@dataclass
class FitNHPPResult:
    initial: np.ndarray
    srm: NHPPModel
    llf: float
    df: int
    convergence: bool
    iter: int
    aerror: float
    rerror: float
    aic: float
    ctime: float
    call: str | None = None

    @property
    def params_dict(self) -> dict:
        names = self.srm.param_names()
        values = np.asarray(self.srm.params, dtype=float)
        if len(names) != len(values):
            names = [f"p{i}" for i in range(len(values))]
        return {k: float(v) for k, v in zip(names, values)}

    def mvf(self, t):
        return self.srm.mvf(t)

    def dmvf(self, t):
        return self.srm.dmvf(t)

    def rate(self, t):
        return self.srm.intensity(t)

    def plot_mvf(self, *args, **kwargs):
        from .plot import plot_mvf
        return plot_mvf(*args, results={self.srm.name: self}, **kwargs)

    def plot_dmvf(self, *args, **kwargs):
        from .plot import plot_dmvf
        return plot_dmvf(*args, results={self.srm.name: self}, **kwargs)

    def plot_rate(self, *args, **kwargs):
        from .plot import plot_rate
        return plot_rate(*args, results={self.srm.name: self}, **kwargs)

    def __repr__(self) -> str:
        lines = []
        lines.append(f"Model name: {self.srm.name}")
        lines.append(f"params: {self.params_dict}")
        lines.append(f"Maximum LLF: {self.llf}")
        lines.append(f"AIC: {self.aic}")
        lines.append(f"Convergence: {self.convergence}")
        lines.append(f"Iter: {self.iter}")
        return "\n".join(lines)


@dataclass
class FitCollection:
    all: dict[str, FitNHPPResult]
    best: FitNHPPResult

    def __getitem__(self, name: str) -> FitNHPPResult:
        return self.all[name]

    def keys(self):
        return self.all.keys()

    def items(self):
        return self.all.items()

    def plot_mvf(self, data, **kwargs):
        from .plot import plot_mvf
        return plot_mvf(data=data, results=self.all, **kwargs)

    def plot_dmvf(self, data, **kwargs):
        from .plot import plot_dmvf
        return plot_dmvf(data=data, results=self.all, **kwargs)

    def plot_rate(self, data, **kwargs):
        from .plot import plot_rate
        return plot_rate(data=data, results=self.all, **kwargs)
