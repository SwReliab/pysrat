from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import t as _t_dist

from .base import NHPPModel
from .data.nhpp import NHPPData

__all__ = [
    "FitResult",
    "emfit",
    "bs_time",
    "bs_group",
    "eic_group",
    "eic_time",
    "eic_sample",
    "eic",
]


@dataclass
class FitResult:
    srm: NHPPModel
    llf: float
    data: NHPPData

    @classmethod
    def from_model(cls, model: NHPPModel, data: NHPPData | None = None) -> "FitResult":
        if data is None:
            if model.data_ is None:
                raise ValueError("data must be provided when model has no fitted data")
            data = model.data_
        if model.llf_ is None:
            raise ValueError("model.llf_ is not set; call fit() first")
        return cls(srm=model, llf=float(model.llf_), data=data)


def _clone_model(model: NHPPModel) -> NHPPModel:
    params = copy.deepcopy(model.get_params())
    return model.__class__(**params)


def emfit(model: NHPPModel, data: NHPPData, *, initialize: bool = True, **kwargs) -> FitResult:
    cloned = _clone_model(model)
    cloned.fit(data, initialize=initialize, **kwargs)
    return FitResult(srm=cloned, llf=float(cloned.llf_), data=data)


def _as_fitresult(obj: FitResult | NHPPModel, data: NHPPData | None = None) -> FitResult:
    if isinstance(obj, FitResult):
        return obj
    if isinstance(obj, NHPPModel):
        return FitResult.from_model(obj, data=data)
    raise TypeError("obj must be FitResult or NHPPModel")


def _intensity_scalar(model: NHPPModel, t: float) -> float:
    return float(np.asarray(model.intensity(np.array([t], dtype=float)))[0])


def bs_time(srm: NHPPModel, data: NHPPData, maximum: float, *, rng: np.random.Generator | None = None) -> NHPPData:
    if maximum <= 0:
        raise ValueError("maximum must be > 0")
    rng = rng if rng is not None else np.random.default_rng()

    cte = float(np.sum(data.time))

    n = rng.poisson(maximum * cte)
    if n == 0:
        raise ValueError("bootstrap produced no events; increase maximum or retry")

    candidates = np.sort(rng.uniform(0.0, cte, size=int(n)))
    keep = srm.intensity(candidates) / maximum > rng.uniform(0.0, 1.0, size=int(n))
    y = candidates[keep]
    if y.size == 0:
        raise ValueError("bootstrap produced no events; increase maximum or retry")

    time = np.diff(np.concatenate([[0.0], y]))
    te_res = float(cte - np.sum(time))
    return NHPPData.from_intervals(time=time, te=te_res)


def bs_group(srm: NHPPModel, data: NHPPData, *, rng: np.random.Generator | None = None) -> NHPPData:
    rng = rng if rng is not None else np.random.default_rng()

    ctime = np.concatenate([[0.0], np.cumsum(np.asarray(data.time, dtype=float))])
    lam = np.diff(srm.mvf(ctime))
    lam = np.clip(lam, 0.0, None)
    fault = rng.poisson(lam)
    return NHPPData.from_counts(fault=fault)


def _ensure_llf(model: NHPPModel):
    if not hasattr(model, "llf"):
        raise NotImplementedError("model.llf is required for EIC bootstrap")


def _faultdata_kind(data: NHPPData) -> str:
    return data.kind


def eic_group(obj: FitResult | NHPPModel, bsample: int = 100, *, initialize: bool = False,
              rng: np.random.Generator | None = None, data: NHPPData | None = None) -> np.ndarray:
    obj = _as_fitresult(obj, data=data)
    _ensure_llf(obj.srm)

    rng = rng if rng is not None else np.random.default_rng()

    b3 = float(obj.llf)
    b1 = np.zeros(int(bsample), dtype=float)
    b2 = np.zeros(int(bsample), dtype=float)
    b4 = np.zeros(int(bsample), dtype=float)

    data = obj.data
    ctime = np.concatenate([[0.0], np.cumsum(np.asarray(data.time, dtype=float))])
    lam = np.diff(obj.srm.mvf(ctime))

    for b in range(int(bsample)):
        fault = rng.poisson(lam)
        sample = NHPPData.from_counts(fault=fault)
        b2[b] = float(obj.srm.llf(sample))
        obj_bs = emfit(obj.srm, sample, initialize=initialize)
        b1[b] = float(obj_bs.llf)
        b4[b] = float(obj_bs.srm.llf(obj.data))

    return b1 - b2 + b3 - b4


def eic_time(obj: FitResult | NHPPModel, bsample: int = 100, *, initialize: bool = False,
             rng: np.random.Generator | None = None, data: NHPPData | None = None) -> np.ndarray:
    obj = _as_fitresult(obj, data=data)
    _ensure_llf(obj.srm)

    rng = rng if rng is not None else np.random.default_rng()

    b3 = float(obj.llf)
    b1 = np.zeros(int(bsample), dtype=float)
    b2 = np.zeros(int(bsample), dtype=float)
    b4 = np.zeros(int(bsample), dtype=float)

    data = obj.data
    cte = float(np.sum(data.time))

    xs = np.linspace(0.0, cte, 2048)
    vals = np.asarray(obj.srm.intensity(xs), dtype=float)
    maximum = float(np.max(vals)) * 1.05

    for b in range(int(bsample)):
        sample = bs_time(obj.srm, data, maximum, rng=rng)
        b2[b] = float(obj.srm.llf(sample))
        obj_bs = emfit(obj.srm, sample, initialize=initialize)
        b1[b] = float(obj_bs.llf)
        b4[b] = float(obj_bs.srm.llf(obj.data))

    return b1 - b2 + b3 - b4


def eic_sample(obj: FitResult | NHPPModel, bsample: int = 100, *, initialize: bool = False,
               rng: np.random.Generator | None = None, data: NHPPData | None = None) -> np.ndarray:
    obj = _as_fitresult(obj, data=data)
    kind = _faultdata_kind(obj.data)
    if kind == "fault_times":
        bias_sample = eic_time(obj, bsample, initialize=initialize, rng=rng)
    elif kind == "counts":
        bias_sample = eic_group(obj, bsample, initialize=initialize, rng=rng)
    else:
        raise ValueError("EIC supports only data from from_counts or from_fault_times.")
    return -2.0 * (float(obj.llf) - bias_sample)


def eic(obj: FitResult | NHPPModel, bsample: int = 100, alpha: float = 0.95, *, initialize: bool = False,
    rng: np.random.Generator | None = None, data: NHPPData | None = None) -> dict:
    obj = _as_fitresult(obj, data=data)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    kind = _faultdata_kind(obj.data)
    if kind == "fault_times":
        bias_sample = eic_time(obj, bsample, initialize=initialize, rng=rng)
    elif kind == "counts":
        bias_sample = eic_group(obj, bsample, initialize=initialize, rng=rng)
    else:
        raise ValueError("EIC supports only data from from_counts or from_fault_times.")

    x = float(np.mean(bias_sample))
    s = float(np.std(bias_sample, ddof=1))
    df = int(bsample) - 1
    talpha = float(_t_dist.ppf(1.0 - (1.0 - alpha) / 2.0, df=df))
    x_interval = x + np.array([-talpha, talpha], dtype=float) * s / np.sqrt(float(bsample))

    return {
        "bias": x,
        "bias.lower": float(x_interval[0]),
        "bias.upper": float(x_interval[1]),
        "eic": -2.0 * (float(obj.llf) - x),
        "eic.lower": -2.0 * (float(obj.llf) - float(x_interval[0])),
        "eic.upper": -2.0 * (float(obj.llf) - float(x_interval[1])),
    }
