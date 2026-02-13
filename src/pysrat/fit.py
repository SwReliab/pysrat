from __future__ import annotations

from .data import NHPPData
from .base import NHPPModel


def compare(models: list[NHPPModel], data: NHPPData, *, criterion: str = "AIC"):
    fitted = [m.fit(data) for m in models]
    crit = criterion.upper()
    if crit != "AIC":
        raise ValueError(f"Unsupported criterion: {criterion}")
    best = min(fitted, key=lambda m: m.aic_)
    return fitted, best
