from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .data import NHPPData
from .base import NHPPModel


def _normalize_results(results):
    if isinstance(results, NHPPModel):
        return {results.name: results}
    if isinstance(results, dict):
        return results
    if isinstance(results, (list, tuple)):
        out = {}
        for item in results:
            if isinstance(item, NHPPModel):
                out[item.name] = item
            else:
                raise TypeError("Unsupported results type")
        return out
    raise TypeError("Unsupported results type")


def plot_mvf(
    data: NHPPData,
    results,
    *,
    xlab: str = "time",
    ylab: str = "# of faults",
    datalab: str = "data",
    xmax=None,
    ymax=None,
    num_points: int = 300,
    ax=None,
):
    results = _normalize_results(results)
    present = float(np.sum(data.time))
    if xmax is None:
        xmax = present * 1.2

    n = data.fault + data.type
    x_data = np.cumsum(data.time)[n != 0]
    y_data = np.cumsum(n)[n != 0]

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(x_data, y_data, label=datalab)
    ax.axvline(present, linestyle="dotted")

    tgrid = np.linspace(0.0, float(xmax), num_points)
    for name, res in results.items():
        ax.plot(tgrid, res.mvf(tgrid), label=name)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    ax.set_xlim(0.0, float(xmax))
    if ymax is not None:
        ax.set_ylim(0.0, float(ymax))

    ax.legend()
    return ax


def plot_dmvf(
    data: NHPPData,
    results,
    *,
    xlab: str = "time",
    ylab: str = "# of faults",
    datalab: str = "data",
    xmax=None,
    ymax=None,
    ax=None,
):
    results = _normalize_results(results)
    t = np.cumsum(data.time).astype(float)
    n = (data.fault + data.type).astype(float)

    if ax is None:
        _, ax = plt.subplots()

    ax.bar(t, n, width=np.maximum(data.time, 1e-12), alpha=0.5, label=datalab, align="center")

    for name, res in results.items():
        ax.plot(t, res.dmvf(t), label=name)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if xmax is not None:
        ax.set_xlim(0.0, float(xmax))
    if ymax is not None:
        ax.set_ylim(0.0, float(ymax))

    ax.legend()
    return ax


def plot_rate(
    data: NHPPData,
    results,
    *,
    xlab: str = "time",
    ylab: str = "detection rate",
    datalab: str = "data",
    xmax=None,
    ymax=None,
    num_points: int = 300,
    ax=None,
):
    results = _normalize_results(results)
    present = float(np.sum(data.time))
    if xmax is None:
        xmax = present * 1.2

    t = np.cumsum(data.time)
    n = data.fault + data.type
    rate_emp = np.divide(n, data.time, where=data.time > 0)

    if ax is None:
        _, ax = plt.subplots()

    ax.bar(t, rate_emp, width=np.maximum(data.time, 1e-12), alpha=0.3, label=datalab)
    ax.axvline(present, linestyle="dotted")

    tgrid = np.linspace(0.1, float(xmax), num_points)
    for name, res in results.items():
        ax.plot(tgrid, res.rate(tgrid), label=name)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    ax.set_xlim(0.1, float(xmax))
    if ymax is not None:
        ax.set_ylim(0.0, float(ymax))

    ax.legend()
    return ax
