from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DMetricsData:
    metrics: np.ndarray  # (n, p)  NOTE: NO intercept column
    offset: np.ndarray   # (n,)
    fault: np.ndarray    # (n,)
    metrics_name: list[str] | None = None
    total: float | None = None  # None -> sum(fault)

    def __post_init__(self):
        self.metrics = np.asarray(self.metrics, dtype=float)
        self.offset = np.asarray(self.offset, dtype=float)
        self.fault = np.asarray(self.fault, dtype=float)

        if self.metrics.ndim != 2:
            raise ValueError("metrics must be 2D array (n, p).")
        n, p = self.metrics.shape

        if self.offset.shape != (n,):
            raise ValueError("offset must have shape (n,).")
        if self.fault.shape != (n,):
            raise ValueError("fault must have shape (n,).")

        if self.metrics_name is None:
            self.metrics_name = [f"m{i+1}" for i in range(p)]
        else:
            if len(self.metrics_name) != p:
                raise ValueError("metrics_name must have length equal to number of metrics.")
            if "intercept" in self.metrics_name:
                raise ValueError(
                    "metrics_name must not include 'intercept'. "
                    "Intercept is controlled by the model (has_intercept), not a column in metrics."
                )

    @property
    def n(self) -> int:
        return int(self.metrics.shape[0])

    @property
    def nmetrics(self) -> int:
        return int(self.metrics.shape[1])

    @property
    def total_faults(self) -> float:
        return float(np.sum(self.fault)) if self.total is None else float(self.total)

    @classmethod
    def from_arrays(
        cls,
        *,
        metrics,
        fault,
        offset=None,
        metrics_name=None,
        total: float | None = None,
    ) -> "DMetricsData":
        metrics = np.asarray(metrics, dtype=float)
        fault = np.asarray(fault, dtype=float)

        if metrics.ndim != 2:
            raise ValueError("metrics must be 2D array (n, p).")

        if offset is None:
            offset = np.zeros(metrics.shape[0], dtype=float)
        else:
            offset = np.asarray(offset, dtype=float)

        return cls(
            metrics=metrics,
            metrics_name=metrics_name,
            offset=offset,
            fault=fault,
            total=total,
        )

    @classmethod
    def from_dataframe(
        cls,
        df,
        *,
        metrics,
        fault,
        offset=None,
        metrics_name=None,
        total: float | None = None,
        intercept: bool | None = None,  # kept for compatibility, but forbidden if True
    ) -> "DMetricsData":
        # Compatibility guard: we do NOT allow intercept columns anymore
        if intercept:
            raise ValueError(
                "intercept=True is no longer supported. "
                "Do not add an intercept column to metrics; use model.has_intercept instead."
            )

        if isinstance(metrics, (list, tuple)):
            metrics_arr = df[list(metrics)].to_numpy(dtype=float)
            if metrics_name is None:
                metrics_name = list(metrics)
        else:
            metrics_arr = np.asarray(metrics, dtype=float)

        if isinstance(fault, str):
            fault_arr = df[fault].to_numpy(dtype=float)
        else:
            fault_arr = np.asarray(fault, dtype=float)

        if offset is None:
            offset_arr = np.zeros(metrics_arr.shape[0], dtype=float)
        elif isinstance(offset, str):
            offset_arr = df[offset].to_numpy(dtype=float)
        else:
            offset_arr = np.asarray(offset, dtype=float)

        return cls(
            metrics=metrics_arr,
            metrics_name=metrics_name,
            offset=offset_arr,
            fault=fault_arr,
            total=total,
        )