from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NHPPData:
    time: np.ndarray
    fault: np.ndarray
    type: np.ndarray
    total: int
    mean: float
    max: float
    kind: str

    @property
    def len(self) -> int:
        return int(np.asarray(self.time).shape[0])

    def to_core_dict(self) -> dict:
        return {
            "len": self.len,
            "time": np.asarray(self.time, dtype=float),
            "fault": np.asarray(self.fault, dtype=float),
            "type": np.asarray(self.type, dtype=np.int64),
        }

    @classmethod
    def from_intervals(cls, *, time=None, fault=None, type=None, te=None) -> "NHPPData":
        time = None if time is None else np.asarray(time, dtype=float)
        fault = None if fault is None else np.asarray(fault, dtype=int)
        type = None if type is None else np.asarray(type, dtype=int)
        te = None if te is None else float(te)

        if time is None:
            if fault is None:
                raise ValueError("Invalid data: Either time or fault is required.")
            if type is None:
                type = np.zeros_like(fault, dtype=int)
            time = np.ones_like(fault, dtype=float)
        else:
            if fault is None:
                if type is None:
                    if te is None:
                        raise ValueError("Invalid data: Either type or te is required when fault is missing.")
                    type = np.ones_like(time, dtype=int)
                    time = np.concatenate([time, np.array([te], dtype=float)])
                    type = np.concatenate([type, np.array([0], dtype=int)])
                    fault = np.zeros_like(time, dtype=int)
                else:
                    fault = np.zeros_like(time, dtype=int)
            else:
                if type is None:
                    type = np.zeros_like(time, dtype=int)

        if time.shape[0] != fault.shape[0] or time.shape[0] != type.shape[0]:
            raise ValueError("Invalid data")
        if np.all((fault == 0) & (type == 0)):
            raise ValueError("Invalid data: no fault.")
        if np.any((time == 0) & (fault != 0) & (type != 0)):
            raise ValueError("Invalid data: zero time exits.")

        tmp = fault + type
        total = int(tmp.sum())
        ct = np.cumsum(time)
        mean = float((ct * tmp).sum() / total)
        maxv = float(ct[tmp >= 1].max())

        return cls(
            time=np.asarray(time, dtype=float),
            fault=np.asarray(fault, dtype=int),
            type=np.asarray(type, dtype=int),
            total=total,
            mean=mean,
            max=maxv,
            kind="intervals",
        )

    @classmethod
    def from_counts(cls, fault, *, type=None) -> "NHPPData":
        data = cls.from_intervals(time=None, fault=fault, type=type, te=None)
        return cls(
            time=data.time,
            fault=data.fault,
            type=data.type,
            total=data.total,
            mean=data.mean,
            max=data.max,
            kind="counts",
        )

    @classmethod
    def from_fault_times(cls, times, *, te) -> "NHPPData":
        data = cls.from_intervals(time=times, fault=None, type=None, te=te)
        return cls(
            time=data.time,
            fault=data.fault,
            type=data.type,
            total=data.total,
            mean=data.mean,
            max=data.max,
            kind="fault_times",
        )

    def __repr__(self) -> str:
        cols = np.vstack([self.time, self.fault, self.type]).T
        head = "NHPPData(time, fault, type)\n"
        body = "\n".join([f"{r[0]:g}\t{int(r[1])}\t{int(r[2])}" for r in cols[:50]])
        if cols.shape[0] > 50:
            body += "\n..."
        meta = f"\nlen={self.len}, total={self.total}, mean={self.mean:g}, max={self.max:g}"
        return head + "time\tfault\ttype\n" + body + meta

@dataclass
class DMetricsData:
    metrics: np.ndarray  # (n, p)
    offset: np.ndarray   # (n,)
    fault: np.ndarray    # (n,)
    metrics_name: list | None = None
    total: float | None = None  # None -> sum(fault)

    def __post_init__(self):
        self.metrics = np.asarray(self.metrics, dtype=float)
        self.offset = np.asarray(self.offset, dtype=float)
        self.fault = np.asarray(self.fault, dtype=float)

        if self.metrics.ndim != 2:
            raise ValueError("metrics must be 2D array (n, p).")
        n = self.metrics.shape[0]
        p = self.metrics.shape[1]
        if self.offset.shape != (n,):
            raise ValueError("offset must have shape (n,).")
        if self.fault.shape != (n,):
            raise ValueError("fault must have shape (n,).")
        if self.metrics_name is None:
            self.metrics_name = [f"m{i+1}" for i in range(p)]
        else:
            if len(self.metrics_name) != p:
                raise ValueError("metrics_name must have length equal to number of metrics.")

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
        intercept: bool = True,
        total: float | None = None,
    ) -> "DMetricsData":
        if isinstance(metrics, (list, tuple)):
            metrics_arr = df[metrics].to_numpy(dtype=float)
            if metrics_name is None:
                metrics_name = list(metrics)
        else:
            metrics_arr = np.asarray(metrics, dtype=float)

        if intercept:
            metrics_arr = np.column_stack([np.ones(metrics_arr.shape[0], dtype=float), metrics_arr])
            if metrics_name is None:
                metrics_name = ["intercept"]
            else:
                metrics_name = ["intercept", *list(metrics_name)]

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

__all__ = ["NHPPData", "DMetricsData"]
