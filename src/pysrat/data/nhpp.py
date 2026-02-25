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

    @property
    def total_time(self) -> float:
        return float(np.sum(self.time))

    @property
    def total_fault(self) -> float:
        return float(np.sum(self.fault) + np.sum(self.type))

    def to_core_dict(self) -> dict:
        return {
            "len": self.len,
            "time": np.asarray(self.time, dtype=float),
            "fault": np.asarray(self.fault, dtype=float),
            "type": np.asarray(self.type, dtype=np.int64),
        }

    @classmethod
    def from_intervals(cls, *, intervals=None, counts=None, on_boundary=None, te=None) -> "NHPPData":
        time = None if intervals is None else np.asarray(intervals, dtype=float)
        fault = None if counts is None else np.asarray(counts, dtype=int)
        type = None if on_boundary is None else np.asarray(on_boundary, dtype=int)
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

        if np.all(type == 0):
            kind = "counts"
        elif np.all(fault == 0) and np.any(type == 1):
            kind = "fault_times"
        else:
            kind = "mixed"

        return cls(
            time=np.asarray(time, dtype=float),
            fault=np.asarray(fault, dtype=int),
            type=np.asarray(type, dtype=int),
            total=total,
            mean=mean,
            max=maxv,
            kind=kind,
        )

    @classmethod
    def from_event_times(cls, intervals, *, te) -> "NHPPData":
        counts = np.zeros_like(intervals, dtype=int)
        on_boundary = np.zeros_like(intervals, dtype=int)
        return cls.from_intervals(intervals=intervals, counts=counts, on_boundary=on_boundary, te=te)

    @classmethod
    def from_dataframe(
        cls,
        df,
        *,
        intervals: str | None = None,
        counts: str | None = None,
        boundary: str | None = None,
        times: str | None = None,
        te: float | None = None,
    ):
        """
        Create NHPPData from a pandas DataFrame.

        Parameters
        ----------
        intervals : column name of interval lengths
        counts : column name of counts
        boundary : column name of boundary indicator
        times : column name of event times (mutually exclusive with intervals)
        te : last observation time (required if times is given)
        """

        import numpy as np

        # ---- event times mode ----
        if times is not None:
            if intervals is not None:
                raise ValueError("Specify either times or intervals, not both.")
            if te is None:
                raise ValueError("te must be provided when using times.")
            return cls.from_event_times(
                times=np.asarray(df[times], dtype=float),
                te=te,
            )

        # ---- intervals mode ----
        if intervals is None and counts is None:
            raise ValueError("At least intervals or counts must be specified.")

        intervals_arr = (
            None if intervals is None
            else np.asarray(df[intervals], dtype=float)
        )

        counts_arr = (
            None if counts is None
            else np.asarray(df[counts], dtype=int)
        )

        boundary_arr = (
            None if boundary is None
            else np.asarray(df[boundary], dtype=int)
        )

        return cls.from_intervals(
            intervals=intervals_arr,
            counts=counts_arr,
            on_boundary=boundary_arr,
        )

    @classmethod
    def from_csv(
        cls,
        path,
        *,
        intervals: str | None = None,
        counts: str | None = None,
        boundary: str | None = None,
        times: str | None = None,
        te: float | None = None,
        **read_csv_kwargs,
    ):
        """
        Create NHPPData from a CSV file.

        Parameters
        ----------
        path : str or path-like
            Path to CSV file.

        intervals : str, optional
            Column name of interval lengths.

        counts : str, optional
            Column name of fault counts.

        boundary : str, optional
            Column name of boundary indicator.

        times : str, optional
            Column name of event times (mutually exclusive with intervals).

        te : float, optional
            Last observation time (required if times is given).

        read_csv_kwargs :
            Additional keyword arguments passed to pandas.read_csv().
        """

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for from_csv(). "
                "Install it with: pip install pandas "
                "or pip install pysrat[io]"
            ) from e

        df = pd.read_csv(path, **read_csv_kwargs)

        return cls.from_dataframe(
            df,
            intervals=intervals,
            counts=counts,
            boundary=boundary,
            times=times,
            te=te,
        )

    def __repr__(self) -> str:
        n = self.len
        head_n = min(n, 20)

        total_time = float(np.sum(self.time)) if n else 0.0

        header = (
            f"NHPPData(kind='{self.kind}', "
            f"len={n}, total_faults={int(self.total)}, "
            f"total_time={total_time:.6g})\n"
            + "-" * 58 + "\n"
            + f"{'idx':>4} {'intervals':>10} {'counts':>8} {'on_boundary':>11}\n"
        )

        rows = []
        for i in range(head_n):
            rows.append(
                f"{i:4d} "
                f"{float(self.time[i]):10.6g} "
                f"{int(self.fault[i]):8d} "
                f"{int(self.type[i]):11d}"
            )

        if n > head_n:
            rows.append("   ...")

        footer = (
            "\n" + "-" * 58 +
            f"\nmean={float(self.mean):.6g}  "
            f"max={float(self.max):.6g}"
        )

        return header + "\n".join(rows) + footer

    def truncate(
        self,
        *,
        t_end: float | None = None,
        frac: float | None = None,
    ) -> "NHPPData":

        if (t_end is None) == (frac is None):
            raise ValueError("Specify exactly one of t_end or frac.")

        total = self.total_time
        if frac is not None:
            if not (0.0 < frac <= 1.0):
                raise ValueError("frac must be in (0, 1].")
            t_end = frac * total

        t_end = float(t_end)
        if t_end <= 0.0:
            raise ValueError("t_end must be positive.")

        ct = np.cumsum(self.time)
        if t_end > float(ct[-1]):
            raise ValueError(f"t_end ({t_end}) exceeds total_time ({float(ct[-1])}).")

        # get the index of the first interval that ends after t_end
        idx = int(np.searchsorted(ct, t_end, side="left"))
        if idx == 0 and t_end < float(ct[0]):
            raise ValueError("t_end is before the first interval.")

        # truncate the data at idx, adjusting the last interval if necessary
        end = idx + 1

        time = np.array(self.time[:end], dtype=float)
        fault = np.array(self.fault[:end], dtype=int)
        type_ = np.array(self.type[:end], dtype=int)

        over = float(ct[idx] - t_end)  # the amount of time to cut from the last interval
        if over > 0.0:
            # If the last interval is cut, we need to adjust the counts and boundary indicators.
            if fault[-1] != 0:
                raise ValueError(
                    "Cannot truncate inside an interval with nonzero counts; "
                    "the correct split of counts is undefined."
                )
            time[-1] -= over
            # If the last interval is on the boundary, we need to move the boundary indicator to the new end point.
            # type_[-1] = 0

        if np.isclose(time[-1], 0.0):
            time = time[:-1]
            fault = fault[:-1]
            type_ = type_[:-1]
            if len(time) == 0:
                raise ValueError("t_end truncates all intervals to zero length.")

        return NHPPData.from_intervals(
            intervals=time,
            counts=fault,
            on_boundary=type_,
        )
