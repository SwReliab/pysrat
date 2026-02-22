from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, List
import numpy as np


@dataclass
class SMetricsData:
    """
    Container for static metrics used in PR-NHPP.

    Parameters
    ----------
    names : (m,)
        Unique module/component names (row keys).
    metrics : (m, q)
        Design matrix for static metrics (WITHOUT intercept column).
    metrics_name : optional (q,)
        Column names for metrics.
    offset : optional (m,)
        Offset for Poisson regression (e.g., log(LOC)).
    standardize : optional (q,)
        0/1 vector indicating which columns to standardize.
    """

    names: Sequence[str]
    metrics: np.ndarray
    metrics_name: Optional[Sequence[str]] = None
    offset: Optional[np.ndarray] = None
    standardize: Optional[np.ndarray] = None

    _index: Dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self.names = np.asarray(self.names, dtype=object)
        self.metrics = np.asarray(self.metrics, dtype=float)

        if self.names.ndim != 1:
            raise ValueError("names must be 1D")

        if self.metrics.ndim != 2:
            raise ValueError("metrics must be 2D")

        m, q = self.metrics.shape

        if self.names.shape[0] != m:
            raise ValueError("metrics rows must match len(names)")

        # uniqueness check
        names_list = [str(nm) for nm in self.names.tolist()]
        if len(set(names_list)) != len(names_list):
            raise ValueError("names must be unique")

        # build index
        self._index = {str(nm): i for i, nm in enumerate(names_list)}

        # metrics_name check
        if self.metrics_name is not None:
            if len(self.metrics_name) != q:
                raise ValueError("metrics_name length must match metrics columns")

        # offset check
        if self.offset is not None:
            self.offset = np.asarray(self.offset, dtype=float)
            if self.offset.shape != (m,):
                raise ValueError("offset must have shape (m,)")

        # standardize check
        if self.standardize is not None:
            self.standardize = np.asarray(self.standardize, dtype=int)
            if self.standardize.shape != (q,):
                raise ValueError("standardize must have shape (q,)")

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df,
        *,
        name_col: Optional[str] = None,
        metrics_cols: Optional[Sequence[str]] = None,
        offset_col: Optional[str] = None,
        standardize_cols: Optional[Sequence[str]] = None,
        use_index_as_name: bool = False,
    ) -> "SMetricsData":
        """
        Construct SMetricsData from a pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
        name_col : column name containing module names
        metrics_cols : columns to use as metrics (default: all numeric except name/offset)
        offset_col : optional column used as offset
        standardize_cols : subset of metrics_cols to standardize
        use_index_as_name : if True, use df.index as names (ignored if name_col given)
        """

        try:
            import pandas as pd  # noqa
        except ImportError:
            raise ImportError("pandas is required for from_dataframe")

        if not hasattr(df, "columns"):
            raise TypeError("df must be a pandas DataFrame")

        # ----------------------------
        # names
        # ----------------------------
        if name_col is not None:
            if name_col not in df.columns:
                raise ValueError(f"{name_col!r} not found in DataFrame")
            names = df[name_col].astype(str).to_numpy()
        elif use_index_as_name:
            names = df.index.astype(str).to_numpy()
        else:
            raise ValueError("Either name_col must be given or use_index_as_name=True")

        # ----------------------------
        # metrics columns
        # ----------------------------
        if metrics_cols is None:
            # default: numeric columns except name_col and offset_col
            exclude = set()
            if name_col is not None:
                exclude.add(name_col)
            if offset_col is not None:
                exclude.add(offset_col)

            metrics_cols = [
                c for c in df.columns
                if c not in exclude and np.issubdtype(df[c].dtype, np.number)
            ]

        if len(metrics_cols) == 0:
            raise ValueError("No metric columns selected")

        metrics = df[list(metrics_cols)].to_numpy(dtype=float)

        # ----------------------------
        # offset
        # ----------------------------
        offset = None
        if offset_col is not None:
            if offset_col not in df.columns:
                raise ValueError(f"{offset_col!r} not found in DataFrame")
            offset = df[offset_col].to_numpy(dtype=float)

        # ----------------------------
        # standardize
        # ----------------------------
        standardize = None
        if standardize_cols is not None:
            standardize = np.zeros(len(metrics_cols), dtype=int)
            for col in standardize_cols:
                if col not in metrics_cols:
                    raise ValueError(f"{col!r} not in metrics_cols")
                idx = metrics_cols.index(col)
                standardize[idx] = 1

        return cls(
            names=names,
            metrics=metrics,
            metrics_name=list(metrics_cols),
            offset=offset,
            standardize=standardize,
        )

    # ------------------------------------------------------------------
    # basic properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return self.metrics.shape[0]

    @property
    def nmetrics(self) -> int:
        return self.metrics.shape[1]

    # ------------------------------------------------------------------
    # indexing helpers
    # ------------------------------------------------------------------

    def index_of(self, name: str) -> int:
        try:
            return self._index[str(name)]
        except KeyError:
            raise KeyError(f"{name!r} not found in SMetricsData") from None

    def row(self, name: str) -> np.ndarray:
        return self.metrics[self.index_of(name)]

    # ------------------------------------------------------------------
    # reordering / subsetting
    # ------------------------------------------------------------------

    def reorder(self, names_order: Sequence[str]) -> "SMetricsData":
        idx = [self.index_of(nm) for nm in names_order]
        return SMetricsData(
            names=[str(nm) for nm in names_order],
            metrics=self.metrics[idx, :],
            metrics_name=self.metrics_name,
            offset=None if self.offset is None else self.offset[idx],
            standardize=self.standardize,
        )

    def subset(self, names_subset: Sequence[str]) -> "SMetricsData":
        return self.reorder(names_subset)

    def select_metrics(self, cols: Sequence[str]) -> "SMetricsData":
        """
        Return a new SMetricsData using only selected metric columns.
        """
        if self.metrics_name is None:
            raise ValueError("metrics_name is required to select by column name")

        idx = []
        for c in cols:
            if c not in self.metrics_name:
                raise KeyError(f"{c!r} not found in metrics_name")
            idx.append(self.metrics_name.index(c))

        new_metrics = self.metrics[:, idx]
        new_standardize = None
        if self.standardize is not None:
            new_standardize = self.standardize[idx]

        return SMetricsData(
            names=self.names,
            metrics=new_metrics,
            metrics_name=[self.metrics_name[i] for i in idx],
            offset=self.offset,
            standardize=new_standardize,
        )

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------

    def as_matrix(self) -> np.ndarray:
        return self.metrics

    def __len__(self):
        return self.n

    def __repr__(self) -> str:
        return (
            f"SMetricsData(n={self.n}, "
            f"nmetrics={self.nmetrics}, "
            f"names={list(self.names)})"
        )