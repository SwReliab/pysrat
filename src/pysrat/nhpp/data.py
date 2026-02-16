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


__all__ = ["NHPPData"]
