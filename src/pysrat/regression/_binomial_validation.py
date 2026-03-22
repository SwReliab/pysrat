from __future__ import annotations

import numpy as np


def prepare_binomial_response(
    y: np.ndarray,
    n_trials: np.ndarray,
    *,
    y_is_proportion: bool,
) -> np.ndarray:
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite")

    if not np.all(np.isfinite(n_trials)):
        raise ValueError("n_trials must be finite")

    if np.any(n_trials < 0.0):
        raise ValueError("n_trials must be >= 0")

    if y_is_proportion:
        if np.any((y < 0.0) | (y > 1.0)):
            raise ValueError("proportions must satisfy 0 <= y <= 1")
        return y * n_trials

    if np.any((y < 0.0) | (y > n_trials)):
        raise ValueError("success counts must satisfy 0 <= y <= n_trials")
    return y