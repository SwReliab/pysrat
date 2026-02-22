from __future__ import annotations

import numpy as np

from ._dglm import DynamicGLMBase

class MFCloglogNHPP(DynamicGLMBase):
    name = "MFCloglogNHPP"
    link = "cloglog"

    def _linkinv(self, eta: np.ndarray) -> np.ndarray:
        # clip to avoid overflow in exp
        eta = np.clip(eta, -700.0, 700.0)
        return 1.0 - np.exp(-np.exp(eta))
