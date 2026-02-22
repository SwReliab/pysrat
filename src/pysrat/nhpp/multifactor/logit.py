from __future__ import annotations

import numpy as np

from ._dglm import DynamicGLMBase

class MFLogitNHPP(DynamicGLMBase):
    name = "MFLogitNHPP"
    link = "logit"

    def _linkinv(self, eta: np.ndarray) -> np.ndarray:
        # clip to avoid overflow in exp
        eta = np.clip(eta, -700.0, 700.0)

        # stable logistic:
        # η >= 0 : 1 / (1 + exp(-η))
        # η < 0  : exp(η) / (1 + exp(η))
        return np.where(
            eta >= 0.0,
            1.0 / (1.0 + np.exp(-eta)),
            np.exp(eta) / (1.0 + np.exp(eta)),
        )
