from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from ._dglm import DynamicGLMBase

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return ndtr(x)


class MFProbitNHPP(DynamicGLMBase):
    name = "MFProbitNHPP"
    link = "probit"

    def _linkinv(self, eta: np.ndarray) -> np.ndarray:
        return _norm_cdf(eta)
