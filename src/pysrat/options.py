from __future__ import annotations

import numpy as np


def nhpp_options():
    return {
        "maxiter": 10000,
        "reltol": float(np.sqrt(np.finfo(float).eps)),
        "abstol": 1.0e200,
        "stopcond": "llf",
        "trace": False,
        "printsteps": 50,
    }
