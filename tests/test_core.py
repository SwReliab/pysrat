import numpy as np
import pysrat


def test_sum_core():
    x = np.array([1.0, 2.0, 3.0])
    assert pysrat.sum(x) == 6.0


def test_em_exp_emstep_core():
    params = np.array([10.0, 0.2], dtype=float)
    data = {
        "len": 3,
        "time": np.array([1.0, 2.0, 1.5], dtype=float),
        "fault": np.array([0.0, 2.0, 1.0], dtype=float),
        "type": np.array([0, 1, 0], dtype=np.int64),
    }
    out = pysrat.em_exp_emstep(params, data)
    assert "param" in out and "pdiff" in out and "llf" in out and "total" in out
    assert out["param"].shape == (2,)
    assert out["pdiff"].shape == (2,)
