from pysrat.data import NHPPData
from pysrat.data import DMetricsData

import numpy as np
import pandas as pd

def test_nhppdata_constructors():
    d1 = NHPPData.from_intervals(time=[1, 1, 1, 1], fault=[0, 1, 0, 5])
    assert d1.len == 4
    assert d1.total == 6

    d2 = NHPPData.from_fault_times([3, 1, 7, 15, 12], te=3)
    assert d2.len == 6

    d3 = NHPPData.from_counts([0, 1, 0, 5])
    assert d3.total == 6

def test_dmetricsdata_from_arrays_defaults():
    rng = np.random.default_rng(2)
    metrics = rng.normal(size=(5, 2))
    fault = rng.poisson(1.2, size=5)

    data = DMetricsData.from_arrays(metrics=metrics, fault=fault)

    assert data.metrics.shape == (5, 2)
    assert data.offset.shape == (5,)
    assert np.allclose(data.offset, 0.0)
    assert data.fault.shape == (5,)

def test_dmetricsdata_from_dataframe():
    df = pd.DataFrame({
        "m1": [0.1, 0.2, -0.1],
        "m2": [0.5, -0.3, 0.0],
        "offset": [0.05, -0.02, 0.01],
        "fault": [2, 0, 1],
    })

    data = DMetricsData.from_dataframe(df, metrics=["m1", "m2"], offset="offset", fault="fault")

    assert data.metrics.shape == (3, 2)
    assert np.allclose(data.offset, [0.05, -0.02, 0.01])
    assert np.array_equal(data.fault, [2, 0, 1])