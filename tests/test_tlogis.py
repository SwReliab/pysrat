import numpy as np
from pysrat.data import NHPPData
from pysrat.dists.tlogis import dtlogis, ptlogis, qtlogis, rtlogis
from pysrat.models.tlogis import TruncatedLogisticNHPP


def test_tlogis_distribution_basic():
    x = np.array([0.0, 0.5, 1.0])
    d = dtlogis(x, location=0.0, scale=1.0)
    p = ptlogis(x, location=0.0, scale=1.0)
    q = qtlogis([0.1, 0.5, 0.9], location=0.0, scale=1.0)
    r = rtlogis(5, location=0.0, scale=1.0)

    assert d.shape == x.shape
    assert p.shape == x.shape
    assert q.shape == (3,)
    assert r.shape == (5,)
    assert np.all(r >= 0.0)


def test_tlogis_model_fit_smoke():
    data = NHPPData.from_intervals(
        time=[1.0, 2.0, 1.5],
        fault=[0.0, 2.0, 1.0],
        type=[0, 1, 0],
    )
    model = TruncatedLogisticNHPP().fit(data, maxiter=5)
    assert model.params_.shape == (3,)
    assert hasattr(model, "aic_")
