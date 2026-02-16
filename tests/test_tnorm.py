import numpy as np
from pysrat.nhpp.data import NHPPData
from pysrat.nhpp.dists.tnorm import dtnorm, ptnorm, qtnorm, rtnorm
from pysrat.nhpp.models.tnorm import TruncatedNormalNHPP


def test_tnorm_distribution_basic():
    x = np.array([0.0, 0.5, 1.0])
    d = dtnorm(x, mean=0.0, sd=1.0)
    p = ptnorm(x, mean=0.0, sd=1.0)
    q = qtnorm([0.1, 0.5, 0.9], mean=0.0, sd=1.0)
    r = rtnorm(5, mean=0.0, sd=1.0)

    assert d.shape == x.shape
    assert p.shape == x.shape
    assert q.shape == (3,)
    assert r.shape == (5,)
    assert np.all(r >= 0.0)


def test_tnorm_model_fit_smoke():
    data = NHPPData.from_intervals(
        time=[1.0, 2.0, 1.5],
        fault=[0.0, 2.0, 1.0],
        type=[0, 1, 0],
    )
    model = TruncatedNormalNHPP().fit(data, maxiter=5)
    assert model.params_.shape == (3,)
    assert hasattr(model, "aic_")
