import numpy as np
from pysrat.data import NHPPData
from pysrat.dists.lnorm import dlnorm, plnorm, qlnorm, rlnorm
from pysrat.models.lnorm import LogNormalNHPP


def test_lnorm_distribution_basic():
    x = np.array([0.5, 1.0, 2.0])
    d = dlnorm(x, meanlog=0.0, sdlog=1.0)
    p = plnorm(x, meanlog=0.0, sdlog=1.0)
    q = qlnorm([0.1, 0.5, 0.9], meanlog=0.0, sdlog=1.0)
    r = rlnorm(5, meanlog=0.0, sdlog=1.0)

    assert d.shape == x.shape
    assert p.shape == x.shape
    assert q.shape == (3,)
    assert r.shape == (5,)
    assert np.all(r > 0.0)


def test_lnorm_model_fit_smoke():
    data = NHPPData.from_intervals(
        time=[1.0, 2.0, 1.5],
        fault=[0.0, 2.0, 1.0],
        type=[0, 1, 0],
    )
    model = LogNormalNHPP().fit(data, maxiter=5)
    assert model.params_.shape == (3,)
    assert hasattr(model, "aic_")
