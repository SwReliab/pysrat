import numpy as np
from pysrat.data import NHPPData
from pysrat.dists.llogis import dllogis, pllogis, qllogis, rllogis
from pysrat.models.llogis import LogLogisticNHPP


def test_llogis_distribution_basic():
    x = np.array([0.5, 1.0, 2.0])
    d = dllogis(x, locationlog=0.0, scalelog=1.0)
    p = pllogis(x, locationlog=0.0, scalelog=1.0)
    q = qllogis([0.1, 0.5, 0.9], locationlog=0.0, scalelog=1.0)
    r = rllogis(5, locationlog=0.0, scalelog=1.0)

    assert d.shape == x.shape
    assert p.shape == x.shape
    assert q.shape == (3,)
    assert r.shape == (5,)
    assert np.all(r > 0.0)


def test_llogis_model_fit_smoke():
    data = NHPPData.from_intervals(
        time=[1.0, 2.0, 1.5],
        fault=[0.0, 2.0, 1.0],
        type=[0, 1, 0],
    )
    model = LogLogisticNHPP().fit(data, maxiter=5)
    assert model.params_.shape == (3,)
    assert hasattr(model, "aic_")
