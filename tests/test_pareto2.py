import pysrat
from pysrat.nhpp.data import NHPPData
from pysrat.nhpp.models.pareto2 import Pareto2NHPP


def test_model_fit_api_pareto2():
    d = NHPPData.from_counts([0, 1, 0, 5])
    model = Pareto2NHPP().fit(d, maxiter=5)
    assert model.params_.shape == (3,)
    assert hasattr(model, "aic_")


def test_plot_accepts_model_pareto2():
    d = NHPPData.from_counts([0, 1, 0, 5])
    model = Pareto2NHPP().fit(d, maxiter=5)
    ax = pysrat.plot_mvf(d, model)
    assert ax is not None
