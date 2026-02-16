import pysrat
from pysrat.nhpp.data import NHPPData
from pysrat.nhpp.models.exp import ExponentialNHPP
from pysrat.nhpp.fit import compare

def test_model_fit_api_exponential():
    d = NHPPData.from_counts([0, 1, 0, 5])
    model = ExponentialNHPP().fit(d, maxiter=5)
    assert model.params_.shape == (2,)
    assert hasattr(model, "aic_")


def test_plot_accepts_model_exponential():
    d = NHPPData.from_counts([0, 1, 0, 5])
    model = ExponentialNHPP().fit(d, maxiter=5)
    ax = pysrat.plot_mvf(d, model)
    assert ax is not None


def test_compare_models_exponential():
    d = NHPPData.from_counts([0, 1, 0, 5])
    fitted, best = compare([ExponentialNHPP()], d)
    assert best is fitted[0]
