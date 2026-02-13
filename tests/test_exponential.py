import pysrat
from pysrat.data import NHPPData
from pysrat.models.exp import ExponentialNHPP
from pysrat.emfit import emfit, compare

def test_emfit_exponential():
    data = NHPPData.from_intervals(
        time=[1.0, 2.0, 1.5],
        fault=[0.0, 2.0, 1.0],
        type=[0, 1, 0],
    )
    model = ExponentialNHPP()
    res = emfit(model, data, maxiter=5)
    assert res.srm.name == "exp"
    assert res.srm.params.shape == (2,)


def test_model_fit_api_exponential():
    d = NHPPData.from_counts([0, 1, 0, 5])
    model = ExponentialNHPP().fit(d, maxiter=5)
    assert model.params.shape == (2,)
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
