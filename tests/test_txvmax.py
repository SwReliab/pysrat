from pysrat.data import NHPPData
from pysrat.models.txvmax import TruncatedExtremeValueMaxNHPP


def test_txvmax_model_fit_smoke():
    data = NHPPData.from_intervals(
        time=[1.0, 2.0, 1.5, 0.5],
        fault=[0.0, 1.0, 0.0, 2.0],
        type=[0, 1, 0, 0],
    )
    model = TruncatedExtremeValueMaxNHPP().fit(data, maxiter=5)
    assert model.params_.shape == (3,)
    assert hasattr(model, "aic_")
