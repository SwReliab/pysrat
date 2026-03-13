from pysrat.data import NHPPData
from pysrat.nhpp.models.lxvmax import LogExtremeValueMaxNHPP


def test_lxvmax_model_fit_smoke():
    data = NHPPData.from_intervals(
        intervals=[1.0, 2.0, 1.5, 0.5],
        counts=[0.0, 1.0, 0.0, 2.0],
        on_boundary=[0, 1, 0, 0],
    )
    model = LogExtremeValueMaxNHPP().fit(data, maxiter=5)
    assert model.params_.shape == (3,)
    assert hasattr(model, "aic_")
