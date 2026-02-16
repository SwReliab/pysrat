import numpy as np
import pytest

from pysrat.nhpp.data import NHPPData
from pysrat.nhpp.models.cf1 import CanonicalPhaseTypeNHPP

pytest.importorskip("pysrat.marlib_cf1")


def _sample_data():
    return NHPPData.from_intervals(
        time=np.array([1.0, 2.0, 1.5], dtype=float),
        fault=np.array([1, 0, 2], dtype=int),
        type=np.array([0, 1, 0], dtype=int),
    )


def test_cf1_model_em_step_and_llf():
    data = _sample_data()
    model = CanonicalPhaseTypeNHPP(3)

    params = model.init_params(data)
    out = model.em_step(params, data)

    assert set(out.keys()) == {"param", "pdiff", "llf", "total"}
    assert out["param"].shape == (1 + 2 * model.n,)
    assert np.isfinite(out["llf"])

    model._set_fitted_params(out["param"])
    llf = model.llf(data)
    assert np.isfinite(llf)


def test_cf1_model_mvf_intensity_shapes():
    model = CanonicalPhaseTypeNHPP(3)
    t = np.array([0.2, 0.5, 1.0], dtype=float)
    model._set_fitted_params(model.init_params(_sample_data()))

    mvf = model.mvf(t)
    intensity = model.intensity(t)

    assert mvf.shape == t.shape
    assert intensity.shape == t.shape
    assert np.all(np.isfinite(mvf))
    assert np.all(np.isfinite(intensity))
