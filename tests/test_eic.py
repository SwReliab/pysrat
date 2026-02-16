import numpy as np
import pytest

from pysrat import NHPPData
from pysrat.nhpp.bootstrap import eic_sample, eic
from pysrat.nhpp.models.cf1 import CanonicalPhaseTypeNHPP

pytest.importorskip("pysrat.marlib_cf1")


def _time_data():
    times = np.linspace(0.2, 4.0, 20)
    return NHPPData.from_fault_times(times=times, te=5.0)


def _group_data():
    return NHPPData.from_counts(fault=np.array([0, 1, 0, 2], dtype=int))


def test_eic_time_sample_and_summary():
    data = _time_data()
    model = CanonicalPhaseTypeNHPP(3)
    model.fit(data, initialize=True, maxiter=50)

    rng = np.random.default_rng(0)
    sample = eic_sample(model, bsample=3, initialize=True, rng=rng)
    assert sample.shape == (3,)
    assert np.all(np.isfinite(sample))

    summary = eic(model, bsample=3, alpha=0.9, initialize=True, rng=rng)
    assert set(summary.keys()) == {
        "bias",
        "bias.lower",
        "bias.upper",
        "eic",
        "eic.lower",
        "eic.upper",
    }
    assert np.isfinite(summary["eic"])


def test_eic_group_sample():
    data = _group_data()
    model = CanonicalPhaseTypeNHPP(3)
    model.fit(data, initialize=True, maxiter=50)

    rng = np.random.default_rng(1)
    sample = eic_sample(model, bsample=3, initialize=True, rng=rng)
    assert sample.shape == (3,)
    assert np.all(np.isfinite(sample))
