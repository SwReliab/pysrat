import numpy as np
import pandas as pd

from pysrat.nhpp.data import DMetricsData
from pysrat.nhpp.models.dglm import dGLMLogit, dGLMProbit, dGLMCloglog


def make_synthetic(seed=0, n=80):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
    offset = 0.2 * rng.normal(size=n)

    beta_true = np.array([0.8, -0.5])
    omega_true = 40.0

    eta = X @ beta_true + offset
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-12, 1 - 1e-12)

    lam = omega_true * mu / max(np.sum(mu), 1e-12)
    fault = rng.poisson(lam)

    return DMetricsData(X, offset, fault)


def _assert_fit(model_cls):
    data = make_synthetic(seed=1, n=120)
    model = model_cls().fit(data, maxiter=50)
    assert model.params_.shape == (1 + data.nmetrics,)
    assert hasattr(model, "aic_")


def test_model_fit_api_dglm_logit():
    _assert_fit(dGLMLogit)


def test_model_fit_api_dglm_probit():
    _assert_fit(dGLMProbit)


def test_model_fit_api_dglm_cloglog():
    _assert_fit(dGLMCloglog)


def test_dmetricsdata_from_arrays_defaults():
    rng = np.random.default_rng(2)
    metrics = rng.normal(size=(5, 2))
    fault = rng.poisson(1.2, size=5)

    data = DMetricsData.from_arrays(metrics=metrics, fault=fault)

    assert data.metrics.shape == (5, 2)
    assert data.offset.shape == (5,)
    assert np.allclose(data.offset, 0.0)
    assert data.fault.shape == (5,)


def test_dmetricsdata_from_dataframe():
    df = pd.DataFrame({
        "m1": [0.1, 0.2, -0.1],
        "m2": [1.0, 1.5, 0.5],
        "fault": [1.0, 0.0, 2.0],
        "offset": [0.05, -0.02, 0.1],
    })

    data = DMetricsData.from_dataframe(
        df,
        metrics=["m1", "m2"],
        fault="fault",
        offset="offset",
    )

    assert data.metrics.shape == (3, 2)
    assert data.offset.shape == (3,)
    assert data.fault.shape == (3,)


def test_dglm_mvf_monotone_and_bounds():
    data = make_synthetic(seed=3, n=80)
    model = dGLMLogit().fit(data, maxiter=50)

    t = np.arange(1, data.n + 1, dtype=int)
    mvf = model.mvf(t)

    assert mvf.shape == (data.n,)
    assert np.all(np.diff(mvf) >= -1e-10)

    omega_hat = float(model.params_[0])
    assert mvf[-1] <= omega_hat + 1e-8