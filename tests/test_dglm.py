import numpy as np
import pandas as pd

from pysrat.data import DMetricsData
from pysrat.nhpp.multifactor import MFCloglogNHPP, MFProbitNHPP, MFLogitNHPP

def make_synthetic(seed=0, n=80):
    rng = np.random.default_rng(seed)

    X = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
    offset = 0.2 * rng.normal(size=n)

    beta_true = np.array([0.8, -0.5])
    omega_true = 40.0

    eta = X @ beta_true + offset

    # numerically stable logistic for generating data
    eta = np.clip(eta, -700.0, 700.0)
    mu = np.where(
        eta >= 0.0,
        1.0 / (1.0 + np.exp(-eta)),
        np.exp(eta) / (1.0 + np.exp(eta)),
    )
    mu = np.clip(mu, 1e-12, 1.0 - 1e-12)

    # expected total faults ~ omega_true (roughly)
    # distribute omega_true across time proportional to mu
    mu_sum = float(np.sum(mu))
    lam = omega_true * mu / max(mu_sum, 1e-12)

    fault = rng.poisson(lam).astype(float)

    return DMetricsData.from_arrays(metrics=X, offset=offset, fault=fault)


def _assert_fit(model_cls, *, has_intercept=True):
    data = make_synthetic(seed=1, n=120)
    model = model_cls(has_intercept=has_intercept).fit(data, maxiter=50)

    p = data.nmetrics
    expected_k = 1 + (1 if has_intercept else 0) + p  # omega + (intercept?) + beta
    assert model.params_.shape == (expected_k,)
    assert hasattr(model, "aic_")
    assert np.all(np.isfinite(model.params_))


def test_model_fit_api_dglm_logit():
    _assert_fit(MFLogitNHPP, has_intercept=True)


def test_model_fit_api_dglm_probit():
    _assert_fit(MFProbitNHPP, has_intercept=True)


def test_model_fit_api_dglm_cloglog():
    _assert_fit(MFCloglogNHPP, has_intercept=True)


def test_model_fit_api_dglm_logit_no_intercept():
    _assert_fit(MFLogitNHPP, has_intercept=False)


def test_dmetricsdata_from_arrays_defaults():
    rng = np.random.default_rng(2)
    metrics = rng.normal(size=(5, 2))
    fault = rng.poisson(1.2, size=5).astype(float)

    data = DMetricsData.from_arrays(metrics=metrics, fault=fault)

    assert data.metrics.shape == (5, 2)
    assert data.offset.shape == (5,)
    assert np.allclose(data.offset, 0.0)
    assert data.fault.shape == (5,)
    assert data.n == 5
    assert data.nmetrics == 2


def test_dmetricsdata_from_dataframe():
    df = pd.DataFrame(
        {
            "m1": [0.1, 0.2, -0.1],
            "m2": [1.0, 1.5, 0.5],
            "fault": [1.0, 0.0, 2.0],
            "offset": [0.05, -0.02, 0.1],
        }
    )

    data = DMetricsData.from_dataframe(
        df,
        metrics=["m1", "m2"],
        fault="fault",
        offset="offset",
    )

    assert data.metrics.shape == (3, 2)
    assert data.offset.shape == (3,)
    assert data.fault.shape == (3,)
    assert data.n == 3
    assert data.nmetrics == 2


def test_dglm_mvf_monotone_and_bounds():
    data = make_synthetic(seed=3, n=80)
    model = MFLogitNHPP(has_intercept=True).fit(data, maxiter=50)

    t = np.arange(1, data.n + 1, dtype=int)
    mvf = model.mvf(t)

    assert mvf.shape == (data.n,)
    # mvf should be (almost) nondecreasing
    assert np.all(np.diff(mvf) >= -1e-10)

    omega_hat = float(model.params_[0])
    # mvf is bounded by omega
    assert mvf[-1] <= omega_hat + 1e-8
    # mvf is nonnegative
    assert np.all(mvf >= -1e-12)