import numpy as np

from pysrat.data import DMetricsData
from pysrat.nhpp.multifactor import MFLogitNHPP
from pysrat.nhpp.stepwise import stepwise


def make_stepwise_data(seed=0, n=120):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(loc=0.0, scale=1.0, size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    X = np.column_stack([x1, x2, x3])

    offset = 0.1 * rng.normal(size=n)
    omega_true = 30.0
    beta_true = np.array([1.2, 0.0, 0.0])

    eta = X @ beta_true + offset
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-12, 1.0 - 1.0e-12)

    lam = omega_true * mu / max(np.sum(mu), 1e-12)
    fault = rng.poisson(lam)

    return DMetricsData.from_arrays(metrics=X, fault=fault, offset=offset)


def _expected_param_len(best_model) -> int:
    # omega + (intercept?) + betas
    has_int = bool(getattr(best_model, "has_intercept", False))
    return 1 + (1 if has_int else 0) + int(best_model.data_.nmetrics)


def test_stepwise_reduces_or_keeps_aic():
    data = make_stepwise_data(seed=2, n=200)
    model = MFLogitNHPP()
    model.fit(data, maxiter=100)

    orig_aic = float(model.aic_)

    best = stepwise(model, verbose=False, max_steps=20)

    assert best.__class__ is model.__class__
    assert hasattr(best, "aic_")
    assert float(best.aic_) <= orig_aic + 1e-8


def test_stepwise_returns_model_with_matching_data():
    data = make_stepwise_data(seed=3, n=100)
    model = MFLogitNHPP().fit(data, maxiter=50)

    best = stepwise(model, data=data, max_steps=10)

    assert best.data_ is not None
    assert best.data_.metrics.shape[0] == data.metrics.shape[0]

    # params length matches (omega + optional intercept + selected betas)
    assert best.params_.shape[0] == _expected_param_len(best)