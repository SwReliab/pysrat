import numpy as np
import pytest

from pysrat.data import SMetricsData
from pysrat.nhpp.regression import fit_pr_nhpp
import pysrat.nhpp.regression.pr_nhpp as pr_nhpp_module


class _DummyModel:
    def __init__(self, total):
        self._total = float(total)
        self.data_ = self._total

    def init_params(self, data):
        return np.array([1.0], dtype=float)

    def em_step(self, params, data, **kwargs):
        return {
            "param": np.asarray(params, dtype=float),
            "llf": 0.0,
            "total": float(data),
        }


def test_fit_pr_nhpp_accepts_offset_override():
    def _glm_poisson_stub(**kwargs):
        X = np.asarray(kwargs["X"], dtype=float)
        return {
            "intercept": 0.0,
            "beta": np.zeros(X.shape[1], dtype=float),
            "converged": True,
            "n_iter": 1,
        }

    original = pr_nhpp_module.glm_poisson
    pr_nhpp_module.glm_poisson = _glm_poisson_stub

    models = {
        "m1": _DummyModel(2.0),
        "m2": _DummyModel(2.0),
        "m3": _DummyModel(2.0),
    }

    sdata = SMetricsData(
        names=["m1", "m2", "m3"],
        metrics=np.zeros((3, 1), dtype=float),
        offset=np.zeros(3, dtype=float),
    )

    try:
        no_offset = fit_pr_nhpp(
            models,
            sdata,
            reg="glm",
            max_outer_iter=1,
            fit_intercept=True,
            initialize=True,
            standardize=np.array([0], dtype=int),
        )

        with_offset = fit_pr_nhpp(
            models,
            sdata,
            reg="glm",
            max_outer_iter=1,
            fit_intercept=True,
            initialize=True,
            offset=np.array([0.0, np.log(2.0), np.log(4.0)], dtype=float),
            standardize=np.array([0], dtype=int),
        )
    finally:
        pr_nhpp_module.glm_poisson = original

    # Without offset and constant design, fitted omega should be almost constant.
    assert np.std(no_offset["omega"]) < 1e-8

    # With offset override, fitted omega should vary by module.
    assert np.std(with_offset["omega"]) > 1e-4


def test_fit_pr_nhpp_uses_smetrics_offset_by_default():
    def _glm_poisson_stub(**kwargs):
        X = np.asarray(kwargs["X"], dtype=float)
        return {
            "intercept": 0.0,
            "beta": np.zeros(X.shape[1], dtype=float),
            "converged": True,
            "n_iter": 1,
        }

    original = pr_nhpp_module.glm_poisson
    pr_nhpp_module.glm_poisson = _glm_poisson_stub

    models = {
        "m1": _DummyModel(2.0),
        "m2": _DummyModel(2.0),
        "m3": _DummyModel(2.0),
    }

    sdata = SMetricsData(
        names=["m1", "m2", "m3"],
        metrics=np.zeros((3, 1), dtype=float),
        offset=np.array([0.0, np.log(2.0), np.log(4.0)], dtype=float),
    )

    try:
        fit = fit_pr_nhpp(
            models,
            sdata,
            reg="glm",
            max_outer_iter=1,
            fit_intercept=True,
            initialize=True,
            standardize=np.array([0], dtype=int),
        )
    finally:
        pr_nhpp_module.glm_poisson = original

    # Function-level offset is None, so sdata.offset should be used.
    assert np.std(fit["omega"]) > 1e-4


def test_fit_pr_nhpp_offset_shape_validation():
    models = {
        "m1": _DummyModel(1.0),
        "m2": _DummyModel(1.0),
    }

    sdata = SMetricsData(
        names=["m1", "m2"],
        metrics=np.zeros((2, 1), dtype=float),
    )

    with pytest.raises(ValueError, match="offset must have shape"):
        fit_pr_nhpp(
            models,
            sdata,
            reg="glm",
            max_outer_iter=1,
            offset=np.array([0.0, 1.0, 2.0], dtype=float),
        )


def test_fit_pr_nhpp_forwards_penalty_and_l2matrix_glm():
    captured = {}

    def _glm_poisson_stub(**kwargs):
        captured.update(kwargs)
        X = np.asarray(kwargs["X"], dtype=float)
        return {
            "intercept": 0.0,
            "beta": np.zeros(X.shape[1], dtype=float),
            "converged": True,
            "n_iter": 1,
        }

    original = pr_nhpp_module.glm_poisson
    pr_nhpp_module.glm_poisson = _glm_poisson_stub

    models = {
        "m1": _DummyModel(2.0),
        "m2": _DummyModel(2.0),
    }
    sdata = SMetricsData(
        names=["m1", "m2"],
        metrics=np.zeros((2, 2), dtype=float),
        offset=np.zeros(2, dtype=float),
    )

    penalty = np.array([0.0, 2.0], dtype=float)
    l2matrix = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=float)

    try:
        fit_pr_nhpp(
            models,
            sdata,
            reg="glm",
            max_outer_iter=1,
            fit_intercept=True,
            initialize=True,
            penalty=penalty,
            l2matrix=l2matrix,
        )
    finally:
        pr_nhpp_module.glm_poisson = original

    assert np.allclose(captured["penalty_factor"], penalty)
    assert np.allclose(captured["lambda_l2_mat"], l2matrix)


def test_fit_pr_nhpp_forwards_penalty_and_l2matrix_elasticnet():
    captured = {}

    def _glmnet_poisson_stub(**kwargs):
        captured.update(kwargs)
        X = np.asarray(kwargs["X"], dtype=float)
        return {
            "intercept": 0.0,
            "beta": np.zeros(X.shape[1], dtype=float),
            "converged": True,
            "n_iter": 1,
            "n_inner": 0,
            "max_delta": 0.0,
            "max_delta_inner": 0.0,
        }

    original = pr_nhpp_module.glm_poisson_elasticnet
    pr_nhpp_module.glm_poisson_elasticnet = _glmnet_poisson_stub

    models = {
        "m1": _DummyModel(2.0),
        "m2": _DummyModel(2.0),
    }
    sdata = SMetricsData(
        names=["m1", "m2"],
        metrics=np.zeros((2, 2), dtype=float),
        offset=np.zeros(2, dtype=float),
    )

    penalty = np.array([0.0, 2.0], dtype=float)
    l2matrix = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=float)

    try:
        fit_pr_nhpp(
            models,
            sdata,
            reg="elasticnet",
            max_outer_iter=1,
            fit_intercept=True,
            initialize=True,
            alpha=0.7,
            lambd=0.3,
            penalty=penalty,
            l2matrix=l2matrix,
        )
    finally:
        pr_nhpp_module.glm_poisson_elasticnet = original

    assert np.allclose(captured["penalty_factor"], penalty)
    assert np.allclose(captured["lambda_l2_mat"], l2matrix)
