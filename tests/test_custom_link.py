"""User-defined link functions for the multi-factor NHPP models.

A subclass that overrides ``_dmu`` supplies its own link, and the M-step is
carried out by the Python IRLS instead of the C++ one.  The tests below are
null controls: a custom link that reproduces a built-in link must give the
same answer as the built-in one, which goes through C++.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from pysrat.data import DMetricsData
from pysrat.nhpp.multifactor import MFCloglogNHPP, MFLogitNHPP, MFProbitNHPP
from pysrat.nhpp.multifactor._dglm import DynamicGLMBase
from pysrat.regression import glm_binomial


def _sigmoid(eta):
    eta = np.clip(eta, -700.0, 700.0)
    return np.where(eta >= 0.0,
                    1.0 / (1.0 + np.exp(-eta)),
                    np.exp(eta) / (1.0 + np.exp(eta)))


class MFCustomLogitNHPP(DynamicGLMBase):
    """The logit link written as a user-defined link."""
    name = "MFCustomLogitNHPP"
    link = "custom-logit"

    def _linkinv(self, eta):
        return _sigmoid(eta)

    def _dmu(self, eta, mu):
        return mu * (1.0 - mu)


class MFCustomProbitNHPP(DynamicGLMBase):
    name = "MFCustomProbitNHPP"
    link = "custom-probit"

    def _linkinv(self, eta):
        return ndtr(eta)

    def _dmu(self, eta, mu):
        return np.exp(-0.5 * eta**2) / np.sqrt(2.0 * np.pi)


class MFCustomCloglogNHPP(DynamicGLMBase):
    name = "MFCustomCloglogNHPP"
    link = "custom-cloglog"

    def _linkinv(self, eta):
        return 1.0 - np.exp(-np.exp(np.clip(eta, -50.0, 50.0)))

    def _dmu(self, eta, mu):
        e = np.clip(eta, -50.0, 50.0)
        return np.exp(e - np.exp(e))


def make_data(seed=0, n=60):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
    eta = X @ np.array([0.8, -0.5])
    mu = _sigmoid(eta)
    surv = np.concatenate([[1.0], np.cumprod(1.0 - mu)[:-1]])
    fault = rng.poisson(40.0 * mu * surv).astype(float)
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "faults": fault})
    return DMetricsData.from_dataframe(df, metrics=["x1", "x2"], fault="faults")


@pytest.mark.parametrize("builtin, custom", [
    (MFLogitNHPP, MFCustomLogitNHPP),
    (MFProbitNHPP, MFCustomProbitNHPP),
    (MFCloglogNHPP, MFCustomCloglogNHPP),
])
def test_custom_link_matches_builtin(builtin, custom):
    """A custom link equal to a built-in one must give the same fit."""
    data = make_data()
    a = builtin(has_intercept=True)
    a.fit(data)
    b = custom(has_intercept=True)
    b.fit(data)

    assert np.isclose(a.llf_, b.llf_, rtol=0.0, atol=1e-8)
    assert np.allclose(np.asarray(a.params_), np.asarray(b.params_),
                       rtol=0.0, atol=1e-6)


def test_builtin_links_still_use_cpp():
    """Models that do not override _dmu keep passing the link name to C++."""
    for cls in (MFLogitNHPP, MFProbitNHPP, MFCloglogNHPP):
        m = cls(has_intercept=True)
        assert m._link_arg() == m.link
        assert isinstance(m._link_arg(), str)


def test_custom_link_passes_callables():
    m = MFCustomLogitNHPP(has_intercept=True)
    arg = m._link_arg()
    assert not isinstance(arg, str)
    assert len(arg) == 3 and all(callable(f) for f in arg)
    # a link defined on the whole line declares no domain
    assert arg[2](np.linspace(-5.0, 5.0, 11)) is None


def test_restricted_domain_is_respected():
    """A step that leaves the link's domain must be rejected."""
    rng = np.random.default_rng(3)
    n, p = 40, 2
    X = rng.normal(size=(n, p))
    n_trials = np.full(n, 10.0)
    y = rng.binomial(10, _sigmoid(X @ np.array([0.6, -0.4]))).astype(float)

    lam = -0.6

    def linkinv(eta):
        z = np.maximum(1.0 + lam * eta, 1e-12)
        return 1.0 / (1.0 + np.clip(z ** (-1.0 / lam), 1e-300, 1e300))

    def dmu(eta, mu):
        return mu * (1.0 - mu) / np.maximum(1.0 + lam * eta, 1e-12)

    def domain(eta):
        return 1.0 + lam * eta > 1e-12

    fit = glm_binomial(X=X, y=y, n_trials=n_trials, fit_intercept=True,
                       intercept0=-2.0, max_iter=50, tol=1e-10,
                       link=(linkinv, dmu, domain))
    eta = fit["intercept"] + X @ fit["beta"]
    assert np.all(1.0 + lam * eta > -1e-6)


def test_glm_binomial_custom_link_matches_builtin():
    """The same holds one level down, in glm_binomial itself."""
    rng = np.random.default_rng(1)
    n, p = 50, 3
    X = rng.normal(size=(n, p))
    n_trials = rng.integers(5, 20, size=n).astype(float)
    mu = _sigmoid(X @ np.array([0.5, -0.3, 0.2]))
    y = rng.binomial(n_trials.astype(int), mu).astype(float)

    kw = dict(X=X, y=y, n_trials=n_trials, fit_intercept=True,
              max_iter=50, tol=1e-10)
    a = glm_binomial(link="logit", **kw)
    b = glm_binomial(link=(_sigmoid, lambda eta, mu: mu * (1.0 - mu)), **kw)

    assert np.isclose(a["intercept"], b["intercept"], atol=1e-8)
    assert np.allclose(a["beta"], b["beta"], atol=1e-8)


def test_parametric_link_reduces_to_logit():
    """A link with a parameter must reduce to the logit at its null value.

    This is the structure used for the Box-Cox and generalized-logit links:
    the parameter is held fixed, so the model stays inside the GLM family.
    """
    class MFBoxCoxNHPP(DynamicGLMBase):
        name = "MFBoxCoxNHPP"
        link = "boxcox"

        def __init__(self, lam=0.0, **kw):
            super().__init__(**kw)
            self.lam = float(lam)

        def _linkinv(self, eta):
            if self.lam == 0.0:
                return _sigmoid(eta)
            z = np.maximum(1.0 + self.lam * eta, 1e-12)
            return 1.0 / (1.0 + z ** (-1.0 / self.lam))

        def _dmu(self, eta, mu):
            if self.lam == 0.0:
                return mu * (1.0 - mu)
            return mu * (1.0 - mu) / np.maximum(1.0 + self.lam * eta, 1e-12)

    data = make_data()
    a = MFLogitNHPP(has_intercept=True)
    a.fit(data)
    b = MFBoxCoxNHPP(lam=0.0, has_intercept=True)
    b.fit(data)
    assert np.isclose(a.llf_, b.llf_, rtol=0.0, atol=1e-8)
