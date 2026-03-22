import numpy as np
import pytest
from pysrat.regression.glm_binomial import glm_binomial
from pysrat.regression.glm_poisson import glm_poisson

def test_glm_binomial_basic():
    np.random.seed(0)
    n, p = 100, 2
    X = np.random.randn(n, p)
    beta_true = np.array([1.0, -0.5])
    eta = X @ beta_true
    p_true = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, p_true)
    beta_hat = glm_binomial(X, y)["beta"]
    assert beta_hat.shape == (p,)
    # Should be close to true beta (loose tolerance)
    assert np.allclose(beta_hat, beta_true, atol=0.5)

def test_glm_binomial_aggregated():
    np.random.seed(1)
    n, p = 50, 3
    X = np.random.randn(n, p)
    beta_true = np.array([0.5, 1.0, -0.5])
    eta = X @ beta_true
    p_true = 1 / (1 + np.exp(-eta))
    n_trials = np.random.randint(5, 20, size=n)
    y = np.random.binomial(n_trials, p_true)
    y_prop = y / n_trials
    beta_hat = glm_binomial(X, y_prop, n_trials=n_trials, y_is_proportion=True)["beta"]
    assert beta_hat.shape == (p,)
    assert np.all(np.isfinite(beta_hat))


def test_glm_binomial_rejects_invalid_binomial_response():
    X = np.array([[0.0], [1.0], [2.0]], dtype=float)
    n_trials = np.array([1.0, 2.0, 1.0], dtype=float)

    with pytest.raises(ValueError, match="0 <= y <= n_trials"):
        glm_binomial(X, np.array([0.0, 3.0, 1.0]), n_trials=n_trials)

    with pytest.raises(ValueError, match="0 <= y <= 1"):
        glm_binomial(
            X,
            np.array([0.0, 1.2, 0.5]),
            n_trials=n_trials,
            y_is_proportion=True,
        )

def test_glm_poisson_basic():
    np.random.seed(2)
    n, p = 200, 2
    X = np.random.randn(n, p)
    beta_true = np.array([0.3, -0.7])

    eta = X @ beta_true
    mu = np.exp(eta)
    y = np.random.poisson(mu)

    fit = glm_poisson(X, y, fit_intercept=False)
    beta_hat = fit["beta"]

    assert beta_hat.shape == (p,)
    assert fit["converged"] is True
    # loose tolerance because Poisson noise can be large
    assert np.allclose(beta_hat, beta_true, atol=0.5)


def test_glm_poisson_with_offset():
    np.random.seed(3)
    n, p = 150, 3
    X = np.random.randn(n, p)
    beta_true = np.array([0.2, -0.4, 0.6])

    exposure = np.random.uniform(0.5, 2.0, size=n)
    offset = np.log(exposure)

    eta = X @ beta_true + offset
    mu = np.exp(eta)
    y = np.random.poisson(mu)

    fit = glm_poisson(X, y, offset=offset, fit_intercept=False)
    beta_hat = fit["beta"]

    assert beta_hat.shape == (p,)
    assert fit["converged"] is True
    assert np.all(np.isfinite(beta_hat))
    assert np.allclose(beta_hat, beta_true, atol=0.5)
