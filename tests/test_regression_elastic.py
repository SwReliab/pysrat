import numpy as np
import pytest
from pysrat.regression.glmnet_binomial import glmnet_binomial
from pysrat.regression.glmnet_poisson import glmnet_poisson

def test_glm_binomial_elasticnet_basic():
    np.random.seed(4)
    n, p = 200, 5
    X = np.random.randn(n, p)

    beta_true = np.array([1.0, -0.5, 0.0, 0.0, 0.0])
    eta = X @ beta_true
    p_true = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, p_true)

    fit = glmnet_binomial(
        X, y,
        alpha=1.0,          # pure LASSO
        lambda_=0.1,
        max_iter=50,
        tol=1e-8,
    )

    beta_hat = fit["beta"]

    assert beta_hat.shape == (p,)
    assert fit["converged"] is True

    # alpha=1.0 (LASSO) should shrink zero components more, but non-zero components should be relatively close to true values
    assert np.allclose(beta_hat[:2], beta_true[:2], atol=0.5)

    # zero components should be shrunk towards zero
    assert np.all(np.abs(beta_hat[2:]) < 0.3)


def test_glm_binomial_elasticnet_aggregated():
    np.random.seed(5)
    n, p = 150, 4
    X = np.random.randn(n, p)

    beta_true = np.array([0.8, -0.4, 0.0, 0.0])
    eta = X @ beta_true
    p_true = 1 / (1 + np.exp(-eta))

    n_trials = np.random.randint(5, 20, size=n)
    y = np.random.binomial(n_trials, p_true)
    y_prop = y / n_trials

    fit = glmnet_binomial(
        X, y_prop,
        n_trials=n_trials,
        y_is_proportion=True,
        alpha=0.8,
        lambda_=0.05,
        max_iter=50,
    )

    beta_hat = fit["beta"]

    assert beta_hat.shape == (p,)
    assert fit["converged"] is True
    assert np.all(np.isfinite(beta_hat))


def test_glm_binomial_elasticnet_penalty_mask():
    np.random.seed(6)
    n, p = 120, 4
    X = np.random.randn(n, p)

    beta_true = np.array([1.0, -0.5, 0.0, 0.0])
    eta = X @ beta_true
    p_true = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, p_true)

    # the first two coefficients are not penalized, while the last two are penalized
    penalty_factor = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)

    fit = glmnet_binomial(
        X, y,
        penalty_factor=penalty_factor,
        alpha=1.0,
        lambda_=0.2,
        max_iter=50,
    )

    beta_hat = fit["beta"]

    assert fit["converged"] is True

    # non-penalized coefficients should be close to true values
    assert np.allclose(beta_hat[:2], beta_true[:2], atol=0.5)

    # penalized coefficients should be small
    assert np.all(np.abs(beta_hat[2:]) < 0.3)


def test_glmnet_binomial_rejects_invalid_binomial_response():
    X = np.array([[0.0], [1.0], [2.0]], dtype=float)
    n_trials = np.array([1.0, 2.0, 1.0], dtype=float)

    with pytest.raises(ValueError, match="0 <= y <= n_trials"):
        glmnet_binomial(X, np.array([0.0, 3.0, 1.0]), n_trials=n_trials)

    with pytest.raises(ValueError, match="0 <= y <= 1"):
        glmnet_binomial(
            X,
            np.array([0.0, 1.2, 0.5]),
            n_trials=n_trials,
            y_is_proportion=True,
        )

def test_glm_poisson_elasticnet_basic():
    np.random.seed(10)
    n, p = 300, 5
    X = np.random.randn(n, p)

    beta_true = np.array([0.6, -0.4, 0.0, 0.0, 0.0])

    # NOTE:
    # Using `eta = X @ beta_true` may trigger spurious RuntimeWarnings
    # ("divide by zero/overflow encountered in matmul") on macOS when
    # NumPy dispatches to Accelerate (BLAS). This appears to be due to
    # floating-point exception flags being set internally, even though
    # the computed result is finite and correct.
    #
    # To avoid environment-dependent BLAS warnings in tests, we compute
    # the linear predictor without using matmul.
    eta = (X * beta_true).sum(axis=1)

    mu = np.exp(eta)
    y = np.random.poisson(mu)

    fit = glmnet_poisson(
        X, y,
        alpha=1.0,
        lambda_=0.1,
        max_iter=50,
        tol=1e-8,
        fit_intercept=False,
    )

    beta_hat = fit["beta"]

    assert beta_hat.shape == (p,)
    assert fit["converged"] or fit["n_iter"] == 50

    # signal part roughly correct
    assert np.linalg.norm(beta_hat[:2] - beta_true[:2]) < 1.0

    # noise part shrunk
    assert np.max(np.abs(beta_hat[2:])) < 0.5

def test_glm_poisson_elasticnet_with_offset():
    np.random.seed(11)
    n, p = 250, 4
    X = np.random.randn(n, p)

    beta_true = np.array([0.5, -0.3, 0.0, 0.0])

    exposure = np.random.uniform(0.5, 2.0, size=n)
    offset = np.log(exposure)

    eta = X @ beta_true + offset
    mu = np.exp(eta)
    y = np.random.poisson(mu)

    fit = glmnet_poisson(
        X, y,
        offset=offset,
        alpha=0.8,
        lambda_=0.05,
        max_iter=50,
        fit_intercept=False,
    )

    beta_hat = fit["beta"]

    assert fit["converged"] is True
    assert beta_hat.shape == (p,)
    assert np.all(np.isfinite(beta_hat))


def test_glm_poisson_elasticnet_penalty_mask():
    np.random.seed(12)
    n, p = 200, 4
    X = np.random.randn(n, p)

    beta_true = np.array([0.7, -0.5, 0.0, 0.0])
    eta = X @ beta_true
    mu = np.exp(eta)
    y = np.random.poisson(mu)

    # first two coefficients are unpenalized
    penalty_factor = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)

    fit = glmnet_poisson(
        X, y,
        penalty_factor=penalty_factor,
        alpha=1.0,
        lambda_=0.2,
        max_iter=50,
        fit_intercept=False,
    )

    beta_hat = fit["beta"]

    assert fit["converged"] is True

    # non-penalized close to true
    assert np.allclose(beta_hat[:2], beta_true[:2], atol=0.5)

    # penalized ones small
    assert np.all(np.abs(beta_hat[2:]) < 0.3)