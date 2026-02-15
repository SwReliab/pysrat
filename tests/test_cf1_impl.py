import numpy as np
import pytest


marlib_cf1 = pytest.importorskip("pysrat.marlib_cf1")


def _params():
    alpha = np.array([0.4, 0.3, 0.2], dtype=float)
    rate = np.array([1.5, 0.5, 1.0], dtype=float)
    return alpha, rate


def _dx():
    return np.array([0.1, 0.5, 1.0], dtype=float)


def test_cf1pdf_and_cf1cdf():
    alpha, rate = _params()
    dx = _dx()

    pdf = marlib_cf1.cf1pdf(dx, alpha, rate)
    assert pdf.shape == dx.shape
    assert np.all(np.isfinite(pdf))

    cdf = marlib_cf1.cf1cdf(dx, alpha, rate)
    assert cdf.shape == dx.shape
    assert np.all(np.isfinite(cdf))
    assert np.all((cdf >= 0.0) & (cdf <= 1.0))

    logpdf = marlib_cf1.cf1pdf(dx, alpha, rate, log=True)
    assert logpdf.shape == dx.shape
    assert np.all(np.isfinite(logpdf))

    logcdf = marlib_cf1.cf1cdf(dx, alpha, rate, log=True, lower=False)
    assert logcdf.shape == dx.shape
    assert np.all(np.isfinite(logcdf))


def test_cf1sample():
    alpha, rate = _params()
    n = 10
    out = marlib_cf1.cf1sample(n, alpha, rate, seed=1234)
    assert out.shape == (n,)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_cf1reform():
    alpha, rate = _params()
    out = marlib_cf1.cf1reform(alpha, rate)
    assert set(out.keys()) == {"alpha", "rate"}

    a = np.asarray(out["alpha"], dtype=float)
    r = np.asarray(out["rate"], dtype=float)
    assert a.shape == alpha.shape
    assert r.shape == rate.shape
    assert np.all(np.diff(r) >= 0.0)


def test_cf1sojourn():
    alpha, rate = _params()
    f = np.array([1.0, 0.5, 0.25], dtype=float)
    b = np.array([0.2, 0.3, 0.4], dtype=float)
    out = marlib_cf1.cf1sojourn(alpha, rate, f, b, t=1.0)
    assert out.shape == (2 * alpha.size,)
    assert np.all(np.isfinite(out))


def test_em_cf1_emstep():
    alpha, rate = _params()
    params = {
        "omega": 10.0,
        "alpha": alpha,
        "rate": rate,
    }
    data = {
        "time": np.array([1.0, 2.0, 3.0], dtype=float),
        "fault": np.array([1, 0, 2], dtype=np.int64),
        "type": np.array([0, 1, 0], dtype=np.int64),
    }

    out = marlib_cf1.em_cf1_emstep(params, data)
    assert set(out.keys()) == {"param", "pdiff", "llf", "total"}

    param = out["param"]
    pdiff = out["pdiff"]

    assert set(param.keys()) == {"omega", "alpha", "rate"}
    assert set(pdiff.keys()) == {"omega", "alpha", "rate"}

    new_alpha = np.asarray(param["alpha"], dtype=float)
    new_rate = np.asarray(param["rate"], dtype=float)

    assert new_alpha.shape == alpha.shape
    assert new_rate.shape == rate.shape
    assert np.isfinite(out["llf"])
    assert np.isfinite(out["total"])
