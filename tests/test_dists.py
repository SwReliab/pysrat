import numpy as np

from pysrat.nhpp.dists import (
    dexp,
    pexp,
    qexp,
    rexp,
    dtnorm,
    ptnorm,
    qtnorm,
    rtnorm,
    dpareto2,
    ppareto2,
    qpareto2,
    rpareto2,
    dlnorm,
    plnorm,
    qlnorm,
    rlnorm,
    dtlogis,
    ptlogis,
    qtlogis,
    rtlogis,
    dllogis,
    pllogis,
    qllogis,
    rllogis,
    dgumbel,
    pgumbel,
    qgumbel,
    rgumbel,
    dtgumbel,
    ptgumbel,
    qtgumbel,
    rtgumbel,
    dlgumbel,
    plgumbel,
    qlgumbel,
    rlgumbel,
    dcf1,
    pcf1,
    rcf1,
)


def test_exp_distribution():
    x = np.array([0.1, 0.5, 1.0])
    d = dexp(x, rate=2.0)
    assert d.shape == x.shape
    assert np.all(d >= 0)

    p = pexp(x, rate=2.0)
    assert np.all((p >= 0) & (p <= 1))
    p_upper = pexp(x, rate=2.0, lower_tail=False)
    assert np.allclose(p + p_upper, 1.0, atol=1e-12)

    q = qexp(p, rate=2.0)
    assert np.allclose(q, x, atol=1e-10)

    dlog = dexp(x, rate=2.0, log=True)
    assert np.allclose(dlog, np.log(d))

    r = rexp(5, rate=2.0, rng=np.random.default_rng(0))
    assert r.shape == (5,)


def test_tnorm_distribution():
    x = np.array([-1.0, 0.0, 1.0])
    d = dtnorm(x, mean=0.0, sd=1.0)
    assert d[0] == 0.0
    assert np.all(d[1:] >= 0)

    p = ptnorm(x, mean=0.0, sd=1.0)
    assert p[0] == 0.0
    p_upper = ptnorm(x, mean=0.0, sd=1.0, lower_tail=False)
    assert p_upper[0] == 1.0

    q = qtnorm(np.array([0.0, 0.5, 1.0]), mean=0.0, sd=1.0)
    assert np.all(q >= 0.0)

    r = rtnorm(4, mean=0.0, sd=1.0, rng=np.random.default_rng(0))
    assert r.shape == (4,)


def test_pareto2_distribution():
    x = np.array([-1.0, 0.0, 1.0])
    d = dpareto2(x, shape=2.0, scale=1.0)
    assert d[0] == 0.0
    p = ppareto2(x, shape=2.0, scale=1.0)
    assert p[0] == 0.0
    p_upper = ppareto2(x, shape=2.0, scale=1.0, lower_tail=False)
    assert p_upper[0] == 1.0

    q = qpareto2(np.array([0.0, 0.5, 1.0]), shape=2.0, scale=1.0)
    assert np.all(q >= 0.0)

    r = rpareto2(3, shape=2.0, scale=1.0, rng=np.random.default_rng(0))
    assert r.shape == (3,)


def test_lognorm_distribution():
    x = np.array([-1.0, 0.1, 1.0])
    d = dlnorm(x, meanlog=0.0, sdlog=1.0)
    assert d[0] == 0.0
    p = plnorm(x, meanlog=0.0, sdlog=1.0)
    assert p[0] == 0.0
    p_upper = plnorm(x, meanlog=0.0, sdlog=1.0, lower_tail=False)
    assert p_upper[0] == 1.0

    q = qlnorm(np.array([0.0, 0.5, 1.0]), meanlog=0.0, sdlog=1.0)
    assert np.all(q >= 0.0)

    r = rlnorm(3, meanlog=0.0, sdlog=1.0, rng=np.random.default_rng(0))
    assert r.shape == (3,)


def test_tlogis_distribution():
    x = np.array([-1.0, 0.0, 1.0])
    d = dtlogis(x, location=0.0, scale=1.0)
    assert d[0] == 0.0
    p = ptlogis(x, location=0.0, scale=1.0)
    assert p[0] == 0.0
    p_upper = ptlogis(x, location=0.0, scale=1.0, lower_tail=False)
    assert p_upper[0] == 1.0

    q = qtlogis(np.array([0.0, 0.5, 1.0]), location=0.0, scale=1.0)
    assert np.all(q >= 0.0)

    r = rtlogis(3, location=0.0, scale=1.0, rng=np.random.default_rng(0))
    assert r.shape == (3,)


def test_llogis_distribution():
    x = np.array([-1.0, 0.1, 1.0])
    d = dllogis(x, locationlog=0.0, scalelog=1.0)
    assert d[0] == 0.0
    p = pllogis(x, locationlog=0.0, scalelog=1.0)
    assert p[0] == 0.0
    p_upper = pllogis(x, locationlog=0.0, scalelog=1.0, lower_tail=False)
    assert p_upper[0] == 1.0

    q = qllogis(np.array([0.0, 0.5, 1.0]), locationlog=0.0, scalelog=1.0)
    assert np.all(q >= 0.0)

    r = rllogis(3, locationlog=0.0, scalelog=1.0, rng=np.random.default_rng(0))
    assert r.shape == (3,)


def test_gumbel_distribution_min_max():
    x = np.array([-1.0, 0.0, 1.0])
    d_max = dgumbel(x, loc=0.0, scale=1.0, min=False)
    d_min = dgumbel(x, loc=0.0, scale=1.0, min=True)
    assert d_max.shape == x.shape and d_min.shape == x.shape

    p_max = pgumbel(x, loc=0.0, scale=1.0, lower=True, min=False)
    p_min = pgumbel(x, loc=0.0, scale=1.0, lower=True, min=True)
    assert np.all((p_max >= 0) & (p_max <= 1))
    assert np.all((p_min >= 0) & (p_min <= 1))

    q_max = qgumbel(np.array([0.1, 0.5, 0.9]), loc=0.0, scale=1.0, min=False)
    q_min = qgumbel(np.array([0.1, 0.5, 0.9]), loc=0.0, scale=1.0, min=True)
    assert q_max.shape == (3,) and q_min.shape == (3,)

    r = rgumbel(3, loc=0.0, scale=1.0, min=False, rng=np.random.default_rng(0))
    assert r.shape == (3,)


def test_tgumbel_distribution_min_max():
    x = np.array([-1.0, 0.0, 1.0])
    d = dtgumbel(x, loc=0.0, scale=1.0, min=False)
    assert d[0] == 0.0
    p = ptgumbel(x, loc=0.0, scale=1.0, min=False)
    assert p[0] == 0.0
    p_upper = ptgumbel(x, loc=0.0, scale=1.0, lower_tail=False, min=False)
    assert p_upper[0] == 1.0

    q = qtgumbel(np.array([0.0, 0.5, 1.0]), loc=0.0, scale=1.0, min=True)
    assert np.all(q >= 0.0)

    r = rtgumbel(3, loc=0.0, scale=1.0, min=True, rng=np.random.default_rng(0))
    assert r.shape == (3,)


def test_lgumbel_distribution_min_max():
    x = np.array([-1.0, 0.1, 1.0])
    d = dlgumbel(x, loclog=0.0, scalelog=1.0, min=False)
    assert d[0] == 0.0
    p = plgumbel(x, loclog=0.0, scalelog=1.0, min=True)
    assert p[0] == 0.0
    p_upper = plgumbel(x, loclog=0.0, scalelog=1.0, lower_tail=False, min=True)
    assert p_upper[0] == 1.0

    q = qlgumbel(np.array([0.1, 0.5, 0.9]), loclog=0.0, scalelog=1.0, min=False)
    assert np.all(q >= 0.0)

    r = rlgumbel(3, loclog=0.0, scalelog=1.0, min=False, rng=np.random.default_rng(0))
    assert r.shape == (3,)


def test_cf1_distribution():
    x = np.array([0.1, 0.5, 1.0])
    alpha = np.array([0.4, 0.3, 0.2])
    rate = np.array([0.5, 1.0, 1.5])

    d = dcf1(x, alpha=alpha, rate=rate)
    assert d.shape == x.shape
    assert np.all(np.isfinite(d))
    assert np.all(d >= 0.0)

    p = pcf1(x, alpha=alpha, rate=rate)
    assert p.shape == x.shape
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))

    p_upper = pcf1(x, alpha=alpha, rate=rate, lower_tail=False)
    assert np.allclose(p + p_upper, 1.0, atol=1e-10)

    r = rcf1(5, alpha=alpha, rate=rate, scramble=False)
    assert r.shape == (5,)
    assert np.all(np.isfinite(r))
