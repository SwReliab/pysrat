import numpy as np

from pysrat import _core


def _sample_data():
    return {
        "len": 3,
        "time": np.array([1.0, 1.0, 1.0], dtype=float),
        "fault": np.array([1.0, 0.0, 2.0], dtype=float),
        "type": np.array([0, 1, 0], dtype=np.int64),
    }


def test_em_exp_emstep():
    params = np.array([10.0, 0.2], dtype=float)
    out = _core.em_exp_emstep(params, _sample_data())
    assert set(out.keys()) == {"param", "pdiff", "llf", "total"}
    assert out["param"].shape == (2,)


def test_em_tnorm_emstep():
    params = np.array([10.0, 0.0, 1.0], dtype=float)
    out = _core.em_tnorm_emstep(params, _sample_data())
    assert set(out.keys()) == {"param", "pdiff", "llf", "total"}
    assert out["param"].shape == (3,)


def test_em_pareto_emstep():
    params = np.array([10.0, 1.5, 2.0], dtype=float)
    out = _core.em_pareto_emstep(params, _sample_data())
    assert set(out.keys()) == {"param", "pdiff", "llf", "total"}
    assert out["param"].shape == (3,)


def test_em_lnorm_emstep():
    params = np.array([10.0, 0.0, 1.0], dtype=float)
    out = _core.em_lnorm_emstep(params, _sample_data())
    assert set(out.keys()) == {"param", "pdiff", "llf", "total"}
    assert out["param"].shape == (3,)


def test_em_tlogis_emstep_mo():
    params = np.array([10.0, 0.0, 1.0], dtype=float)
    out = _core.em_tlogis_emstep_mo(params, _sample_data())
    assert set(out.keys()) == {"param", "pdiff", "llf", "total"}
    assert out["param"].shape == (3,)


def test_em_tlogis_estep():
    params = np.array([10.0, 0.0, 1.0], dtype=float)
    out = _core.em_tlogis_estep(params, _sample_data())
    assert set(out.keys()) == {"llf", "omega", "w0", "w1", "total"}


def test_em_tlogis_pllf():
    params = np.array([0.0, 1.0], dtype=float)
    out = _core.em_tlogis_pllf(params, _sample_data(), 1.0, 1.0)
    assert np.isfinite(out)


def test_em_llogis_emstep():
    params = np.array([10.0, 0.0, 1.0], dtype=float)
    out = _core.em_llogis_emstep(params, _sample_data())
    assert set(out.keys()) == {"param", "pdiff", "llf", "total", "residual"}
    assert out["param"].shape == (3,)


def test_em_llogis_estep():
    params = np.array([10.0, 0.0, 1.0], dtype=float)
    out = _core.em_llogis_estep(params, _sample_data())
    assert set(out.keys()) == {"llf", "omega", "w1", "total"}


def test_em_llogis_pllf():
    params = np.array([0.0, 1.0], dtype=float)
    out = _core.em_llogis_pllf(params, _sample_data(), 1.0)
    assert np.isfinite(out)


def test_em_gamma_emstep():
    params = np.array([10.0, 1.5, 2.0], dtype=float)
    out = _core.em_gamma_emstep(params, _sample_data(), divide=5, eps=1.0e-8)
    assert set(out.keys()) == {"param", "pdiff", "llf", "total"}
    assert out["param"].shape == (3,)
