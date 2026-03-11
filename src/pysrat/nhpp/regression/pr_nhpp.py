# pysrat/nhpp/regression/pr_nhpp.py
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Literal

import numpy as np

from ...data import SMetricsData
from ...regression.glm_poisson import glm_poisson

# If you have these already, import them. (Otherwise remove elasticnet branch or add stubs.)
try:
    from ...regression.glm_poisson_elasticnet import glm_poisson_elasticnet  # type: ignore
except Exception:  # pragma: no cover
    glm_poisson_elasticnet = None  # type: ignore


def _normalize_models(
    models: Union[Mapping[str, Any], Sequence[Any]],
    *,
    names: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Normalize models input into (names_list, models_dict).

    - If models is a dict, preserve insertion order (Python 3.7+).
    - If models is a list/tuple, `names` is required and defines the module names.
    """
    if isinstance(models, Mapping):
        names_list = [str(k) for k in models.keys()]
        return names_list, {str(k): v for k, v in models.items()}

    if names is None:
        raise ValueError("names must be provided when models is a sequence")

    models_list = list(models)
    if len(models_list) != len(names):
        raise ValueError("len(models) must match len(names)")

    names_list = list(map(str, names))
    return names_list, {nm: models_list[i] for i, nm in enumerate(names_list)}


def _get_model_data(model: Any, name: str, data_by_name: Optional[Mapping[str, Any]]) -> Any:
    """
    Use model-attached data_ by default; optionally override with data_by_name[name].
    """
    if data_by_name is not None:
        if name not in data_by_name:
            raise ValueError(f"data_by_name is missing key {name!r}")
        d = data_by_name[name]
        if hasattr(model, "set_data") and callable(getattr(model, "set_data")):
            model.set_data(d)
        else:
            setattr(model, "data_", d)
        return d

    d = getattr(model, "data_", None)
    if d is None:
        raise ValueError(
            f"Model {name!r} has no data_ set. "
            "Set model.set_data(data) beforehand or pass data_by_name."
        )
    return d


def _poisson_mu_log_link(
    intercept: float,
    beta: np.ndarray,
    X: np.ndarray,
    offset: Optional[np.ndarray],
) -> np.ndarray:
    """
    Fitted means for Poisson log link:
        mu = exp(intercept + X @ beta + offset)
    """
    eta = float(intercept) + (np.asarray(X, dtype=float) @ np.asarray(beta, dtype=float))
    if offset is not None:
        eta = eta + np.asarray(offset, dtype=float)
    eta = np.clip(eta, -700.0, 700.0)
    return np.exp(eta)


def _as_mask_01(mask: Optional[np.ndarray], p: int, *, name: str) -> np.ndarray:
    """
    Convert None or array-like to int{0,1} mask of shape (p,).
    """
    if mask is None:
        return np.ones(p, dtype=int)
    m = np.asarray(mask, dtype=int).reshape(-1)
    if m.shape != (p,):
        raise ValueError(f"{name} must have shape ({p},), got {m.shape}")
    # normalize to 0/1
    m = (m != 0).astype(int)
    return m


def fit_pr_nhpp(
    models: Union[Mapping[str, Any], Sequence[Any]],
    sdata: SMetricsData,
    *,
    names: Optional[Sequence[str]] = None,
    data_by_name: Optional[Mapping[str, Any]] = None,
    offset: Optional[np.ndarray] = None,
    fit_intercept: bool = True,
    initialize: bool = True,
    # outer loop
    max_outer_iter: int = 2000,
    outer_tol: float = 1e-6,
    # regression method
    reg: Literal["glm", "elasticnet"] = "elasticnet",
    # regression (shared)
    max_glm_iter: int = 25,
    glm_tol: float = 1e-8,
    ridge: float = 1e-12,
    eps_mu: float = 1e-15,
    # elasticnet specific
    alpha: float = 0.5,
    lambd: float = 0.0,
    penalty: Optional[np.ndarray] = None,     # (q,) 1=penalize, 0=free (beta only)
    standardize: Optional[np.ndarray] = None,      # (q,) override sdata.standardize if not None
    # inner EM kwargs forwarded to each model.em_step(...)
    inner_kwargs: Optional[Dict[str, Any]] = None,
    # write back final params to model objects
    update_models: bool = True,
    # diagnostics
    trace: bool = False,
) -> Dict[str, Any]:
    """
    Fit PR-NHPP framework (static metrics regression for module-wise total faults).

    This is a *framework* (outer alternating procedure), not a single NHPPModel.

    Parameters
    ----------
    models:
        dict[name -> NHPPModel-like] or list of NHPPModel-like objects.

        Each submodel must implement:
          - init_params(data) -> np.ndarray
          - em_step(params, data, ...) -> dict with keys: "param", "llf", "total"

        Assumption (per your note): params[0] is omega.

    sdata:
        SMetricsData (names + metrics (+ optional offset/standardize)).
        Row order is aligned to model-name order via sdata.reorder(names_list).

    offset:
        Optional override for regression offset, shape (m,).
        If None, uses ``sdata.offset``.

    reg:
        "glm" uses glm_poisson (IRLS WLS).
        "elasticnet" uses glm_poisson_elasticnet if available.
        Tip: set lambd=0.0 to match unpenalized behavior (if your ENet solver supports it).

    standardize:
        Optional override for sdata.standardize. If None, use sdata.standardize.

    Returns
    -------
    dict with keys:
      - models: dict[name -> model] (same objects)
      - names: list[str] order used
      - coef: regression coefficients (intercept first if fit_intercept else just beta)
      - omega: fitted omega per module, shape (m,)
      - total: EM-derived totals used in last regression, shape (m,)
      - llf: sum of submodel llf from last outer iteration
      - converged: whether outer loop converged
      - n_iter: number of outer iterations used
      - glm_converged, glm_n_iter: from last regression call
      - reg_info: (optional) extra info from elasticnet solver if provided
    """
    if inner_kwargs is None:
        inner_kwargs = {}

    names_list, models_dict = _normalize_models(models, names=names)

    # Align s-metrics rows to model order (name-based join, order-safe)
    s_aligned = sdata.reorder(names_list)

    X = np.asarray(s_aligned.metrics, dtype=float)

    m, q = X.shape
    if m != len(names_list):
        raise ValueError(
            "Aligned smetrics rows must match number of models "
            "(bug in reorder or names mismatch)."
        )

    if offset is None:
        offset_arr = None if s_aligned.offset is None else np.asarray(s_aligned.offset, dtype=float)
    else:
        offset_arr = np.asarray(offset, dtype=float).reshape(-1)
        if offset_arr.shape != (m,):
            raise ValueError(f"offset must have shape ({m},), got {offset_arr.shape}")

    std_mask = standardize if standardize is not None else s_aligned.standardize
    std_mask01 = _as_mask_01(std_mask, q, name="standardize") if std_mask is not None else None

    pen_mask01 = _as_mask_01(penalty, q, name="penalty")

    if reg == "elasticnet" and glm_poisson_elasticnet is None:
        raise RuntimeError(
            "reg='elasticnet' requested, but glm_poisson_elasticnet import failed. "
            "Ensure pysrat/regression/glm_poisson_elasticnet.py is available."
        )

    # Ensure data is attached (or overridden)
    data_list: List[Any] = []
    for nm in names_list:
        data_list.append(_get_model_data(models_dict[nm], nm, data_by_name))

    # Initialize per-model parameter vectors
    params: Dict[str, np.ndarray] = {}
    for nm, d in zip(names_list, data_list):
        model = models_dict[nm]

        if initialize:
            p0 = model.init_params(d)
            params[nm] = np.asarray(p0, dtype=float)
        else:
            p = getattr(model, "params_", None)
            if p is None:
                p0 = model.init_params(d)
                params[nm] = np.asarray(p0, dtype=float)
            else:
                params[nm] = np.asarray(p, dtype=float)

        if params[nm].size == 0:
            raise ValueError(f"init params for model {nm!r} is empty")

    # Initialize regression coefficients (warm start)
    k = (1 + q) if fit_intercept else q
    coef = np.zeros(k, dtype=float)

    converged_outer = False
    total = np.zeros(m, dtype=float)
    omega = np.zeros(m, dtype=float)
    llf_sum = float("nan")

    glm_converged = False
    glm_n_iter = 0
    reg_info: Dict[str, Any] = {}

    for outer_it in range(1, max_outer_iter + 1):
        # ---- Inner: one EM step per submodel -> updated params, llf, total
        llf_sum = 0.0
        params_after_em: Dict[str, np.ndarray] = {}

        for i, nm in enumerate(names_list):
            model = models_dict[nm]
            d = data_list[i]

            r = model.em_step(params[nm], d, **inner_kwargs)
            if not isinstance(r, Mapping) or "param" not in r or "llf" not in r or "total" not in r:
                raise ValueError(
                    f"em_step for model {nm!r} must return dict with keys: 'param', 'llf', 'total'"
                )

            p_em = np.asarray(r["param"], dtype=float)
            if p_em.size == 0:
                raise ValueError(f"em_step returned empty params for model {nm!r}")

            params_after_em[nm] = p_em
            llf_sum += float(r["llf"])
            total[i] = float(r["total"])

            if trace:
                print(f"  After EM, model {nm}: total={total[i]:.4f}, llf={float(r['llf']):.4f}")

        # ---- Regression: total ~ Poisson(log) with smetrics
        if fit_intercept:
            intercept0 = float(coef[0])
            beta0 = np.asarray(coef[1:], dtype=float)
        else:
            intercept0 = 0.0
            beta0 = np.asarray(coef, dtype=float)

        if reg == "glm":
            reg_res = glm_poisson(
                X=X,
                y=total,
                offset=offset_arr,
                intercept0=intercept0,
                beta0=beta0,
                fit_intercept=fit_intercept,
                max_iter=int(max_glm_iter),
                tol=float(glm_tol),
                standardize=std_mask01,
                ridge=float(ridge),
                eps_mu=float(eps_mu),
            )
            reg_info = {}
        else:
            # elasticnet
            # Expect glm_poisson_elasticnet signature similar to binomial one:
            #   (..., alpha=..., lambd=..., ridge=..., eps_mu=..., standardize=..., penalty=...)
            reg_res = glm_poisson_elasticnet(  # type: ignore[misc]
                X=X,
                y=total,
                offset=offset_arr,
                intercept0=intercept0,
                beta0=beta0,
                fit_intercept=fit_intercept,
                max_iter=int(max_glm_iter),
                tol=float(glm_tol),
                standardize=std_mask01,
                alpha=float(alpha),
                lambd=float(lambd),
                penalty=pen_mask01,
                ridge=float(ridge),
                eps_mu=float(eps_mu),
            )
            # allow extra diagnostics
            reg_info = {k: reg_res[k] for k in reg_res.keys() if k not in ("intercept", "beta", "converged", "n_iter")}

        reg_intercept = float(reg_res["intercept"]) if fit_intercept else 0.0
        reg_beta = np.asarray(reg_res["beta"], dtype=float)
        if reg_beta.shape != (q,):
            raise ValueError(
                f"Regression returned beta with unexpected shape: {reg_beta.shape}, expected {(q,)}"
            )

        glm_converged = bool(reg_res["converged"])
        glm_n_iter = int(reg_res["n_iter"])

        coef_new = np.concatenate([[reg_intercept], reg_beta]) if fit_intercept else reg_beta.copy()

        # Fitted omega (mu) from log link
        omega_new = _poisson_mu_log_link(reg_intercept, reg_beta, X, offset_arr)

        # ---- Update omega in each model's parameter vector (params[0] is omega)
        new_params: Dict[str, np.ndarray] = {}
        for i, nm in enumerate(names_list):
            p = params_after_em[nm]
            p = np.asarray(p, dtype=float).copy()
            p[0] = float(omega_new[i])
            new_params[nm] = p

        # ---- Convergence check (outer): max(|Δcoef|, |Δomega|)
        dcoef = float(np.max(np.abs(coef_new - coef))) if coef_new.size else 0.0
        domega = float(np.max(np.abs(omega_new - omega))) if omega_new.size else 0.0
        diff = max(dcoef, domega)

        if trace:
            print(f"  Regression: coef={coef_new}")
            print(f"  omega range: {float(omega_new.min()):.6g} .. {float(omega_new.max()):.6g}")
            print(f"  outer diff: {diff:.3e}")

        # Commit
        coef = coef_new
        omega = omega_new
        params = new_params

        if diff < outer_tol:
            converged_outer = True
            break

    # ---- Write back final params to models (best-effort)
    if update_models:
        for nm in names_list:
            model = models_dict[nm]
            p = params[nm]

            if hasattr(model, "_set_fitted_params") and callable(getattr(model, "_set_fitted_params")):
                model._set_fitted_params(p)
            else:
                setattr(model, "params_", np.asarray(p, dtype=float))

            if hasattr(model, "_fitted"):
                try:
                    model._fitted = True
                except Exception:
                    pass

    out: Dict[str, Any] = {
        "models": models_dict,
        "names": names_list,
        "coef": np.asarray(coef, dtype=float),
        "omega": np.asarray(omega, dtype=float),
        "total": np.asarray(total, dtype=float),
        "llf": float(llf_sum),
        "converged": bool(converged_outer),
        "n_iter": int(outer_it),
        "glm_converged": bool(glm_converged),
        "glm_n_iter": int(glm_n_iter),
        "reg": str(reg),
    }
    if reg_info:
        out["reg_info"] = reg_info
    return out