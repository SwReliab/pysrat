# pysrat

pysrat is a Python package for fitting and analysing NHPP (Non-Homogeneous Poisson
Process) software reliability models. It provides a clean scikit-learn–style API
(`model.fit(data)`, `params_`, `llf_`, `aic_`) and high-performance EM core
implementations exposed from C++ via pybind11.

### Features

- Scikit-learn–style estimators (`model.fit(data)`, `params_`, `llf_`, `aic_`)
- Fast EM updates implemented in C++ and exposed via pybind11
- A collection of classical NHPP models and CF1 (canonical phase-type)
- Plotting helpers for visualization (`plot_mvf`, `plot_dmvf`, `plot_rate`)
- Metrics-based (multi-factor) models in which software metrics enter as covariates
  (`MFLogitNHPP`, `MFProbitNHPP`, `MFCloglogNHPP`)
- User-defined link functions: supply the inverse link and its derivative, and the
  EM algorithm, likelihood, AIC and variable selection are inherited
- Offsets, which express Cox proportional-hazards models without dedicated code
- AIC-based variable selection (`stepwise`)
- A Poisson regression–based framework (`fit_pr_nhpp`)

### Installation

```bash
pip install pysrat
```

To build from source (recommended during development):

```bash
git clone https://github.com/SwReliab/pysrat.git
cd pysrat
pip install -e .
```

Requirements

- Python ≥ 3.9
- A C++20-capable compiler (required for some extension modules such as CF1)

Optional / developer tools

- `ninja` (recommended build backend for faster CMake builds)
- `pytest` for development and tests
- `Python headers` and development toolchain for building native extensions

### Quick start

```python
import numpy as np
from pysrat.data.nhpp import NHPPData
from pysrat.nhpp.models import ExponentialNHPP
from pysrat.nhpp.plot import plot_mvf

# Create NHPP data from grouped intervals
data = NHPPData.from_intervals(intervals=[1, 1, 1, 1], counts=[0, 1, 0, 5])

# Fit model
model = ExponentialNHPP().fit(data)

print("Parameters:", model.params_)
print("Log-likelihood:", model.llf_)
print("AIC:", model.aic_)

# Plot mean value function
plot_mvf(data, model)
```

### Working with data

pysrat accepts several NHPP data formats via `pysrat.data.nhpp.NHPPData`:

- Interval/grouped data: `NHPPData.from_intervals(intervals=..., counts=..., on_boundary=...)`
- Counts per interval: `NHPPData.from_intervals(counts=...)`
- Time data (without counts): `NHPPData.from_intervals(intervals=..., on_boundary=...)` or `NHPPData.from_intervals(intervals=..., te=...)`

### Available models

Common models are exported under `pysrat.nhpp.models`:

- `ExponentialNHPP`
- `GammaNHPP`
- `Pareto2NHPP`
- `TruncatedNormalNHPP`
- `LogNormalNHPP`
- `TruncatedLogisticNHPP`
- `LogLogisticNHPP`
- `TruncatedExtremeValueMaxNHPP`
- `LogExtremeValueMaxNHPP`
- `TruncatedExtremeValueMinNHPP`
- `LogExtremeValueMinNHPP`
- `CanonicalPhaseTypeNHPP` (CF1)

### CF1 example

```python
from pysrat.data.nhpp import NHPPData
from pysrat.nhpp.models import CanonicalPhaseTypeNHPP

data = NHPPData.from_intervals(intervals=[1, 2, 1.5], counts=[1, 0, 2], on_boundary=[0, 1, 0])
model = CanonicalPhaseTypeNHPP(3).fit(data)
print(model.params_)
```

### Plotting

Use the plotting helpers from `pysrat.nhpp.plot`:

```python
from pysrat.nhpp.plot import plot_mvf, plot_rate
plot_mvf(data, model)
plot_rate(data, model)
```

You can pass a dictionary of models to compare multiple MVFs:

```python
models = {
    "Exp": ExponentialNHPP().fit(data),
    "Gamma": GammaNHPP().fit(data),
}
plot_mvf(data, models)
```

### Model comparison

Compare models using `aic_` or `llf_`:

```python
from pysrat.nhpp.models import ExponentialNHPP, GammaNHPP
data = NHPPData.from_intervals(counts=[0, 1, 0, 5])
m1 = ExponentialNHPP().fit(data)
m2 = GammaNHPP().fit(data)
best = min((m1, m2), key=lambda m: m.aic_)
print("Best model:", best.name)
```

### Metrics-based (multi-factor) models

These models let software metrics enter as covariates of the fault-detection
probability. The data are held by `DMetricsData`, and each model differs only in its
link function.

```python
import importlib.resources as resources
import pandas as pd

from pysrat.data import DMetricsData
from pysrat.nhpp import MFLogitNHPP, MFProbitNHPP, MFCloglogNHPP, stepwise

# a bundled data set: one row per testing period
path = resources.files("pysrat").joinpath("datasets/dmetrics/dmetrics1.csv")
df = pd.read_csv(path)

data = DMetricsData.from_dataframe(
    df, metrics=["day", "tc", "ctc", "cov", "ccov"], fault="fault")

model = MFLogitNHPP().fit(data)
print(model.aic_)          # 82.895

best = stepwise(model)     # AIC-based variable selection
print(best.aic_, best.data_.metrics_name)   # 79.074 ['day', 'tc', 'ctc']
```

Switching the link function is a matter of replacing the class, so `MFProbitNHPP`
and `MFCloglogNHPP` are used in exactly the same way.

Two points are worth noting.

- **Do not put a constant column in `metrics`.** The intercept is handled by the model
  through `has_intercept`; passing `intercept=True` raises `ValueError`.
- **An offset turns the cloglog model into a Cox proportional-hazards model.** The
  offset is the complementary log-log of the baseline hazard rate, so a discrete
  Weibull baseline with shape `m` needs no dedicated class:

```python
import numpy as np

k = np.arange(1, len(df) + 1, dtype=float)
df["off"] = np.log(k**m - (k - 1.0)**m)      # Weibull offset, up to a constant
data = DMetricsData.from_dataframe(
    df, metrics=[...], fault="fault", offset="off")
model = MFCloglogNHPP().fit(data)
```

See [`docs/metrics-models.md`](docs/metrics-models.md) for the formulation.

### Adding a link function

The E-step of the EM algorithm does not depend on the link function, and the M-step is
a fit of a generalized linear model to weighted binomial data. Within that fit, the link
function enters through one step only: computing the fault-detection probability
`mu = linkinv(z)` and its derivative `dmu/dz` from the linear predictor `z`.

`DynamicGLMBase` therefore exposes these two as its extension points. The example below
is the logit link under a Box-Cox transformation, which has one shape parameter `lam`.

```python
import numpy as np
from pysrat.nhpp.multifactor._dglm import DynamicGLMBase

EPS = 1e-12

class MFBoxCoxNHPP(DynamicGLMBase):
    link = "boxcox"

    def __init__(self, lam=0.0, has_intercept=True):
        super().__init__(has_intercept=has_intercept)
        self.lam = float(lam)

    def _linkinv(self, z):
        if self.lam == 0.0:
            return 1.0 / (1.0 + np.exp(-z))
        w = np.maximum(1.0 + self.lam * z, EPS)
        return 1.0 / (1.0 + w**(-1.0 / self.lam))

    def _dmu(self, z, mu):
        if self.lam == 0.0:
            return mu * (1.0 - mu)
        return mu * (1.0 - mu) / (1.0 + self.lam * z)

    def _link_domain(self, z):          # only for a restricted domain
        return 1.0 + self.lam * z > EPS
```

Everything else -- the EM algorithm, the log likelihood, AIC and `stepwise` -- is
inherited. `_link_domain` is needed only when the link is undefined on part of the real
line, as here, where `1 + lam * z > 0` is required; the M-step then never steps outside
that domain.

A link function with a parameter such as `lam` stays inside the GLM family once the
parameter is fixed, so an outer one-dimensional search over the parameter, with the EM
algorithm run to convergence at each value, estimates it.

The built-in links (`logit`, `probit`, `cloglog`) keep using the C++ IRLS; only
user-defined links go through the Python one.

### PR-NHPP regression framework

`fit_pr_nhpp` fits module-wise NHPP models with a Poisson regression outer loop. Each
module has its own fault data and thus its own NHPP model, while the static metrics of
the modules explain the differences between their expected fault counts.

```python
import importlib.resources as resources
import pandas as pd

from pysrat.data import NHPPData, SMetricsData
from pysrat.nhpp import GammaNHPP
from pysrat.nhpp.regression import fit_pr_nhpp

root = resources.files("pysrat").joinpath("datasets/smetrics")
names = ["catalina", "connector", "jasper", "servlets", "tester", "webapps"]

# one NHPP model per module, fitted to that module's fault data
models = {
    nm: GammaNHPP().fit(
        NHPPData.from_csv(root.joinpath(f"tomcat5_{nm}.csv"),
                          intervals="time", counts="fault"))
    for nm in names
}

# static metrics of the modules; the row index gives the module names
sdata = SMetricsData.from_dataframe(
    pd.read_csv(root.joinpath("tomcat5_smetrics.csv"), index_col=0),
    use_index_as_name=True)

# 7 metrics + intercept = 8 coefficients from 6 modules, so a penalty is required
fit = fit_pr_nhpp(models, sdata, reg="glm", lambd=50.0)
print(fit["converged"], fit["n_iter"])   # True 5743
print(fit["coef"])
```

The keys of `models` must match the names in `sdata`. `reg="glm"` fits the outer
regression by IRLS and `reg="elasticnet"` adds an L1/L2 mix controlled by `alpha`;
both accept `lambd` as the penalty strength.

`SMetricsData` can also be built directly, which is useful when the metrics do not come
from a data frame. The offset is optional and may be overridden at call time.

```python
import numpy as np

sdata = SMetricsData(
    names=["mod_a", "mod_b", "mod_c"],
    metrics=np.array([[1.2, 0.3], [0.7, 1.0], [1.5, 0.4]], dtype=float),
    offset=np.log(np.array([1200.0, 900.0, 1500.0], dtype=float)),  # optional
)
```

**Check `fit["converged"]` before using the result.** Two things make the outer loop
stop short.

- **Identifiability.** The regression estimates `1 + q` coefficients, where `q` is the
  number of metrics, from `m` modules. When `1 + q >= m` the model is not identified and
  the loop never converges, however many iterations it is given. A penalty (`lambd > 0`)
  restores convergence. The bundled Tomcat data has 6 modules and 7 metrics, hence the
  `lambd=50.0` above.
- **Iteration count.** The loop runs one EM step per iteration, so it converges slowly.
  A few thousand iterations are normal even when the model is well posed; the default
  `max_outer_iter` is 10000.

A `RuntimeWarning` is issued if the loop ends without converging.

## Examples

See the `examples/` notebooks included in the repository for end-to-end
workflows (basic fitting, comparison, CF1 usage and plotting):

- `examples/example1.ipynb` — Exponential NHPP
- `examples/example2.ipynb` — CF1 fitting and comparison
- `examples/example3.ipynb` — Multi-factor models (MFLogitNHPP)
- `examples/example4.ipynb` — Poisson regression–based NHPP fitting (`fit_pr_nhpp`)

## Contributing

Contributions are welcome. Please open an issue describing the change or a PR
with tests and documentation updates.

## License

This project is distributed under the terms of the MIT License. See `LICENSE`.

