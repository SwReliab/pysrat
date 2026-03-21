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
- Multi-factor models (MFLogitNHPP, MFProbitNHPP, MFCloglogNHPP)
- A Poisson regression–based framework (`fit_pr_nhpp`)

### Installation

From source (recommended during development):

```bash
pip install pysrat
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
- Time-only data with final observation time: `NHPPData.from_intervals(intervals=..., te=...)`

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

### PR-NHPP regression framework

`fit_pr_nhpp` fits module-wise NHPP models with a Poisson regression outer loop.

```python
import numpy as np
from pysrat.data import SMetricsData
from pysrat.nhpp.regression import fit_pr_nhpp

# names must align with your model dictionary keys
sdata = SMetricsData(
    names=["mod_a", "mod_b", "mod_c"],
    metrics=np.array([[1.2, 0.3], [0.7, 1.0], [1.5, 0.4]], dtype=float),
    offset=np.log(np.array([1200.0, 900.0, 1500.0], dtype=float)),  # optional
)

fit = fit_pr_nhpp(models, sdata, reg="glm")

# You can also override offset at call-time (shape = number of modules)
fit2 = fit_pr_nhpp(models, sdata, reg="glm", offset=np.log(np.array([1.0, 2.0, 4.0])))
```

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

