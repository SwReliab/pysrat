# pysrat

pysrat is a Python package for fitting and analysing NHPP (Non-Homogeneous Poisson
Process) software reliability models. It provides a clean scikit-learn–style API
(`model.fit(data)`, `params_`, `llf_`, `aic_`) and high-performance EM core
implementations exposed from C++ via pybind11.

---

### Features

- Scikit-learn–style estimators (`model.fit(data)`, `params_`, `llf_`, `aic_`)
- Fast EM updates implemented in C++ and exposed via pybind11
- A collection of classical NHPP models and CF1 (canonical phase-type)
- Plotting helpers for visualization (`plot_mvf`, `plot_dmvf`, `plot_rate`)

---

### Installation

From source (recommended during development):

```bash
python -m pip install -U pip
pip install -e .
```

Requirements

- Python ≥ 3.9
- A C++20-capable compiler (required for some extension modules such as CF1)

Optional / developer tools

- `ninja` (recommended build backend for faster CMake builds)
- `pytest`, `black`, `ruff` for development and tests
- `Python headers` and development toolchain for building native extensions

---

### Quick start

```python
import numpy as np
from pysrat.data.nhpp import NHPPData
from pysrat.nhpp.models import ExponentialNHPP
from pysrat.nhpp.plot import plot_mvf

# Create NHPP data from grouped intervals
data = NHPPData.from_intervals(time=[1, 1, 1, 1], fault=[0, 1, 0, 5])

# Fit model
model = ExponentialNHPP().fit(data)

print("Parameters:", model.params_)
print("Log-likelihood:", model.llf_)
print("AIC:", model.aic_)

# Plot mean value function
plot_mvf(data, model)
```

---

### Working with data

pysrat accepts several NHPP data formats via `pysrat.data.nhpp.NHPPData`:

- Interval/grouped data: `NHPPData.from_intervals(time=..., fault=..., type=..., te=...)`
- Counts per interval: `NHPPData.from_counts(fault=...)`
- Exact fault times: `NHPPData.from_fault_times(times=..., te=... )`

---

### Available models

Common models are exported under `pysrat.nhpp.models`, for example:

- `ExponentialNHPP`, `GammaNHPP`, `Pareto2NHPP`, `LogNormalNHPP`
- `TruncatedNormalNHPP`, `LogLogisticNHPP`, `TruncatedLogisticNHPP`
- `TruncatedExtremeValueMaxNHPP`, `LogExtremeValueMaxNHPP`
- `TruncatedExtremeValueMinNHPP`, `LogExtremeValueMinNHPP`
- `CanonicalPhaseTypeNHPP` (CF1)

---

### CF1 example

```python
from pysrat.data.nhpp import NHPPData
from pysrat.nhpp.models import CanonicalPhaseTypeNHPP

data = NHPPData.from_intervals(time=[1, 2, 1.5], fault=[1, 0, 2], type=[0, 1, 0])
model = CanonicalPhaseTypeNHPP(3).fit(data)
print(model.params_)
```

---

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

---

### Model comparison

There is no single `compare()` helper. Compare models using `aic_` or `llf_`:

```python
from pysrat.nhpp.models import ExponentialNHPP, GammaNHPP
data = NHPPData.from_counts([0, 1, 0, 5])
m1 = ExponentialNHPP().fit(data)
m2 = GammaNHPP().fit(data)
best = min((m1, m2), key=lambda m: m.aic_)
print("Best model:", best.name)
```

---

### Hyperparameters (sklearn-style)

Models expose `get_params()` and `set_params()`:

```python
model = ExponentialNHPP(omega0=1.0, rate0=1.0)
model.set_params(rate0=0.5)
print(model.get_params())
```

---

## Regression / GLM helpers

Low-level GLM implementations and penalties live under `src/pysrat/regression` and
the performant IRLS/elastic-net solvers are provided by a compiled extension
(`_glm`) exposed through the `pysrat.regression` package.

Example (binomial GLM wrapper):

```python
from pysrat.regression.glm_binomial import glm_binomial
fit = glm_binomial(X=..., y=..., n_trials=..., offset=..., fit_intercept=True)
print(fit['intercept'], fit['beta'])
```

---

## Examples

See the `examples/` notebooks included in the repository for end-to-end
workflows (basic fitting, comparison, CF1 usage and plotting):

- `examples/example1.ipynb` — quick start and Exponential NHPP
- `examples/example2.ipynb` — GLM-based dGLM usage
- `examples/example3.ipynb` — CF1 fitting
- `examples/example4.ipynb` — visualization and bootstrapping

---

## Development and testing

Install developer dependencies and run tests:

```bash
pip install -e .[dev]
pytest -q
```

Notes

- When building extensions repeatedly during development, use `ninja` and a
    persistent virtual environment for faster iteration.
- To reduce thread-pool contention in parallel workloads, set
    `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` in the environment.

---

## Concurrency notes

- `NHPPData` objects are immutable dataclasses and can be shared across threads.
- `model.fit()` mutates model instance state (`params_`, `llf_`, `aic_`, etc.).
    Calling `fit()` concurrently on the **same** model instance is not thread-safe;
    use separate model instances for parallel fits or protect access with a lock.

---

## Contributing

Contributions are welcome. Please open an issue describing the change or a PR
with tests and documentation updates.

---

## License

This project is distributed under the terms of the MIT License. See `LICENSE`.

---

### Distribution helpers

R-like distribution helpers are available under `pysrat.dists` (e.g. `dcf1`, `pcf1`, `rcf1`).

---

### Examples

See the `examples/` notebooks shipped with the repo (example1–4.ipynb) for end-to-end
workflows demonstrating fitting, comparison, CF1 usage, and visualization.

---

### Development

Install dev dependencies and run tests:

```bash
pip install -e .[dev]
pytest -q
```

---

### Release

Tag a release and push tags to trigger CI/publishing:

```bash
git tag v0.1.0
git push --tags
```

Use `PYPI_API_TOKEN` or other CI secrets for PyPI uploads.

---

### Philosophy

pysrat aims to provide a clean Python interface, high-performance core routines,
and an extensible platform for NHPP research and industrial workflows.
