# pysrat

Python implementation of NHPP (non-homogeneous Poisson process) software reliability models with a C/pybind11 core and a Pythonic estimator API.

## Features

- Scikit-learn–style estimators: `model.fit(data)` and `params_`/`llf_`/`aic_`
- NHPP data helpers: `NHPPData.from_intervals(...)`, `from_counts(...)`, `from_fault_times(...)`
- Fast EM updates in C via pybind11
- CF1 (canonical phase-type) NHPP model and distribution helpers
- Simple plotting helpers (`plot_mvf`, `plot_dmvf`, `plot_rate`)

## Installation

```bash
python -m pip install -U pip
pip install -e .
```

Notes:
- A C++20-capable compiler is required to build the CF1 extension module.

## Quick start

```python
import numpy as np
from pysrat import NHPPData, ExponentialNHPP, plot_mvf

data = NHPPData.from_intervals(time=[1, 1, 1, 1], fault=[0, 1, 0, 5])

model = ExponentialNHPP().fit(data)
print(model.params_)
print(model.aic_)

plot_mvf(data, model)
```

CF1 example:

```python
import numpy as np
from pysrat import NHPPData
from pysrat.models.cf1 import CanonicalPhaseTypeNHPP

data = NHPPData.from_intervals(time=[1, 2, 1.5], fault=[1, 0, 2], type=[0, 1, 0])
model = CanonicalPhaseTypeNHPP(3).fit(data)
print(model.params_)
```

## API overview

### Data

- `NHPPData.from_intervals(time=..., fault=..., type=..., te=...)`
- `NHPPData.from_counts(fault=...)`
- `NHPPData.from_fault_times(times=..., te=...)`

### Models

- `ExponentialNHPP` (exponential)
- `TruncatedNormalNHPP`, `Pareto2NHPP`, `GammaNHPP`, `LogNormalNHPP`
- `TruncatedLogisticNHPP`, `LogLogisticNHPP`
- `TruncatedExtremeValueMaxNHPP`, `LogExtremeValueMaxNHPP`
- `TruncatedExtremeValueMinNHPP`, `LogExtremeValueMinNHPP`
- `CanonicalPhaseTypeNHPP` (CF1)

### Model comparison

```python
from pysrat import compare, ExponentialNHPP, NHPPData

data = NHPPData.from_counts([0, 1, 0, 5])
fitted, best = compare([ExponentialNHPP()], data, criterion="AIC")
```

### Distributions

R-like helpers live under `pysrat.dists`, e.g.:

```python
from pysrat.dists import dcf1, pcf1, rcf1

alpha = [0.4, 0.3, 0.3]
rate = [0.5, 1.0, 1.5]
print(dcf1([0.1, 0.5], alpha=alpha, rate=rate))
print(pcf1([0.1, 0.5], alpha=alpha, rate=rate))
print(rcf1(5, alpha=alpha, rate=rate))
```

### Plotting

- `plot_mvf(data, model_or_results)`
- `plot_dmvf(data, model_or_results)`
- `plot_rate(data, model_or_results)`

Note: Only the `plot_*` functions above are exported; legacy aliases like `mvfplot`, `dmvfplot`, and `rateplot` are not provided.

`model_or_results` can be an `NHPPModel`, a `dict[str, NHPPModel]`, or a list of models.

### Hyperparameters

Models expose sklearn-style hyperparameters via `get_params()`/`set_params()`. For example:

```python
model = ExponentialNHPP(omega0=1.0, rate0=1.0)
model.set_params(rate0=0.5)
```

## Development

```bash
python -m pip install -U pip
pip install -e .[dev]
pytest -q
```
