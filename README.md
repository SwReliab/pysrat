# pysrat

Python implementation of NHPP (non-homogeneous Poisson process) software reliability models with a C/pybind11 core and a Pythonic estimator API.

## Features

- Scikit-learn–style estimators: `model.fit(data)`
- NHPP data helpers: `NHPPData.from_intervals(...)`, `from_counts(...)`, `from_fault_times(...)`
- Fast EM updates in C via pybind11
- Simple plotting helpers (`plot_mvf`, `plot_dmvf`, `plot_rate`)

## Installation

```bash
python -m pip install -U pip
pip install -e .
```

## Quick start

```python
import numpy as np
from pysrat import NHPPData, ExponentialNHPP, plot_mvf

data = NHPPData.from_intervals(time=[1, 1, 1, 1], fault=[0, 1, 0, 5])

model = ExponentialNHPP().fit(data)
print(model.params)
print(model.aic_)

plot_mvf(data, model)
```

## API overview

### Data

- `NHPPData.from_intervals(time=..., fault=..., type=..., te=...)`
- `NHPPData.from_counts(fault=...)`
- `NHPPData.from_fault_times(times=..., te=...)`

### Models

- `ExponentialNHPP` (NHPP with exponential distribution)

### Model comparison

```python
from pysrat import compare, ExponentialNHPP, NHPPData

data = NHPPData.from_counts([0, 1, 0, 5])
fitted, best = compare([ExponentialNHPP()], data, criterion="AIC")
```

### Plotting

- `plot_mvf(data, model_or_results)`
- `plot_dmvf(data, model_or_results)`
- `plot_rate(data, model_or_results)`

`model_or_results` can be an `NHPPModel`, a `FitNHPPResult`, a `FitCollection`, a `dict[str, FitNHPPResult]`, or a list of models/results.

## Development

```bash
python -m pip install -U pip
pip install -e .[dev]
pytest -q
```
