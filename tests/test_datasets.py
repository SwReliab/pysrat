import csv
import importlib.resources as resources

import pandas as pd

from pysrat.data import NHPPData
from pysrat.nhpp.models.exp import ExponentialNHPP


def test_musa_dataset_available():
    data_path = resources.files("pysrat").joinpath("datasets/musa/ss1a.csv")
    assert data_path.is_file()

    with data_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        first_row = next(reader)

    assert header == ["time", "fault", "indicator"]
    assert len(first_row) == 3


def test_musa_ss1a_exp_fit():
    data_path = resources.files("pysrat").joinpath("datasets/musa/ss1a.csv")
    df = pd.read_csv(data_path)
    data = NHPPData.from_intervals(
        intervals=df["time"].to_numpy(dtype=float),
        counts=df["fault"].to_numpy(dtype=float),
        on_boundary=df["indicator"].to_numpy(dtype=int),
    )
    model = ExponentialNHPP().fit(data, maxiter=5)
    assert model.params_.shape == (2,)
    assert hasattr(model, "aic_")

def test_smetrics_from_dataframe():
    from pysrat.data.smetrics import SMetricsData

    data = {
        "name": ["A", "B", "C"],
        "metric1": [0.1, 0.2, 0.3],
        "metric2": [1.0, 0.5, 0.0],
        "offset": [10, 20, 30],
    }
    df = pd.DataFrame(data)

    smetrics = SMetricsData.from_dataframe(
        df,
        name_col="name",
        metrics_cols=["metric1", "metric2"],
        offset_col="offset",
        standardize_cols=["metric1"],
    )

    assert smetrics.names.shape == (3,)
    assert smetrics.metrics.shape == (3, 2)
    assert smetrics.offset.shape == (3,)
    assert smetrics.standardize.shape == (2,)
    assert smetrics.standardize[0] == 1
    assert smetrics.standardize[1] == 0