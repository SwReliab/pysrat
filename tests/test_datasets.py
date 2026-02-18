import csv
import importlib.resources as resources

import pandas as pd

from pysrat.nhpp.data import NHPPData
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
        time=df["time"].to_numpy(dtype=float),
        fault=df["fault"].to_numpy(dtype=float),
        type=df["indicator"].to_numpy(dtype=int),
    )
    model = ExponentialNHPP().fit(data, maxiter=5)
    assert model.params_.shape == (2,)
    assert hasattr(model, "aic_")
