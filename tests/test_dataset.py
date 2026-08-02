from unittest.mock import patch

import numpy as np
import pandas as pd

from o2org.analysis import Dataset, quick_report
from o2org.client import OrgClient

from .helpers import FakeResponse


def _sample_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(size=60),
        "y": rng.normal(size=60),
        "z": rng.normal(size=60),
    })


def test_dataset_chains_and_returns_self():
    ds = Dataset(_sample_df())
    result = ds.impute().normalize()
    assert result is ds
    assert ds.history == ["impute(method='mean')", "normalize(method='zscore')"]


def test_dataset_pca_returns_result_not_self():
    ds = Dataset(_sample_df())
    result = ds.pca(plot=False)
    assert result is not ds
    assert result.components.shape[1] == 2


def test_quick_report_runs_end_to_end_without_target(capsys):
    quick_report(_sample_df())
    captured = capsys.readouterr()
    assert "Dataset: 60 rows" in captured.out


def test_quick_report_runs_regression_when_target_is_numeric(capsys):
    df = _sample_df()
    df["target"] = 2 * df["x"] + df["y"]
    quick_report(df, target="target")
    captured = capsys.readouterr()
    assert "RegressionResult" in captured.out


def test_dataset_accepts_raw_list_of_dicts_like_resource_list_returns():
    records = [{"canonical_name": "Glucose", "mz": 180.06}, {"canonical_name": "Fructose", "mz": 180.06}]
    ds = Dataset(records)
    assert list(ds.df.columns) == ["canonical_name", "mz"]
    assert len(ds.df) == 2


def test_dataset_accepts_single_dict_like_resource_get_returns():
    ds = Dataset({"canonical_name": "Glucose", "mz": 180.06})
    assert len(ds.df) == 1


def test_quick_report_accepts_raw_list_of_dicts(capsys):
    records = [{"x": i, "y": i * 2, "z": i * 3} for i in range(20)]
    quick_report(records)
    captured = capsys.readouterr()
    assert "Dataset: 20 rows" in captured.out


def test_quick_report_skips_classification_gracefully_when_target_has_one_class(capsys):
    df = _sample_df()
    df["label"] = True  # every row the same class - can't classify, shouldn't crash
    ds = quick_report(df, target="label")
    captured = capsys.readouterr()
    assert "Skipping classification" in captured.out
    assert ds is not None  # still returns the Dataset instead of raising


def test_dataset_accepts_an_unmaterialized_list_query_directly():
    client = OrgClient("https://dma.example.org", token="preset-token")
    payload = [{"canonical_name": "Glucose"}, {"canonical_name": "Fructose"}]
    with patch("requests.Session.request", return_value=FakeResponse(200, payload)):
        ds = Dataset(client.compounds.list())  # still-lazy query, not .get() first
    assert len(ds.df) == 2


def test_quick_report_handles_numeric_looking_string_columns_from_real_apis(capsys):
    # e.g. Laravel decimal casts serialized as JSON strings
    records = [
        {"mz": "180.1600000000", "rt": "3000.0000000000", "sample_group": "food"},
        {"mz": "150.0000000000", "rt": "12.5000000000", "sample_group": "environment"},
        {"mz": "220.3300000000", "rt": "8.1000000000", "sample_group": "food"},
    ]
    ds = quick_report(records)
    assert pd.api.types.is_numeric_dtype(ds.df["mz"])
    assert pd.api.types.is_numeric_dtype(ds.df["rt"])
    captured = capsys.readouterr()
    assert "Need at least 2 numeric columns" not in captured.out
