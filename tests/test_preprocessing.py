import numpy as np
import pandas as pd
import pytest

from o2org.analysis import preprocessing


def test_normalize_zscore_has_mean_zero_std_one():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = preprocessing.normalize(df, method="zscore")
    assert out["x"].mean() == pytest.approx(0.0, abs=1e-9)
    assert out["x"].std() == pytest.approx(1.0, abs=1e-9)


def test_normalize_minmax_bounds_zero_to_one():
    df = pd.DataFrame({"x": [10, 20, 30]})
    out = preprocessing.normalize(df, method="minmax")
    assert out["x"].min() == 0.0
    assert out["x"].max() == 1.0


def test_normalize_does_not_mutate_input():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    preprocessing.normalize(df, method="minmax")
    assert df["x"].tolist() == [1.0, 2.0, 3.0]


def test_impute_mean_fills_missing():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    out = preprocessing.impute(df, columns=["x"], method="mean")
    assert out["x"].isna().sum() == 0
    assert out["x"].iloc[1] == pytest.approx(2.0)


def test_impute_drop_removes_rows_with_missing():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    out = preprocessing.impute(df, columns=["x"], method="drop")
    assert len(out) == 2


def test_encode_onehot_expands_columns():
    df = pd.DataFrame({"group": ["a", "b", "a"]})
    out = preprocessing.encode_categorical(df, columns=["group"], method="onehot")
    assert "group_a" in out.columns and "group_b" in out.columns


def test_encode_label_produces_integers():
    df = pd.DataFrame({"group": ["a", "b", "a"]})
    out = preprocessing.encode_categorical(df, columns=["group"], method="label")
    assert pd.api.types.is_integer_dtype(out["group"])


def test_normalize_accepts_raw_list_of_dicts_like_resource_list_returns():
    records = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    out = preprocessing.normalize(records, method="minmax")
    assert isinstance(out, pd.DataFrame)
    assert out["x"].max() == 1.0
