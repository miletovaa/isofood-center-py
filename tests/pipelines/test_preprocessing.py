import numpy as np
import pandas as pd
import pytest

from isofood_center.pipelines import preprocessing


def test_log10_transform_clips_at_pseudocount():
    df = pd.DataFrame({"a": [100.0, 0.0, -5.0]})
    result = preprocessing.log10_transform(df, ["a"], pseudocount=1e-6)
    assert result["a"].iloc[0] == pytest.approx(2.0)
    assert result["a"].iloc[1] == pytest.approx(np.log10(1e-6))
    assert result["a"].iloc[2] == pytest.approx(np.log10(1e-6))  # clipped, not NaN/-inf


def test_pareto_scale_centers_and_scales():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = preprocessing.pareto_scale(df, ["a"])
    assert result["a"].mean() == pytest.approx(0.0, abs=1e-9)


def test_pareto_scale_zero_variance_column_fills_zero_not_nan():
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
    result = preprocessing.pareto_scale(df, ["a"])
    assert (result["a"] == 0).all()


def test_is_normalize_divides_by_internal_standard():
    df = pd.DataFrame({"is": [2.0, 4.0], "feat": [10.0, 10.0]})
    result = preprocessing.is_normalize(df, "is", columns=["feat"])
    assert result["feat"].tolist() == [5.0, 2.5]


def test_is_normalize_treats_zero_standard_as_missing():
    df = pd.DataFrame({"is": [0.0], "feat": [10.0]})
    result = preprocessing.is_normalize(df, "is", columns=["feat"])
    assert pd.isna(result["feat"].iloc[0])


def test_batch_median_normalize_divides_within_each_batch():
    df = pd.DataFrame({"batch": ["a", "a", "b", "b"], "feat": [2.0, 4.0, 10.0, 30.0]})
    result = preprocessing.batch_median_normalize(df, ["feat"], batch_col="batch")
    # batch a median = 3 -> [2/3, 4/3]; batch b median = 20 -> [0.5, 1.5]
    assert result["feat"].tolist() == pytest.approx([2 / 3, 4 / 3, 0.5, 1.5])


def test_lod_half_impute_fills_zeros_and_nans_with_half_the_min_real_value():
    df = pd.DataFrame({"feat": [0.0, np.nan, 4.0, 8.0]})
    imputed, counts = preprocessing.lod_half_impute(df, ["feat"])
    assert imputed["feat"].tolist() == [2.0, 2.0, 4.0, 8.0]  # min real value 4 -> half is 2
    assert counts["feat"] == 2


def test_lod_half_impute_is_batch_aware():
    df = pd.DataFrame({"batch": ["a", "a", "b", "b"], "feat": [0.0, 4.0, 0.0, 100.0]})
    imputed, counts = preprocessing.lod_half_impute(df, ["feat"], batch_col="batch")
    assert imputed["feat"].tolist() == [2.0, 4.0, 50.0, 100.0]
    assert counts["feat"] == 2


def test_zero_nan_threshold_drops_columns_above_threshold():
    df = pd.DataFrame({"keep": [1.0, 2.0, 3.0, 0.0], "drop": [0.0, 0.0, 0.0, 1.0]})
    result = preprocessing.zero_nan_threshold(df, ["keep", "drop"], threshold=0.5, axis="columns")
    assert list(result.columns) == ["keep"]


def test_zero_nan_threshold_drops_rows_above_threshold():
    df = pd.DataFrame({"a": [1.0, 0.0], "b": [2.0, np.nan]})
    result = preprocessing.zero_nan_threshold(df, ["a", "b"], threshold=0.4, axis="rows")
    assert len(result) == 1
    assert result["a"].iloc[0] == 1.0


def test_zero_nan_threshold_rejects_unknown_axis():
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError):
        preprocessing.zero_nan_threshold(df, ["a"], axis="diagonal")


def test_grubbs_outliers_flags_the_extreme_value():
    values = np.array([10.0, 11.0, 9.0, 10.5, 9.5, 100.0])
    mask = preprocessing.grubbs_outliers(values, alpha=0.05)
    assert mask.tolist() == [False, False, False, False, False, True]


def test_grubbs_outliers_flags_nothing_on_uniform_data():
    values = np.array([10.0, 10.1, 9.9, 10.05, 9.95])
    mask = preprocessing.grubbs_outliers(values, alpha=0.05)
    assert not mask.any()


def test_grubbs_outliers_preserves_nan_positions():
    values = np.array([10.0, np.nan, 11.0, 100.0])
    mask = preprocessing.grubbs_outliers(values, alpha=0.05)
    assert len(mask) == 4
    assert mask[1] == False  # NaN position never flagged
