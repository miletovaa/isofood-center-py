import numpy as np
import pandas as pd
import pytest

from o2org.analysis import stats


def test_summary_reports_missing_and_dtype():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "label": ["a", "b", "c"]})
    out = stats.summary(df)
    assert out.loc["x", "missing"] == 1
    assert out.loc["label", "missing"] == 0


def test_correlation_perfectly_correlated_columns():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    corr = stats.correlation(df)
    assert corr.loc["x", "y"] == pytest.approx(1.0)


def test_ttest_detects_clear_difference():
    df = pd.DataFrame({
        "group": ["a"] * 20 + ["b"] * 20,
        "value": list(np.random.default_rng(0).normal(0, 1, 20)) +
                 list(np.random.default_rng(0).normal(10, 1, 20)),
    })
    result = stats.ttest(df, "group", "value")
    assert result["pvalue"] < 0.05
    assert result["significant_at_0.05"] is True


def test_anova_detects_group_difference():
    df = pd.DataFrame({
        "group": ["a"] * 10 + ["b"] * 10 + ["c"] * 10,
        "value": list(np.random.default_rng(1).normal(0, 1, 10)) +
                 list(np.random.default_rng(1).normal(5, 1, 10)) +
                 list(np.random.default_rng(1).normal(10, 1, 10)),
    })
    result = stats.anova(df, "group", "value")
    assert result["pvalue"] < 0.05


def test_summary_accepts_raw_list_of_dicts_like_resource_list_returns():
    records = [{"x": 1.0, "label": "a"}, {"x": 2.0, "label": "b"}]
    out = stats.summary(records)
    assert isinstance(out, pd.DataFrame)
    assert "x" in out.index


def test_summary_handles_columns_with_unhashable_nested_values():
    # e.g. a "compound" or "synonyms" field straight from an API response
    records = [
        {"id": 1, "compound": {"id": 10, "name": "Glucose"}, "tags": ["a", "b"]},
        {"id": 2, "compound": None, "tags": []},
        {"id": 3, "compound": {"id": 10, "name": "Glucose"}, "tags": ["a"]},
    ]
    out = stats.summary(records)
    # nunique() drops missing values by default (consistent with the hashable-column
    # path) — one distinct non-null dict value among the three rows
    assert out.loc["compound", "unique"] == 1
    assert out.loc["tags", "missing"] == 0


def test_chi_square_returns_contingency_table():
    df = pd.DataFrame({
        "a": ["x", "x", "y", "y"] * 10,
        "b": ["p", "q", "p", "q"] * 10,
    })
    result = stats.chi_square(df, "a", "b")
    assert result["contingency_table"].shape == (2, 2)
