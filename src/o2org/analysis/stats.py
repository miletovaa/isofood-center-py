"""Descriptive and inferential statistics shortcuts, returning plain
DataFrames/dicts rather than library-specific result objects.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from scipy import stats as _scipy_stats

from ._coerce import AnalysisInput, as_dataframe


def _nunique(series: pd.Series) -> int:
    """`series.nunique()`, tolerant of columns holding unhashable values
    (nested dicts/lists — e.g. a "compound" or "synonyms" field straight from
    an API response). Falls back to counting distinct string representations.
    """
    try:
        return int(series.nunique())
    except TypeError:
        return int(series.astype(str).nunique())


def summary(df: AnalysisInput) -> pd.DataFrame:
    """One row per column: dtype, missing count, and (for numeric columns) the
    usual descriptive statistics. The single "what does my data look like" call.

    Args:
        df: Data to summarize (any `AnalysisInput` shape).

    Returns:
        A DataFrame indexed by column name, with columns `dtype`, `missing`,
        `unique`, and (for numeric columns) `mean`/`std`/`min`/`25%`/`50%`/`75%`/`max`.
    """
    df = as_dataframe(df)
    rows = []
    for col in df.columns:
        series = df[col]
        row = {
            "column": col,
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": _nunique(series),
        }
        if pd.api.types.is_numeric_dtype(series):
            described = series.describe()
            row.update({
                "mean": described.get("mean"),
                "std": described.get("std"),
                "min": described.get("min"),
                "25%": described.get("25%"),
                "50%": described.get("50%"),
                "75%": described.get("75%"),
                "max": described.get("max"),
            })
        rows.append(row)
    return pd.DataFrame(rows).set_index("column")


def correlation(
    df: AnalysisInput,
    columns: list[str] | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> pd.DataFrame:
    """Correlation matrix over numeric columns.

    Args:
        df: Data to correlate (any `AnalysisInput` shape).
        columns: Columns to include. Defaults to every numeric column.
        method: 'pearson', 'spearman', or 'kendall'.

    Returns:
        A square DataFrame of pairwise correlations.
    """
    df = as_dataframe(df)
    numeric = df[columns] if columns is not None else df.select_dtypes(include="number")
    return numeric.corr(method=method)


def ttest(df: AnalysisInput, group_col: str, value_col: str) -> dict[str, Any]:
    """Independent two-sample t-test comparing `value_col` between the two
    groups in `group_col` (which must have exactly two distinct values).

    Args:
        df: Data to test (any `AnalysisInput` shape).
        group_col: Column identifying the two groups to compare.
        value_col: Numeric column being compared between groups.

    Returns:
        A dict with `groups`, `means`, `statistic`, `pvalue`, and `significant_at_0.05`.

    Raises:
        ValueError: `group_col` does not have exactly two distinct values.
    """
    df = as_dataframe(df)
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"{group_col!r} must have exactly 2 groups, found {len(groups)}: {list(groups)}")
    a = df.loc[df[group_col] == groups[0], value_col].dropna()
    b = df.loc[df[group_col] == groups[1], value_col].dropna()
    statistic, pvalue = _scipy_stats.ttest_ind(a, b, equal_var=False)
    return {
        "groups": tuple(groups),
        "means": (float(a.mean()), float(b.mean())),
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "significant_at_0.05": bool(pvalue < 0.05),
    }


def anova(df: AnalysisInput, group_col: str, value_col: str) -> dict[str, Any]:
    """One-way ANOVA: does `value_col` differ across the groups in `group_col`?

    Args:
        df: Data to test (any `AnalysisInput` shape).
        group_col: Column identifying the groups to compare (2 or more).
        value_col: Numeric column being compared across groups.

    Returns:
        A dict with `groups`, `statistic`, `pvalue`, and `significant_at_0.05`.
    """
    df = as_dataframe(df)
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    statistic, pvalue = _scipy_stats.f_oneway(*groups)
    return {
        "groups": list(df[group_col].dropna().unique()),
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "significant_at_0.05": bool(pvalue < 0.05),
    }


def chi_square(df: AnalysisInput, col1: str, col2: str) -> dict[str, Any]:
    """Chi-square test of independence between two categorical columns.

    Args:
        df: Data to test (any `AnalysisInput` shape).
        col1: First categorical column.
        col2: Second categorical column.

    Returns:
        A dict with `contingency_table`, `statistic`, `pvalue`,
        `degrees_of_freedom`, and `significant_at_0.05`.
    """
    df = as_dataframe(df)
    table = pd.crosstab(df[col1], df[col2])
    statistic, pvalue, dof, expected = _scipy_stats.chi2_contingency(table)
    return {
        "contingency_table": table,
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "degrees_of_freedom": int(dof),
        "significant_at_0.05": bool(pvalue < 0.05),
    }
