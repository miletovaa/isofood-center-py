"""Standalone preprocessing transforms — ported from the reference
lcms/peptide pipelines, generalized to any wide DataFrame (rows = samples,
columns = features).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def log10_transform(df: pd.DataFrame, columns: list[str] | None = None, pseudocount: float = 1e-6) -> pd.DataFrame:
    """log10(x), clipped at `pseudocount` to avoid -inf on zeros.

    Args:
        df: Wide DataFrame.
        columns: Columns to transform; defaults to all of `df`'s columns.
        pseudocount: Floor applied before taking the log.
    """
    columns = columns or list(df.columns)
    result = df.copy()
    result[columns] = np.log10(df[columns].apply(pd.to_numeric, errors="coerce").clip(lower=pseudocount))
    return result


def pareto_scale(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Mean-center each column and divide by sqrt(std) (Pareto scaling) —
    the standard pre-PCA scaling for metabolomics-style data, less aggressive
    than z-scoring.
    """
    columns = columns or list(df.columns)
    result = df.copy()
    X = df[columns].apply(pd.to_numeric, errors="coerce")
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1).replace(0, np.nan)
    result[columns] = ((X - mu) / np.sqrt(sd)).fillna(0)
    return result


def is_normalize(df: pd.DataFrame, is_column: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Divide each feature column by an internal-standard column (ratio to a
    reference feature).

    Args:
        df: Wide DataFrame.
        is_column: Column holding the internal-standard measurement.
        columns: Columns to normalize; defaults to every column except `is_column`.
    """
    columns = columns or [c for c in df.columns if c != is_column]
    result = df.copy()
    is_vector = pd.to_numeric(df[is_column], errors="coerce").replace(0, np.nan)
    result[columns] = df[columns].apply(pd.to_numeric, errors="coerce").div(is_vector, axis=0)
    return result


def batch_median_normalize(df: pd.DataFrame, columns: list[str], batch_col: str = "batch") -> pd.DataFrame:
    """Divide each feature by its own batch's median.

    Args:
        df: Wide DataFrame.
        columns: Feature columns to normalize.
        batch_col: Column identifying each row's batch/run.
    """
    result = df.copy()
    normalized = df.groupby(batch_col, group_keys=False)[columns].apply(lambda g: g / g.median())
    result[columns] = normalized.replace([np.inf, -np.inf], 0)
    return result


def lod_half_impute(
    df: pd.DataFrame, columns: list[str], batch_col: str | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Impute zeros/NaNs with half the minimum detected (non-zero, non-NaN)
    value for that feature — done per-batch if `batch_col` is given.

    Returns:
        `(imputed_df, imputed_counts)` — `imputed_counts` is a Series (one
        entry per feature) of how many values were imputed.
    """
    result = df.copy()
    imputed_counts = pd.Series(0, index=columns, dtype=int)

    def _impute(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        for col in columns:
            to_impute = int(group[col].isna().sum() + group[col].eq(0).sum())
            real_values = group[col][(group[col] != 0) & group[col].notna()]
            if not real_values.empty:
                lod_half = real_values.min() / 2
                group[col] = group[col].apply(lambda x: lod_half if pd.isna(x) or x == 0 else x)
            imputed_counts[col] += to_impute
        return group

    if batch_col is not None:
        result[columns] = df.groupby(batch_col, group_keys=False)[columns].apply(_impute)
    else:
        result[columns] = _impute(df[columns])
    return result, imputed_counts


def zero_nan_threshold(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 0.2,
    axis: Literal["columns", "rows"] = "columns",
) -> pd.DataFrame:
    """Drop features (or rows) whose fraction of zero/NaN values exceeds
    `threshold`.

    Args:
        df: Wide DataFrame.
        columns: Feature columns to consider; defaults to all of `df`'s columns.
        threshold: Maximum allowed zero/NaN fraction.
        axis: "columns" drops features (lcms-style); "rows" drops
            samples/compounds (peptide-style).
    """
    columns = columns or list(df.columns)
    if axis == "columns":
        zero_fractions = (df[columns].eq(0) | df[columns].isna()).sum() / len(df)
        drop = set(zero_fractions[zero_fractions > threshold].index)
        return df[[c for c in df.columns if c not in drop]]
    if axis == "rows":
        zero_fractions = (df[columns].eq(0) | df[columns].isna()).sum(axis=1) / len(columns)
        return df.loc[zero_fractions <= threshold].copy()
    raise ValueError(f"axis must be 'columns' or 'rows', got {axis!r}")


def grubbs_outliers(values, alpha: float = 0.05) -> np.ndarray:
    """Iterative (ESD-style) Grubbs' test: repeatedly flags the most extreme
    remaining value as an outlier as long as it exceeds the critical value
    for the shrinking sample.

    Used in place of LOQ/2 imputation for isotope-ratio data, which isn't
    censored the way LC-MS peak areas are.

    Args:
        values: 1D array of numeric values (NaNs are dropped).
        alpha: Significance level.

    Returns:
        A boolean array the same length as the non-NaN input, True where
        that value was flagged as an outlier.
    """
    values = np.asarray(values, dtype=float)
    non_nan_idx = np.where(~np.isnan(values))[0]
    working = values[non_nan_idx].copy()
    mask = np.zeros(len(working), dtype=bool)
    remaining_idx = np.arange(len(working))

    while len(working) >= 3:
        n = len(working)
        sd = working.std(ddof=1)
        if sd == 0:
            break
        mean = working.mean()
        abs_dev = np.abs(working - mean)
        i = int(np.argmax(abs_dev))
        g_stat = abs_dev[i] / sd
        t_crit = scipy_stats.t.ppf(1 - alpha / (2 * n), n - 2)
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        if g_stat <= g_crit:
            break
        mask[remaining_idx[i]] = True
        working = np.delete(working, i)
        remaining_idx = np.delete(remaining_idx, i)

    full_mask = np.zeros(len(values), dtype=bool)
    full_mask[non_nan_idx] = mask
    return full_mask
