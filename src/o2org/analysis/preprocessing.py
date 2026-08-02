"""Cleaning and normalization shortcuts. Every function accepts data in
whatever shape it was retrieved in — a list of dicts, a single dict, a
DataFrame, or a Dataset — and returns a new DataFrame (the input is never
mutated), so they chain safely.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from ._coerce import AnalysisInput, as_dataframe


def numeric_columns(df: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    """Column names to treat as numeric: `columns` if given, else every numeric
    dtype column in `df`. Shared by normalize()/pca()/kmeans_cluster()/etc. so
    "pick the numeric columns" means the same thing everywhere in the library.

    Args:
        df: The DataFrame to inspect.
        columns: Explicit column names to use instead of auto-detecting.

    Returns:
        The resolved list of column names.
    """
    if columns is not None:
        return list(columns)
    return list(df.select_dtypes(include="number").columns)


def normalize(
    df: AnalysisInput,
    columns: list[str] | None = None,
    method: Literal["zscore", "minmax", "robust"] = "zscore",
) -> pd.DataFrame:
    """Scale numeric columns.

    Args:
        df: Data to normalize (any `AnalysisInput` shape).
        columns: Columns to scale. Defaults to every numeric column.
        method: 'zscore' (mean 0, std 1), 'minmax' (0-1), or 'robust'
            (median/IQR — resists outliers).

    Returns:
        A new DataFrame with the given columns scaled.
    """
    df = as_dataframe(df).copy()
    columns = numeric_columns(df, columns)

    for col in columns:
        series = df[col].astype(float)
        if method == "zscore":
            std = series.std()
            df[col] = (series - series.mean()) / std if std else 0.0
        elif method == "minmax":
            span = series.max() - series.min()
            df[col] = (series - series.min()) / span if span else 0.0
        elif method == "robust":
            median = series.median()
            iqr = series.quantile(0.75) - series.quantile(0.25)
            df[col] = (series - median) / iqr if iqr else 0.0
        else:
            raise ValueError("method must be 'zscore', 'minmax', or 'robust'")
    return df


def impute(
    df: AnalysisInput,
    columns: list[str] | None = None,
    method: Literal["mean", "median", "mode", "constant", "drop"] = "mean",
    value: Any = None,
) -> pd.DataFrame:
    """Fill (or drop) missing values.

    Args:
        df: Data to impute (any `AnalysisInput` shape).
        columns: Columns to fill/check. Defaults to every column.
        method: 'mean', 'median', 'mode', 'constant' (fills with `value`), or
            'drop' (drops rows with any missing value in `columns`).
        value: Fill value, used only when `method="constant"`.

    Returns:
        A new DataFrame with missing values handled.
    """
    df = as_dataframe(df).copy()
    columns = columns if columns is not None else list(df.columns)

    if method == "drop":
        return df.dropna(subset=columns)

    for col in columns:
        if method == "mean":
            fill = df[col].mean()
        elif method == "median":
            fill = df[col].median()
        elif method == "mode":
            modes = df[col].mode()
            fill = modes.iloc[0] if not modes.empty else np.nan
        elif method == "constant":
            fill = value
        else:
            raise ValueError("method must be 'mean', 'median', 'mode', 'constant', or 'drop'")
        df[col] = df[col].fillna(fill)
    return df


def encode_categorical(
    df: AnalysisInput,
    columns: list[str] | None = None,
    method: Literal["onehot", "label"] = "onehot",
) -> pd.DataFrame:
    """Turn text/category columns into numbers.

    Args:
        df: Data to encode (any `AnalysisInput` shape).
        columns: Columns to encode. Defaults to every non-numeric column.
        method: 'onehot' (one 0/1 column per category) or 'label' (one integer
            column per original column).

    Returns:
        A new DataFrame with the given columns encoded.
    """
    df = as_dataframe(df).copy()
    columns = columns if columns is not None else list(df.select_dtypes(exclude="number").columns)

    if method == "onehot":
        return pd.get_dummies(df, columns=columns)
    elif method == "label":
        for col in columns:
            df[col] = df[col].astype("category").cat.codes
        return df
    raise ValueError("method must be 'onehot' or 'label'")
