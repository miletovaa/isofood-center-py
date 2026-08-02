"""The beginner-facing entry point: a chainable Dataset wrapper, plus
quick_report() for a zero-configuration "just show me my data" pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from . import ml, preprocessing, stats, viz
from ._coerce import AnalysisInput, as_dataframe

if TYPE_CHECKING:
    from ..resources import Resource


class Dataset:
    """Wraps retrieved data so cleaning/analysis steps chain:

        ds = Dataset(client.compounds.list()).impute().normalize()
        ds.correlation()
        ds.pca()

    `data` can be exactly what a Resource hands back (a list of dicts, or a
    single dict) as well as a plain DataFrame — it's converted once, here,
    and every method after that just works with `self.df`.

    Every method returns `self` unless it produces a result (summary,
    correlation matrix, PCA/cluster/regression/classification result), in
    which case that result is returned instead so it can be inspected or
    re-plotted.

    Attributes:
        df: The underlying DataFrame. Mutated in place by `impute()`,
            `normalize()`, and `encode()`.
        history: One string per cleaning step applied so far, in order.
    """

    def __init__(self, data: AnalysisInput) -> None:
        """Args:
            data: Data to wrap — a list of dicts, a single dict, a DataFrame,
                or another Dataset.
        """
        self.df = as_dataframe(data).copy()
        self.history: list[str] = []

    @classmethod
    def from_resource(cls, resource: Resource, **params: Any) -> Dataset:
        """Build straight from an o2org Resource.

        Args:
            resource: A resource attached to a client, e.g. `client.compounds`.
            **params: Passed through to `resource.list(...)` (filters, `all_pages`, `page`, ...).

        Returns:
            A new `Dataset` wrapping `resource.list(**params)`.
        """
        return cls(resource.list(**params))

    def head(self, n: int = 5) -> pd.DataFrame:
        """The first `n` rows, as a plain DataFrame."""
        return self.df.head(n)

    def impute(
        self,
        columns: list[str] | None = None,
        method: Literal["mean", "median", "mode", "constant", "drop"] = "mean",
        value: Any = None,
    ) -> Dataset:
        """Fill (or drop) missing values in place. See `preprocessing.impute`. Returns `self`."""
        self.df = preprocessing.impute(self.df, columns=columns, method=method, value=value)
        self.history.append(f"impute(method={method!r})")
        return self

    def normalize(
        self,
        columns: list[str] | None = None,
        method: Literal["zscore", "minmax", "robust"] = "zscore",
    ) -> Dataset:
        """Scale numeric columns in place. See `preprocessing.normalize`. Returns `self`."""
        self.df = preprocessing.normalize(self.df, columns=columns, method=method)
        self.history.append(f"normalize(method={method!r})")
        return self

    def encode(
        self,
        columns: list[str] | None = None,
        method: Literal["onehot", "label"] = "onehot",
    ) -> Dataset:
        """Turn text/category columns into numbers in place. See
        `preprocessing.encode_categorical`. Returns `self`.
        """
        self.df = preprocessing.encode_categorical(self.df, columns=columns, method=method)
        self.history.append(f"encode(method={method!r})")
        return self

    def summary(self) -> pd.DataFrame:
        """Per-column stats: dtype, missing count, and (for numeric columns) mean/std/min/max/etc.
        See `stats.summary`.
        """
        return stats.summary(self.df)

    def correlation(
        self,
        columns: list[str] | None = None,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        plot: bool = True,
    ) -> pd.DataFrame:
        """Correlation matrix over numeric columns, plotted as a heatmap by default.
        See `stats.correlation`.

        Args:
            columns: Columns to include. Defaults to every numeric column.
            method: 'pearson', 'spearman', or 'kendall'.
            plot: Whether to also draw a correlation heatmap.
        """
        corr = stats.correlation(self.df, columns=columns, method=method)
        if plot:
            viz.correlation_heatmap(corr)
        return corr

    def ttest(self, group_col: str, value_col: str) -> dict[str, object]:
        """Two-sample t-test between the two groups in `group_col`. See `stats.ttest`."""
        return stats.ttest(self.df, group_col, value_col)

    def anova(self, group_col: str, value_col: str) -> dict[str, object]:
        """One-way ANOVA across the groups in `group_col`. See `stats.anova`."""
        return stats.anova(self.df, group_col, value_col)

    def chi_square(self, col1: str, col2: str) -> dict[str, object]:
        """Chi-square test of independence between two categorical columns. See `stats.chi_square`."""
        return stats.chi_square(self.df, col1, col2)

    def pca(
        self,
        columns: list[str] | None = None,
        n_components: int = 2,
        scale: bool = True,
        plot: bool = True,
        hue: str | None = None,
    ) -> ml.PCAResult:
        """Reduce numeric columns to principal components, plotted as a PC1-vs-PC2
        scatter by default. See `ml.pca`.

        Args:
            columns: Columns to use as PCA input. Defaults to every numeric column.
            n_components: Number of principal components to keep.
            scale: Whether to standardize columns before fitting.
            plot: Whether to also draw the PC1-vs-PC2 scatter.
            hue: Column to color points by, if `plot=True`. Can be any column
                in `self.df`, not just one PCA was fit on.
        """
        result = ml.pca(self.df, columns=columns, n_components=n_components, scale=scale)
        if plot:
            result.plot(hue=hue)
        return result

    def cluster(
        self,
        columns: list[str] | None = None,
        k: int = 3,
        scale: bool = True,
        plot: bool = True,
    ) -> ml.ClusterResult:
        """Group rows into `k` clusters, plotted as a colored scatter by default.
        See `ml.kmeans_cluster`.
        """
        result = ml.kmeans_cluster(self.df, columns=columns, k=k, scale=scale)
        if plot:
            result.plot()
        return result

    def regression(
        self,
        target: str,
        features: list[str] | None = None,
        plot: bool = True,
    ) -> ml.RegressionResult:
        """Fit a linear model predicting `target`, plotted as predicted-vs-actual
        by default. See `ml.linear_regression`.
        """
        result = ml.linear_regression(self.df, target=target, features=features)
        if plot:
            result.plot()
        return result

    def classify(
        self,
        target: str,
        features: list[str] | None = None,
        model: Literal["logistic", "random_forest"] = "logistic",
        plot: bool = True,
    ) -> ml.ClassificationResult:
        """Fit a classifier predicting `target`. See `ml.classify`.

        Args:
            target: Column to predict (treated as a category).
            features: Predictor columns. Defaults to every other numeric column.
            model: 'logistic' or 'random_forest'.
            plot: Whether to also print the classification report.
        """
        result = ml.classify(self.df, target=target, features=features, model=model)
        if plot:
            result.plot()
        return result

    def __repr__(self) -> str:
        return f"<Dataset {self.df.shape[0]} rows x {self.df.shape[1]} cols>"


def quick_report(data: AnalysisInput, target: str | None = None) -> Dataset:
    """The one-call pipeline for someone who just wants to see their data.

    Prints a summary, plots a correlation heatmap and a PCA scatter (if there
    are at least 2 numeric columns), and — if `target` is given — fits and
    plots a regression (numeric target) or classification (categorical
    target) using every other numeric column as a feature.

    Args:
        data: Data to report on. Can be exactly what a Resource hands back —
            `client.compounds.list()`, `client.samples.list()`, etc. — a
            DataFrame, or a Dataset; no conversion step needed.
        target: Column to also fit a model for. A numeric column with more
            than 10 distinct values is treated as a regression target;
            anything else (including a numeric 0/1 flag) is treated as a
            classification target.

    Returns:
        The underlying `Dataset`, so you can keep going from there.
    """
    ds = data if isinstance(data, Dataset) else Dataset(data)
    print(f"Dataset: {ds.df.shape[0]} rows, {ds.df.shape[1]} columns")
    print(ds.summary())

    numeric = ds.df.select_dtypes(include="number")
    if numeric.shape[1] >= 2:
        ds.correlation()
        ds.pca()
    else:
        print("Need at least 2 numeric columns for a correlation/PCA view — skipping.")

    if target is not None:
        if target not in ds.df.columns:
            print(f"target {target!r} not found in columns — skipping regression/classification.")
        elif pd.api.types.is_numeric_dtype(ds.df[target]) and ds.df[target].nunique() > 10:
            # A numeric column with few distinct values (0/1 flags, star ratings,
            # small class codes) is almost always a category, not a quantity.
            print(ds.regression(target=target))
        else:
            try:
                print(ds.classify(target=target))
            except ValueError as exc:
                print(f"Skipping classification: {exc}")

    return ds
