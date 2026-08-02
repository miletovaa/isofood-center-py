"""Machine-learning shortcuts. Each function returns a small result object that
prints a plain-English summary and knows how to plot itself, so someone who
doesn't code can do `result = pca(df)` then `result.plot()` and be done.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as _PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression as _LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import viz
from ._coerce import AnalysisInput, as_dataframe
from .preprocessing import numeric_columns

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class PCAResult:
    """Result of `pca(...)`. `.components` holds the reduced data; `.plot()` re-draws it."""

    def __init__(
        self,
        components: pd.DataFrame,
        explained_variance_ratio: np.ndarray,
        model: _PCA,
        feature_names: list[str],
        source_df: pd.DataFrame,
    ) -> None:
        self.components = components  # DataFrame with PC1, PC2, ... columns
        self.explained_variance_ratio = explained_variance_ratio
        self.model = model
        self.feature_names = feature_names
        self._source_df = source_df  # for looking up a hue column at plot time

    def plot(self, hue: str | None = None, ax: Axes | None = None) -> Axes:
        """Scatter PC1 vs PC2.

        Args:
            hue: Column to color points by. Can be any column from the
                original data passed to `pca()`, not just one of the columns
                PCA was fit on — e.g. a category PCA itself never saw.
            ax: Existing matplotlib Axes to draw on; a new figure is created if omitted.
        """
        data = self.components if hue is None else self.components.join(self._source_df[[hue]])
        return viz.pca_scatter(data, hue=hue, ax=ax)

    def __repr__(self) -> str:
        pct = ", ".join(f"{v:.1%}" for v in self.explained_variance_ratio)
        return (f"<PCAResult {len(self.feature_names)} features -> "
                f"{self.components.shape[1]} components, explained variance: {pct}>")


def pca(
    df: AnalysisInput,
    columns: list[str] | None = None,
    n_components: int = 2,
    scale: bool = True,
) -> PCAResult:
    """Reduce numeric columns to `n_components` principal components.

    Args:
        df: Data to reduce (any `AnalysisInput` shape).
        columns: Columns to use as PCA input. Defaults to every numeric column.
        n_components: Number of principal components to keep.
        scale: Whether to standardize columns (mean 0, std 1) before fitting —
            usually what you want when columns are on different scales.

    Returns:
        A `PCAResult` with `.components` (a DataFrame of PC1, PC2, ...) and
        `.explained_variance_ratio`.
    """
    df = as_dataframe(df)
    columns = numeric_columns(df, columns)
    data = df[columns].dropna()
    X = StandardScaler().fit_transform(data) if scale else data.values

    model = _PCA(n_components=n_components)
    transformed = model.fit_transform(X)
    components = pd.DataFrame(
        transformed, index=data.index,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    return PCAResult(components, model.explained_variance_ratio_, model, columns, df)


class ClusterResult:
    """Result of `kmeans_cluster(...)`. `.labels` holds the cluster assignment per row."""

    def __init__(self, df: pd.DataFrame, columns: list[str], labels: np.ndarray, model: KMeans) -> None:
        self.df = df
        self.columns = columns
        self.labels = labels
        self.model = model
        self.inertia = model.inertia_

    def plot(self, x: str | None = None, y: str | None = None, ax: Axes | None = None) -> Axes:
        """Scatter two of the clustered columns, colored by cluster.

        Args:
            x: Column for the x-axis. Defaults to the first clustered column.
            y: Column for the y-axis. Defaults to the second clustered column.
            ax: Existing matplotlib Axes to draw on; a new figure is created if omitted.
        """
        x = x or self.columns[0]
        y = y or self.columns[1]
        return viz.cluster_scatter(self.df, x, y, self.labels, ax=ax)

    def __repr__(self) -> str:
        k = len(set(self.labels))
        return f"<ClusterResult k={k} inertia={self.inertia:.2f}>"


def kmeans_cluster(
    df: AnalysisInput,
    columns: list[str] | None = None,
    k: int = 3,
    scale: bool = True,
    random_state: int = 0,
) -> ClusterResult:
    """Group rows into `k` clusters by similarity over numeric columns.

    Args:
        df: Data to cluster (any `AnalysisInput` shape).
        columns: Columns to cluster on. Defaults to every numeric column.
        k: Number of clusters.
        scale: Whether to standardize columns before clustering.
        random_state: Seed for reproducible cluster assignment.

    Returns:
        A `ClusterResult` with `.labels` (one cluster id per row) and `.inertia`.
    """
    df = as_dataframe(df)
    columns = numeric_columns(df, columns)
    data = df[columns].dropna()
    X = StandardScaler().fit_transform(data) if scale else data.values

    model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = model.fit_predict(X)
    return ClusterResult(data, columns, labels, model)


class RegressionResult:
    """Result of `linear_regression(...)`."""

    def __init__(
        self,
        model: _LinearRegression,
        features: list[str],
        target: str,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        r2: float,
    ) -> None:
        self.model = model
        self.features = features
        self.target = target
        self.y_test = y_test
        self.y_pred = y_pred
        self.r2 = r2

    @property
    def coefficients(self) -> pd.Series:
        """One fitted coefficient per feature, indexed by feature name."""
        return pd.Series(self.model.coef_, index=self.features, name="coefficient")

    def plot(self, ax: Axes | None = None) -> Axes:
        """Scatter predicted vs. actual values on the held-out test split."""
        return viz.regression_plot(self.y_test, self.y_pred, ax=ax)

    def __repr__(self) -> str:
        return f"<RegressionResult target={self.target!r} r2={self.r2:.3f}>"


def linear_regression(
    df: AnalysisInput,
    target: str,
    features: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 0,
) -> RegressionResult:
    """Fit a linear model predicting `target` from the other numeric columns
    (or `features`, if given). Reports R^2 on a held-out test split.

    Args:
        df: Data to fit on (any `AnalysisInput` shape).
        target: Column to predict.
        features: Predictor columns. Defaults to every other numeric column.
        test_size: Fraction of rows held out for evaluation.
        random_state: Seed for the train/test split.

    Returns:
        A `RegressionResult` with `.r2`, `.coefficients`, and `.plot()`.
    """
    df = as_dataframe(df)
    features = features or [c for c in numeric_columns(df) if c != target]
    data = df[features + [target]].dropna()
    X, y = data[features], data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = _LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return RegressionResult(model, features, target, y_test.values, y_pred, r2_score(y_test, y_pred))


class ClassificationResult:
    """Result of `classify(...)`."""

    def __init__(
        self,
        model: LogisticRegression | RandomForestClassifier,
        features: list[str],
        target: str,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        accuracy: float,
        report: str,
    ) -> None:
        self.model = model
        self.features = features
        self.target = target
        self.y_test = y_test
        self.y_pred = y_pred
        self.accuracy = accuracy
        self.report = report

    def plot(self, ax: Axes | None = None) -> None:
        """Print the classification report (precision/recall/f1 per class).

        A confusion-style plot isn't drawn here — there's no generic way to
        pick "the two most informative features" to plot against, so the
        text report is the primary output for this result type.
        """
        print(self.report)
        return None

    def __repr__(self) -> str:
        return f"<ClassificationResult target={self.target!r} accuracy={self.accuracy:.3f}>"


def classify(
    df: AnalysisInput,
    target: str,
    features: list[str] | None = None,
    model: Literal["logistic", "random_forest"] = "logistic",
    test_size: float = 0.2,
    random_state: int = 0,
) -> ClassificationResult:
    """Fit a classifier predicting `target`.

    Args:
        df: Data to fit on (any `AnalysisInput` shape).
        target: Column to predict (treated as a category, not a quantity).
        features: Predictor columns. Defaults to every other numeric column.
        model: 'logistic' (logistic regression) or 'random_forest'.
        test_size: Fraction of rows held out for evaluation.
        random_state: Seed for the train/test split and (for `random_forest`) the model.

    Returns:
        A `ClassificationResult` with `.accuracy`, `.report`, and `.plot()`.

    Raises:
        ValueError: `target` has fewer than 2 distinct values (after dropping
            missing rows) — there's nothing to classify between.
    """
    df = as_dataframe(df)
    features = features or [c for c in numeric_columns(df) if c != target]
    data = df[features + [target]].dropna()
    X, y = data[features], data[target]
    if y.nunique() < 2:
        raise ValueError(
            f"{target!r} has only one distinct value ({y.unique()[0]!r}) after "
            "dropping missing rows — classify() needs at least 2 classes."
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None,
    )

    estimator: LogisticRegression | RandomForestClassifier
    if model == "logistic":
        estimator = LogisticRegression(max_iter=1000)
    elif model == "random_forest":
        estimator = RandomForestClassifier(random_state=random_state)
    else:
        raise ValueError("model must be 'logistic' or 'random_forest'")

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    report = classification_report(y_test, y_pred)
    return ClassificationResult(estimator, features, target, y_test.values, y_pred,
                                 accuracy_score(y_test, y_pred), report)
