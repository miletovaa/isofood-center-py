"""Plotting shortcuts. One function per chart type, each returning the Axes it
drew on so callers can keep customizing.

Colors follow the o2org default palette: a fixed-order 8-hue categorical set
for identity, a single blue ramp for magnitude, and a blue<->red diverging pair
(gray midpoint) for correlation/signed data. Hues are never reassigned by
rank/filter, and dual (twin) y-axes are never used - two differently-scaled
measures get two charts instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from ._coerce import AnalysisInput, as_dataframe

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "o2org_sequential", ["#cde2fb", "#3987e5", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "o2org_diverging", ["#2a78d6", "#f0efec", "#e34948"]
)


def apply_style() -> None:
    """Set matplotlib rcParams so every chart in this library looks consistent.
    Called once at import time; safe to call again after plt.style.use(...).
    """
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRIDLINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


apply_style()


def histogram(df: AnalysisInput, column: str, bins: int = 30, ax: Axes | None = None, **kwargs: Any) -> Axes:
    """Histogram of one numeric column.

    Args:
        df: Data to plot (any `AnalysisInput` shape).
        column: Column to bin.
        bins: Number of histogram bins.
        ax: Existing matplotlib Axes to draw on; a new figure is created if omitted.
        **kwargs: Passed through to `Axes.hist`.
    """
    df = as_dataframe(df)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
    ax.hist(df[column].dropna(), bins=bins, color=CATEGORICAL[0], edgecolor=SURFACE, **kwargs)
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    ax.set_title(f"Distribution of {column}")
    return ax


def boxplot(df: AnalysisInput, column: str, by: str | None = None, ax: Axes | None = None) -> Axes:
    """Boxplot of one numeric column, optionally split by a categorical column.

    Args:
        df: Data to plot (any `AnalysisInput` shape).
        column: Numeric column to summarize.
        by: If given, draw one box per distinct value of this column instead of a single box.
        ax: Existing matplotlib Axes to draw on; a new figure is created if omitted.
    """
    df = as_dataframe(df)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
    if by is None:
        ax.boxplot(df[column].dropna(), tick_labels=[column],
                   patch_artist=True, boxprops={"facecolor": CATEGORICAL[0]})
    else:
        groups = df[by].dropna().unique()
        data = [df.loc[df[by] == g, column].dropna() for g in groups]
        bp = ax.boxplot(data, tick_labels=[str(g) for g in groups], patch_artist=True)
        for patch, color in zip(bp["boxes"], CATEGORICAL * (len(groups) // len(CATEGORICAL) + 1)):
            patch.set_facecolor(color)
        ax.set_xlabel(by)
    ax.set_ylabel(column)
    ax.set_title(f"{column} by {by}" if by else f"{column}")
    return ax


def scatter(df: AnalysisInput, x: str, y: str, hue: str | None = None, ax: Axes | None = None) -> Axes:
    """Scatter plot of two numeric columns, optionally colored by a third.

    Args:
        df: Data to plot (any `AnalysisInput` shape).
        x: Column for the x-axis.
        y: Column for the y-axis.
        hue: Column to color points by (categorical).
        ax: Existing matplotlib Axes to draw on; a new figure is created if omitted.
    """
    df = as_dataframe(df)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
    if hue is None:
        ax.scatter(df[x], df[y], color=CATEGORICAL[0], alpha=0.8, s=28, edgecolor=SURFACE)
    else:
        categories = df[hue].astype("category")
        for level, color in zip(categories.cat.categories, CATEGORICAL):
            mask = categories == level
            ax.scatter(df.loc[mask, x], df.loc[mask, y], color=color, alpha=0.8,
                       s=28, edgecolor=SURFACE, label=str(level))
        ax.legend(title=hue, frameon=False)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    return ax


def correlation_heatmap(corr_df: pd.DataFrame, ax: Axes | None = None, annotate: bool = True) -> Axes:
    """Heatmap of a correlation matrix (as returned by `o2org.analysis.stats.correlation`).

    Args:
        corr_df: A square correlation matrix.
        ax: Existing matplotlib Axes to draw on; a new figure is created if omitted.
        annotate: Whether to print each correlation value on its cell.
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
    im = ax.imshow(corr_df.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.grid(False)
    if annotate:
        for i in range(len(corr_df.index)):
            for j in range(len(corr_df.columns)):
                value = corr_df.values[i, j]
                text_color = INK_PRIMARY if abs(value) < 0.6 else SURFACE
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        color=text_color, fontsize=8)
    ax.figure.colorbar(im, ax=ax, label="correlation")
    ax.set_title("Correlation matrix")
    return ax


def pca_scatter(components_df: pd.DataFrame, hue: str | None = None, ax: Axes | None = None) -> Axes:
    """Scatter PC1 vs PC2.

    Args:
        components_df: Must have at least two columns named PC1, PC2 (as
            returned by `o2org.analysis.ml.pca(...).components`), plus `hue`
            itself if given.
        hue: Column to color points by.
        ax: Existing matplotlib Axes to draw on; a new figure is created if omitted.
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
    scatter(components_df, "PC1", "PC2", hue=hue, ax=ax)
    ax.set_title("PCA: PC1 vs PC2")
    return ax


def cluster_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    labels: np.ndarray,
    centers: np.ndarray | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Scatter two columns, colored by cluster assignment.

    Args:
        df: Data containing at least columns `x` and `y`.
        x: Column for the x-axis.
        y: Column for the y-axis.
        labels: One cluster id per row of `df` (as returned by
            `o2org.analysis.ml.kmeans_cluster(...).labels`).
        centers: Optional cluster-center coordinates to mark, shape (k, 2+).
        ax: Existing matplotlib Axes to draw on; a new figure is created if omitted.
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
    labels = np.asarray(labels)
    for i, cluster_id in enumerate(sorted(set(labels))):
        mask = labels == cluster_id
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.scatter(df.loc[mask, x], df.loc[mask, y], color=color, alpha=0.8,
                   s=28, edgecolor=SURFACE, label=f"cluster {cluster_id}")
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=140,
                   color=INK_PRIMARY, edgecolor=SURFACE, label="centers", zorder=5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Cluster assignments")
    ax.legend(frameon=False)
    return ax


def regression_plot(y_true: np.ndarray, y_pred: np.ndarray, ax: Axes | None = None) -> Axes:
    """Predicted vs. actual, with the perfect-prediction diagonal for reference.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values, same length as `y_true`.
        ax: Existing matplotlib Axes to draw on; a new figure is created if omitted.
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
    ax.scatter(y_true, y_pred, color=CATEGORICAL[0], alpha=0.8, s=28, edgecolor=SURFACE)
    lo, hi = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linestyle="--", linewidth=1.5, label="perfect prediction")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title("Predicted vs. actual")
    ax.legend(frameon=False)
    return ax
