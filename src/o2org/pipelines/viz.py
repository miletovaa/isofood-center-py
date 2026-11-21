"""Visualizations — Plotly throughout (not matplotlib), so every figure gets
hover tooltips and the same dual PNG+HTML export via
`o2org.pipelines.utils.output.save_figure`. Structure and statistics are
ported directly from the reference lcms/peptide pipelines; the matplotlib
plotting calls there are transliterated to Plotly equivalents.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.metrics import auc, confusion_matrix, roc_curve

_DEFAULT_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def _group_colors(groups: list[Any]) -> dict[Any, str]:
    return {g: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i, g in enumerate(groups)}


def _per_feature_subplots(df, feature_cols, group_col, titles=None):
    feature_cols = list(feature_cols)
    titles = titles or [str(f) for f in feature_cols]
    groups = sorted(df[group_col].dropna().unique(), key=str)
    colors = _group_colors(groups)
    fig = make_subplots(rows=1, cols=len(feature_cols), shared_yaxes=False, subplot_titles=titles)
    return fig, feature_cols, groups, colors


def box_plot(df: pd.DataFrame, feature_cols: list[str], group_col: str, titles: list[str] | None = None) -> go.Figure:
    """Box plot per feature, one panel per feature, grouped by `group_col`
    within each panel (direct port of peptide_analysis.py's significant-
    peptide box plots).
    """
    fig, feature_cols, groups, colors = _per_feature_subplots(df, feature_cols, group_col, titles)
    for col_idx, feature in enumerate(feature_cols, start=1):
        for g in groups:
            values = df.loc[df[group_col] == g, feature].dropna().astype(float).values
            fig.add_trace(
                go.Box(y=values, name=str(g), boxpoints="outliers", showlegend=(col_idx == 1), marker_color=colors[g]),
                row=1, col=col_idx,
            )
    fig.update_layout(title="Box plots per feature", width=max(350 * len(feature_cols), 800), height=400)
    fig.update_xaxes(title_text=group_col)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    return fig


def median_bar_plot(df: pd.DataFrame, feature_cols: list[str], group_col: str, titles: list[str] | None = None) -> go.Figure:
    """Median bar chart per feature, one panel per feature, one bar per group."""
    fig, feature_cols, groups, colors = _per_feature_subplots(df, feature_cols, group_col, titles)
    for col_idx, feature in enumerate(feature_cols, start=1):
        for g in groups:
            values = df.loc[df[group_col] == g, feature].dropna().astype(float).values
            median_val = float(np.median(values)) if len(values) else np.nan
            fig.add_trace(
                go.Bar(x=[str(g)], y=[median_val], name=str(g), marker_color=colors[g], showlegend=(col_idx == 1)),
                row=1, col=col_idx,
            )
    fig.update_layout(barmode="group", title="Group medians per feature", width=max(300 * len(feature_cols), 300), height=350)
    fig.update_xaxes(title_text=group_col)
    fig.update_yaxes(title_text="Median value", row=1, col=1)
    return fig


def kde_density_plot(df: pd.DataFrame, feature_cols: list[str], group_col: str, titles: list[str] | None = None) -> go.Figure:
    """KDE density/distribution plot per feature, one curve per group."""
    fig, feature_cols, groups, colors = _per_feature_subplots(df, feature_cols, group_col, titles)
    for col_idx, feature in enumerate(feature_cols, start=1):
        group_vals = {}
        all_vals = []
        for g in groups:
            vals = df.loc[df[group_col] == g, feature].dropna().astype(float).values
            if len(vals) < 2:
                continue
            group_vals[g] = vals
            all_vals.append(vals)
        if not all_vals:
            continue
        concatenated = np.concatenate(all_vals)
        x_min, x_max = float(concatenated.min()), float(concatenated.max())
        if x_min == x_max:
            x_min, x_max = x_min - 1e-6, x_max + 1e-6
        xs = np.linspace(x_min, x_max, 200)
        for g, vals in group_vals.items():
            ys = gaussian_kde(vals)(xs)
            fig.add_trace(
                go.Scatter(x=xs, y=ys, mode="lines", name=str(g), line=dict(color=colors[g]), showlegend=(col_idx == 1)),
                row=1, col=col_idx,
            )
    fig.update_layout(title="Density distributions per feature", width=max(400 * len(feature_cols), 400), height=350)
    fig.update_xaxes(title_text="Value")
    fig.update_yaxes(title_text="Density", row=1, col=1)
    return fig


def pca_score_scatter(pca_result, group: pd.Series | None = None, pcs: tuple[str, str] = ("PC1", "PC2")) -> go.Figure:
    """PCA score scatter, colored by `group` if given."""
    scores = pca_result.scores
    fig = go.Figure()
    if group is not None:
        colors = _group_colors(sorted(group.dropna().unique(), key=str))
        for g in colors:
            idx = group == g
            fig.add_trace(go.Scatter(
                x=scores.loc[idx, pcs[0]], y=scores.loc[idx, pcs[1]], mode="markers", name=str(g),
                marker=dict(size=10, opacity=0.8, color=colors[g]), text=scores.index[idx].astype(str),
                hovertemplate="Sample: %{text}<br>%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter(x=scores[pcs[0]], y=scores[pcs[1]], mode="markers", marker=dict(size=10, opacity=0.8)))

    evr = pca_result.explained_variance_ratio
    pc_idx = {name: i for i, name in enumerate(scores.columns)}
    fig.update_layout(
        title="PCA score scatter",
        xaxis_title=f"{pcs[0]} ({evr[pc_idx[pcs[0]]] * 100:.1f}% var)",
        yaxis_title=f"{pcs[1]} ({evr[pc_idx[pcs[1]]] * 100:.1f}% var)",
        width=800, height=700,
    )
    return fig


def pca_loadings_plot(pca_result, top_n: int = 20, pcs: tuple[str, str] = ("PC1", "PC2")) -> go.Figure:
    """PCA loadings plot: the top `top_n` features by loading magnitude on `pcs`."""
    loadings = pca_result.loadings.copy()
    loadings["magnitude"] = np.sqrt(loadings[pcs[0]] ** 2 + loadings[pcs[1]] ** 2)
    top = loadings.nlargest(min(top_n, len(loadings)), "magnitude")

    fig = go.Figure()
    for feature, row in top.iterrows():
        fig.add_shape(type="line", x0=0, y0=0, x1=row[pcs[0]], y1=row[pcs[1]], line=dict(color="#BBBBBB", width=1))
    fig.add_trace(go.Scatter(
        x=top[pcs[0]], y=top[pcs[1]], mode="markers+text", text=top.index.astype(str), textposition="top center",
        marker=dict(size=9, color="#333333"), showlegend=False,
        hovertemplate="Feature: %{text}<br>%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="PCA loadings", xaxis_title=f"{pcs[0]} loading", yaxis_title=f"{pcs[1]} loading",
        xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="#333"),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="#333"),
        width=800, height=700,
    )
    return fig


def pca_biplot(pca_result, group: pd.Series | None = None, top_n: int = 20, pcs: tuple[str, str] = ("PC1", "PC2")) -> go.Figure:
    """PCA biplot: scores + top-loading feature vectors combined."""
    scores = pca_result.scores
    loadings = pca_result.loadings.copy()
    loadings["magnitude"] = np.sqrt(loadings[pcs[0]] ** 2 + loadings[pcs[1]] ** 2)
    top = loadings.nlargest(min(top_n, len(loadings)), "magnitude")

    score_max = float(np.max(np.abs(scores[list(pcs)].values))) if len(scores) else 1.0
    loading_max = float(np.max(np.abs(top[list(pcs)].values))) if len(top) else 1.0
    scale = (score_max / loading_max) * 0.7 if loading_max else 1.0
    scaled = top[list(pcs)] * scale

    fig = pca_score_scatter(pca_result, group=group, pcs=pcs)
    for feature, row in scaled.iterrows():
        fig.add_shape(type="line", x0=0, y0=0, x1=row[pcs[0]], y1=row[pcs[1]], line=dict(color="#444444", width=1))
    fig.add_trace(go.Scatter(
        x=scaled[pcs[0]], y=scaled[pcs[1]], mode="markers+text", text=scaled.index.astype(str), textposition="top center",
        marker=dict(color="#444444", size=9, symbol="diamond"), name="Feature loadings",
        hovertemplate="Feature: %{text}<br>Scaled %{xaxis.title.text}: %{x:.3f}<br>Scaled %{yaxis.title.text}: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(title="PCA biplot")
    return fig


def rsd_comparison_plot(rsd_by_stage: dict[str, pd.Series]) -> go.Figure:
    """RSD line/marker comparison plot: one trace per normalization stage,
    overlaid (direct port of s01prep.py's fig1).
    """
    symbols = ["circle", "x", "square", "diamond", "triangle-up"]
    fig = go.Figure()
    for i, (name, series) in enumerate(rsd_by_stage.items()):
        fig.add_trace(go.Scatter(
            y=series.values, text=series.index, mode="lines+markers", name=name,
            marker=dict(symbol=symbols[i % len(symbols)], opacity=0.7),
        ))
    fig.update_layout(
        title="RSD per feature: raw vs. normalizations", xaxis_title="Feature index", yaxis_title="RSD (%)",
        template="plotly_white", legend=dict(x=0.01, y=0.99), width=900, height=600,
    )
    return fig


def rsd_comparison_subplots(rsd_by_stage: dict[str, pd.Series]) -> go.Figure:
    """RSD comparison on separate stacked panels, one per stage (direct port
    of s01prep.py's fig2).
    """
    symbols = ["circle", "x", "diamond", "square", "triangle-up"]
    names = list(rsd_by_stage.keys())
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=names)
    for i, name in enumerate(names, start=1):
        series = rsd_by_stage[name]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(series))), y=series.values, text=series.index,
                hovertemplate="Feature: %{text}<br>RSD: %{y}<extra></extra>",
                mode="lines+markers", name=name, marker=dict(symbol=symbols[(i - 1) % len(symbols)], opacity=0.7),
            ),
            row=i, col=1,
        )
    fig.update_layout(title="RSD per feature comparison", width=700, height=350 * len(names))
    fig.update_yaxes(title_text="RSD (%)")
    fig.update_xaxes(title_text="Feature index")
    return fig


def rsd_scatter_multi(rsd_raw: pd.Series, rsd_by_stage: dict[str, pd.Series]) -> go.Figure:
    """RSD scatter, normalized vs. raw, one panel per normalization stage
    (direct port of s01prep.py's fig3).
    """
    names = list(rsd_by_stage.keys())
    fig = make_subplots(
        rows=1, cols=len(names), shared_yaxes=True, horizontal_spacing=0.05,
        subplot_titles=[f"{name} vs Raw" for name in names],
    )
    for i, name in enumerate(names, start=1):
        data = rsd_by_stage[name]
        max_val = max(rsd_raw.max(), data.max())
        fig.add_trace(go.Scatter(x=rsd_raw, y=data, mode="markers", marker=dict(opacity=0.5, size=6), name=name), row=1, col=i)
        fig.update_xaxes(title_text="Raw RSD (%)", range=[0, max_val], row=1, col=i)
        fig.update_yaxes(title_text="Normalized RSD (%)", range=[0, max_val], row=1, col=i)
        fig.add_trace(
            go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(color="gray", dash="dash", width=1), showlegend=False),
            row=1, col=i,
        )
    fig.update_layout(title="Feature-wise RSD comparison after normalizations", template="plotly_white", width=500 * len(names), height=450, showlegend=False)
    return fig


def rsd_scatter_single(rsd_raw: pd.Series, rsd_by_stage: dict[str, pd.Series]) -> go.Figure:
    """RSD scatter, every normalization stage vs. raw on one grid (direct
    port of s01prep.py's fig4).
    """
    fig = go.Figure()
    max_val = rsd_raw.max()
    for i, (name, series) in enumerate(rsd_by_stage.items()):
        max_val = max(max_val, series.max())
        fig.add_trace(go.Scatter(
            x=rsd_raw, y=series, mode="markers", marker=dict(color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)], opacity=0.5, size=6), name=name,
        ))
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(color="gray", dash="dash", width=1), name="y=x"))
    fig.update_layout(
        title="All normalizations vs. raw (feature-wise RSD)", xaxis_title="Raw RSD (%)", yaxis_title="Normalized RSD (%)",
        template="plotly_white", width=600, height=600, legend=dict(x=0.02, y=0.98),
    )
    return fig


def imputed_counts_plot(imputed_counts: pd.Series) -> go.Figure:
    """Bar chart of imputed value counts per feature, sorted descending."""
    sorted_counts = imputed_counts.sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=sorted_counts.index.astype(str), y=sorted_counts.values))
    fig.update_layout(
        title="Imputed 0s/NaNs per feature (LOD/2 imputation)", yaxis_title="Number of imputed values",
        width=max(16 * len(sorted_counts), 600), height=400,
    )
    return fig


def roc_plot(y_true, y_proba) -> go.Figure:
    """ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC = {roc_auc:.2f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), showlegend=False))
    fig.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR", width=700, height=600)
    return fig


def threshold_sensitivity_plot(tuned: pd.DataFrame, threshold: float | None = None) -> go.Figure:
    """Precision/recall/F1 vs. decision threshold, for both classes (from
    `ml.tune_threshold`'s output).
    """
    fig = go.Figure()
    for metric, dash in (("precision", "dash"), ("recall", "solid"), ("f1", "dashdot")):
        fig.add_trace(go.Scatter(x=tuned["threshold"], y=tuned[f"{metric}_0"], mode="lines", name=f"{metric} (class 0)", line=dict(dash=dash)))
        fig.add_trace(go.Scatter(x=tuned["threshold"], y=tuned[f"{metric}_1"], mode="lines", name=f"{metric} (class 1)", line=dict(dash=dash), opacity=0.5))
    if threshold is not None:
        fig.add_vline(x=threshold, line=dict(color="gray", dash="dot"), annotation_text=f"threshold={threshold}")
    fig.update_layout(title="Precision/recall/F1 vs. decision threshold", xaxis_title="Decision threshold", yaxis_title="Score", width=900, height=500)
    return fig


def confusion_matrix_table(y_true, y_pred, target_names: tuple[str, str]) -> pd.DataFrame:
    """Confusion matrix as a plain DataFrame (table/CSV, not a figure)."""
    matrix = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(matrix, index=list(target_names), columns=list(target_names))


def mdi_elbow_plot(importances_sorted, elbow_idx: int, cutoff: float) -> go.Figure:
    """MDI feature importance plot with the elbow point marked."""
    y_vals = np.asarray(importances_sorted, dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_vals))), y=y_vals, mode="lines+markers", name="MDI importance"))
    fig.add_vline(x=elbow_idx, line=dict(color="red", dash="dash"), annotation_text=f"elbow @ {cutoff:.5f}")
    fig.update_layout(title="MDI feature importances with elbow point", xaxis_title="Feature rank", yaxis_title="Importance", width=800, height=500)
    return fig


def top_features_barh(importances_df: pd.DataFrame, cutoff: float) -> go.Figure:
    """Horizontal bar chart of features above the elbow cutoff."""
    top = importances_df.loc[importances_df["importance"] >= cutoff].sort_values("importance")
    fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"].astype(str), orientation="h"))
    fig.update_layout(
        title="Top features above elbow cutoff", xaxis_title="Importance", width=900,
        height=max(400, 20 * len(top)),
    )
    return fig


def isotope_biplot(df: pd.DataFrame, x_analyte: str, y_analyte: str, group_col: str = "sample_group") -> go.Figure:
    """Raw scatter of two isotope analytes (e.g. δ13C vs δ15N) colored by
    group — the standard view for an isotope panel with ≤6 dimensions,
    used in place of a full PCA.
    """
    groups = sorted(df[group_col].dropna().unique(), key=str)
    colors = _group_colors(groups)
    fig = go.Figure()
    for g in groups:
        sub = df[df[group_col] == g]
        fig.add_trace(go.Scatter(
            x=sub[x_analyte], y=sub[y_analyte], mode="markers", name=str(g),
            marker=dict(size=10, opacity=0.8, color=colors[g]),
        ))
    fig.update_layout(title=f"Isotope biplot: {x_analyte} vs {y_analyte}", xaxis_title=x_analyte, yaxis_title=y_analyte, width=700, height=650)
    return fig


def van_krevelen_plot(df: pd.DataFrame, h_c_col: str = "h_c", o_c_col: str = "o_c", group_col: str = "sample_group") -> go.Figure:
    """Van Krevelen plot (H/C vs O/C, colored by group) — the field-standard
    view for organic-matter characterization from elemental composition.
    """
    groups = sorted(df[group_col].dropna().unique(), key=str)
    colors = _group_colors(groups)
    fig = go.Figure()
    for g in groups:
        sub = df[df[group_col] == g]
        fig.add_trace(go.Scatter(
            x=sub[o_c_col], y=sub[h_c_col], mode="markers", name=str(g),
            marker=dict(size=10, opacity=0.8, color=colors[g]),
        ))
    fig.update_layout(title="Van Krevelen plot", xaxis_title="O/C", yaxis_title="H/C", width=700, height=650)
    return fig


# --------------------------------------------------------------------------
# Plots for the methods added after the O2/ISOFood publication survey.
# --------------------------------------------------------------------------


def scores_scatter(scores: pd.DataFrame, group_col: str, x: str | None = None, y: str | None = None,
                   title: str = "Scores", x_label: str | None = None, y_label: str | None = None) -> go.Figure:
    """Generic 2D score scatter (LDA, PLS-DA, PCA, ...), one colour per group.
    Falls back to a 1D strip plot when there's only one score column."""
    score_cols = [c for c in scores.columns if c != group_col]
    x = x or score_cols[0]
    y = y or (score_cols[1] if len(score_cols) > 1 else None)
    groups = sorted(scores[group_col].dropna().unique(), key=str)
    colors = _group_colors(groups)
    fig = go.Figure()
    for g in groups:
        sub = scores[scores[group_col] == g]
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub[y] if y else [str(g)] * len(sub), mode="markers", name=str(g),
            marker=dict(color=colors[g], size=10, line=dict(width=0.5, color="white")),
            hovertext=[str(i) for i in sub.index],
        ))
    fig.update_layout(title=title, xaxis_title=x_label or x, yaxis_title=(y_label or y) if y else group_col,
                      width=750, height=600)
    return fig


def correlation_heatmap(r: pd.DataFrame, p: pd.DataFrame | None = None, alpha: float = 0.05,
                        title: str = "Correlation matrix") -> go.Figure:
    """Correlation heatmap (−1 blue … +1 red); cells with p ≥ `alpha` are
    left unlabelled when a p-value matrix is given."""
    text = r.round(2).astype(str)
    if p is not None:
        text = text.where(p < alpha, "")
        np.fill_diagonal(text.values, "")
    fig = go.Figure(go.Heatmap(
        z=r.values, x=[str(c) for c in r.columns], y=[str(c) for c in r.index], zmin=-1, zmax=1,
        colorscale="RdBu_r", text=text.values, texttemplate="%{text}", colorbar=dict(title="r"),
    ))
    size = max(450, 40 * len(r) + 200)
    fig.update_layout(title=title, width=size, height=size, yaxis_autorange="reversed")
    return fig


def dendrogram_plot(hca_result, color_threshold: float | None = None, title: str = "Hierarchical clustering") -> go.Figure:
    """Dendrogram for an `ml.hierarchical_cluster` result."""
    from scipy.cluster.hierarchy import dendrogram

    d = dendrogram(hca_result.linkage, labels=hca_result.sample_labels, no_plot=True, color_threshold=color_threshold)
    # scipy names colours matplotlib-style ("C0", "C1", ...); map them onto our palette
    def _color(c: str) -> str:
        return _DEFAULT_COLORS[int(c[1:]) % len(_DEFAULT_COLORS)] if c.startswith("C") and c[1:].isdigit() else c

    fig = go.Figure()
    for xs, ys, c in zip(d["icoord"], d["dcoord"], d["color_list"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=_color(c), width=1.5), hoverinfo="skip", showlegend=False))
    ticks = [5 + 10 * i for i in range(len(d["ivl"]))]
    fig.update_layout(title=f"{title} ({hca_result.method} linkage)", yaxis_title="Distance",
                      xaxis=dict(tickvals=ticks, ticktext=d["ivl"], tickangle=-90),
                      width=max(700, 14 * len(ticks)), height=500)
    return fig


def confusion_heatmap(confusion: pd.DataFrame, title: str = "Confusion matrix (cross-validated)") -> go.Figure:
    """Heatmap of a confusion matrix DataFrame (rows = true, columns = predicted)."""
    fig = go.Figure(go.Heatmap(
        z=confusion.values, x=[str(c) for c in confusion.columns], y=[str(c) for c in confusion.index],
        colorscale="Blues", text=confusion.values, texttemplate="%{text}", showscale=False,
    ))
    n = len(confusion)
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="True", yaxis_autorange="reversed",
                      width=max(420, 70 * n + 200), height=max(400, 70 * n + 150))
    return fig


def importance_bar(importances: pd.DataFrame, value_col: str = "importance", top_n: int = 25,
                   threshold: float | None = None, title: str = "Feature importance") -> go.Figure:
    """Horizontal bar chart of the top features (permutation importance,
    VIP scores, MDI...). `threshold` draws a reference line (e.g. VIP = 1)."""
    top = importances.nlargest(top_n, value_col).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top[value_col], y=top["feature"].astype(str), orientation="h",
        error_x=dict(type="data", array=top["std"]) if "std" in top else None, marker_color="#636EFA",
    ))
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color="red")
    fig.update_layout(title=title, xaxis_title=value_col, width=700, height=max(350, 22 * len(top) + 120))
    return fig


def permutation_hist(perm: dict, title: str = "Permutation test") -> go.Figure:
    """Distribution of shuffled-label scores vs the real score (`ml.permutation_test`)."""
    fig = go.Figure(go.Histogram(x=perm["permutation_scores"], nbinsx=30, name="Permuted labels", marker_color="#B0B7C3"))
    fig.add_vline(x=perm["score"], line_color="#EF553B", line_width=3,
                  annotation_text=f"model = {perm['score']:.2f} (p = {perm['pvalue']:.3f})")
    fig.update_layout(title=title, xaxis_title="Balanced accuracy (CV)", yaxis_title="Count", width=700, height=400)
    return fig


def model_comparison_plot(comparison: pd.DataFrame, metric: str = "balanced_accuracy") -> go.Figure:
    """Bar chart of `ml.compare_classifiers` output."""
    ok = comparison.dropna(subset=[metric]).sort_values(metric)
    fig = go.Figure(go.Bar(x=ok[metric], y=ok["label"], orientation="h", marker_color="#00CC96",
                           text=ok[metric].round(3), textposition="outside"))
    fig.update_layout(title=f"Model comparison — {metric} (cross-validated)", xaxis=dict(range=[0, 1.1]),
                      width=750, height=max(350, 35 * len(ok) + 120))
    return fig


def regression_plot(df: pd.DataFrame, x: str, y: str, group_col: str | None = None, model=None,
                    title: str | None = None) -> go.Figure:
    """Scatter of `y` vs `x` with an OLS fit line (pass a fitted
    `stats.linear_regression` model, or one is fitted for you)."""
    from . import stats as _stats

    sub = df[[x, y] + ([group_col] if group_col else [])].dropna()
    model = model or _stats.linear_regression(sub, y, x)
    fig = go.Figure()
    if group_col:
        colors = _group_colors(sorted(sub[group_col].unique(), key=str))
        for g, color in colors.items():
            s = sub[sub[group_col] == g]
            fig.add_trace(go.Scatter(x=s[x], y=s[y], mode="markers", name=str(g), marker=dict(color=color, size=9)))
    else:
        fig.add_trace(go.Scatter(x=sub[x], y=sub[y], mode="markers", name="samples", marker=dict(size=9)))
    xs = np.linspace(sub[x].astype(float).min(), sub[x].astype(float).max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=model.params.iloc[0] + model.params.iloc[1] * xs, mode="lines", name="OLS fit",
                             line=dict(color="black", dash="dash")))
    fig.update_layout(title=title or f"{y} ~ {x}  (R² = {model.rsquared:.3f}, p = {model.pvalues.iloc[1]:.2e})",
                      xaxis_title=x, yaxis_title=y, width=750, height=550)
    return fig


def interaction_plot(df: pd.DataFrame, value_col: str, factor_a: str, factor_b: str) -> go.Figure:
    """Mean ± SE of `value_col` for each level of `factor_a`, one line per
    level of `factor_b` — plot this before interpreting a two-way ANOVA."""
    agg = df.groupby([factor_b, factor_a])[value_col].agg(["mean", "sem"]).reset_index()
    colors = _group_colors(sorted(agg[factor_b].unique(), key=str))
    fig = go.Figure()
    for g, color in colors.items():
        s = agg[agg[factor_b] == g]
        fig.add_trace(go.Scatter(x=s[factor_a].astype(str), y=s["mean"], error_y=dict(type="data", array=s["sem"]),
                                 mode="lines+markers", name=str(g), line=dict(color=color)))
    fig.update_layout(title=f"Interaction: {value_col} by {factor_a} × {factor_b}", xaxis_title=factor_a,
                      yaxis_title=f"mean {value_col}", legend_title=factor_b, width=750, height=500)
    return fig


def scree_plot(pca_result) -> go.Figure:
    """Explained variance per principal component, with the cumulative line."""
    evr = np.asarray(pca_result.explained_variance_ratio) * 100
    pcs = [f"PC{i + 1}" for i in range(len(evr))]
    fig = go.Figure(go.Bar(x=pcs, y=evr, name="per PC", marker_color="#636EFA"))
    fig.add_trace(go.Scatter(x=pcs, y=np.cumsum(evr), mode="lines+markers", name="cumulative", line=dict(color="#EF553B")))
    fig.update_layout(title="Scree plot", yaxis_title="Explained variance (%)", width=650, height=420)
    return fig
