"""Family B: elemental composition (`result_elemental_composition`).

Long format (element, value, stdev, unit) over the C/H/N/O/S panel — same
shape as the isotopes family, plus a distinctive Van Krevelen view.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from . import ml, stats, viz
from ._report import PipelineResult, describe_group_tests
from .utils.extract import extract, pivot_wide

RECORD_TYPE = "result_elemental_composition"

#: Standard atomic weights, used to convert mass fractions to molar ratios.
_ATOMIC_WEIGHTS = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999, "S": 32.06}


def _wide(client, project_id, grouping) -> tuple[pd.DataFrame, list[str]]:
    df_long = extract(client, RECORD_TYPE, project_id)
    index_cols = ["sample_id", grouping]
    df_wide = pivot_wide(df_long, index_cols=index_cols, analyte_col="details_element", value_col="details_value")
    elements = [c for c in df_wide.columns if c not in index_cols]
    return df_wide, elements


def molar_ratios(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Compute H/C, O/C, N/C molar ratios from element mass-fraction columns
    ("C", "H", "N", "O", "S") — the standard organic-matter-characterization
    view, added on top of whatever the reference repos did (neither computes
    elemental ratios).

    Args:
        df_wide: Wide DataFrame with element columns from `pivot_wide`.

    Returns:
        `df_wide` with `h_c`/`o_c`/`n_c` columns added (only the ratios whose
        both elements are present).
    """
    result = df_wide.copy()
    moles = {
        element: pd.to_numeric(df_wide[element], errors="coerce") / weight
        for element, weight in _ATOMIC_WEIGHTS.items() if element in df_wide.columns
    }
    if "H" in moles and "C" in moles:
        result["h_c"] = moles["H"] / moles["C"]
    if "O" in moles and "C" in moles:
        result["o_c"] = moles["O"] / moles["C"]
    if "N" in moles and "C" in moles:
        result["n_c"] = moles["N"] / moles["C"]
    return result


def stats_pipeline(
    client, project_id: Any, grouping: str = "sample_group", alpha: float = 0.01,
    correction: Literal["bonferroni", "fdr_bh", None] = "fdr_bh", include_figures: bool = True,
) -> PipelineResult:
    """Shapiro -> Kruskal-Wallis/ANOVA -> correction -> box/density plots per
    element, plus a Van Krevelen plot (H/C vs O/C, colored by group) as the
    primary view in place of PCA.

    Returns a `PipelineResult` — see `isotopes.stats_pipeline`'s docstring
    for what that means: one call already shows the full report by default.

    Args:
        client: A connected `OrgClient`.
        project_id: The project's id.
        grouping: Which sample attribute to compare groups on.
        alpha: Significance threshold.
        correction: Multiple-testing correction across elements.
        include_figures: Set False to skip building/embedding any plots.

    Returns:
        `wide` (with molar ratios added), `normality`, `tests`,
        `significant_elements`, `figures` (box/kde/van_krevelen), `van_krevelen`.
    """
    df_wide, elements = _wide(client, project_id, grouping)
    df_wide = molar_ratios(df_wide)

    normality = pd.concat(
        [stats.shapiro_by_group(df_wide, e, grouping, alpha=alpha).assign(element=e) for e in elements],
        ignore_index=True,
    ) if elements else pd.DataFrame()

    test_results = stats.test_all_features(df_wide, elements, grouping, test="auto", alpha=alpha, correction=correction)
    significant = test_results.loc[test_results["significant"], "feature"].tolist() if not test_results.empty else []

    summary_lines = describe_group_tests(test_results, df_wide, grouping, alpha, feature_label="element")

    figures: dict[str, Any] = {}
    van_krevelen = None
    if include_figures:
        if significant:
            figures["box"] = viz.box_plot(df_wide, significant, grouping)
            figures["kde"] = viz.kde_density_plot(df_wide, significant, grouping)

        if {"h_c", "o_c"}.issubset(df_wide.columns):
            van_krevelen = viz.van_krevelen_plot(df_wide, group_col=grouping)
            figures["van_krevelen"] = van_krevelen
            summary_lines.append("Van Krevelen plot: H/C vs O/C.")
        else:
            summary_lines.append("Van Krevelen plot skipped — need both H and O measured to compute H/C and O/C.")

    return PipelineResult(
        {
            "wide": df_wide,
            "normality": normality,
            "tests": test_results,
            "significant_elements": significant,
            "figures": figures,
            "van_krevelen": van_krevelen,
        },
        title=f"elemental.stats_pipeline — project {project_id}",
        summary_lines=summary_lines,
        figures=figures,
    )


def ml_pipeline(
    client, project_id: Any, grouping: str = "sample_group", method: Literal["kmeans", "random_forest"] = "kmeans",
    k: int = 3, positive_class: Any | None = None, negative_class: Any | None = None, include_figures: bool = True,
) -> PipelineResult:
    """`method="kmeans"` (default): unsupervised clustering over the H/C-O/C-
    N/C ratio space, closer exploration than a supervised classifier for
    finding compositional clusters not captured by an existing group label.
    `method="random_forest"`: supervised classification on element ratios —
    with only 4-5 elements, feature-importance ranking is trivial, so
    elbow-cutoff is skipped by default (just report ranked importance).

    Returns a `PipelineResult` — see `isotopes.stats_pipeline`'s docstring
    for what that means: one call already shows the full report by default.

    Args:
        client: A connected `OrgClient`.
        project_id: The project's id.
        grouping: Which sample attribute defines the classes (random_forest only).
        method: "kmeans" or "random_forest".
        k: Number of clusters (kmeans only).
        positive_class: For `method="random_forest"` only.
        negative_class: See `positive_class`.
        include_figures: Set False to skip building/embedding any plots.
    """
    df_wide, elements = _wide(client, project_id, grouping)
    df_wide = molar_ratios(df_wide)
    ratio_cols = [c for c in ("h_c", "o_c", "n_c") if c in df_wide.columns]
    title = f"elemental.ml_pipeline ({method}) — project {project_id}"

    if method == "kmeans":
        cluster_result = ml.kmeans_cluster(df_wide, ratio_cols, k=k)
        counts = pd.Series(cluster_result.labels).value_counts().sort_index()
        summary_lines = [
            f"{len(df_wide)} samples clustered over {', '.join(ratio_cols)} into k={k} clusters.",
            "Cluster sizes: " + ", ".join(f"cluster {c}: {n}" for c, n in counts.items()),
            f"Inertia: {cluster_result.inertia:.4f}.",
        ]
        figures = {}
        if include_figures and len(ratio_cols) >= 2:
            import plotly.graph_objects as go
            fig = go.Figure()
            labels = pd.Series(cluster_result.labels, index=df_wide.index)
            for cluster_id in sorted(labels.unique()):
                mask = labels == cluster_id
                fig.add_trace(go.Scatter(
                    x=df_wide.loc[mask, ratio_cols[0]], y=df_wide.loc[mask, ratio_cols[1]],
                    mode="markers", name=f"cluster {cluster_id}", marker=dict(size=10),
                ))
            fig.update_layout(title="K-means clusters", xaxis_title=ratio_cols[0], yaxis_title=ratio_cols[1], width=700, height=650)
            figures["clusters"] = fig
        return PipelineResult(
            {"wide": df_wide, "method": "kmeans", "result": cluster_result, "columns": ratio_cols},
            title=title, summary_lines=summary_lines, figures=figures,
        )

    if method == "random_forest":
        classes = df_wide[grouping].dropna().unique().tolist()
        positive_class = positive_class if positive_class is not None else (classes[0] if classes else None)
        negative_class = negative_class if negative_class is not None else (classes[1] if len(classes) > 1 else None)
        rf_result = ml.random_forest_classify(df_wide, elements, grouping, positive_class, negative_class)
        importances = ml.mdi_importance(rf_result.model, elements)
        accuracy = (rf_result.y_pred == rf_result.y_true).mean()
        note = "Elbow-cutoff skipped — trivial to rank on only 4-5 elements."
        summary_lines = [
            f"Classifying {rf_result.target_names[1]!r} vs {rf_result.target_names[0]!r} ({len(rf_result.y_true)} samples).",
            f"Cross-validated accuracy: {accuracy:.1%}.",
            "Feature importance (MDI): " + ", ".join(f"{r.feature}={r.importance:.3f}" for r in importances.itertuples()),
            note,
        ]
        figures = {}
        if include_figures:
            figures["roc"] = viz.roc_plot(rf_result.y_true, rf_result.y_proba)
        return PipelineResult(
            {"wide": df_wide, "method": "random_forest", "result": rf_result, "importances": importances, "note": note},
            title=title, summary_lines=summary_lines, figures=figures,
        )

    raise ValueError(f"method must be 'kmeans' or 'random_forest', got {method!r}")
