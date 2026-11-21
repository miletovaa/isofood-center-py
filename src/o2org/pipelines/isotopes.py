"""Family A: stable isotopes (`result_stable_isotopes`).

Already long-format (one row per analyte: δ13C, δ15N, δ34S, δ2H, δ18O, ...)
— structurally closest to the peptide_analysis reference pipeline.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go

from . import ml, preprocessing, stats, viz
from ._report import PipelineResult, describe_group_tests
from .utils.extract import extract, pivot_wide

RECORD_TYPE = "result_stable_isotopes"

#: Preferred analyte pairs for the isotope biplot, tried in order.
_BIPLOT_PAIRS = (("d13c", "d15n"), ("d34s", "d18o"))
_MULTIVARIATE_ANALYTES = ("d13c", "d2h", "d15n", "d18o", "d34s")


def _wide(client, project_id, grouping) -> tuple[pd.DataFrame, list[str]]:
    df_long = extract(client, RECORD_TYPE, project_id)
    index_cols = ["sample_id", grouping]
    df_wide = pivot_wide(df_long, index_cols=index_cols, analyte_col="details_analyte", value_col="details_value")
    analytes = [c for c in df_wide.columns if c not in index_cols]
    return df_wide, analytes


def stats_pipeline(
    client, project_id: Any, grouping: str = "sample_group", alpha: float = 0.01,
    correction: Literal["bonferroni", "fdr_bh", None] = "fdr_bh", grubbs_alpha: float = 0.05,
    include_figures: bool = True,
) -> PipelineResult:
    """Shapiro -> Kruskal-Wallis/ANOVA -> correction -> box/KDE plots, plus
    an isotope biplot (or full PCA when enough dimensions are present)
    instead of the peptide pipeline's mandatory PCA step.

    Returns a `PipelineResult` — a dict (`result["tests"]`, `result["wide"]`,
    etc. all work as documented below) that *also* renders as a full report
    (sample counts, per-analyte significance, and every figure) when it's the
    last expression in a Jupyter cell, or via `print(result)` elsewhere. One
    call with no extra steps already shows the whole analysis.

    Args:
        client: A connected `OrgClient`.
        project_id: The project's id.
        grouping: Which sample attribute to compare groups on — see
            `o2org.pipelines.utils.GROUPING_CANDIDATES`.
        alpha: Significance threshold for normality and group-difference tests.
        correction: Multiple-testing correction across analytes.
        grubbs_alpha: Significance level for outlier flagging (isotope values
            aren't LOQ-censored, so Grubbs' test replaces LOD/2 imputation).
        include_figures: Set False to skip building/embedding any plots (the
            data and test results are unaffected) — faster for headless/batch use.

    Returns:
        `wide`, `outliers` (per analyte), `normality`, `tests`,
        `significant_analytes`, `figures` (box/kde/isotope_biplot/pca_score),
        `isotope_biplot`, `pca` (only set when ≥4 of C/H/N/O/S analytes are
        present together).
    """
    df_wide, analytes = _wide(client, project_id, grouping)

    outliers = {
        analyte: preprocessing.grubbs_outliers(df_wide[analyte].values, alpha=grubbs_alpha)
        for analyte in analytes
    }

    normality = pd.concat(
        [stats.shapiro_by_group(df_wide, a, grouping, alpha=alpha).assign(analyte=a) for a in analytes],
        ignore_index=True,
    ) if analytes else pd.DataFrame()

    test_results = stats.test_all_features(df_wide, analytes, grouping, test="auto", alpha=alpha, correction=correction)
    significant = test_results.loc[test_results["significant"], "feature"].tolist() if not test_results.empty else []

    summary_lines = describe_group_tests(test_results, df_wide, grouping, alpha, feature_label="analyte")

    figures: dict[str, Any] = {}
    biplot = None
    pca_result = None

    if include_figures:
        if significant:
            figures["box"] = viz.box_plot(df_wide, significant, grouping)
            figures["kde"] = viz.kde_density_plot(df_wide, significant, grouping)

        for x, y in _BIPLOT_PAIRS:
            if x in analytes and y in analytes:
                biplot = viz.isotope_biplot(df_wide, x, y, group_col=grouping)
                figures["isotope_biplot"] = biplot
                summary_lines.append(f"Isotope biplot: {x} vs {y}.")
                break

        present = [a for a in _MULTIVARIATE_ANALYTES if a in analytes]
        if len(present) >= 4:
            pca_result = ml.pca(df_wide, columns=present)
            figures["pca_score"] = viz.pca_score_scatter(pca_result, group=df_wide[grouping])
            summary_lines.append(f"PCA over {len(present)} isotopes: PC1+PC2 explain {pca_result.explained_variance_ratio[:2].sum() * 100:.1f}% of variance.")
        else:
            summary_lines.append(f"PCA skipped — only {len(present)} of C/H/N/O/S isotopes present together (need ≥4).")

    return PipelineResult(
        {
            "wide": df_wide,
            "outliers": outliers,
            "normality": normality,
            "tests": test_results,
            "significant_analytes": significant,
            "figures": figures,
            "isotope_biplot": biplot,
            "pca": pca_result,
        },
        title=f"isotopes.stats_pipeline — project {project_id}",
        summary_lines=summary_lines,
        figures=figures,
    )


def ml_pipeline(
    client, project_id: Any, grouping: str = "sample_group", method: Literal["lda", "random_forest"] = "lda",
    positive_class: Any | None = None, negative_class: Any | None = None, include_figures: bool = True,
) -> PipelineResult:
    """Provenance classification. `method="lda"` (default) is the practical
    substitute for a Bayesian mixing model (MixSIAR has no Python port) on a
    handful of isotope features; `method="random_forest"` is a valid
    feature-importance sanity check but expect low value from elbow-cutoff
    on only 5-6 features.

    Returns a `PipelineResult` — see `stats_pipeline`'s docstring for what
    that means: one call already shows the full report by default.

    Args:
        client: A connected `OrgClient`.
        project_id: The project's id.
        grouping: Which sample attribute defines the classes.
        method: "lda" or "random_forest".
        positive_class: For `method="random_forest"` only — defaults to the
            first two distinct `grouping` values if omitted.
        negative_class: See `positive_class`.
        include_figures: Set False to skip building/embedding plots.
    """
    df_wide, analytes = _wide(client, project_id, grouping)
    title = f"isotopes.ml_pipeline ({method}) — project {project_id}"

    if method == "lda":
        result = ml.lda_classify(df_wide, analytes, grouping)
        accuracy = (result.encoder.inverse_transform(result.y_pred) == df_wide[grouping].values).mean()
        summary_lines = [
            f"{len(df_wide)} samples across {len(result.encoder.classes_)} classes: {', '.join(result.encoder.classes_)}.",
            f"LDA classification accuracy (resubstitution, not cross-validated): {accuracy:.1%}.",
        ]
        figures = {}
        ld_cols = [c for c in result.scores.columns if c != grouping]
        if include_figures and len(ld_cols) >= 2:
            colors = viz._group_colors(sorted(df_wide[grouping].dropna().unique(), key=str))
            fig = go.Figure()
            for g, color in colors.items():
                sub = result.scores[result.scores[grouping] == g]
                fig.add_trace(go.Scatter(x=sub[ld_cols[0]], y=sub[ld_cols[1]], mode="markers", name=str(g), marker=dict(color=color, size=10)))
            fig.update_layout(title="LDA scores", xaxis_title=ld_cols[0], yaxis_title=ld_cols[1], width=700, height=650)
            figures["lda_scores"] = fig
        elif include_figures and len(ld_cols) == 1:
            # exactly 2 classes -> LDA has only one discriminant axis (LD1);
            # a 2D score scatter isn't possible, so show its distribution per group instead.
            figures["lda_scores"] = viz.kde_density_plot(result.scores, ld_cols, grouping)
        return PipelineResult(
            {"wide": df_wide, "method": "lda", "result": result},
            title=title, summary_lines=summary_lines, figures=figures,
        )

    if method == "random_forest":
        classes = df_wide[grouping].dropna().unique().tolist()
        positive_class = positive_class if positive_class is not None else (classes[0] if classes else None)
        negative_class = negative_class if negative_class is not None else (classes[1] if len(classes) > 1 else None)
        rf_result = ml.random_forest_classify(df_wide, analytes, grouping, positive_class, negative_class)
        importances = ml.mdi_importance(rf_result.model, analytes)
        cutoff, elbow_idx = ml.elbow_cutoff(importances["importance"].values)
        accuracy = (rf_result.y_pred == rf_result.y_true).mean()
        note = (
            "Feature-importance sanity check only — with 5-6 isotope "
            "features there's no thousands-of-features pool to cut down "
            "from, so the elbow point is not especially informative."
        )
        summary_lines = [
            f"Classifying {rf_result.target_names[1]!r} vs {rf_result.target_names[0]!r} ({len(rf_result.y_true)} samples).",
            f"Cross-validated accuracy: {accuracy:.1%}.",
            "Feature importance (MDI): " + ", ".join(f"{r.feature}={r.importance:.3f}" for r in importances.itertuples()),
            note,
        ]
        figures = {}
        if include_figures:
            figures["roc"] = viz.roc_plot(rf_result.y_true, rf_result.y_proba)
            figures["mdi_elbow"] = viz.mdi_elbow_plot(importances["importance"].values, elbow_idx, cutoff)
        return PipelineResult(
            {
                "wide": df_wide, "method": "random_forest", "result": rf_result,
                "importances": importances, "elbow_cutoff": cutoff, "elbow_index": elbow_idx, "note": note,
            },
            title=title, summary_lines=summary_lines, figures=figures,
        )

    raise ValueError(f"method must be 'lda' or 'random_forest', got {method!r}")
