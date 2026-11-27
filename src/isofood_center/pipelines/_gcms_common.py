"""Internal: the normalize -> impute -> RSD -> RF chain shared by the
MK GC-MS/IRMS and VOC GC-MS/IRMS ML pipelines (direct reuse of the reference
lcms pipeline's `s01prep.py` + `s03rforest.py`, factored out since both
families run it verbatim over different feature panels).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import ml, preprocessing, stats, viz
from ._report import PipelineResult


def run_normalize_impute_rf(
    df_wide: pd.DataFrame,
    feature_cols: list[str],
    grouping: str,
    batch_col: str,
    is_column: str | None,
    zero_threshold: float | None,
    positive_class: Any | None,
    negative_class: Any | None,
    title: str,
    feature_label: str = "compound",
    include_figures: bool = True,
) -> PipelineResult:
    raw = df_wide.copy()

    normalized = df_wide
    if is_column is not None and is_column in feature_cols:
        normalize_cols = [c for c in feature_cols if c != is_column]
        normalized = preprocessing.is_normalize(normalized, is_column, columns=normalize_cols)
        working_cols = normalize_cols
    else:
        working_cols = list(feature_cols)

    dropped = []
    if zero_threshold is not None:
        normalized = preprocessing.zero_nan_threshold(normalized, working_cols, threshold=zero_threshold, axis="columns")
        dropped = [c for c in working_cols if c not in normalized.columns]
        working_cols = [c for c in working_cols if c in normalized.columns]

    imputed, imputed_counts = preprocessing.lod_half_impute(normalized, working_cols, batch_col=batch_col)
    final = preprocessing.batch_median_normalize(imputed, working_cols, batch_col=batch_col)

    rsd_raw = stats.rsd(raw, working_cols)
    rsd_stages = {
        "IS normalized" if is_column else "Normalized": stats.rsd(normalized, working_cols),
        "IS + batch median" if is_column else "Batch median": stats.rsd(final, working_cols),
    }

    classes = final[grouping].dropna().unique().tolist()
    positive_class = positive_class if positive_class is not None else (classes[0] if classes else None)
    negative_class = negative_class if negative_class is not None else (classes[1] if len(classes) > 1 else None)

    rf_result = ml.random_forest_classify(final, working_cols, grouping, positive_class, negative_class)
    importances = ml.mdi_importance(rf_result.model, working_cols)
    cutoff, elbow_idx = ml.elbow_cutoff(importances["importance"].values)
    tuned = ml.tune_threshold(rf_result.y_true, rf_result.y_proba)
    accuracy = (rf_result.y_pred == rf_result.y_true).mean()
    top_features = importances.loc[importances["importance"] >= cutoff, "feature"].tolist()

    figures = {}
    if include_figures:
        figures = {
            "rsd_comparison": viz.rsd_comparison_plot({"Raw": rsd_raw, **rsd_stages}),
            "rsd_comparison_subplots": viz.rsd_comparison_subplots({"Raw": rsd_raw, **rsd_stages}),
            "rsd_scatter_multi": viz.rsd_scatter_multi(rsd_raw, rsd_stages),
            "rsd_scatter_single": viz.rsd_scatter_single(rsd_raw, rsd_stages),
            "imputed_counts": viz.imputed_counts_plot(imputed_counts),
            "roc": viz.roc_plot(rf_result.y_true, rf_result.y_proba),
            "threshold_sensitivity": viz.threshold_sensitivity_plot(tuned),
            "mdi_elbow": viz.mdi_elbow_plot(importances["importance"].values, elbow_idx, cutoff),
            "top_features": viz.top_features_barh(importances, cutoff),
        }

    counts = final[grouping].value_counts()
    summary_lines = [
        f"{len(final)} samples: " + ", ".join(f"{g} ({n})" for g, n in counts.items()),
        f"{len(feature_cols)} {feature_label}s extracted"
        + (f", {len(dropped)} dropped below the zero/NaN threshold ({zero_threshold})" if dropped else "")
        + f", {len(working_cols)} carried through to the RF classifier.",
        f"Classifying {rf_result.target_names[1]!r} vs {rf_result.target_names[0]!r}: cross-validated accuracy {accuracy:.1%}.",
        f"{len(top_features)} {feature_label}s above the MDI elbow cutoff ({cutoff:.4f}): " + ", ".join(map(str, top_features[:10]))
        + (", ..." if len(top_features) > 10 else ""),
        f"RSD (median): raw {rsd_raw.median():.1f}% -> normalized {list(rsd_stages.values())[-1].median():.1f}%.",
    ]

    return PipelineResult(
        {
            "raw": raw,
            "normalized": final,
            "imputed_counts": imputed_counts,
            "rf": rf_result,
            "importances": importances,
            "elbow_cutoff": cutoff,
            "elbow_index": elbow_idx,
            "tuned_thresholds": tuned,
            "figures": figures,
        },
        title=title,
        summary_lines=summary_lines,
        figures=figures,
    )
