"""Family C: MK GC-MS / MK GC-IRMS (`result_mk_gc_ms` / `result_mk_gc_irms`).

A fixed fatty-acid panel — wide, closed feature set — structurally the
closest match to the reference lcms pipeline's bounded ANP/RP feature set.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from . import stats, viz
from ._gcms_common import run_normalize_impute_rf
from ._report import PipelineResult, describe_group_tests
from .utils.extract import extract, pivot_wide

RECORD_TYPES = {"gc_ms": "result_mk_gc_ms", "gc_irms": "result_mk_gc_irms"}


def _wide(client, project_id, grouping, technique) -> tuple[pd.DataFrame, list[str]]:
    record_type = RECORD_TYPES[technique]
    df_long = extract(client, record_type, project_id)
    index_cols = ["sample_id", grouping, "performed_at", "instrument"]
    index_cols = [c for c in index_cols if c in df_long.columns]
    df_wide = pivot_wide(df_long, index_cols=index_cols, analyte_col="details_compound_id", value_col="details_value")
    compounds = [c for c in df_wide.columns if c not in index_cols]
    return df_wide, compounds


def stats_pipeline(
    client, project_id: Any, technique: Literal["gc_ms", "gc_irms"] = "gc_ms", grouping: str = "sample_group",
    alpha: float = 0.01, correction: Literal["bonferroni", "fdr_bh", None] = "fdr_bh", include_figures: bool = True,
) -> PipelineResult:
    """Shapiro -> Kruskal-Wallis/ANOVA -> correction -> box/density plots,
    applied per fatty acid — same shape as the isotopes/elemental stats
    pipelines.

    Returns a `PipelineResult` — see `isotopes.stats_pipeline`'s docstring
    for what that means: one call already shows the full report by default.

    Args:
        client: A connected `OrgClient`.
        project_id: The project's id.
        technique: "gc_ms" (result_mk_gc_ms) or "gc_irms" (result_mk_gc_irms).
        grouping: Which sample attribute to compare groups on.
        alpha: Significance threshold.
        correction: Multiple-testing correction across fatty acids.
        include_figures: Set False to skip building/embedding any plots.
    """
    df_wide, compounds = _wide(client, project_id, grouping, technique)

    normality = pd.concat(
        [stats.shapiro_by_group(df_wide, c, grouping, alpha=alpha).assign(compound_id=c) for c in compounds],
        ignore_index=True,
    ) if compounds else pd.DataFrame()

    test_results = stats.test_all_features(df_wide, compounds, grouping, test="auto", alpha=alpha, correction=correction)
    significant = test_results.loc[test_results["significant"], "feature"].tolist() if not test_results.empty else []

    summary_lines = describe_group_tests(test_results, df_wide, grouping, alpha, feature_label="fatty acid")

    figures = {}
    if include_figures and significant:
        figures["box"] = viz.box_plot(df_wide, significant, grouping)
        figures["kde"] = viz.kde_density_plot(df_wide, significant, grouping)

    return PipelineResult(
        {"wide": df_wide, "normality": normality, "tests": test_results, "significant_compounds": significant, "figures": figures},
        title=f"mk_gcms.stats_pipeline ({technique}) — project {project_id}",
        summary_lines=summary_lines,
        figures=figures,
    )


def ml_pipeline(
    client, project_id: Any, technique: Literal["gc_ms", "gc_irms"] = "gc_ms", grouping: str = "sample_group",
    batch_col: str = "performed_at", is_column: str | None = None, zero_threshold: float | None = None,
    positive_class: Any | None = None, negative_class: Any | None = None, include_figures: bool = True,
) -> PipelineResult:
    """IS normalization (or median fallback) -> zero-threshold filtering
    (off by default) -> batch-wise LOD/2 imputation -> batch-wise median
    normalization -> RSD diagnostics -> RF classification with stratified
    CV, ROC/AUC, MDI importance + elbow. Direct reuse of the reference
    lcms pipeline (`s01prep.py` + `s03rforest.py`).

    Args:
        client: A connected `OrgClient`.
        project_id: The project's id.
        technique: "gc_ms" or "gc_irms".
        grouping: Which sample attribute defines the RF classes (e.g.
            sample_group/matrix_group/storage_condition).
        batch_col: Column identifying each run's batch — defaults to
            `performed_at` (no literal `batch` column exists); pass
            `"instrument"` once that field is populated for these records.
        is_column: A compound_id to use as internal standard, if one was
            spiked; if omitted, falls back to batch-median normalization only.
        zero_threshold: Drop compounds with more than this fraction of
            zero/NaN values. `None` (default) skips filtering — with a fixed
            ~10-20 compound panel there's no smaller informative subset to
            discover by dropping features.
        positive_class: RF positive class; defaults to the first two
            distinct `grouping` values if omitted.
        negative_class: RF negative class.
        include_figures: Set False to skip building/embedding any plots.

    Returns a `PipelineResult` — see `isotopes.stats_pipeline`'s docstring
    for what that means: one call already shows the full report by default.
    """
    df_wide, compounds = _wide(client, project_id, grouping, technique)
    return run_normalize_impute_rf(
        df_wide, compounds, grouping, batch_col=batch_col, is_column=is_column, zero_threshold=zero_threshold,
        positive_class=positive_class, negative_class=negative_class,
        title=f"mk_gcms.ml_pipeline ({technique}) — project {project_id}", feature_label="fatty acid",
        include_figures=include_figures,
    )
