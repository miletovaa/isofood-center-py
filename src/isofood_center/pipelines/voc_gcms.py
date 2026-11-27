"""Family D: VOC GC-MS / VOC GC-IRMS (`result_voc_gc_ms` / `result_voc_gc_irms`).

Open-ended compound identification (combobox, untargeted-style) — the true
analog of both reference repos combined: peptide_analysis's per-compound
statistics chain, at the lcms repo's thousands-of-features RF-discovery scale.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from . import ml, preprocessing, stats, viz
from ._gcms_common import run_normalize_impute_rf
from ._report import PipelineResult, describe_group_tests
from .utils.extract import extract, pivot_wide, resolve_compound_names

RECORD_TYPES = {"gc_ms": "result_voc_gc_ms", "gc_irms": "result_voc_gc_irms"}


def _compound_names(client, df_long: pd.DataFrame) -> dict[Any, str]:
    resolved = resolve_compound_names(client, df_long)
    return (
        resolved.drop_duplicates("details_compound_id")
        .set_index("details_compound_id")["compound_name"]
        .to_dict()
    )


def _wide(client, project_id, grouping, technique, extra_index_cols: list[str] = ()):
    record_type = RECORD_TYPES[technique]
    df_long = extract(client, record_type, project_id)
    names = _compound_names(client, df_long)

    index_cols = ["sample_id", grouping, *extra_index_cols]
    index_cols = [c for c in dict.fromkeys(index_cols) if c in df_long.columns]
    df_wide = pivot_wide(df_long, index_cols=index_cols, analyte_col="details_compound_id", value_col="details_value")
    compounds = [c for c in df_wide.columns if c not in index_cols]
    return df_wide, compounds, names


def _row_pareto_scale(matrix: pd.DataFrame) -> pd.DataFrame:
    """Pareto-scale each row using stats computed across its own columns —
    the compound-level PCA needs each compound normalized across samples,
    the opposite axis from the standard (sample-level) `preprocessing.pareto_scale`.
    """
    row_means = matrix.mean(axis=1)
    row_std = matrix.std(axis=1, ddof=1).replace(0, np.nan)
    scaled = matrix.sub(row_means, axis=0).div(np.sqrt(row_std), axis=0)
    return scaled.fillna(0)


def stats_pipeline(
    client, project_id: Any, technique: Literal["gc_ms", "gc_irms"] = "gc_ms", grouping: str = "sample_group",
    zero_threshold: float = 0.6, alpha: float = 0.01, correction: Literal["bonferroni", "fdr_bh", None] = "fdr_bh",
    include_figures: bool = True,
) -> PipelineResult:
    """Essentially peptide_analysis.py unmodified in structure: row-threshold
    filter (here, column filter — our pivot is samples x compounds, not
    peptide_analysis's peptides x samples) -> LOQ/2 impute -> Shapiro on raw
    and log10-transformed data -> Kruskal-Wallis/ANOVA -> Bonferroni+BH ->
    box/median-bar/KDE plots -> sample-level and compound-level PCA.

    Returns a `PipelineResult` — see `isotopes.stats_pipeline`'s docstring
    for what that means: one call already shows the full report by default.

    Args:
        client: A connected `OrgClient`.
        project_id: The project's id.
        technique: "gc_ms" (result_voc_gc_ms) or "gc_irms" (result_voc_gc_irms).
        grouping: Which sample attribute to compare groups on.
        zero_threshold: Drop compounds missing in more than this fraction of
            samples (peptide_analysis's default is 0.6).
        alpha: Significance threshold.
        correction: Multiple-testing correction across compounds.
        include_figures: Set False to skip building/embedding any plots.

    Returns:
        `wide`, `compound_names` (id -> canonical_name), `imputed_counts`,
        `normality_raw`, `normality_log`, `tests`, `significant_compounds`,
        `figures` (box/median_bar/kde + sample-level and compound-level PCA).
    """
    df_wide, compounds, names = _wide(client, project_id, grouping, technique)
    dropped = compounds

    df_wide = preprocessing.zero_nan_threshold(df_wide, compounds, threshold=zero_threshold, axis="columns")
    compounds = [c for c in compounds if c in df_wide.columns]
    dropped = [c for c in dropped if c not in compounds]

    df_wide, imputed_counts = preprocessing.lod_half_impute(df_wide, compounds, batch_col=None)

    normality_raw = pd.concat(
        [stats.shapiro_by_group(df_wide, c, grouping, alpha=alpha).assign(compound_id=c) for c in compounds],
        ignore_index=True,
    ) if compounds else pd.DataFrame()

    df_log = preprocessing.log10_transform(df_wide, compounds)
    normality_log = pd.concat(
        [stats.shapiro_by_group(df_log, c, grouping, alpha=alpha).assign(compound_id=c) for c in compounds],
        ignore_index=True,
    ) if compounds else pd.DataFrame()

    test_results = stats.test_all_features(df_wide, compounds, grouping, test="auto", alpha=alpha, correction=correction)
    significant = test_results.loc[test_results["significant"], "feature"].tolist() if not test_results.empty else []
    titles = [str(names.get(c, c)) for c in significant]

    summary_lines = describe_group_tests(test_results, df_wide, grouping, alpha, feature_label="compound", label_map=names)
    if dropped:
        summary_lines.insert(1, f"{len(dropped)} compound(s) dropped (missing in >{zero_threshold:.0%} of samples): {', '.join(str(names.get(c, c)) for c in dropped[:10])}" + (", ..." if len(dropped) > 10 else ""))

    figures = {}
    sample_pca = compound_pca = None
    if include_figures:
        if significant:
            figures["box"] = viz.box_plot(df_wide, significant, grouping, titles=titles)
            figures["median_bar"] = viz.median_bar_plot(df_wide, significant, grouping, titles=titles)
            figures["kde"] = viz.kde_density_plot(df_wide, significant, grouping, titles=titles)

        if compounds:
            sample_pca = ml.pca(df_log, columns=compounds, scale="pareto")
            figures["pca_score"] = viz.pca_score_scatter(sample_pca, group=df_wide[grouping])
            figures["pca_loadings"] = viz.pca_loadings_plot(sample_pca)
            figures["pca_biplot"] = viz.pca_biplot(sample_pca, group=df_wide[grouping])
            summary_lines.append(f"Sample-level PCA over {len(compounds)} compounds: PC1+PC2 explain {sample_pca.explained_variance_ratio[:2].sum() * 100:.1f}% of variance.")

            compound_matrix = df_log.set_index("sample_id")[compounds].T
            compound_matrix_scaled = _row_pareto_scale(compound_matrix)
            compound_pca = ml.pca(compound_matrix_scaled, scale=None)
            figures["compound_pca_score"] = viz.pca_score_scatter(compound_pca)

    return PipelineResult(
        {
            "wide": df_wide,
            "compound_names": names,
            "imputed_counts": imputed_counts,
            "normality_raw": normality_raw,
            "normality_log": normality_log,
            "tests": test_results,
            "significant_compounds": significant,
            "figures": figures,
            "sample_pca": sample_pca,
            "compound_pca": compound_pca,
        },
        title=f"voc_gcms.stats_pipeline ({technique}) — project {project_id}",
        summary_lines=summary_lines,
        figures=figures,
    )


def ml_pipeline(
    client, project_id: Any, technique: Literal["gc_ms", "gc_irms"] = "gc_ms", grouping: str = "sample_group",
    batch_col: str = "instrument", is_column: str | None = None, zero_threshold: float | None = 0.2,
    positive_class: Any | None = None, negative_class: Any | None = None, include_figures: bool = True,
) -> PipelineResult:
    """Essentially the reference lcms pipeline (`s01prep.py` + `s03rforest.py`)
    unmodified in structure — the pipeline where full RF + elbow-cutoff
    feature selection earns its keep, since VOC panels can run into hundreds
    of unique compounds. `batch_col` defaults to "instrument" (falling back
    to "performed_at" for rows where it's still null, since existing data
    predates that column).

    Args:
        client: A connected `OrgClient`.
        project_id: The project's id.
        technique: "gc_ms" or "gc_irms".
        grouping: Which sample attribute defines the RF classes.
        batch_col: Column identifying each run's batch.
        is_column: A compound_id to use as internal standard, if spiked.
        zero_threshold: Drop compounds with more than this fraction of
            zero/NaN values (default 0.2, matching the reference lcms pipeline
            — VOC panels benefit from filtering unlike the fixed MK panel).
        positive_class: RF positive class; defaults to the first two
            distinct `grouping` values if omitted.
        negative_class: RF negative class.
        include_figures: Set False to skip building/embedding any plots.

    Returns a `PipelineResult` — see `isotopes.stats_pipeline`'s docstring
    for what that means: one call already shows the full report by default.
    """
    df_wide, compounds, names = _wide(client, project_id, grouping, technique, extra_index_cols=["performed_at", "instrument"])

    if batch_col == "instrument" and "instrument" in df_wide.columns and "performed_at" in df_wide.columns:
        df_wide = df_wide.assign(_batch=df_wide["instrument"].fillna(df_wide["performed_at"]))
        batch_col = "_batch"

    result = run_normalize_impute_rf(
        df_wide, compounds, grouping, batch_col=batch_col, is_column=is_column, zero_threshold=zero_threshold,
        positive_class=positive_class, negative_class=negative_class,
        title=f"voc_gcms.ml_pipeline ({technique}) — project {project_id}", feature_label="compound",
        include_figures=include_figures,
    )
    result["compound_names"] = names
    return result
