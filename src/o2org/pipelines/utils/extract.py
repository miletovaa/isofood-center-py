"""Shared extraction & pivot layer, used by every pipeline in `o2org.pipelines`.

org_dma's analysis data starts as EAV rows in `experiment_records.details`
(JSON), not a CSV/Excel file. Every pipeline needs the same first step before
any statistics run:

    extract() -> one row per experiment_record, `details` flattened into
                 `details_<key>` columns, sample_group/sample_subgroup/
                 matrix_group/storage_condition/project_id already joined in
                 server-side (see ExperimentRecords.list_by_project).
    pivot_wide() -> long (one row per analyte/compound) to wide (one row per
                    sample, one column per analyte/compound) — the EAV
                    equivalent of a CSV-orientation transpose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ...client import OrgClient


def extract(
    client: "OrgClient",
    record_type: str,
    project_id: Any,
    schema: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Long-format extraction for one record_type within one project.

    Args:
        client: A connected `OrgClient`.
        record_type: e.g. "result_stable_isotopes".
        project_id: The project's id.
        schema: The EAV field schema for `record_type` (as returned by
            `client.record_type_schema()`). Fetched automatically if omitted
            — pass it explicitly to avoid a repeat request across many calls.

    Returns:
        One row per experiment_record, `details` flattened into
        `details_<key>` columns (numeric fields coerced per the schema),
        plus the record's own fields and the joined sample_group/
        sample_subgroup/matrix_group/storage_condition/project_id.
    """
    if schema is None:
        schema = client.record_type_schema(record_type)["fields"]
    records = client.experiment_records.list_by_project(project_id, record_type=record_type).get()
    return flatten_details(records, schema=schema)


def flatten_details(records: list[dict[str, Any]], schema: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    """Blow each record's `details` dict into `details_<key>` columns.

    Args:
        records: Experiment records as returned by the API (each with a
            `details` dict).
        schema: The EAV field schema for these records' record_type — fields
            with `type == "number"` are coerced with `pd.to_numeric` instead
            of guessed at from the values (Laravel serializes decimal-cast
            columns as JSON strings, e.g. "180.16").

    Returns:
        A DataFrame with one row per record.
    """
    rows = []
    for record in records:
        details = record.get("details") or {}
        row = {k: v for k, v in record.items() if k != "details"}
        for key, value in details.items():
            row[f"details_{key}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    if schema:
        for field in schema:
            if field.get("type") != "number":
                continue
            column = f"details_{field['key']}"
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def pivot_wide(
    df_long: pd.DataFrame,
    index_cols: list[str],
    analyte_col: str,
    value_col: str = "details_value",
    agg: str = "mean",
) -> pd.DataFrame:
    """Long (one row per analyte/compound) -> wide (one row per sample).

    Args:
        df_long: A DataFrame as returned by `extract()`.
        index_cols: Columns that identify a sample and stay constant across
            its analyte rows, e.g. `["sample_id", "sample_group"]`.
        analyte_col: Column whose distinct values become the new columns,
            e.g. "details_analyte", "details_element", "details_compound_id".
        value_col: Column holding the measurement.
        agg: Aggregation used if a sample has more than one row for the same
            analyte (e.g. replicate measurements).

    Returns:
        One row per unique `index_cols` combination, one column per distinct
        `analyte_col` value.
    """
    # groupby(...).unstack(...) rather than pivot_table: pivot_table builds
    # its row MultiIndex from the cartesian product of each index column's
    # unique values, so any index column that's NaN for every row (e.g.
    # `instrument`, often unpopulated on older records) makes the *entire*
    # result vanish, and dropna=False on pivot_table only trades that away
    # for the opposite bug — spurious all-NaN rows for index combinations
    # that never actually co-occurred. groupby(dropna=False) only ever
    # groups combinations observed in the data, NaN key or not, and
    # unstack() only introduces NaN for a genuinely missing (group, analyte)
    # pair — never for outer-index combinations that never existed.
    grouped = df_long.groupby([*index_cols, analyte_col], dropna=False)[value_col].agg(agg)
    wide = grouped.unstack(analyte_col).reset_index()
    wide.columns.name = None
    return wide


def resolve_compound_names(
    client: "OrgClient", df: pd.DataFrame, id_col: str = "details_compound_id", name_col: str = "compound_name"
) -> pd.DataFrame:
    """Resolve VOC `compound_id` (inside `details`, not a real Eloquent FK)
    to `compounds.canonical_name`, client-side.

    Args:
        client: A connected `OrgClient`.
        df: A DataFrame with an `id_col` column of compound ids.
        id_col: Column holding the compound id.
        name_col: Name for the new column of resolved compound names.

    Returns:
        `df` with `name_col` added.
    """
    compounds = client.compounds.list(all_pages=True).get()
    id_to_name = {c["id"]: c.get("canonical_name") for c in compounds}
    result = df.copy()
    result[name_col] = result[id_col].map(id_to_name)
    return result
