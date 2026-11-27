from unittest.mock import patch

from isofood_center.client import OrgClient
from isofood_center.pipelines.utils.extract import extract, flatten_details, pivot_wide, resolve_compound_names

from ..helpers import FakeResponse

ISOTOPE_SCHEMA = [
    {"key": "analyte", "label": "Analyte", "type": "select"},
    {"key": "value", "label": "Value", "type": "number"},
    {"key": "stdev", "label": "Stdev", "type": "number"},
    {"key": "unit", "label": "Unit", "type": "text"},
]


def _authed_client():
    return OrgClient("https://dma.example.org", token="preset-token")


def test_flatten_details_expands_details_into_columns():
    records = [
        {"id": 1, "sample_id": 6, "details": {"analyte": "d13c", "value": "-24.8", "stdev": "0.09"}},
        {"id": 2, "sample_id": 6, "details": {"analyte": "d18o", "value": "26.1", "stdev": "0.15"}},
    ]
    df = flatten_details(records, schema=ISOTOPE_SCHEMA)
    assert list(df["details_analyte"]) == ["d13c", "d18o"]
    # numeric fields coerced per schema, not left as Laravel's JSON strings
    assert df["details_value"].dtype.kind == "f"
    assert df["details_value"].tolist() == [-24.8, 26.1]


def test_flatten_details_without_schema_leaves_values_untouched():
    records = [{"id": 1, "details": {"value": "1.5"}}]
    df = flatten_details(records, schema=None)
    assert df["details_value"].iloc[0] == "1.5"  # no schema -> no coercion


def test_extract_fetches_schema_then_records_by_project():
    client = _authed_client()
    schema_response = FakeResponse(200, {"record_type": "result_stable_isotopes", "fields": ISOTOPE_SCHEMA})
    records_response = FakeResponse(200, {
        "data": [{"id": 10, "sample_id": 6, "sample_group": "food", "details": {"analyte": "d13c", "value": "-24.8"}}],
        "meta": {"current_page": 1, "last_page": 1, "total": 1},
    })
    with patch("requests.Session.request", side_effect=[schema_response, records_response]) as mock_request:
        df = extract(client, "result_stable_isotopes", project_id=12)

    urls = [call.args[1] for call in mock_request.call_args_list]
    assert urls == [
        "https://dma.example.org/api/record-types/result_stable_isotopes/schema",
        "https://dma.example.org/api/projects/12/experiment-records",
    ]
    assert df["details_value"].iloc[0] == -24.8


def test_extract_reuses_schema_when_given_explicitly():
    client = _authed_client()
    records_response = FakeResponse(200, [{"id": 10, "sample_id": 6, "details": {"value": "1.0"}}])
    with patch("requests.Session.request", return_value=records_response) as mock_request:
        extract(client, "result_stable_isotopes", project_id=12, schema=ISOTOPE_SCHEMA)
    assert mock_request.call_count == 1  # no schema request made


def test_pivot_wide_keeps_rows_when_an_index_column_is_all_nan():
    # regression test: pivot_table's default groupby drops any row whose
    # index column is NaN (e.g. `instrument`, unpopulated on older records) —
    # pivot_wide must not silently lose samples because of that.
    import pandas as pd

    df_long = pd.DataFrame({
        "sample_id": [1, 1],
        "sample_group": ["food", "food"],
        "instrument": [None, None],
        "details_analyte": ["d13c", "d18o"],
        "details_value": [1.0, 2.0],
    })
    wide = pivot_wide(df_long, index_cols=["sample_id", "sample_group", "instrument"], analyte_col="details_analyte")
    assert len(wide) == 1
    assert wide.loc[0, "d13c"] == 1.0
    assert wide.loc[0, "d18o"] == 2.0


def test_pivot_wide_one_row_per_sample():
    import pandas as pd

    df_long = pd.DataFrame({
        "sample_id": [1, 1, 2, 2],
        "sample_group": ["food", "food", "environment", "environment"],
        "details_analyte": ["d13c", "d18o", "d13c", "d18o"],
        "details_value": [1.0, 2.0, 3.0, 4.0],
    })
    wide = pivot_wide(df_long, index_cols=["sample_id", "sample_group"], analyte_col="details_analyte")
    assert sorted(wide.columns) == ["d13c", "d18o", "sample_group", "sample_id"]
    assert len(wide) == 2


def test_resolve_compound_names_maps_id_to_canonical_name():
    import pandas as pd

    client = _authed_client()
    compounds_response = FakeResponse(200, [{"id": 5, "canonical_name": "Limonene"}, {"id": 9, "canonical_name": "Linalool"}])
    df = pd.DataFrame({"details_compound_id": [5, 9, 5]})
    with patch("requests.Session.request", return_value=compounds_response):
        result = resolve_compound_names(client, df)
    assert result["compound_name"].tolist() == ["Limonene", "Linalool", "Limonene"]
