from unittest.mock import patch

import pandas as pd

from o2org.analysis._coerce import as_dataframe
from o2org.client import OrgClient

from .helpers import FakeResponse


def test_numeric_looking_string_column_is_coerced_to_float():
    # e.g. Laravel decimal casts come back as JSON strings ("180.1600000000")
    records = [{"mz": "180.1600000000"}, {"mz": "150.0000000000"}]
    df = as_dataframe(records)
    assert pd.api.types.is_numeric_dtype(df["mz"])
    assert df["mz"].tolist() == [180.16, 150.0]


def test_numeric_string_column_with_missing_values_is_still_coerced():
    records = [{"mz": "180.16"}, {"mz": None}, {"mz": "150.0"}]
    df = as_dataframe(records)
    assert pd.api.types.is_numeric_dtype(df["mz"])
    assert df["mz"].isna().sum() == 1


def test_genuine_text_column_is_left_alone():
    records = [{"canonical_name": "D-Glucose"}, {"canonical_name": "Fructose"}]
    df = as_dataframe(records)
    assert not pd.api.types.is_numeric_dtype(df["canonical_name"])
    assert df["canonical_name"].tolist() == ["D-Glucose", "Fructose"]


def test_partially_numeric_column_is_not_coerced():
    # one row that doesn't parse as a number - too risky to guess, leave as text
    records = [{"code": "123"}, {"code": "abc"}]
    df = as_dataframe(records)
    assert not pd.api.types.is_numeric_dtype(df["code"])


def test_nested_dict_column_is_left_alone():
    records = [{"compound": {"id": 1, "name": "Glucose"}}, {"compound": None}]
    df = as_dataframe(records)
    assert df["compound"].dtype == object
    assert df["compound"].iloc[0] == {"id": 1, "name": "Glucose"}


def test_nested_list_column_is_left_alone():
    records = [{"synonyms": ["a", "b"]}, {"synonyms": []}]
    df = as_dataframe(records)
    assert df["synonyms"].dtype == object
    assert df["synonyms"].tolist() == [["a", "b"], []]


def test_all_null_column_is_not_coerced_to_numeric():
    # e.g. a "notes" field that happens to be empty on every retrieved row -
    # vacuously "every non-null value parses as a number" (there are none),
    # but treating it as numeric would sweep an all-NaN column into PCA/etc.
    records = [{"notes": None}, {"notes": None}]
    df = as_dataframe(records)
    assert not pd.api.types.is_numeric_dtype(df["notes"])


def test_already_a_dataframe_is_not_touched():
    # coercion only applies when building a new DataFrame from records - an
    # already-prepared DataFrame's dtypes are the caller's own choice
    df_in = pd.DataFrame({"mz": ["180.16", "150.0"]})
    df_out = as_dataframe(df_in)
    assert not pd.api.types.is_numeric_dtype(df_out["mz"])
    assert df_out is df_in


def test_unmaterialized_list_query_is_fetched_and_coerced():
    # a still-lazy .list() result must work exactly like the records it
    # would materialize to - this is the point where "I want the data" is
    # unambiguous, so fetching here (rather than raising) is correct
    client = OrgClient("https://dma.example.org", token="preset-token")
    payload = [{"mz": "180.1600000000"}, {"mz": "150.0000000000"}]
    with patch("requests.Session.request", return_value=FakeResponse(200, payload)):
        query = client.compounds.list()
        df = as_dataframe(query)
    assert pd.api.types.is_numeric_dtype(df["mz"])
    assert df["mz"].tolist() == [180.16, 150.0]
