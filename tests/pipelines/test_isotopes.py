from unittest.mock import patch

import numpy as np

from isofood_center.client import OrgClient
from isofood_center.pipelines import isotopes

from ..helpers import FakeResponse

SCHEMA = {
    "record_type": "result_stable_isotopes",
    "fields": [
        {"key": "analyte", "label": "Analyte", "type": "select"},
        {"key": "value", "label": "Value", "type": "number"},
        {"key": "stdev", "label": "Stdev", "type": "number"},
        {"key": "unit", "label": "Unit", "type": "text"},
    ],
}


def _authed_client():
    return OrgClient("https://dma.example.org", token="preset-token")


def _isotope_records(n_per_group=15, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    record_id = 1
    for group, (c_mean, n_mean) in (("food", (-25.0, 8.0)), ("environment", (-20.0, 12.0))):
        for sample in range(n_per_group):
            sample_id = f"{group}-{sample}"
            for analyte, mean in (("d13c", c_mean), ("d15n", n_mean)):
                records.append({
                    "id": record_id, "sample_id": sample_id, "sample_group": group,
                    "details": {"analyte": analyte, "value": float(rng.normal(mean, 0.5)), "stdev": 0.1, "unit": "‰"},
                })
                record_id += 1
    return records


def _mock_responses(records):
    return [
        FakeResponse(200, SCHEMA),
        FakeResponse(200, {"data": records, "meta": {"current_page": 1, "last_page": 1, "total": len(records)}}),
    ]


def test_stats_pipeline_finds_significant_analyte_and_builds_biplot():
    client = _authed_client()
    with patch("requests.Session.request", side_effect=_mock_responses(_isotope_records())):
        result = isotopes.stats_pipeline(client, project_id=12, alpha=0.01)

    assert set(result["wide"]["sample_group"]) == {"food", "environment"}
    assert "d13c" in result["significant_analytes"]
    assert "d15n" in result["significant_analytes"]
    assert result["isotope_biplot"] is not None  # d13c + d15n present
    assert "box" in result["figures"]


def test_ml_pipeline_lda_separates_the_two_groups():
    client = _authed_client()
    with patch("requests.Session.request", side_effect=_mock_responses(_isotope_records())):
        result = isotopes.ml_pipeline(client, project_id=12, method="lda")

    assert result["method"] == "lda"
    accuracy = (result["result"].encoder.inverse_transform(result["result"].y_pred) == result["wide"]["sample_group"].values).mean()
    assert accuracy > 0.8


def test_ml_pipeline_random_forest_reports_low_value_note():
    client = _authed_client()
    with patch("requests.Session.request", side_effect=_mock_responses(_isotope_records())):
        result = isotopes.ml_pipeline(client, project_id=12, method="random_forest")

    assert result["method"] == "random_forest"
    assert "elbow" in result["note"].lower()
    assert set(result["importances"]["feature"]) == {"d13c", "d15n"}


def test_ml_pipeline_rejects_unknown_method():
    client = _authed_client()
    with patch("requests.Session.request", side_effect=_mock_responses(_isotope_records())):
        try:
            isotopes.ml_pipeline(client, project_id=12, method="bogus")
            assert False, "expected ValueError"
        except ValueError:
            pass
