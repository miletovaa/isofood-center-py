from unittest.mock import patch

import numpy as np

from isofood_center.client import OrgClient
from isofood_center.pipelines import elemental

from ..helpers import FakeResponse

SCHEMA = {
    "record_type": "result_elemental_composition",
    "fields": [
        {"key": "element", "label": "Element", "type": "select"},
        {"key": "value", "label": "Value", "type": "number"},
        {"key": "stdev", "label": "Stdev", "type": "number"},
        {"key": "unit", "label": "Unit", "type": "text"},
    ],
}


def _authed_client():
    return OrgClient("https://dma.example.org", token="preset-token")


def _elemental_records(n_per_group=15, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    record_id = 1
    element_means = {"C": 50.0, "H": 6.0, "N": 4.0, "O": 40.0}
    for group in ("environment", "food"):
        for sample in range(n_per_group):
            sample_id = f"{group}-{sample}"
            for element, mean in element_means.items():
                records.append({
                    "id": record_id, "sample_id": sample_id, "sample_group": group,
                    "details": {"element": element, "value": float(rng.normal(mean, 1.0)), "stdev": 0.2, "unit": "%"},
                })
                record_id += 1
    return records


def _mock_responses(records):
    return [
        FakeResponse(200, SCHEMA),
        FakeResponse(200, {"data": records, "meta": {"current_page": 1, "last_page": 1, "total": len(records)}}),
    ]


def test_stats_pipeline_computes_molar_ratios_and_van_krevelen():
    client = _authed_client()
    with patch("requests.Session.request", side_effect=_mock_responses(_elemental_records())):
        result = elemental.stats_pipeline(client, project_id=19)

    assert {"h_c", "o_c", "n_c"}.issubset(result["wide"].columns)
    assert result["van_krevelen"] is not None


def test_ml_pipeline_kmeans_default_clusters_ratio_space():
    client = _authed_client()
    with patch("requests.Session.request", side_effect=_mock_responses(_elemental_records())):
        result = elemental.ml_pipeline(client, project_id=19, k=2)

    assert result["method"] == "kmeans"
    assert len(result["result"].labels) == len(result["wide"])


def test_ml_pipeline_random_forest_skips_elbow_by_default():
    client = _authed_client()
    with patch("requests.Session.request", side_effect=_mock_responses(_elemental_records())):
        result = elemental.ml_pipeline(client, project_id=19, method="random_forest")

    assert result["method"] == "random_forest"
    assert "elbow" in result["note"].lower()
    assert "elbow_cutoff" not in result
