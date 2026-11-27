from unittest.mock import patch

import numpy as np

from isofood_center.client import OrgClient
from isofood_center.pipelines import mk_gcms

from ..helpers import FakeResponse

SCHEMA = {
    "record_type": "result_mk_gc_ms",
    "fields": [
        {"key": "compound_id", "label": "Fatty acid", "type": "fatty_acid_select"},
        {"key": "unit", "label": "Unit", "type": "select"},
        {"key": "value", "label": "Value", "type": "number"},
    ],
}

FATTY_ACIDS = ["C16:0", "C18:0", "C18:1", "C18:2", "C20:0"]


def _authed_client():
    return OrgClient("https://dma.example.org", token="preset-token")


def _mk_records(n_per_group=15, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    record_id = 1
    for group, shift in (("food", 0.0), ("environment", 5.0)):
        for sample in range(n_per_group):
            sample_id = f"{group}-{sample}"
            batch = "batch1" if sample % 2 == 0 else "batch2"
            for fa in FATTY_ACIDS:
                records.append({
                    "id": record_id, "sample_id": sample_id, "sample_group": group,
                    "performed_at": f"2026-01-{(sample % 28) + 1:02d}T00:00:00Z", "instrument": batch,
                    "details": {"compound_id": fa, "unit": "%", "value": float(rng.normal(10 + shift, 1.0))},
                })
                record_id += 1
    return records


def _mock_responses(records):
    return [
        FakeResponse(200, SCHEMA),
        FakeResponse(200, {"data": records, "meta": {"current_page": 1, "last_page": 1, "total": len(records)}}),
    ]


def test_stats_pipeline_tests_each_fatty_acid():
    client = _authed_client()
    with patch("requests.Session.request", side_effect=_mock_responses(_mk_records())):
        result = mk_gcms.stats_pipeline(client, project_id=19)

    assert set(result["tests"]["feature"]) == set(FATTY_ACIDS)
    assert len(result["significant_compounds"]) > 0


def test_ml_pipeline_runs_full_normalize_impute_rf_chain():
    client = _authed_client()
    with patch("requests.Session.request", side_effect=_mock_responses(_mk_records())):
        result = mk_gcms.ml_pipeline(client, project_id=19, batch_col="performed_at")

    assert set(result["normalized"].columns) >= set(FATTY_ACIDS)
    assert result["rf"].y_proba.shape[0] == len(result["normalized"])
    assert set(result["figures"]) >= {"roc", "threshold_sensitivity", "mdi_elbow", "top_features"}


def test_ml_pipeline_skips_zero_threshold_filtering_by_default():
    client = _authed_client()
    with patch("requests.Session.request", side_effect=_mock_responses(_mk_records())):
        result = mk_gcms.ml_pipeline(client, project_id=19, batch_col="performed_at")
    # a fixed 5-compound panel should keep every compound with zero_threshold=None
    assert set(result["importances"]["feature"]) == set(FATTY_ACIDS)
