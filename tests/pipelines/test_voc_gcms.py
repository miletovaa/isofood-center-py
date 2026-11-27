from unittest.mock import patch

import numpy as np

from isofood_center.client import OrgClient
from isofood_center.pipelines import voc_gcms

from ..helpers import FakeResponse

SCHEMA = {
    "record_type": "result_voc_gc_ms",
    "fields": [
        {"key": "compound_id", "label": "Compound", "type": "compound_combobox"},
        {"key": "value", "label": "Value", "type": "number"},
        {"key": "unit", "label": "Unit", "type": "text"},
    ],
}

COMPOUNDS = [{"id": i, "canonical_name": f"Compound {i}"} for i in range(1, 9)]


def _authed_client():
    return OrgClient("https://dma.example.org", token="preset-token")


def _voc_records(n_per_group=15, seed=0, n_compounds=8):
    rng = np.random.default_rng(seed)
    records = []
    record_id = 1
    for group, shift in (("food", 0.0), ("environment", 6.0)):
        for sample in range(n_per_group):
            sample_id = f"{group}-{sample}"
            instrument = "gc-a" if sample % 2 == 0 else "gc-b"
            for compound_id in range(1, n_compounds + 1):
                records.append({
                    "id": record_id, "sample_id": sample_id, "sample_group": group,
                    "performed_at": "2026-01-01T00:00:00Z", "instrument": instrument,
                    "details": {"compound_id": compound_id, "value": float(rng.normal(100 + shift, 5.0)), "unit": "area"},
                })
                record_id += 1
    return records


def _mock_responses(records, n_resolve_calls=1):
    responses = [FakeResponse(200, SCHEMA), FakeResponse(200, {"data": records, "meta": {"current_page": 1, "last_page": 1, "total": len(records)}})]
    responses += [FakeResponse(200, COMPOUNDS)] * n_resolve_calls
    return responses


def test_stats_pipeline_resolves_compound_names_and_runs_sample_and_compound_pca():
    client = _authed_client()
    with patch("requests.Session.request", side_effect=_mock_responses(_voc_records())):
        result = voc_gcms.stats_pipeline(client, project_id=1)

    assert result["compound_names"][1] == "Compound 1"
    assert len(result["tests"]) == 8
    assert result["sample_pca"] is not None
    assert result["compound_pca"] is not None
    assert "pca_score" in result["figures"]
    assert "compound_pca_score" in result["figures"]


def test_ml_pipeline_uses_instrument_as_batch_by_default():
    client = _authed_client()
    with patch("requests.Session.request", side_effect=_mock_responses(_voc_records())):
        result = voc_gcms.ml_pipeline(client, project_id=1, zero_threshold=None)

    assert result["rf"].y_proba.shape[0] == len(result["normalized"])
    assert result["compound_names"][1] == "Compound 1"


def test_ml_pipeline_zero_threshold_drops_sparse_compounds():
    client = _authed_client()
    records = _voc_records(n_compounds=8)
    # blank out compound 8 for all but one sample -> should get filtered at threshold=0.2
    sparse_id = 8
    kept = 0
    filtered = []
    for r in records:
        if r["details"]["compound_id"] == sparse_id:
            if kept == 0:
                filtered.append(r)
                kept += 1
            # else: drop this record entirely (missing -> NaN after pivot)
        else:
            filtered.append(r)

    with patch("requests.Session.request", side_effect=_mock_responses(filtered)):
        result = voc_gcms.ml_pipeline(client, project_id=1, zero_threshold=0.2)

    assert 8 not in result["normalized"].columns
