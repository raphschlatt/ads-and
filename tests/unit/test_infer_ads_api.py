from __future__ import annotations

from pathlib import Path

import pytest

from src import cli
from src.infer_ads_api import InferAdsRequest, run_infer_ads


def test_run_infer_ads_requires_exactly_one_model_source():
    with pytest.raises(ValueError, match="exactly one"):
        run_infer_ads(InferAdsRequest(dataset_id="ads_2026"))

    with pytest.raises(ValueError, match="exactly one"):
        run_infer_ads(
            InferAdsRequest(
                dataset_id="ads_2026",
                model_run_id="full_2026",
                model_bundle="artifacts/models/full_2026/bundle_v1",
            )
        )


def test_run_infer_ads_returns_typed_result(monkeypatch):
    captured = {}

    def _fake_impl(args):
        captured["args"] = args
        return {
            "run_id": "infer_ads_api_test",
            "go": True,
            "metrics_dir": "/tmp/metrics",
            "clusters_path": "/tmp/clusters.parquet",
            "publication_authors_path": "/tmp/publication_authors.parquet",
            "publications_disambiguated_path": "/tmp/publications.disambiguated.jsonl",
            "references_disambiguated_path": None,
            "stage_metrics_path": "/tmp/05_stage_metrics_infer_ads.json",
            "go_no_go_path": "/tmp/05_go_no_go_infer_ads.json",
        }

    monkeypatch.setattr(cli, "_run_infer_ads_impl", _fake_impl)

    request = InferAdsRequest(
        dataset_id="ads_prod_current",
        model_run_id="full_2026",
        uid_scope="dataset",
        uid_namespace="ads_prod_current",
        progress=False,
    )
    result = run_infer_ads(request)

    assert result.run_id == "infer_ads_api_test"
    assert result.go is True
    assert result.metrics_dir == Path("/tmp/metrics")
    assert result.clusters_path == Path("/tmp/clusters.parquet")

    args = captured["args"]
    assert args.dataset_id == "ads_prod_current"
    assert args.model_run_id == "full_2026"
    assert args.model_bundle is None
    assert args.uid_scope == "dataset"
    assert args.uid_namespace == "ads_prod_current"
    assert args.progress is False
