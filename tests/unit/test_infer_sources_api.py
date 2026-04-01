from __future__ import annotations

from pathlib import Path

from author_name_disambiguation.infer_sources import InferSourcesRequest, InferSourcesResult, run_infer_sources


def test_run_infer_sources_returns_typed_result(monkeypatch, tmp_path: Path):
    captured = {}

    def _fake_run(request):
        captured["request"] = request
        return InferSourcesResult(
            run_id="infer_sources_test",
            go=True,
            output_root=tmp_path,
            publications_disambiguated_path=tmp_path / "publications_disambiguated.parquet",
            references_disambiguated_path=None,
            source_author_assignments_path=tmp_path / "source_author_assignments.parquet",
            author_entities_path=tmp_path / "author_entities.parquet",
            mention_clusters_path=tmp_path / "mention_clusters.parquet",
            stage_metrics_path=tmp_path / "05_stage_metrics_infer_sources.json",
            go_no_go_path=tmp_path / "05_go_no_go_infer_sources.json",
        )

    monkeypatch.setattr("author_name_disambiguation.infer_sources.run_source_inference", _fake_run)

    request = InferSourcesRequest(
        publications_path=tmp_path / "publications.parquet",
        output_root=tmp_path / "out",
        dataset_id="ads_prod_current",
        model_bundle=tmp_path / "bundle",
        uid_scope="dataset",
        uid_namespace="ads_prod_current",
        runtime_mode="cpu",
        specter_runtime_backend="onnx_fp32",
        progress=False,
    )
    result = run_infer_sources(request)

    assert result.run_id == "infer_sources_test"
    assert result.go is True
    assert result.output_root == tmp_path

    captured_request = captured["request"]
    assert captured_request.dataset_id == "ads_prod_current"
    assert captured_request.uid_scope == "dataset"
    assert captured_request.uid_namespace == "ads_prod_current"
    assert captured_request.runtime_mode == "cpu"
    assert captured_request.specter_runtime_backend == "onnx_fp32"
    assert captured_request.progress is False
