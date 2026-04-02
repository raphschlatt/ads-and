from __future__ import annotations

import sys
from pathlib import Path

from author_name_disambiguation.api import disambiguate_sources, evaluate_lspo_quality, train_lspo_model
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


def test_run_infer_sources_accepts_keyword_arguments(monkeypatch, tmp_path: Path):
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
            summary_path=tmp_path / "summary.json",
        )

    monkeypatch.setattr("author_name_disambiguation.infer_sources.run_source_inference", _fake_run)
    monkeypatch.setattr(
        "author_name_disambiguation.infer_sources.resolve_fixed_model_bundle_path",
        lambda: tmp_path / "packaged_bundle",
    )

    result = run_infer_sources(
        publications_path=tmp_path / "publications.parquet",
        output_root=tmp_path / "out",
        dataset_id="ads_prod_current",
        infer_stage="incremental",
        progress=False,
    )

    assert result.summary_path == tmp_path / "summary.json"
    captured_request = captured["request"]
    assert captured_request.model_bundle == tmp_path / "packaged_bundle"
    assert captured_request.infer_stage == "incremental"
    assert captured_request.progress is False
    assert captured_request.runtime_mode in {"cpu", "gpu"}


def test_disambiguate_sources_uses_auto_runtime_and_packaged_bundle(monkeypatch, tmp_path: Path):
    captured = {}

    def _fake_run(request=None, **kwargs):
        captured["kwargs"] = kwargs
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
            summary_path=tmp_path / "summary.json",
        )

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    created_ui = {}

    class _FakeUi:
        def __init__(self, total_steps, progress, progress_style):
            created_ui["args"] = {
                "total_steps": total_steps,
                "progress": progress,
                "progress_style": progress_style,
            }

        def close(self):
            created_ui["closed"] = True

    monkeypatch.setattr("author_name_disambiguation.api.run_infer_sources", _fake_run)
    monkeypatch.setattr("author_name_disambiguation.api.resolve_fixed_model_bundle_path", lambda: tmp_path / "bundle")
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setattr("author_name_disambiguation.api.CliUI", _FakeUi)
    monkeypatch.setattr("author_name_disambiguation.api.get_active_ui", lambda: None)

    result = disambiguate_sources(
        publications_path=tmp_path / "publications.parquet",
        output_dir=tmp_path / "out",
        progress=False,
        progress_style="verbose",
    )

    assert result.summary_path == tmp_path / "summary.json"
    kwargs = captured["kwargs"]
    assert kwargs["dataset_id"] == "out"
    assert kwargs["runtime_mode"] == "cpu"
    assert kwargs["model_bundle"] == tmp_path / "bundle"
    assert created_ui["args"]["progress_style"] == "verbose"
    assert created_ui["closed"] is True


def test_evaluate_lspo_quality_defaults_to_fixed_model_baseline(monkeypatch, tmp_path: Path):
    captured = {}

    class _FakeCli:
        @staticmethod
        def cmd_run_cluster_test_report(args):
            captured["args"] = args

    monkeypatch.setattr("author_name_disambiguation.api._load_cli_module", lambda: _FakeCli)
    result = evaluate_lspo_quality(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        raw_lspo_parquet=tmp_path / "LSPO_v1.parquet",
        progress=False,
        progress_style="verbose",
    )

    assert result.model_run_id == "full_20260218T111506Z_cli02681429"
    args = captured["args"]
    assert args.allow_legacy_lspo_compat is True
    assert args.progress is False
    assert args.progress_style == "verbose"


def test_train_lspo_model_uses_simple_defaults(monkeypatch, tmp_path: Path):
    captured = {}

    class _FakeCli:
        @staticmethod
        def cmd_run_train_stage(args):
            captured["args"] = args

    monkeypatch.setattr("author_name_disambiguation.api._load_cli_module", lambda: _FakeCli)
    monkeypatch.setattr("author_name_disambiguation.api.default_train_run_id", lambda stage: f"{stage}_auto_id")

    result = train_lspo_model(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        raw_lspo_parquet=tmp_path / "LSPO_v1.parquet",
        progress=False,
        progress_style="verbose",
    )

    assert result.run_id == "full_auto_id"
    args = captured["args"]
    assert args.run_stage == "full"
    assert args.progress is False
    assert args.progress_style == "verbose"


def test_disambiguate_sources_reuses_active_ui_without_creating_new_one(monkeypatch, tmp_path: Path):
    existing_ui = object()
    created = {"called": False}

    def _fake_run(request=None, **kwargs):
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
            summary_path=tmp_path / "summary.json",
        )

    class _FakeUi:
        def __init__(self, *args, **kwargs):
            created["called"] = True

        def close(self):
            return None

    monkeypatch.setattr("author_name_disambiguation.api.run_infer_sources", _fake_run)
    monkeypatch.setattr("author_name_disambiguation.api.resolve_fixed_model_bundle_path", lambda: tmp_path / "bundle")
    monkeypatch.setattr("author_name_disambiguation.api.CliUI", _FakeUi)
    monkeypatch.setattr("author_name_disambiguation.api.get_active_ui", lambda: existing_ui)

    disambiguate_sources(
        publications_path=tmp_path / "publications.parquet",
        output_dir=tmp_path / "out",
        progress=True,
    )

    assert created["called"] is False
