from __future__ import annotations

import json
from pathlib import Path

import author_name_disambiguation
from author_name_disambiguation import public_cli
from author_name_disambiguation._modal_backend import ModalCostResult
from author_name_disambiguation.infer_sources import InferSourcesResult


def test_public_package_root_exports_inference_only():
    exported = set(author_name_disambiguation.__all__)
    assert "disambiguate_sources" in exported
    assert "resolve_modal_cost" in exported
    assert "run_infer_sources" in exported
    assert "InferSourcesRequest" in exported
    assert "ModalCostResult" in exported
    assert "evaluate_lspo_quality" not in exported
    assert "train_lspo_model" not in exported
    assert "precompute_source_embeddings" not in exported


def test_public_cli_exposes_only_infer_command():
    parser = public_cli.build_parser()
    commands = set(parser._subparsers._group_actions[0].choices.keys())
    assert commands == {"infer", "cost"}


def test_public_cli_infer_parser_defaults():
    parser = public_cli.build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--publications-path",
            "publications.parquet",
            "--output-dir",
            "out",
        ]
    )

    assert args.command == "infer"
    assert args.references_path is None
    assert args.backend == "local"
    assert args.modal_gpu is None
    assert args.dataset_id is None
    assert args.runtime == "auto"
    assert args.infer_stage == "full"
    assert args.progress is True
    assert args.progress_style == "compact"
    assert args.json_output is False


def test_public_cli_cost_parser_defaults():
    parser = public_cli.build_parser()
    args = parser.parse_args(["cost", "--output-dir", "out"])

    assert args.command == "cost"
    assert args.output_dir == "out"
    assert args.json_output is False


def test_public_cli_infer_emits_summary(monkeypatch, tmp_path: Path, capsys):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "go": True,
                "runtime_mode": "cpu",
                "resolved_device": "cpu",
                "runtime_backend": "cpu_auto_transformers",
                "clustering_backend": "sklearn_cpu",
                "counts": {
                    "publications": 2,
                    "references": 1,
                    "mentions": 3,
                    "clusters": 2,
                    "authors_mapped": 3,
                    "authors_total": 3,
                },
                "stage_seconds": {
                    "load_inputs": 0.1,
                    "preflight": 0.1,
                    "name_embeddings": 0.1,
                    "text_embeddings": 0.2,
                    "pair_inference": 0.3,
                    "clustering": 0.1,
                    "export": 0.1,
                    "total": 1.0,
                },
                "output_root": str(tmp_path),
                "outputs": {
                    "publications_disambiguated_path": str(tmp_path / "publications_disambiguated.parquet"),
                    "references_disambiguated_path": None,
                    "source_author_assignments_path": str(tmp_path / "source_author_assignments.parquet"),
                },
                "summary_path": str(summary_path),
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    def _fake_disambiguate_sources(**_kwargs):
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
            summary_path=summary_path,
        )

    monkeypatch.setattr("author_name_disambiguation.public_cli.disambiguate_sources", _fake_disambiguate_sources)

    public_cli.main(
        [
            "infer",
            "--publications-path",
            "publications.parquet",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out
    assert "ADS inference complete" in output
    assert "mode=cpu" in output


def test_public_cli_infer_passes_modal_gpu(monkeypatch, tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"summary_path": str(summary_path), "output_root": str(tmp_path)}), encoding="utf-8")
    captured = {}

    def _fake_disambiguate_sources(**kwargs):
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
            summary_path=summary_path,
        )

    monkeypatch.setattr("author_name_disambiguation.public_cli.disambiguate_sources", _fake_disambiguate_sources)

    public_cli.main(
        [
            "infer",
            "--publications-path",
            "publications.parquet",
            "--output-dir",
            str(tmp_path),
            "--backend",
            "modal",
            "--modal-gpu",
            "l4",
            "--json",
        ]
    )

    assert captured["kwargs"]["modal_gpu"] == "l4"


def test_public_cli_infer_summary_includes_modal_block(tmp_path: Path):
    summary = {
        "go": True,
        "runtime_mode": "gpu",
        "resolved_device": "cuda:0",
        "runtime_backend": "transformers",
        "clustering_backend": "sklearn_cpu",
        "counts": {"publications": 10, "references": 5, "mentions": 15, "clusters": 5, "authors_mapped": 5, "authors_total": 5},
        "stage_seconds": {"total": 12.3},
        "output_root": str(tmp_path),
        "outputs": {
            "publications_disambiguated_path": str(tmp_path / "pubs.parquet"),
            "references_disambiguated_path": None,
            "source_author_assignments_path": str(tmp_path / "assignments.parquet"),
        },
        "summary_path": str(tmp_path / "summary.json"),
        "modal": {
            "app_id": "ap-uM9ApHoH8Be0F6Ydvofw7A",
            "app_name": "ads-and-modal",
            "gpu_type": "L4",
            "mode": "ephemeral_app_run",
            "billing_resolution": "h",
            "query_start_utc": "2026-04-13T14:00:00Z",
            "query_end_exclusive_utc": "2026-04-13T15:00:00Z",
            "exact_cost_available_after_utc": "2026-04-13T15:10:00Z",
            "publications_staging_bytes": 4300,
            "references_staging_bytes": 4100,
        },
    }
    rendered = public_cli._build_infer_human_summary(summary)
    assert "Modal: app=ads-and-modal (ephemeral_app_run) | gpu=L4" in rendered
    assert "app_id=ap-uM9ApHoH8Be0F6Ydvofw7A" in rendered
    assert "staging=pubs " in rendered
    assert "refs " in rendered
    assert "billing window=2026-04-13T14:00:00Z → 2026-04-13T15:00:00Z (resolution h)" in rendered
    assert "exact cost available after 2026-04-13T15:10:00Z" in rendered
    assert "ads-and cost --output-dir" in rendered


def test_public_cli_infer_parser_accepts_modal_gpu():
    parser = public_cli.build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--publications-path",
            "publications.parquet",
            "--output-dir",
            "out",
            "--backend",
            "modal",
            "--modal-gpu",
            "l4",
        ]
    )

    assert args.backend == "modal"
    assert args.modal_gpu == "l4"


def test_public_cli_cost_emits_summary(monkeypatch, tmp_path: Path, capsys):
    cost_report_path = tmp_path / "modal_cost_report.json"

    def _fake_resolve_modal_cost(**_kwargs):
        return ModalCostResult(
            status="complete",
            app_id="ap-test",
            exact_cost_available_after_utc="2026-04-13T12:10:00Z",
            actual_cost_usd=0.1234,
            cost_report_path=cost_report_path,
        )

    monkeypatch.setattr("author_name_disambiguation.public_cli.resolve_modal_cost", _fake_resolve_modal_cost)

    public_cli.main(["cost", "--output-dir", str(tmp_path)])

    output = capsys.readouterr().out
    assert "Modal cost lookup: complete" in output
    assert "0.1234" in output
