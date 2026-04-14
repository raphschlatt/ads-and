from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from author_name_disambiguation import cli
from author_name_disambiguation.api import LspoQualityResult, LspoTrainingResult
from author_name_disambiguation.infer_sources import InferSourcesResult


def _write_infer_summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "run_id": "infer_sources_test",
                "dataset_id": "ads_demo",
                "output_root": str(path.parent),
                "summary_path": str(path),
                "go": True,
                "infer_stage_requested": "full",
                "infer_stage_effective": "full",
                "runtime_mode": "gpu",
                "runtime_backend": "transformers",
                "resolved_device": "cuda",
                "precision_mode": "amp_bf16",
                "clustering_backend": "sklearn_cpu",
                "counts": {
                    "publications": 10,
                    "references": 5,
                    "canonical_records": 12,
                    "specter_sources": 12,
                    "mentions": 20,
                    "clusters": 8,
                    "authors_total": 24,
                    "authors_mapped": 24,
                    "authors_unmapped": 0,
                },
                "stage_seconds": {
                    "bootstrap": 0.1,
                    "load_inputs": 1.0,
                    "preflight": 0.5,
                    "name_embeddings": 2.0,
                    "text_embeddings": 3.0,
                    "pair_inference": 4.0,
                    "clustering": 5.0,
                    "export": 6.0,
                    "total": 21.6,
                },
                "warnings": ["singleton_ratio"],
                "blockers": [],
                "outputs": {
                    "publications_disambiguated_path": str(path.parent / "publications_disambiguated.parquet"),
                    "references_disambiguated_path": str(path.parent / "references_disambiguated.parquet"),
                    "author_entities_path": str(path.parent / "author_entities.parquet"),
                    "source_author_assignments_path": str(path.parent / "source_author_assignments.parquet"),
                    "mention_clusters_path": str(path.parent / "mention_clusters.parquet"),
                    "stage_metrics_path": str(path.parent / "05_stage_metrics_infer_sources.json"),
                    "go_no_go_path": str(path.parent / "05_go_no_go_infer_sources.json"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_infer_cli_prints_human_summary_by_default(monkeypatch, tmp_path: Path, capsys):
    summary_path = tmp_path / "summary.json"
    _write_infer_summary(summary_path)

    def _fake_disambiguate_sources(*args, **kwargs):
        return InferSourcesResult(
            run_id="infer_sources_test",
            go=True,
            output_root=tmp_path,
            publications_disambiguated_path=tmp_path / "publications_disambiguated.parquet",
            references_disambiguated_path=tmp_path / "references_disambiguated.parquet",
            source_author_assignments_path=tmp_path / "source_author_assignments.parquet",
            author_entities_path=tmp_path / "author_entities.parquet",
            mention_clusters_path=tmp_path / "mention_clusters.parquet",
            stage_metrics_path=tmp_path / "05_stage_metrics_infer_sources.json",
            go_no_go_path=tmp_path / "05_go_no_go_infer_sources.json",
            summary_path=summary_path,
        )

    monkeypatch.setattr("author_name_disambiguation.api.disambiguate_sources", _fake_disambiguate_sources)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--publications-path",
            "publications.parquet",
            "--output-dir",
            str(tmp_path),
            "--no-progress",
        ]
    )
    args.func(args)

    out = capsys.readouterr().out
    assert "ADS Full Inference Run complete" in out
    assert "GO: True" in out
    assert '"run_id"' not in out


def test_infer_cli_prints_json_when_requested(monkeypatch, tmp_path: Path, capsys):
    summary_path = tmp_path / "summary.json"
    _write_infer_summary(summary_path)

    def _fake_disambiguate_sources(*args, **kwargs):
        return InferSourcesResult(
            run_id="infer_sources_test",
            go=True,
            output_root=tmp_path,
            publications_disambiguated_path=tmp_path / "publications_disambiguated.parquet",
            references_disambiguated_path=tmp_path / "references_disambiguated.parquet",
            source_author_assignments_path=tmp_path / "source_author_assignments.parquet",
            author_entities_path=tmp_path / "author_entities.parquet",
            mention_clusters_path=tmp_path / "mention_clusters.parquet",
            stage_metrics_path=tmp_path / "05_stage_metrics_infer_sources.json",
            go_no_go_path=tmp_path / "05_go_no_go_infer_sources.json",
            summary_path=summary_path,
        )

    monkeypatch.setattr("author_name_disambiguation.api.disambiguate_sources", _fake_disambiguate_sources)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--publications-path",
            "publications.parquet",
            "--output-dir",
            str(tmp_path),
            "--no-progress",
            "--json",
        ]
    )
    args.func(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "infer_sources_test"
    assert payload["runtime_mode"] == "gpu"


def test_quality_and_train_cli_print_human_summaries(monkeypatch, tmp_path: Path, capsys):
    quality_report = tmp_path / "06_clustering_test_report.json"
    quality_report.write_text(
        json.dumps(
            {
                "wall_seconds": 17.5,
                "variants": {
                    "dbscan_with_constraints": {"f1_mean": 0.97},
                    "dbscan_no_constraints": {"f1_mean": 0.96},
                },
                "delta_with_constraints_minus_no_constraints": {"f1": 0.01},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "author_name_disambiguation.api.evaluate_lspo_quality",
        lambda **kwargs: LspoQualityResult(
            model_run_id="full_demo",
            metrics_dir=tmp_path,
            report_json_path=quality_report,
            summary_csv_path=tmp_path / "summary.csv",
            per_seed_csv_path=tmp_path / "per_seed.csv",
            report_markdown_path=tmp_path / "report.md",
            wall_seconds=17.5,
        ),
    )
    monkeypatch.setattr(
        "author_name_disambiguation.api.train_lspo_model",
        lambda **kwargs: LspoTrainingResult(
            run_id="train_demo",
            metrics_dir=tmp_path,
            train_manifest_path=tmp_path / "03_train_manifest.json",
            stage_metrics_path=tmp_path / "05_stage_metrics_full.json",
            go_no_go_path=tmp_path / "05_go_no_go_full.json",
            cluster_config_used_path=tmp_path / "04_clustering_config_used.json",
        ),
    )
    (tmp_path / "05_stage_metrics_full.json").write_text(
        json.dumps({"threshold": 0.5, "lspo_pairwise_f1": 0.97}),
        encoding="utf-8",
    )
    (tmp_path / "05_go_no_go_full.json").write_text(json.dumps({"go": True, "warnings": []}), encoding="utf-8")

    parser = cli.build_parser()
    quality_args = parser.parse_args(["quality-lspo", "--model-run-id", "full_demo", "--no-progress"])
    quality_args.func(quality_args)
    quality_out = capsys.readouterr().out
    assert "LSPO Quality Run complete" in quality_out
    assert "Wall time: 17.5s" in quality_out

    train_args = parser.parse_args(["train-lspo", "--no-progress"])
    train_args.func(train_args)
    train_out = capsys.readouterr().out
    assert "LSPO Training Run complete" in train_out


def test_quality_lspo_rejects_missing_target(tmp_path: Path):
    parser = cli.build_parser()
    args = parser.parse_args(["quality-lspo", "--no-progress"])

    with pytest.raises(ValueError, match="quality-lspo requires an explicit target"):
        args.func(args)


def test_doctor_command_prints_checklist_and_json(tmp_path: Path, capsys):
    parser = cli.build_parser()

    missing_args = parser.parse_args(["doctor"])
    with pytest.raises(SystemExit, match="1"):
        missing_args.func(missing_args)
    missing_out = capsys.readouterr().out
    assert "Research doctor: missing_prerequisites" in missing_out
    assert "Packaged fixed bundle:" in missing_out
    assert "infer-only" in missing_out
    assert "Quality target: Pass --model-run-id <run_id>" in missing_out

    raw_h5 = tmp_path / "data" / "raw" / "lspo" / "LSPO_v1.h5"
    raw_h5.parent.mkdir(parents=True, exist_ok=True)
    raw_h5.write_text("stub", encoding="utf-8")
    metrics_dir = tmp_path / "artifacts" / "metrics" / "full_demo"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "00_context.json").write_text(
        json.dumps({"run_stage": "full", "pipeline_scope": "train"}),
        encoding="utf-8",
    )
    (metrics_dir / "03_train_manifest.json").write_text(
        json.dumps(
            {
                "best_threshold": 0.5,
                "runs": [{"seed": 1, "checkpoint": str(tmp_path / "artifacts" / "checkpoints" / "full_demo" / "seed1.pt")}],
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "04_clustering_config_used.json").write_text(
        json.dumps({"cluster_config_used": {"eps": 0.35}}),
        encoding="utf-8",
    )
    (metrics_dir / "05_stage_metrics_full.json").write_text(json.dumps({"subset_cache_key": "abc"}), encoding="utf-8")
    checkpoint = tmp_path / "artifacts" / "checkpoints" / "full_demo" / "seed1.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")

    ok_args = parser.parse_args(
        [
            "doctor",
            "--data-root",
            str(tmp_path / "data"),
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--raw-lspo-h5",
            str(raw_h5),
            "--model-run-id",
            "full_demo",
            "--json",
        ]
    )
    ok_args.func(ok_args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["raw_lspo"]["selected_source"] == "h5"
    assert payload["quality_target"]["provided"] is True
    assert payload["mandatory_train_artifacts"][0]["exists"] is True


def test_module_entrypoints_do_not_emit_runpy_warning():
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    for mod in ["author_name_disambiguation", "author_name_disambiguation.cli"]:
        proc = subprocess.run(
            [sys.executable, "-m", mod, "infer", "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0
        assert "found in sys.modules" not in proc.stderr
