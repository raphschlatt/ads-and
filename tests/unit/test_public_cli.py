from __future__ import annotations

import json
from pathlib import Path

import author_name_disambiguation
from author_name_disambiguation import public_cli
from author_name_disambiguation.infer_sources import InferSourcesResult


def test_public_package_root_exports_inference_only():
    exported = set(author_name_disambiguation.__all__)
    assert "disambiguate_sources" in exported
    assert "run_infer_sources" in exported
    assert "InferSourcesRequest" in exported
    assert "evaluate_lspo_quality" not in exported
    assert "train_lspo_model" not in exported
    assert "precompute_source_embeddings" not in exported


def test_public_cli_exposes_only_infer_command():
    parser = public_cli.build_parser()
    commands = set(parser._subparsers._group_actions[0].choices.keys())
    assert commands == {"infer"}


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
    assert args.dataset_id is None
    assert args.runtime == "auto"
    assert args.infer_stage == "full"
    assert args.progress is True
    assert args.progress_style == "compact"
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
