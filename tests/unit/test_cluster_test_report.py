from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from author_name_disambiguation import cli


def test_compute_mean_sem_handles_empty_single_and_multi():
    mean, sem = cli._compute_mean_sem([])
    assert mean is None
    assert sem is None

    mean, sem = cli._compute_mean_sem([3.0])
    assert mean == 3.0
    assert sem is None

    mean, sem = cli._compute_mean_sem([1.0, 3.0, 5.0])
    assert mean == 3.0
    assert sem is not None
    expected = math.sqrt(4.0) / math.sqrt(3.0)
    assert sem == expected


def test_build_cluster_variant_config_does_not_mutate_input():
    base = {
        "eps": 0.35,
        "min_samples": 1,
        "metric": "precomputed",
        "constraints": {
            "enabled": True,
            "name_conflict_mode": "hard",
        },
    }

    out_disabled = cli._build_cluster_variant_config(base, enable_constraints=False)
    out_enabled = cli._build_cluster_variant_config(base, enable_constraints=True)

    assert base["constraints"]["enabled"] is True
    assert out_disabled["constraints"]["enabled"] is False
    assert out_enabled["constraints"]["enabled"] is True
    assert out_disabled["constraints"]["name_conflict_mode"] == "hard"


def test_summarize_cluster_test_rows_and_markdown():
    per_seed_rows = [
        {
            "seed": 1,
            "checkpoint": "/tmp/seed1.pt",
            "threshold": 0.5,
            "variant": "dbscan_no_constraints",
            "accuracy": 0.90,
            "precision": 0.91,
            "recall": 0.92,
            "f1": 0.93,
            "n_pairs": 100,
        },
        {
            "seed": 2,
            "checkpoint": "/tmp/seed2.pt",
            "threshold": 0.5,
            "variant": "dbscan_no_constraints",
            "accuracy": 0.80,
            "precision": 0.81,
            "recall": 0.82,
            "f1": 0.83,
            "n_pairs": 120,
        },
        {
            "seed": 1,
            "checkpoint": "/tmp/seed1.pt",
            "threshold": 0.5,
            "variant": "dbscan_with_constraints",
            "accuracy": 0.95,
            "precision": 0.96,
            "recall": 0.97,
            "f1": 0.98,
            "n_pairs": 100,
        },
        {
            "seed": 2,
            "checkpoint": "/tmp/seed2.pt",
            "threshold": 0.5,
            "variant": "dbscan_with_constraints",
            "accuracy": 0.85,
            "precision": 0.86,
            "recall": 0.87,
            "f1": 0.88,
            "n_pairs": 120,
        },
    ]
    summary = cli._summarize_cluster_test_rows(per_seed_rows)
    assert "dbscan_no_constraints" in summary
    assert "dbscan_with_constraints" in summary
    assert summary["dbscan_no_constraints"]["seed_count"] == 2
    assert summary["dbscan_no_constraints"]["n_pairs_total"] == 220
    assert summary["dbscan_no_constraints"]["f1_mean"] == 0.88
    assert summary["dbscan_no_constraints"]["f1_sem"] is not None

    report = {
        "model_run_id": "full_2026abc",
        "run_stage": "full",
        "generated_utc": "2026-02-25T00:00:00+00:00",
        "selected_eps": 0.35,
        "min_samples": 1,
        "metric": "precomputed",
        "seeds_expected": [1, 2],
        "seeds_evaluated": [1, 2],
        "variants": summary,
        "per_seed_rows": per_seed_rows,
        "delta_with_constraints_minus_no_constraints": {
            "accuracy": 0.05,
            "precision": 0.05,
            "recall": 0.05,
            "f1": 0.05,
        },
    }
    md = cli._build_cluster_test_report_markdown(report)
    assert "# Final Clustering Test Report" in md
    assert "dbscan_no_constraints" in md
    assert "dbscan_with_constraints" in md
    assert "## Per Seed" in md


def test_deep_merge_dict_merges_nested_without_mutating_inputs():
    base = {
        "eps": 0.35,
        "constraints": {"enabled": True, "name_conflict_mode": "hard"},
    }
    override = {
        "constraints": {"enabled": False, "year_gap_mode": "soft"},
        "new_section": {"value": 7},
    }
    merged = cli._deep_merge_dict(base, override)
    assert base["constraints"]["enabled"] is True
    assert "year_gap_mode" not in base["constraints"]
    assert merged["constraints"]["enabled"] is False
    assert merged["constraints"]["name_conflict_mode"] == "hard"
    assert merged["constraints"]["year_gap_mode"] == "soft"
    assert merged["new_section"]["value"] == 7


def test_sanitize_report_tag_accepts_valid_and_rejects_invalid():
    assert cli._sanitize_report_tag(None) is None
    assert cli._sanitize_report_tag("epsbkt_v1") == "epsbkt_v1"
    assert cli._sanitize_report_tag("  eps.bkt-1  ") == "eps.bkt-1"
    with pytest.raises(ValueError, match="non-empty"):
        cli._sanitize_report_tag("   ")
    with pytest.raises(ValueError, match="Invalid report_tag"):
        cli._sanitize_report_tag("eps bkt")


def test_resolve_report_paths_supports_default_and_tagged(tmp_path: Path):
    default_paths = cli._resolve_report_paths(tmp_path, report_tag=None)
    assert default_paths["json"].name == "06_clustering_test_report.json"
    assert default_paths["summary_csv"].name == "06_clustering_test_summary.csv"
    tagged_paths = cli._resolve_report_paths(tmp_path, report_tag="epsbkt_v1")
    assert tagged_paths["json"].name == "06_clustering_test_report__epsbkt_v1.json"
    assert tagged_paths["per_seed_csv"].name == "06_clustering_test_per_seed__epsbkt_v1.csv"


def test_apply_cluster_config_override_merges_and_reports_ignored_fields(tmp_path: Path):
    override_path = tmp_path / "cluster_override.yaml"
    override_payload = {
        "eps": 0.41,
        "selected_eps": 0.41,
        "eps_mode": "val_sweep",
        "constraints": {"enabled": False, "name_conflict_mode": "soft"},
        "eps_block_policy": {"enabled": True},
    }
    override_path.write_text(yaml.safe_dump(override_payload, sort_keys=False), encoding="utf-8")

    base_cfg = {
        "eps": 0.35,
        "min_samples": 1,
        "constraints": {"enabled": True, "name_conflict_mode": "hard"},
    }
    merged, source_mode, resolved_path, ignored = cli._apply_cluster_config_override(
        base_cluster_config=base_cfg,
        override_path=str(override_path),
    )
    assert source_mode == "train_plus_override"
    assert resolved_path == str(override_path.resolve())
    assert ignored == ["eps", "selected_eps", "eps_mode"]
    assert merged["constraints"]["enabled"] is False
    assert merged["constraints"]["name_conflict_mode"] == "soft"
    assert merged["eps_block_policy"]["enabled"] is True

    base_only, mode_base, path_base, ignored_base = cli._apply_cluster_config_override(
        base_cluster_config=base_cfg,
        override_path=None,
    )
    assert mode_base == "train_only"
    assert path_base is None
    assert ignored_base == []
    assert base_only == base_cfg
