from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "ops" / "freeze_infer_baseline.py"


def test_freeze_infer_baseline_rejects_runtime_regression_and_writes_decision(tmp_path: Path):
    metrics_root = tmp_path / "exports"
    baseline_dir = metrics_root / "bench_full_v22_fix2"
    candidate_dir = metrics_root / "bench_full_perf_pkg1"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    compare_payload = {
        "compare_scope": "infer",
        "go_baseline": True,
        "go_current": True,
        "warnings_baseline": ["singleton_ratio"],
        "warnings_current": ["singleton_ratio"],
        "blockers_current": [],
        "ads_clusters_delta": -7.0,
        "ads_mentions_delta": 0.0,
        "ads_cluster_assignments_delta": 0.0,
        "singleton_ratio_delta": 0.00001,
        "source_coverage_rate_current": 1.0,
        "mention_cluster_compare_status": "ok",
        "mention_cluster_changed_mentions": 62,
        "mention_cluster_changed_blocks": 13,
        "mention_cluster_top_changed_blocks": [{"block_key": "h.neuberger", "changed_mentions": 18}],
        "runtime_seconds_compare": {
            "metrics": {
                "clustering.dbscan_seconds_total": {
                    "baseline": 331.77,
                    "current": 610.16,
                    "delta": 278.39,
                    "speedup": 0.54,
                }
            }
        },
    }
    (candidate_dir / "99_compare_infer_to_baseline.json").write_text(
        json.dumps(compare_payload),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--baseline-run-id",
            "bench_full_v22_fix2",
            "--candidate-run-id",
            "bench_full_perf_pkg1",
            "--metrics-root",
            str(metrics_root),
            "--max-abs-cluster-delta",
            "20",
            "--max-changed-mentions",
            "100",
            "--runtime-metric-max-delta",
            "clustering.dbscan_seconds_total=0",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr

    decision_path = candidate_dir / "98_infer_baseline_decision.json"
    markdown_path = candidate_dir / "98_infer_baseline_decision.md"
    assert decision_path.exists()
    assert markdown_path.exists()
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    assert payload["decision"]["decision"] == "keep_baseline"
    assert payload["decision"]["passed"] is False
    assert any("clustering.dbscan_seconds_total" in item for item in payload["decision"]["failures"])


def test_freeze_infer_baseline_can_promote_and_write_manifest(tmp_path: Path):
    metrics_root = tmp_path / "exports"
    baseline_dir = metrics_root / "bench_full_v22_fix2"
    candidate_dir = metrics_root / "bench_full_perf_pkg2"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    (candidate_dir / "00_context.json").write_text(
        json.dumps(
            {
                "dataset_id": "ads_full_run_20260305",
                "run_id": "infer_sources_context",
                "source_model_run_id": "full_20260218T111506Z_cli02681429",
                "model_bundle": "artifacts/models/full_20260218T111506Z_cli02681429/bundle_v1",
                "precision_mode": "fp32",
                "device": "auto",
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "05_stage_metrics_infer_sources.json").write_text(
        json.dumps(
            {
                "run_id": "infer_sources_candidate",
                "counts": {
                    "ads_mentions": 10,
                    "ads_clusters": 4,
                    "ads_cluster_assignments": 10,
                    "ads_blocks": 3,
                },
                "runtime": {
                    "specter": {
                        "effective_precision_mode": "amp_bf16",
                        "effective_batch_size": 256,
                        "tokenize_seconds_total": 12.5,
                    },
                    "clustering": {"dbscan_seconds_total": 300.0},
                },
                "source_export": {"coverage_rate": 1.0},
                "singleton_ratio": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "05_go_no_go_infer_sources.json").write_text(
        json.dumps({"go": True, "warnings": [], "blockers": []}),
        encoding="utf-8",
    )
    (candidate_dir / "99_compare_infer_to_baseline.json").write_text(
        json.dumps(
            {
                "compare_scope": "infer",
                "go_baseline": True,
                "go_current": True,
                "warnings_baseline": [],
                "warnings_current": [],
                "blockers_current": [],
                "ads_clusters_delta": 0.0,
                "ads_mentions_delta": 0.0,
                "ads_cluster_assignments_delta": 0.0,
                "singleton_ratio_delta": 0.0,
                "source_coverage_rate_current": 1.0,
                "mention_cluster_compare_status": "ok",
                "mention_cluster_changed_mentions": 0,
                "mention_cluster_changed_blocks": 0,
                "mention_cluster_top_changed_blocks": [],
                "runtime_seconds_compare": {
                    "metrics": {
                        "clustering.dbscan_seconds_total": {
                            "baseline": 331.77,
                            "current": 300.0,
                            "delta": -31.77,
                            "speedup": 1.10,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "docs" / "baselines" / "infer_ads_full_run_20260305_v23.json"
    active_path = tmp_path / "docs" / "baselines" / "infer_ads_active.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--baseline-run-id",
            "bench_full_v22_fix2",
            "--candidate-run-id",
            "bench_full_perf_pkg2",
            "--metrics-root",
            str(metrics_root),
            "--runtime-metric-max-delta",
            "clustering.dbscan_seconds_total=0",
            "--promote-manifest-path",
            str(manifest_path),
            "--active-baseline-path",
            str(active_path),
            "--keep-artifact",
            "bench_full_v22_fix2",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads((candidate_dir / "98_infer_baseline_decision.json").read_text(encoding="utf-8"))
    assert payload["decision"]["decision"] == "promote_candidate"
    assert payload["decision"]["passed"] is True
    assert payload["decision"]["promoted_manifest_path"] == str(manifest_path)
    assert payload["references"]["active_baseline_path"] == str(active_path)

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["canonical_baseline"]["artifact_dir"] == "bench_full_perf_pkg2"
    assert "bench_full_v22_fix2" in manifest_payload["artifact_keep_set"]

    active_payload = json.loads(active_path.read_text(encoding="utf-8"))
    assert active_payload["baseline_run_id"] == "bench_full_perf_pkg2"
    assert active_payload["artifact_dir"] == "bench_full_perf_pkg2"
    assert active_payload["manifest_path"] == str(manifest_path)