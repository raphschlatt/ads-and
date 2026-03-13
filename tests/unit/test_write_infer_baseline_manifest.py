from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "ops" / "write_infer_baseline_manifest.py"


def test_write_infer_baseline_manifest_writes_versioned_manifest(tmp_path: Path):
    run_dir = tmp_path / "bench_full_v99"
    run_dir.mkdir()
    (run_dir / "00_context.json").write_text(
        json.dumps(
            {
                "dataset_id": "ads_full_run_20260305",
                "run_id": "infer_sources_fake_context",
                "source_model_run_id": "full_20260218T111506Z_cli02681429",
                "model_bundle": "artifacts/models/full_20260218T111506Z_cli02681429/bundle_v1",
                "precision_mode": "fp32",
                "device": "auto",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "05_stage_metrics_infer_sources.json").write_text(
        json.dumps(
            {
                "run_id": "infer_sources_fake_stage",
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
                    "chars2vec": {"wall_seconds": 4.0},
                },
                "source_export": {"coverage_rate": 1.0},
                "singleton_ratio": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "05_go_no_go_infer_sources.json").write_text(
        json.dumps({"go": True, "warnings": ["singleton_ratio"], "blockers": []}),
        encoding="utf-8",
    )
    compare_path = tmp_path / "99_compare_infer_to_baseline.json"
    compare_path.write_text(
        json.dumps({"baseline_run_id": "bench_full_v22_fix2", "current_run_id": "bench_full_v99"}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "docs" / "baselines" / "infer_ads_full_run_20260305_v99.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--run-dir",
            str(run_dir),
            "--manifest-path",
            str(manifest_path),
            "--compare-report",
            str(compare_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["manifest_scope"] == "infer_baseline"
    assert payload["manifest_version"] == 2
    assert payload["canonical_baseline"]["artifact_dir"] == "bench_full_v99"
    assert payload["canonical_baseline"]["run_id"] == "infer_sources_fake_stage"
    assert payload["canonical_baseline"]["runtime"]["specter_effective_batch_size"] == 256
    assert payload["comparison_to_previous_baseline"]["baseline_run_id"] == "bench_full_v22_fix2"
