from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "ops" / "prune_infer_run.py"


def test_prune_infer_run_keeps_only_json_retention_set(tmp_path: Path):
    run_dir = tmp_path / "bench_full_candidate"
    run_dir.mkdir()
    keep_names = [
        "00_context.json",
        "01_input_summary.json",
        "02_preflight_infer.json",
        "03_pairs_qc.json",
        "04_embeddings_qc.json",
        "05_stage_metrics_infer_sources.json",
        "05_go_no_go_infer_sources.json",
        "98_infer_baseline_decision.json",
        "98_infer_baseline_decision.md",
        "99_compare_infer_to_baseline.json",
        "99_compare_infer_to_speedup_wave.json",
        "summary.json",
    ]
    for name in keep_names:
        (run_dir / name).write_text("{}", encoding="utf-8")
    (run_dir / "publications_disambiguated.parquet").write_text("big", encoding="utf-8")
    (run_dir / "mention_clusters.parquet").write_text("big", encoding="utf-8")
    scratch_dir = run_dir / "scratch"
    scratch_dir.mkdir()
    (scratch_dir / "mention_embeddings.npy").write_text("big", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--run-dir",
            str(run_dir),
            "--mode",
            "json-only",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(proc.stdout)
    assert payload["removed_files"] == ["mention_clusters.parquet", "publications_disambiguated.parquet"]
    assert payload["removed_dirs"] == ["scratch"]
    assert sorted(path.name for path in run_dir.iterdir()) == sorted(keep_names)


def test_prune_infer_run_product_only_keeps_final_products_and_metadata(tmp_path: Path):
    run_dir = tmp_path / "ads_full_candidate"
    run_dir.mkdir()
    keep_names = [
        "00_context.json",
        "05_stage_metrics_infer_sources.json",
        "05_go_no_go_infer_sources.json",
        "98_infer_baseline_decision.json",
        "98_infer_baseline_decision.md",
        "99_compare_infer_to_baseline.json",
        "publications_disambiguated.parquet",
        "references_disambiguated.parquet",
        "source_author_assignments.parquet",
        "author_entities.parquet",
        "mention_clusters.parquet",
    ]
    for name in keep_names:
        (run_dir / name).write_text("{}", encoding="utf-8")
    for dirname in ("scratch", "interim", "processed", "embeddings"):
        path = run_dir / dirname
        path.mkdir()
        (path / "payload.bin").write_text("big", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--run-dir",
            str(run_dir),
            "--mode",
            "product-only",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(proc.stdout)
    assert payload["removed_dirs"] == ["embeddings", "interim", "processed", "scratch"]
    assert payload["removed_files"] == []
    assert sorted(path.name for path in run_dir.iterdir()) == sorted(keep_names)
