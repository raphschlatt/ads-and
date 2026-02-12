from pathlib import Path

from src.common.config import (
    build_run_dirs,
    resolve_run_id,
    write_latest_run_context,
    write_run_consistency,
)


def test_run_context_flow_without_placeholder(tmp_path: Path):
    data_cfg = {
        "subset_cache_dir": str(tmp_path / "data/subsets/cache"),
        "subset_manifest_dir": str(tmp_path / "data/subsets/manifests"),
        "interim_dir": str(tmp_path / "data/interim"),
        "processed_dir": str(tmp_path / "data/processed"),
    }
    artifacts_cfg = {
        "metrics_dir": str(tmp_path / "artifacts/metrics"),
        "checkpoints_dir": str(tmp_path / "artifacts/checkpoints"),
        "pair_scores_dir": str(tmp_path / "artifacts/pair_scores"),
        "clusters_dir": str(tmp_path / "artifacts/clusters"),
        "embeddings_dir": str(tmp_path / "artifacts/embeddings"),
    }
    run_id = "smoke_20260212T000000Z_abcd1234"

    run_dirs = build_run_dirs(data_cfg, artifacts_cfg, run_id)
    for p in run_dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    latest = tmp_path / "artifacts/metrics/latest_run.json"
    write_latest_run_context(run_id=run_id, run_dirs=run_dirs, output_path=latest, stage="smoke")

    resolved = resolve_run_id(None, latest, allow_placeholder=False)
    assert resolved == run_id
    assert "replace_with_run_id_from_00" not in resolved

    for stage_file in ["00", "01", "02", "03", "04", "05"]:
        write_run_consistency(
            run_id=resolved,
            run_stage="smoke",
            run_dirs=run_dirs,
            output_path=run_dirs["metrics"] / f"{stage_file}_run_consistency.json",
            extras={"stage_file": stage_file},
        )

    for stage_file in ["00", "01", "02", "03", "04", "05"]:
        assert (run_dirs["metrics"] / f"{stage_file}_run_consistency.json").exists()
