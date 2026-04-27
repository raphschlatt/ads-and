from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_script_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "ops"
        / "check_baseline_integrity.py"
    )
    spec = importlib.util.spec_from_file_location("check_baseline_integrity", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Failed to load check_baseline_integrity module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_operational_integrity_check_uses_manifest_and_current_srcb2_state(tmp_path, monkeypatch):
    module = _load_script_module()
    repo_root = tmp_path / "repo"
    report_path = repo_root / "artifacts" / "metrics" / "run" / "06_report.json"
    compare_path = repo_root / "artifacts" / "metrics" / "run" / "99_compare.json"
    context_path = repo_root / "artifacts" / "metrics" / "run" / "00_context.json"
    run_config_path = repo_root / "configs" / "run.yaml"
    interim_path = repo_root / "data" / "interim" / "lspo_mentions.parquet"
    shared_subset_path = repo_root / "data" / "cache" / "_shared" / "subsets" / "lspo_mentions.parquet"
    shared_embed_path = repo_root / "data" / "cache" / "_shared" / "embeddings" / "lspo_chars2vec_cpu.npy"
    manifest_path = repo_root / "docs" / "baselines" / "lspo_quality_operational.json"

    for path in (
        report_path,
        compare_path,
        context_path,
        run_config_path,
        interim_path,
        shared_subset_path,
        shared_embed_path,
        manifest_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "run_stage": "full",
                "source_context_path": str(context_path),
                "lspo_source_paths": {"interim_lspo_mentions": str(interim_path)},
                "lspo_source_fingerprint": "b2c9203fe342",
                "subset_cache_key_computed": "full_seed11_targetfull_cfg0dbcdaf9_srcb2c9203fe342",
                "subset_verification_mode": "legacy_compat",
                "seeds_expected": [1, 2, 3, 4, 5],
                "seeds_evaluated": [1, 2, 3, 4, 5],
            }
        ),
        encoding="utf-8",
    )
    compare_path.write_text(
        json.dumps({"decision": "rollback_to_baseline", "candidate_status": "ok", "baseline_status": "ok"}),
        encoding="utf-8",
    )
    context_path.write_text(
        json.dumps(
            {
                "run_config": "configs/runs/full.yaml",
                "run_config_payload": {"stage": "full"},
                "model_config": "configs/model/nand_best.yaml",
                "model_config_payload": {"name": "nand_best"},
                "cluster_config": "configs/clustering/dbscan_paper.yaml",
                "cluster_config_payload": {"method": "dbscan"},
                "run_stage": "full",
            }
        ),
        encoding="utf-8",
    )
    run_config_path.write_text("stage: full\n", encoding="utf-8")
    interim_path.write_text("placeholder", encoding="utf-8")
    shared_subset_path.write_text("placeholder", encoding="utf-8")
    shared_embed_path.write_text("placeholder", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "baseline_run_id": "full_20260218T111506Z_cli02681429",
                "report_path": str(report_path.relative_to(repo_root)),
                "compare_report_path": str(compare_path.relative_to(repo_root)),
                "expected_seeds": [1, 2, 3, 4, 5],
                "expected_source_fingerprint": "b2c9203fe342",
                "expected_subset_cache_key": "full_seed11_targetfull_cfg0dbcdaf9_srcb2c9203fe342",
                "expected_subset_verification_mode": "legacy_compat",
                "required_paths": [
                    str(report_path.relative_to(repo_root)),
                    str(compare_path.relative_to(repo_root)),
                    str(interim_path.relative_to(repo_root)),
                    str(shared_subset_path.relative_to(repo_root)),
                    str(shared_embed_path.relative_to(repo_root)),
                ],
                "shared_keep_paths": [
                    str(shared_subset_path.relative_to(repo_root)),
                    str(shared_embed_path.relative_to(repo_root)),
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(module, "compute_lspo_source_fp", lambda path: "b2c9203fe342")
    monkeypatch.setattr(
        module,
        "compute_subset_identity",
        lambda **kwargs: SimpleNamespace(subset_tag="full_seed11_targetfull_cfg0dbcdaf9_srcb2c9203fe342"),
    )

    old_argv = sys.argv
    sys.argv = [
        "check_baseline_integrity.py",
        "--mode",
        "operational",
        "--manifest",
        str(manifest_path),
    ]
    try:
        rc = module.main()
    finally:
        sys.argv = old_argv

    assert rc == 0


def test_operational_integrity_check_requires_embedded_context_payloads(tmp_path, monkeypatch):
    module = _load_script_module()
    repo_root = tmp_path / "repo"
    metrics_dir = repo_root / "artifacts" / "metrics" / "run"
    report_path = metrics_dir / "06_report.json"
    context_path = metrics_dir / "00_context.json"
    manifest_path = repo_root / "docs" / "baselines" / "lspo_quality_operational.json"
    for path in (report_path, context_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "source_context_path": str(context_path.relative_to(repo_root)),
                "seeds_expected": [1, 2, 3, 4, 5],
                "seeds_evaluated": [1, 2, 3, 4, 5],
            }
        ),
        encoding="utf-8",
    )
    context_path.write_text(
        json.dumps(
            {
                "run_config": "configs/runs/full.yaml",
                "model_config": "configs/model/nand_best.yaml",
                "cluster_config": "configs/clustering/dbscan_paper.yaml",
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "report_path": str(report_path.relative_to(repo_root)),
                "expected_seeds": [1, 2, 3, 4, 5],
                "required_paths": [
                    str(report_path.relative_to(repo_root)),
                    str(context_path.relative_to(repo_root)),
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "_repo_root", lambda: repo_root)

    summary = module._run_operational_check(
        SimpleNamespace(
            manifest=str(manifest_path),
            expected_seeds="1,2,3,4,5",
            interim_lspo_mentions="data/interim/lspo_mentions.parquet",
        ),
        repo_root,
    )

    assert summary["ok"] is False
    assert any("run_config_payload" in failure for failure in summary["failures"])
    assert any("model_config_payload" in failure for failure in summary["failures"])
    assert any("cluster_config_payload" in failure for failure in summary["failures"])
