#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from author_name_disambiguation.common.subset_artifacts import compute_lspo_source_fp, compute_subset_identity


DEFAULT_BASELINE_RUN_ID = "full_20260218T111506Z_cli02681429"

BASELINE_KEEP_PATHS = [
    "artifacts/metrics/{run_id}",
    "artifacts/checkpoints/{run_id}",
    "artifacts/models/{run_id}",
    "artifacts/embeddings/{run_id}",
    "data/interim/lspo_mentions.parquet",
    "data/cache/_shared/subsets/lspo_mentions_full_seed11_targetfull_cfg0dbcdaf9_srcd52b159f766e.parquet",
    "data/cache/_shared/embeddings/lspo_chars2vec_05757fec0582.npy",
    "data/cache/_shared/embeddings/lspo_specter_05757fec0582.npy",
    "data/cache/_shared/pairs/lspo_mentions_split_978ea2bd7512.parquet",
    "data/cache/_shared/pairs/lspo_pairs_978ea2bd7512.parquet",
    "data/cache/_shared/pairs/split_balance_978ea2bd7512.json",
    "data/cache/_shared/pairs/pairs_qc_train_978ea2bd7512.json",
    "data/cache/_shared/eps_sweeps/eps_sweep_4f69281cae15.json",
]

BASELINE_BENCHMARK_FILES = [
    "03_train_manifest.json",
    "04_clustering_config_used.json",
    "05_stage_metrics_full.json",
    "06_clustering_test_report.json",
    "06_clustering_test_summary.csv",
    "06_clustering_test_per_seed.csv",
    "06_clustering_test_report.md",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_expected_seeds(raw: str) -> list[int]:
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if not items:
        raise ValueError("--expected-seeds cannot be empty.")
    return sorted(int(x) for x in items)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lean integrity check for a canonical train baseline run.",
    )
    parser.add_argument(
        "--baseline-run-id",
        default=DEFAULT_BASELINE_RUN_ID,
        help=f"Canonical baseline run id (default: {DEFAULT_BASELINE_RUN_ID}).",
    )
    parser.add_argument(
        "--expected-seeds",
        default="1,2,3,4,5",
        help="Comma-separated seed list expected in 06_clustering_test_report.json.",
    )
    parser.add_argument(
        "--paths-config",
        default="configs/paths.local.yaml",
        help="Paths config fallback for interim path resolution.",
    )
    return parser.parse_args()


def _resolve_interim_path(
    *,
    repo_root: Path,
    report: dict[str, Any] | None,
    paths_cfg_path: Path,
) -> Path:
    if report:
        source_paths = report.get("lspo_source_paths") or {}
        from_report = str(source_paths.get("interim_lspo_mentions", "")).strip()
        if from_report:
            return Path(from_report)
    cfg = _load_yaml(paths_cfg_path)
    data_cfg = dict(cfg.get("data", {}) or {})
    interim_dir_raw = str(data_cfg.get("interim_dir", "data/interim"))
    interim_dir = Path(interim_dir_raw)
    if not interim_dir.is_absolute():
        interim_dir = repo_root / interim_dir
    return interim_dir / "lspo_mentions.parquet"


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    run_id = str(args.baseline_run_id).strip()
    expected_seeds = _parse_expected_seeds(str(args.expected_seeds))
    paths_cfg_path = Path(args.paths_config)
    if not paths_cfg_path.is_absolute():
        paths_cfg_path = repo_root / paths_cfg_path

    failures: list[str] = []

    # 1) Check explicit keep-set.
    missing_keep_paths: list[str] = []
    for rel in BASELINE_KEEP_PATHS:
        p = repo_root / rel.format(run_id=run_id)
        if not p.exists():
            missing_keep_paths.append(str(p))
    if missing_keep_paths:
        failures.append(f"Missing baseline keep-set paths: {len(missing_keep_paths)}")

    metrics_dir = repo_root / "artifacts" / "metrics" / run_id

    # 2) Check benchmark/report completeness.
    missing_benchmark_files: list[str] = []
    for name in BASELINE_BENCHMARK_FILES:
        p = metrics_dir / name
        if not p.exists():
            missing_benchmark_files.append(str(p))
    if missing_benchmark_files:
        failures.append(f"Missing required metric/report files: {len(missing_benchmark_files)}")

    context_path = metrics_dir / "00_context.json"
    stage_metrics_path = metrics_dir / "05_stage_metrics_full.json"
    report_path = metrics_dir / "06_clustering_test_report.json"
    cache_refs_path = metrics_dir / "00_cache_refs.json"

    report: dict[str, Any] | None = None
    if report_path.exists():
        report = _load_json(report_path)

    # 3) Validate subset_cache_key reproducibility.
    subset_cache_key_expected = None
    subset_cache_key_computed = None
    if context_path.exists() and stage_metrics_path.exists():
        context = _load_json(context_path)
        stage_metrics = _load_json(stage_metrics_path)
        subset_cache_key_expected = str(stage_metrics.get("subset_cache_key", "")).strip() or None
        run_cfg_ref = str(context.get("run_config", "")).strip()
        if not run_cfg_ref:
            failures.append(f"Missing run_config in {context_path}.")
        elif not subset_cache_key_expected:
            failures.append(f"Missing subset_cache_key in {stage_metrics_path}.")
        else:
            run_cfg_path = Path(run_cfg_ref)
            if not run_cfg_path.is_absolute():
                run_cfg_path = repo_root / run_cfg_path
            if not run_cfg_path.exists():
                failures.append(f"Run config not found: {run_cfg_path}")
            else:
                run_cfg = _load_yaml(run_cfg_path)
                run_stage = str(context.get("run_stage") or run_cfg.get("stage") or "full")
                run_cfg["stage"] = run_stage
                interim_path = _resolve_interim_path(
                    repo_root=repo_root,
                    report=report,
                    paths_cfg_path=paths_cfg_path,
                )
                if not interim_path.exists():
                    failures.append(f"Interim LSPO mentions path not found: {interim_path}")
                else:
                    source_fp = compute_lspo_source_fp(interim_path)
                    subset_identity = compute_subset_identity(
                        run_cfg=run_cfg,
                        source_fp=source_fp,
                        sampler_version="v3",
                    )
                    subset_cache_key_computed = subset_identity.subset_tag
                    if subset_cache_key_computed != subset_cache_key_expected:
                        failures.append(
                            "subset_cache_key mismatch: "
                            f"expected={subset_cache_key_expected}, computed={subset_cache_key_computed}"
                        )

    # 4) Validate final clustering report status + seeds.
    report_status = None
    report_seeds_expected = None
    report_seeds_evaluated = None
    if report:
        report_status = str(report.get("status", ""))
        report_seeds_expected = sorted(int(x) for x in (report.get("seeds_expected") or []))
        report_seeds_evaluated = sorted(int(x) for x in (report.get("seeds_evaluated") or []))
        if report_status != "ok":
            failures.append(f"06 report status is not ok: {report_status!r}")
        if report_seeds_evaluated != expected_seeds:
            failures.append(
                "06 report seeds_evaluated mismatch: "
                f"expected={expected_seeds}, got={report_seeds_evaluated}"
            )
        if report_seeds_expected != expected_seeds:
            failures.append(
                "06 report seeds_expected mismatch: "
                f"expected={expected_seeds}, got={report_seeds_expected}"
            )
        if report_seeds_expected != report_seeds_evaluated:
            failures.append(
                "06 report seeds_expected and seeds_evaluated differ: "
                f"{report_seeds_expected} vs {report_seeds_evaluated}"
            )

    # 5) Validate shared cache refs from 00_cache_refs.json.
    missing_cache_ref_shared_paths: list[str] = []
    cache_ref_rows = 0
    if cache_refs_path.exists():
        cache_refs = _load_json(cache_refs_path)
        rows = cache_refs.get("cache_refs", [])
        cache_ref_rows = len(rows)
        for row in rows:
            shared_path_raw = str(row.get("shared_path", "")).strip()
            if not shared_path_raw:
                continue
            shared_path = Path(shared_path_raw)
            if not shared_path.exists():
                missing_cache_ref_shared_paths.append(shared_path_raw)
        if missing_cache_ref_shared_paths:
            failures.append(
                f"Missing shared cache refs from 00_cache_refs.json: {len(missing_cache_ref_shared_paths)}"
            )

    summary = {
        "baseline_run_id": run_id,
        "ok": len(failures) == 0,
        "failures": failures,
        "missing_keep_paths": missing_keep_paths,
        "missing_required_metric_files": missing_benchmark_files,
        "subset_cache_key_expected": subset_cache_key_expected,
        "subset_cache_key_computed": subset_cache_key_computed,
        "report_status": report_status,
        "report_seeds_expected": report_seeds_expected,
        "report_seeds_evaluated": report_seeds_evaluated,
        "cache_ref_rows": cache_ref_rows,
        "missing_cache_ref_shared_paths": missing_cache_ref_shared_paths,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
