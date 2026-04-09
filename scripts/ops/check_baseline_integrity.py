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
DEFAULT_OPERATIONAL_MANIFEST = "docs/baselines/lspo_quality_operational.json"

HISTORICAL_BASELINE_KEEP_PATHS = [
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

HISTORICAL_BASELINE_BENCHMARK_FILES = [
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
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
    return payload


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Integrity check for the active LSPO quality reference. "
            "Default mode validates the current operational srcb2... compat state."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("operational", "historical"),
        default="operational",
        help=(
            "Validation target. 'operational' checks the current reproducible srcb2... "
            "reference set. 'historical' checks the original srcd52... baseline as an advisory."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_OPERATIONAL_MANIFEST,
        help=(
            "Operational manifest path. Only used in operational mode "
            f"(default: {DEFAULT_OPERATIONAL_MANIFEST})."
        ),
    )
    parser.add_argument(
        "--baseline-run-id",
        default=DEFAULT_BASELINE_RUN_ID,
        help=f"Historical baseline run id (default: {DEFAULT_BASELINE_RUN_ID}).",
    )
    parser.add_argument(
        "--expected-seeds",
        default="1,2,3,4,5",
        help="Comma-separated seed list expected in LSPO clustering reports.",
    )
    parser.add_argument(
        "--interim-lspo-mentions",
        default="data/interim/lspo_mentions.parquet",
        help="Direct fallback path for interim LSPO mentions when the report does not embed one.",
    )
    return parser


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _resolve_interim_path(
    *,
    repo_root: Path,
    report: dict[str, Any] | None,
    fallback_path: str,
) -> Path:
    if report:
        source_paths = report.get("lspo_source_paths") or {}
        from_report = str(source_paths.get("interim_lspo_mentions", "")).strip()
        if from_report:
            return _resolve_path(repo_root, from_report)
    return _resolve_path(repo_root, fallback_path)


def _resolve_run_config_path(repo_root: Path, context: dict[str, Any]) -> Path | None:
    run_cfg_ref = str(context.get("run_config", "")).strip()
    if not run_cfg_ref:
        return None
    return _resolve_path(repo_root, run_cfg_ref)


def _compute_subset_key(
    *,
    repo_root: Path,
    report: dict[str, Any],
    interim_lspo_mentions: str,
    failures: list[str],
) -> tuple[str | None, str | None]:
    context_path_raw = str(report.get("source_context_path", "")).strip()
    if not context_path_raw:
        failures.append("Missing source_context_path in report.")
        return None, None
    context_path = _resolve_path(repo_root, context_path_raw)
    if not context_path.exists():
        failures.append(f"Context path not found: {context_path}")
        return None, None
    context = _load_json(context_path)
    run_cfg_path = _resolve_run_config_path(repo_root, context)
    if run_cfg_path is None:
        failures.append(f"Missing run_config in {context_path}.")
        return None, None
    if not run_cfg_path.exists():
        failures.append(f"Run config not found: {run_cfg_path}")
        return None, None
    run_cfg = _load_yaml(run_cfg_path)
    run_stage = str(report.get("run_stage") or context.get("run_stage") or run_cfg.get("stage") or "full")
    run_cfg["stage"] = run_stage
    interim_path = _resolve_interim_path(
        repo_root=repo_root,
        report=report,
        fallback_path=interim_lspo_mentions,
    )
    if not interim_path.exists():
        failures.append(f"Interim LSPO mentions path not found: {interim_path}")
        return None, None
    source_fp = compute_lspo_source_fp(interim_path)
    subset_identity = compute_subset_identity(
        run_cfg=run_cfg,
        source_fp=source_fp,
        sampler_version="v3",
    )
    return source_fp, subset_identity.subset_tag


def _run_operational_check(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    manifest_path = _resolve_path(repo_root, str(args.manifest))
    if not manifest_path.exists():
        raise FileNotFoundError(f"Operational manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    failures: list[str] = []
    expected_seeds = sorted(int(x) for x in (manifest.get("expected_seeds") or []))
    if not expected_seeds:
        expected_seeds = _parse_expected_seeds(str(args.expected_seeds))

    required_paths = [_resolve_path(repo_root, rel) for rel in manifest.get("required_paths", [])]
    missing_required_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_required_paths:
        failures.append(f"Missing required operational paths: {len(missing_required_paths)}")

    shared_keep_paths = [_resolve_path(repo_root, rel) for rel in manifest.get("shared_keep_paths", [])]
    missing_shared_keep_paths = [str(path) for path in shared_keep_paths if not path.exists()]
    if missing_shared_keep_paths:
        failures.append(f"Missing shared keep-set paths: {len(missing_shared_keep_paths)}")

    report_path = _resolve_path(repo_root, str(manifest["report_path"]))
    compare_report_path = _resolve_path(repo_root, str(manifest["compare_report_path"]))
    report = _load_json(report_path)
    compare_report = _load_json(compare_report_path)

    report_status = str(report.get("status", ""))
    if report_status != "ok":
        failures.append(f"Operational report status is not ok: {report_status!r}")

    report_seeds_expected = sorted(int(x) for x in (report.get("seeds_expected") or []))
    report_seeds_evaluated = sorted(int(x) for x in (report.get("seeds_evaluated") or []))
    if report_seeds_expected != expected_seeds:
        failures.append(
            "Operational report seeds_expected mismatch: "
            f"expected={expected_seeds}, got={report_seeds_expected}"
        )
    if report_seeds_evaluated != expected_seeds:
        failures.append(
            "Operational report seeds_evaluated mismatch: "
            f"expected={expected_seeds}, got={report_seeds_evaluated}"
        )

    expected_subset_mode = str(manifest.get("expected_subset_verification_mode", "")).strip() or None
    report_subset_mode = str(report.get("subset_verification_mode", "")).strip() or None
    if expected_subset_mode and report_subset_mode != expected_subset_mode:
        failures.append(
            "subset_verification_mode mismatch: "
            f"expected={expected_subset_mode}, got={report_subset_mode}"
        )

    expected_source_fingerprint = str(manifest.get("expected_source_fingerprint", "")).strip() or None
    report_source_fingerprint = str(report.get("lspo_source_fingerprint", "")).strip() or None
    if expected_source_fingerprint and report_source_fingerprint != expected_source_fingerprint:
        failures.append(
            "lspo_source_fingerprint mismatch in report: "
            f"expected={expected_source_fingerprint}, got={report_source_fingerprint}"
        )

    expected_subset_cache_key = str(manifest.get("expected_subset_cache_key", "")).strip() or None
    report_subset_cache_key = str(report.get("subset_cache_key_computed", "")).strip() or None
    if expected_subset_cache_key and report_subset_cache_key != expected_subset_cache_key:
        failures.append(
            "subset_cache_key_computed mismatch in report: "
            f"expected={expected_subset_cache_key}, got={report_subset_cache_key}"
        )

    computed_source_fingerprint, computed_subset_cache_key = _compute_subset_key(
        repo_root=repo_root,
        report=report,
        interim_lspo_mentions=str(args.interim_lspo_mentions),
        failures=failures,
    )
    if expected_source_fingerprint and computed_source_fingerprint != expected_source_fingerprint:
        failures.append(
            "computed lspo_source_fingerprint mismatch: "
            f"expected={expected_source_fingerprint}, got={computed_source_fingerprint}"
        )
    if expected_subset_cache_key and computed_subset_cache_key != expected_subset_cache_key:
        failures.append(
            "computed subset_cache_key mismatch: "
            f"expected={expected_subset_cache_key}, got={computed_subset_cache_key}"
        )

    summary = {
        "mode": "operational",
        "manifest_path": str(manifest_path),
        "baseline_run_id": str(manifest.get("baseline_run_id") or ""),
        "ok": len(failures) == 0,
        "failures": failures,
        "historical_note": manifest.get("historical_note"),
        "missing_required_paths": missing_required_paths,
        "missing_shared_keep_paths": missing_shared_keep_paths,
        "report_path": str(report_path),
        "compare_report_path": str(compare_report_path),
        "report_status": report_status,
        "compare_decision": compare_report.get("decision"),
        "expected_seeds": expected_seeds,
        "report_seeds_expected": report_seeds_expected,
        "report_seeds_evaluated": report_seeds_evaluated,
        "expected_subset_verification_mode": expected_subset_mode,
        "report_subset_verification_mode": report_subset_mode,
        "expected_source_fingerprint": expected_source_fingerprint,
        "report_source_fingerprint": report_source_fingerprint,
        "computed_source_fingerprint": computed_source_fingerprint,
        "expected_subset_cache_key": expected_subset_cache_key,
        "report_subset_cache_key": report_subset_cache_key,
        "computed_subset_cache_key": computed_subset_cache_key,
    }
    return summary


def _run_historical_check(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    run_id = str(args.baseline_run_id).strip()
    expected_seeds = _parse_expected_seeds(str(args.expected_seeds))
    failures: list[str] = []
    missing_keep_paths: list[str] = []
    for rel in HISTORICAL_BASELINE_KEEP_PATHS:
        p = repo_root / rel.format(run_id=run_id)
        if not p.exists():
            missing_keep_paths.append(str(p))
    if missing_keep_paths:
        failures.append(f"Missing baseline keep-set paths: {len(missing_keep_paths)}")

    metrics_dir = repo_root / "artifacts" / "metrics" / run_id
    missing_benchmark_files: list[str] = []
    for name in HISTORICAL_BASELINE_BENCHMARK_FILES:
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

    subset_cache_key_expected = None
    subset_cache_key_computed = None
    if context_path.exists() and stage_metrics_path.exists():
        context = _load_json(context_path)
        stage_metrics = _load_json(stage_metrics_path)
        subset_cache_key_expected = str(stage_metrics.get("subset_cache_key", "")).strip() or None
        if subset_cache_key_expected:
            computed_source_fingerprint, computed_subset_cache_key = _compute_subset_key(
                repo_root=repo_root,
                report=report or {},
                interim_lspo_mentions=str(args.interim_lspo_mentions),
                failures=failures,
            )
            _ = computed_source_fingerprint
            subset_cache_key_computed = computed_subset_cache_key
            if subset_cache_key_computed != subset_cache_key_expected:
                failures.append(
                    "subset_cache_key mismatch: "
                    f"expected={subset_cache_key_expected}, computed={subset_cache_key_computed}"
                )
        else:
            failures.append(f"Missing subset_cache_key in {stage_metrics_path}.")
    else:
        if not context_path.exists():
            failures.append(f"Missing context file: {context_path}")
        if not stage_metrics_path.exists():
            failures.append(f"Missing stage metrics file: {stage_metrics_path}")

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

    return {
        "mode": "historical",
        "baseline_run_id": run_id,
        "ok": len(failures) == 0,
        "failures": failures,
        "advisory": True,
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


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = _repo_root()

    if args.mode == "operational":
        summary = _run_operational_check(args, repo_root)
    else:
        summary = _run_historical_check(args, repo_root)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
