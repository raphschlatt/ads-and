#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _flatten_runtime(prefix: str, payload: dict[str, Any], out: dict[str, float]) -> None:
    for key, value in payload.items():
        next_prefix = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten_runtime(next_prefix, value, out)
        elif isinstance(value, (int, float)) and "seconds" in str(key):
            out[next_prefix] = float(value)


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a versioned ADS infer baseline manifest for a promoted run."
    )
    parser.add_argument("--run-dir", required=True, help="Path to artifacts/exports/<run_id> directory.")
    parser.add_argument("--manifest-path", required=True, help="Output path in docs/baselines/.")
    parser.add_argument("--compare-report", default=None, help="Optional compare JSON against previous baseline.")
    parser.add_argument(
        "--keep-artifact",
        action="append",
        default=[],
        help="Additional artifact path or run dir name to list in artifact_keep_set.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]

    context = _load_json(run_dir / "00_context.json")
    stage = _load_json(run_dir / "05_stage_metrics_infer_sources.json")
    go = _load_json(run_dir / "05_go_no_go_infer_sources.json")
    compare = _load_json(Path(args.compare_report).expanduser().resolve()) if args.compare_report else None

    runtime_seconds: dict[str, float] = {}
    _flatten_runtime("", dict(stage.get("runtime") or {}), runtime_seconds)

    counts = dict(stage.get("counts") or {})
    source_export = dict(stage.get("source_export") or {})
    artifact_keep_set = [run_dir.name]
    artifact_keep_set.extend(str(value).strip() for value in args.keep_artifact if str(value).strip())

    payload = {
        "manifest_scope": "infer_baseline",
        "manifest_version": 2,
        "generated_utc": _utc_now(),
        "dataset_id": context.get("dataset_id"),
        "metrics_root": _relative_to_repo(run_dir.parent, repo_root),
        "canonical_baseline": {
            "artifact_dir": run_dir.name,
            "run_id": stage.get("run_id") or context.get("run_id"),
            "source_model_run_id": context.get("source_model_run_id"),
            "model_bundle": context.get("model_bundle"),
            "runtime": {
                "precision_mode": context.get("precision_mode"),
                "device": context.get("device"),
                "specter_effective_precision_mode": ((stage.get("runtime") or {}).get("specter") or {}).get(
                    "effective_precision_mode"
                ),
                "specter_effective_batch_size": ((stage.get("runtime") or {}).get("specter") or {}).get(
                    "effective_batch_size"
                ),
            },
            "counts": {
                "ads_mentions": counts.get("ads_mentions"),
                "ads_clusters": counts.get("ads_clusters"),
                "ads_cluster_assignments": counts.get("ads_cluster_assignments"),
                "ads_blocks": counts.get("ads_blocks"),
            },
            "quality": {
                "go": go.get("go"),
                "warnings": go.get("warnings", []),
                "blockers": go.get("blockers", []),
                "source_coverage_rate": source_export.get("coverage_rate"),
                "singleton_ratio": stage.get("singleton_ratio"),
            },
            "runtime_seconds": runtime_seconds,
        },
        "comparison_to_previous_baseline": compare,
        "artifact_keep_set": sorted(dict.fromkeys(artifact_keep_set)),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
