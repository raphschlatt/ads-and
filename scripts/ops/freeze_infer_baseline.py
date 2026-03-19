#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DECISION_JSON = "98_infer_baseline_decision.json"
DEFAULT_DECISION_MD = "98_infer_baseline_decision.md"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_active_baseline(
    *,
    path: Path,
    candidate_run_id: str,
    candidate_dir: Path,
    manifest_path: str | None,
    compare_path: Path,
    previous_baseline_run_id: str,
) -> None:
    payload = {
        "baseline_run_id": str(candidate_run_id),
        "artifact_dir": candidate_dir.name,
        "manifest_path": manifest_path,
        "compare_report": str(compare_path),
        "previous_baseline_run_id": str(previous_baseline_run_id),
        "updated_utc": _utc_now(),
    }
    _write_json(path, payload)


def _parse_metric_gate(raw: str) -> tuple[str, float]:
    text = str(raw).strip()
    if not text or "=" not in text:
        raise ValueError(
            "Runtime metric gates must use '<metric>=<max_delta_seconds>', "
            f"got: {raw!r}"
        )
    metric, limit = text.split("=", 1)
    metric_name = metric.strip()
    if not metric_name:
        raise ValueError(f"Runtime metric name is empty in gate: {raw!r}")
    return metric_name, float(limit.strip())


def _resolve_dir(reference: str, *, metrics_root: Path) -> Path:
    candidate = Path(str(reference)).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return (metrics_root / str(reference)).resolve()


def _build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    decision = dict(payload.get("decision") or {})
    policy = dict(payload.get("policy") or {})
    observed = dict(payload.get("observed") or {})
    runtime_gate_rows = list(observed.get("runtime_metric_gates") or [])
    lines.append("# Infer Baseline Decision")
    lines.append("")
    lines.append(f"- generated_utc: `{payload['generated_utc']}`")
    lines.append(f"- baseline_run_id: `{payload['baseline_run_id']}`")
    lines.append(f"- candidate_run_id: `{payload['candidate_run_id']}`")
    lines.append(f"- decision: `{decision.get('decision')}`")
    lines.append("")
    lines.append("## Policy")
    lines.append("")
    lines.append(f"- require_go_current: `{policy.get('require_go_current')}`")
    lines.append(f"- max_abs_cluster_delta: `{policy.get('max_abs_cluster_delta')}`")
    lines.append(f"- max_changed_mentions: `{policy.get('max_changed_mentions')}`")
    lines.append(f"- require_source_coverage_rate: `{policy.get('require_source_coverage_rate')}`")
    lines.append("")
    if runtime_gate_rows:
        lines.append("## Runtime Gates")
        lines.append("")
        lines.append("| metric | max_delta_seconds | observed_delta_seconds | passed |")
        lines.append("|---|---:|---:|---:|")
        for row in runtime_gate_rows:
            observed_delta = row.get("observed_delta_seconds")
            observed_cell = "null" if observed_delta is None else f"{float(observed_delta):.6f}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("metric")),
                        f"{float(row.get('max_delta_seconds')):.6f}",
                        observed_cell,
                        str(bool(row.get("passed", False))),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.append("## Observed")
    lines.append("")
    lines.append(f"- go_current: `{observed.get('go_current')}`")
    lines.append(f"- ads_clusters_delta: `{observed.get('ads_clusters_delta')}`")
    lines.append(f"- mention_cluster_changed_mentions: `{observed.get('mention_cluster_changed_mentions')}`")
    lines.append(f"- mention_cluster_changed_blocks: `{observed.get('mention_cluster_changed_blocks')}`")
    lines.append(f"- source_coverage_rate_current: `{observed.get('source_coverage_rate_current')}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    for failure in list(decision.get("failures") or []):
        lines.append(f"- failure: {failure}")
    if not list(decision.get("failures") or []):
        lines.append("- failure: none")
    promoted_manifest = decision.get("promoted_manifest_path")
    if promoted_manifest:
        lines.append(f"- promoted_manifest_path: `{promoted_manifest}`")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    for command in list(payload.get("reproduction", {}).get("commands") or []):
        lines.append(f"- `{command}`")
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an ADS infer candidate against the active baseline and write a decision record."
    )
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument("--metrics-root", default="artifacts/exports")
    parser.add_argument("--compare-report", default=None)
    parser.add_argument("--decision-json", default=None)
    parser.add_argument("--decision-md", default=None)
    parser.add_argument("--promote-manifest-path", default=None)
    parser.add_argument("--active-baseline-path", default=None)
    parser.add_argument("--keep-artifact", action="append", default=[])
    parser.add_argument("--require-go-current", dest="require_go_current", action="store_true")
    parser.add_argument("--no-require-go-current", dest="require_go_current", action="store_false")
    parser.set_defaults(require_go_current=True)
    parser.add_argument("--max-abs-cluster-delta", type=float, default=20.0)
    parser.add_argument("--max-changed-mentions", type=int, default=100)
    parser.add_argument("--require-source-coverage-rate", type=float, default=1.0)
    parser.add_argument(
        "--runtime-metric-max-delta",
        action="append",
        default=[],
        help="Gate in the form '<metric>=<max_delta_seconds>'. Can be repeated.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = _repo_root()

    metrics_root = Path(str(args.metrics_root)).expanduser()
    if not metrics_root.is_absolute():
        metrics_root = repo_root / metrics_root
    metrics_root = metrics_root.resolve()

    baseline_dir = _resolve_dir(str(args.baseline_run_id), metrics_root=metrics_root)
    candidate_dir = _resolve_dir(str(args.candidate_run_id), metrics_root=metrics_root)
    if not baseline_dir.exists() or not baseline_dir.is_dir():
        raise FileNotFoundError(f"Baseline run directory does not exist: {baseline_dir}")
    if not candidate_dir.exists() or not candidate_dir.is_dir():
        raise FileNotFoundError(f"Candidate run directory does not exist: {candidate_dir}")

    compare_path = (
        Path(str(args.compare_report)).expanduser().resolve()
        if args.compare_report is not None
        else (candidate_dir / "99_compare_infer_to_baseline.json").resolve()
    )
    if not compare_path.exists():
        raise FileNotFoundError(
            f"Compare report not found: {compare_path}. Run compare-infer-baseline first."
        )

    decision_json = (
        Path(str(args.decision_json)).expanduser().resolve()
        if args.decision_json is not None
        else (candidate_dir / DEFAULT_DECISION_JSON).resolve()
    )
    decision_md = (
        Path(str(args.decision_md)).expanduser().resolve()
        if args.decision_md is not None
        else (candidate_dir / DEFAULT_DECISION_MD).resolve()
    )

    compare_payload = _load_json(compare_path)
    runtime_compare = dict(compare_payload.get("runtime_seconds_compare") or {})
    runtime_metrics = dict(runtime_compare.get("metrics") or {})

    failures: list[str] = []
    runtime_gate_rows: list[dict[str, Any]] = []

    if bool(args.require_go_current) and not bool(compare_payload.get("go_current", False)):
        failures.append("go_current=false")

    ads_clusters_delta = compare_payload.get("ads_clusters_delta")
    if ads_clusters_delta is None:
        failures.append("ads_clusters_delta missing in compare report")
    elif abs(float(ads_clusters_delta)) > float(args.max_abs_cluster_delta):
        failures.append(
            "ads_clusters_delta="
            f"{float(ads_clusters_delta):.6f} exceeds max_abs_cluster_delta={float(args.max_abs_cluster_delta):.6f}"
        )

    changed_mentions = compare_payload.get("mention_cluster_changed_mentions")
    if changed_mentions is None:
        failures.append("mention_cluster_changed_mentions missing in compare report")
    elif int(changed_mentions) > int(args.max_changed_mentions):
        failures.append(
            f"mention_cluster_changed_mentions={int(changed_mentions)} exceeds max_changed_mentions={int(args.max_changed_mentions)}"
        )

    source_coverage = compare_payload.get("source_coverage_rate_current")
    required_coverage = float(args.require_source_coverage_rate)
    if source_coverage is None:
        failures.append("source_coverage_rate_current missing in compare report")
    elif float(source_coverage) < required_coverage:
        failures.append(
            f"source_coverage_rate_current={float(source_coverage):.6f} < require_source_coverage_rate={required_coverage:.6f}"
        )

    metric_gates = [_parse_metric_gate(raw) for raw in list(args.runtime_metric_max_delta or [])]
    for metric_name, max_delta in metric_gates:
        metric_payload = runtime_metrics.get(metric_name)
        row: dict[str, Any] = {
            "metric": metric_name,
            "max_delta_seconds": float(max_delta),
            "observed_delta_seconds": None,
            "passed": False,
        }
        if not isinstance(metric_payload, dict):
            failures.append(f"runtime metric missing in compare report: {metric_name}")
        else:
            observed_delta = metric_payload.get("delta")
            row["observed_delta_seconds"] = observed_delta
            if observed_delta is None:
                failures.append(f"runtime metric delta missing for: {metric_name}")
            elif float(observed_delta) > float(max_delta):
                failures.append(
                    f"runtime metric {metric_name} delta={float(observed_delta):.6f} exceeds max_delta={float(max_delta):.6f}"
                )
            else:
                row["passed"] = True
        runtime_gate_rows.append(row)

    decision_name = "promote_candidate" if not failures else "keep_baseline"
    manifest_path: str | None = None
    active_baseline_path: str | None = None

    if args.promote_manifest_path is not None and not failures:
        manifest_target = Path(str(args.promote_manifest_path)).expanduser().resolve()
        command = [
            sys.executable,
            str(repo_root / "scripts" / "ops" / "write_infer_baseline_manifest.py"),
            "--run-dir",
            str(candidate_dir),
            "--manifest-path",
            str(manifest_target),
            "--compare-report",
            str(compare_path),
        ]
        for keep_artifact in list(args.keep_artifact or []):
            if str(keep_artifact).strip():
                command.extend(["--keep-artifact", str(keep_artifact).strip()])
        proc = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            text=True,
            capture_output=True,
            env={**os.environ, **{"PYTHONPATH": "."}},
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to write promoted infer baseline manifest.\n"
                f"command: {' '.join(command)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        manifest_path = str(manifest_target)

    if args.active_baseline_path is not None and not failures:
        active_target = Path(str(args.active_baseline_path)).expanduser().resolve()
        _write_active_baseline(
            path=active_target,
            candidate_run_id=str(args.candidate_run_id),
            candidate_dir=candidate_dir,
            manifest_path=manifest_path,
            compare_path=compare_path,
            previous_baseline_run_id=str(args.baseline_run_id),
        )
        active_baseline_path = str(active_target)

    payload: dict[str, Any] = {
        "baseline_run_id": str(args.baseline_run_id),
        "candidate_run_id": str(args.candidate_run_id),
        "generated_utc": _utc_now(),
        "compare_scope": str(compare_payload.get("compare_scope", "infer")),
        "policy": {
            "require_go_current": bool(args.require_go_current),
            "max_abs_cluster_delta": float(args.max_abs_cluster_delta),
            "max_changed_mentions": int(args.max_changed_mentions),
            "require_source_coverage_rate": required_coverage,
            "runtime_metric_max_delta": {metric: limit for metric, limit in metric_gates},
        },
        "references": {
            "metrics_root": str(metrics_root),
            "baseline_dir": str(baseline_dir),
            "candidate_dir": str(candidate_dir),
            "compare_report": str(compare_path),
            "decision_json": str(decision_json),
            "decision_markdown": str(decision_md),
            "promote_manifest_path": manifest_path,
            "active_baseline_path": active_baseline_path,
        },
        "observed": {
            "go_baseline": compare_payload.get("go_baseline"),
            "go_current": compare_payload.get("go_current"),
            "warnings_baseline": compare_payload.get("warnings_baseline", []),
            "warnings_current": compare_payload.get("warnings_current", []),
            "blockers_current": compare_payload.get("blockers_current", []),
            "ads_mentions_delta": compare_payload.get("ads_mentions_delta"),
            "ads_clusters_delta": ads_clusters_delta,
            "ads_cluster_assignments_delta": compare_payload.get("ads_cluster_assignments_delta"),
            "singleton_ratio_delta": compare_payload.get("singleton_ratio_delta"),
            "source_coverage_rate_current": source_coverage,
            "mention_cluster_compare_status": compare_payload.get("mention_cluster_compare_status"),
            "mention_cluster_changed_mentions": changed_mentions,
            "mention_cluster_changed_blocks": compare_payload.get("mention_cluster_changed_blocks"),
            "mention_cluster_top_changed_blocks": compare_payload.get("mention_cluster_top_changed_blocks", []),
            "runtime_metric_gates": runtime_gate_rows,
        },
        "decision": {
            "decision": decision_name,
            "passed": len(failures) == 0,
            "failures": failures,
            "promoted_manifest_path": manifest_path,
        },
        "reproduction": {
            "commands": [
                "author-name-disambiguation compare-infer-baseline "
                f"--baseline-run-id {args.baseline_run_id} "
                f"--current-run-id {args.candidate_run_id} "
                f"--metrics-root {metrics_root}",
                "python scripts/ops/freeze_infer_baseline.py "
                f"--baseline-run-id {args.baseline_run_id} "
                f"--candidate-run-id {args.candidate_run_id} "
                f"--metrics-root {metrics_root}",
            ],
        },
    }

    _write_json(decision_json, payload)
    _write_text(decision_md, _build_markdown(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())