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


def _parse_seeds(report: dict[str, Any], *, label: str) -> list[int]:
    values = report.get("seeds_evaluated")
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"{label}: missing or empty seeds_evaluated.")
    return sorted(int(v) for v in values)


def _extract_variant_metrics(report: dict[str, Any], *, variant: str, label: str) -> dict[str, float]:
    variants = report.get("variants")
    if not isinstance(variants, dict):
        raise ValueError(f"{label}: missing variants object.")
    row = variants.get(variant)
    if not isinstance(row, dict):
        raise ValueError(f"{label}: variant {variant!r} not found in variants.")

    out: dict[str, float] = {}
    for key in ("f1_mean", "precision_mean", "recall_mean", "accuracy_mean"):
        value = row.get(key)
        if value is None:
            raise ValueError(f"{label}: {variant}.{key} is missing.")
        out[key] = float(value)
    return out


def _default_output(candidate_report: Path) -> Path:
    stem = candidate_report.stem
    prefix = "06_clustering_test_report__"
    if stem.startswith(prefix):
        tag = stem[len(prefix) :].strip()
        if tag:
            return candidate_report.parent / f"99_compare_cluster_report_to_baseline__{tag}.json"
    return candidate_report.parent / "99_compare_cluster_report_to_baseline.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare LSPO 06_clustering_test_report.json against baseline with hard pass/fail gate."
    )
    parser.add_argument("--baseline-report", required=True, help="Path to baseline 06_clustering_test_report.json")
    parser.add_argument("--candidate-report", required=True, help="Path to candidate 06_clustering_test_report.json")
    parser.add_argument("--variant", default="dbscan_with_constraints")
    parser.add_argument("--min-delta-f1", type=float, default=0.0)
    parser.add_argument("--max-precision-drop", type=float, default=0.001)
    parser.add_argument("--output-path", default=None, help="Output JSON path; defaults next to candidate report.")
    parser.add_argument("--output", dest="output_path_legacy", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    baseline_path = Path(args.baseline_report)
    candidate_path = Path(args.candidate_report)
    output_arg = args.output_path if args.output_path is not None else args.output_path_legacy
    output_path = Path(output_arg) if output_arg else _default_output(candidate_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = _load_json(baseline_path)
    candidate = _load_json(candidate_path)

    failures: list[str] = []
    baseline_status = str(baseline.get("status", ""))
    candidate_status = str(candidate.get("status", ""))
    if baseline_status != "ok":
        failures.append(f"baseline status is not ok: {baseline_status!r}")
    if candidate_status != "ok":
        failures.append(f"candidate status is not ok: {candidate_status!r}")

    baseline_seeds = _parse_seeds(baseline, label="baseline")
    candidate_seeds = _parse_seeds(candidate, label="candidate")
    seeds_match = baseline_seeds == candidate_seeds
    if not seeds_match:
        failures.append(f"seeds mismatch: baseline={baseline_seeds}, candidate={candidate_seeds}")

    variant = str(args.variant)
    baseline_metrics = _extract_variant_metrics(baseline, variant=variant, label="baseline")
    candidate_metrics = _extract_variant_metrics(candidate, variant=variant, label="candidate")

    delta_f1 = float(candidate_metrics["f1_mean"] - baseline_metrics["f1_mean"])
    delta_precision = float(candidate_metrics["precision_mean"] - baseline_metrics["precision_mean"])
    delta_recall = float(candidate_metrics["recall_mean"] - baseline_metrics["recall_mean"])
    delta_accuracy = float(candidate_metrics["accuracy_mean"] - baseline_metrics["accuracy_mean"])

    min_delta_f1 = float(args.min_delta_f1)
    max_precision_drop = float(args.max_precision_drop)
    if delta_f1 <= min_delta_f1:
        failures.append(
            f"delta_f1_mean={delta_f1:.6f} must be > {min_delta_f1:.6f}"
        )
    if delta_precision < (-1.0 * max_precision_drop):
        failures.append(
            f"delta_precision_mean={delta_precision:.6f} must be >= {-1.0 * max_precision_drop:.6f}"
        )

    gate_passed = len(failures) == 0
    payload = {
        "generated_utc": _utc_now(),
        "baseline_report_path": str(baseline_path),
        "candidate_report_path": str(candidate_path),
        "variant": variant,
        "baseline_run_id": baseline.get("model_run_id"),
        "candidate_run_id": candidate.get("model_run_id"),
        "baseline_status": baseline_status,
        "candidate_status": candidate_status,
        "baseline_seeds": baseline_seeds,
        "candidate_seeds": candidate_seeds,
        "seeds_match": bool(seeds_match),
        "gate": {
            "min_delta_f1": min_delta_f1,
            "max_precision_drop": max_precision_drop,
        },
        "metrics": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta": {
                "f1_mean": delta_f1,
                "precision_mean": delta_precision,
                "recall_mean": delta_recall,
                "accuracy_mean": delta_accuracy,
            },
        },
        "passed": gate_passed,
        "decision": "promote_candidate" if gate_passed else "rollback_to_baseline",
        "failures": failures,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
