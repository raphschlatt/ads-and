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


DEFAULT_BASELINE_RUN_ID = "full_20260218T111506Z_cli02681429"
DEFAULT_TAGS = ("epsbkt_v1", "epsbkt_v2", "epsbkt_v3")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _parse_tags(raw: str) -> list[str]:
    out = [t.strip() for t in str(raw).split(",") if t.strip()]
    if not out:
        raise ValueError("--tags must include at least one tag.")
    return out


def _run_baseline_integrity_check(*, repo_root: Path, run_id: str, paths_config: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "ops" / "check_baseline_integrity.py"),
        "--baseline-run-id",
        run_id,
        "--paths-config",
        paths_config,
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, **{"PYTHONPATH": "."}},
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Baseline integrity check failed before freeze.\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    payload = json.loads(proc.stdout)
    if not bool(payload.get("ok", False)):
        raise RuntimeError(f"Baseline integrity returned ok=false: {payload}")
    return payload


def _extract_variant_metrics(report: dict[str, Any], *, variant: str) -> dict[str, float]:
    variants = report.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("Missing variants object in report.")
    row = variants.get(variant)
    if not isinstance(row, dict):
        raise ValueError(f"Variant {variant!r} missing in report.")
    out: dict[str, float] = {}
    for key in ("accuracy_mean", "precision_mean", "recall_mean", "f1_mean"):
        value = row.get(key)
        if value is None:
            raise ValueError(f"Missing {key!r} in report variant {variant!r}.")
        out[key] = float(value)
    return out


def _parse_seeds(report: dict[str, Any], *, label: str) -> list[int]:
    values = report.get("seeds_evaluated")
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"{label}: missing or empty seeds_evaluated.")
    return sorted(int(v) for v in values)


def _build_markdown(manifest: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# EPS Experiments Freeze Manifest")
    lines.append("")
    lines.append(f"- baseline_run_id: `{manifest['baseline_run_id']}`")
    lines.append(f"- generated_utc: `{manifest['generated_utc']}`")
    lines.append(f"- active_variant: `{manifest['decision']['active_variant']}`")
    lines.append(f"- promoted_variant: `{manifest['decision']['promoted_variant']}`")
    lines.append("")
    lines.append("## Gate Policy")
    lines.append("")
    gate = manifest["policy"]["gate"]
    lines.append(f"- min_delta_f1: `{gate['min_delta_f1']}`")
    lines.append(f"- max_precision_drop: `{gate['max_precision_drop']}`")
    lines.append("")
    lines.append("## Metrics (with constraints)")
    lines.append("")
    lines.append("| variant | status | passed_gate | decision | delta_f1 | delta_precision | delta_recall | delta_accuracy |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|")
    for row in manifest["variants"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["report_status"]),
                    str(row["gate_passed"]),
                    str(row["gate_decision"]),
                    f"{row['delta_vs_base']['f1_mean']:.9f}",
                    f"{row['delta_vs_base']['precision_mean']:.9f}",
                    f"{row['delta_vs_base']['recall_mean']:.9f}",
                    f"{row['delta_vs_base']['accuracy_mean']:.9f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- reason: {manifest['decision']['decision_reason']}")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    for item in manifest["reproduction"]["commands"]:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze EPS experiment outcomes and mark baseline as active (soft rollback)."
    )
    parser.add_argument("--baseline-run-id", default=DEFAULT_BASELINE_RUN_ID)
    parser.add_argument("--metrics-root", default="artifacts/metrics")
    parser.add_argument("--paths-config", default="configs/paths.local.yaml")
    parser.add_argument("--tags", default=",".join(DEFAULT_TAGS))
    parser.add_argument("--variant", default="dbscan_with_constraints")
    parser.add_argument("--min-delta-f1", type=float, default=0.0)
    parser.add_argument("--max-precision-drop", type=float, default=0.001)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = _repo_root()
    baseline_run_id = str(args.baseline_run_id).strip()
    variant = str(args.variant).strip()
    tags = _parse_tags(args.tags)

    integrity = _run_baseline_integrity_check(
        repo_root=repo_root,
        run_id=baseline_run_id,
        paths_config=str(args.paths_config),
    )

    metrics_root = Path(str(args.metrics_root))
    if not metrics_root.is_absolute():
        metrics_root = repo_root / metrics_root
    metrics_dir = metrics_root / baseline_run_id
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics dir not found: {metrics_dir}")

    base_report_path = metrics_dir / "06_clustering_test_report.json"
    if not base_report_path.exists():
        raise FileNotFoundError(f"Missing baseline report: {base_report_path}")
    base_report = _load_json(base_report_path)
    base_status = str(base_report.get("status", "")).strip().lower()
    if base_status != "ok":
        raise ValueError(f"Baseline report status must be 'ok', got: {base_status!r}")
    base_seeds = _parse_seeds(base_report, label="base")

    base_metrics = _extract_variant_metrics(base_report, variant=variant)

    variants_payload: list[dict[str, Any]] = []
    references: dict[str, str] = {
        "base_report": str(base_report_path),
    }

    variants_payload.append(
        {
            "variant": "base",
            "tag": None,
            "report_path": str(base_report_path),
            "compare_path": None,
            "report_status": str(base_report.get("status", "")),
            "gate_passed": None,
            "gate_decision": None,
            "gate_failures": [],
            "metrics": base_metrics,
            "delta_vs_base": {
                "accuracy_mean": 0.0,
                "precision_mean": 0.0,
                "recall_mean": 0.0,
                "f1_mean": 0.0,
            },
        }
    )

    reproduction_commands: list[str] = []
    any_promoted = False
    non_promoted_tags: list[str] = []

    for tag in tags:
        candidate_report_path = metrics_dir / f"06_clustering_test_report__{tag}.json"
        compare_path = metrics_dir / f"99_compare_cluster_report_to_baseline__{tag}.json"
        if not candidate_report_path.exists():
            raise FileNotFoundError(f"Missing candidate report for tag={tag}: {candidate_report_path}")
        if not compare_path.exists():
            raise FileNotFoundError(f"Missing compare report for tag={tag}: {compare_path}")

        candidate_report = _load_json(candidate_report_path)
        compare_report = _load_json(compare_path)
        cand_status = str(candidate_report.get("status", "")).strip().lower()
        if cand_status != "ok":
            raise ValueError(f"Candidate report status must be 'ok' for tag={tag}, got: {cand_status!r}")
        cand_seeds = _parse_seeds(candidate_report, label=f"candidate:{tag}")
        if cand_seeds != base_seeds:
            raise ValueError(
                f"Seed mismatch for tag={tag}: candidate={cand_seeds}, baseline={base_seeds}"
            )

        candidate_metrics = _extract_variant_metrics(candidate_report, variant=variant)
        delta = {
            key: float(candidate_metrics[key] - base_metrics[key])
            for key in ("accuracy_mean", "precision_mean", "recall_mean", "f1_mean")
        }
        compare_seeds_match = bool(compare_report.get("seeds_match", False))
        if not compare_seeds_match:
            raise ValueError(f"Compare report indicates seeds mismatch for tag={tag}: {compare_path}")
        gate_passed = bool(compare_report.get("passed", False))
        gate_decision = str(compare_report.get("decision", ""))
        failures = list(compare_report.get("failures", []) or [])

        variants_payload.append(
            {
                "variant": tag,
                "tag": tag,
                "report_path": str(candidate_report_path),
                "compare_path": str(compare_path),
                "report_status": str(candidate_report.get("status", "")),
                "gate_passed": gate_passed,
                "gate_decision": gate_decision,
                "gate_failures": failures,
                "metrics": candidate_metrics,
                "delta_vs_base": delta,
            }
        )
        references[f"{tag}_report"] = str(candidate_report_path)
        references[f"{tag}_compare"] = str(compare_path)

        override_path = str(candidate_report.get("cluster_config_override_path", "")).strip()
        if override_path:
            reproduction_commands.append(
                "author-name-disambiguation run-cluster-test-report "
                f"--model-run-id {baseline_run_id} "
                "--paths-config configs/paths.local.yaml "
                f"--cluster-config-override {override_path} "
                f"--report-tag {tag} --device auto --precision-mode fp32"
            )
        reproduction_commands.append(
            "PYTHONPATH=. python scripts/ops/compare_cluster_test_reports.py "
            f"--baseline-report {base_report_path} "
            f"--candidate-report {candidate_report_path} "
            f"--variant {variant} "
            f"--min-delta-f1 {float(args.min_delta_f1)} "
            f"--max-precision-drop {float(args.max_precision_drop)} "
            f"--output {compare_path}"
        )

        if gate_passed:
            any_promoted = True
        else:
            non_promoted_tags.append(tag)

    decision_reason = (
        "all candidates failed gate (delta_f1 <= 0 or precision drop below threshold)"
        if not any_promoted
        else "at least one candidate passed gate; manual promotion required"
    )

    manifest = {
        "baseline_run_id": baseline_run_id,
        "generated_utc": _utc_now(),
        "policy": {
            "name": "f1_must_increase_and_precision_safe",
            "variant": variant,
            "gate": {
                "min_delta_f1": float(args.min_delta_f1),
                "max_precision_drop": float(args.max_precision_drop),
            },
        },
        "integrity_preflight": integrity,
        "references": references,
        "variants": variants_payload,
        "decision": {
            "promoted_variant": None,
            "active_variant": "baseline",
            "decision_reason": decision_reason,
            "experiments_not_promoted": non_promoted_tags,
        },
        "reproduction": {
            "commands": reproduction_commands,
        },
    }

    manifest_json_path = metrics_dir / "98_eps_experiments_manifest.json"
    manifest_md_path = metrics_dir / "98_eps_experiments_manifest.md"
    active_baseline_path = metrics_dir / "98_active_baseline.json"

    _write_json(manifest_json_path, manifest)
    manifest_md_path.write_text(_build_markdown(manifest), encoding="utf-8")
    _write_json(
        active_baseline_path,
        {
            "baseline_run_id": baseline_run_id,
            "active_variant": "baseline",
            "active_cluster_report": "06_clustering_test_report.json",
            "experiments_not_promoted": non_promoted_tags,
            "updated_utc": _utc_now(),
        },
    )

    print(
        json.dumps(
            {
                "ok": True,
                "baseline_run_id": baseline_run_id,
                "manifest_json": str(manifest_json_path),
                "manifest_md": str(manifest_md_path),
                "active_baseline_json": str(active_baseline_path),
                "non_promoted_tags": non_promoted_tags,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
