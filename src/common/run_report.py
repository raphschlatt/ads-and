from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

import pandas as pd

from src.common.config import load_resolved_config


def load_gate_config(path: str | Path = "configs/gates.yaml") -> Dict:
    return load_resolved_config(path)


def _default_gate_config() -> Dict:
    return {
        "defaults": {
            "threshold_bounds": {"min": -0.95, "max": 0.95},
            "mention_coverage_min": 1.0,
            "uid_uniqueness_max": 1,
        },
        "stages": {
            "smoke": {"f1_min": 0.80, "min_neg_val": 20, "min_neg_test": 20},
            "mini": {"f1_min": 0.88, "min_neg_val": 50, "min_neg_test": 50},
            "mid": {"f1_min": 0.90, "min_neg_val": 200, "min_neg_test": 200},
            "full": {"f1_min": 0.90, "min_neg_val": 500, "min_neg_test": 500},
        },
    }


def load_stage_metrics(metrics_file: str | Path) -> Dict:
    p = Path(metrics_file)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_go_no_go(stage_metrics: Dict, gate_config: Dict | None = None) -> Dict:
    gates = gate_config or _default_gate_config()
    stage = str(stage_metrics.get("stage", "smoke"))
    stage_gates = dict(gates.get("stages", {}).get(stage, {}))
    defaults = dict(gates.get("defaults", {}))

    f1_min = float(stage_gates.get("f1_min", 0.70))
    min_neg_val = int(stage_gates.get("min_neg_val", 0))
    min_neg_test = int(stage_gates.get("min_neg_test", 0))
    threshold_bounds = defaults.get("threshold_bounds", {"min": -0.95, "max": 0.95})
    thr_min = float(threshold_bounds.get("min", -0.95))
    thr_max = float(threshold_bounds.get("max", 0.95))
    coverage_min = float(defaults.get("mention_coverage_min", 1.0))
    uid_uniqueness_max = int(defaults.get("uid_uniqueness_max", 1))

    checks: List[Dict] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add_check(
        "schema_valid",
        stage_metrics.get("schema_valid", False),
        "All required schemas validated" if stage_metrics.get("schema_valid") else "Schema check failed",
    )
    add_check(
        "determinism_valid",
        stage_metrics.get("determinism_valid", False),
        "Determinism checks passed" if stage_metrics.get("determinism_valid") else "Determinism mismatch",
    )
    add_check(
        "uid_uniqueness",
        stage_metrics.get("uid_uniqueness_valid", False),
        "Each mention_id mapped to one author_uid" if stage_metrics.get("uid_uniqueness_valid") else "UID mapping issue",
    )
    add_check(
        "run_id_consistent",
        stage_metrics.get("run_id_consistent", False),
        "Run ID consistent across stage artifacts" if stage_metrics.get("run_id_consistent") else "Run ID mismatch",
    )

    mention_coverage = stage_metrics.get("mention_coverage")
    add_check(
        "mention_coverage",
        mention_coverage is not None and float(mention_coverage) >= coverage_min,
        f"Observed coverage={mention_coverage}, required>={coverage_min}",
    )
    uid_uniqueness_max_observed = stage_metrics.get("uid_uniqueness_max")
    add_check(
        "uid_uniqueness_max",
        uid_uniqueness_max_observed is not None and int(uid_uniqueness_max_observed) <= uid_uniqueness_max,
        f"Observed max={uid_uniqueness_max_observed}, required<={uid_uniqueness_max}",
    )

    val_counts = stage_metrics.get("val_class_counts", {}) or {}
    test_counts = stage_metrics.get("test_class_counts", {}) or {}
    val_neg = int(val_counts.get("neg", 0))
    test_neg = int(test_counts.get("neg", 0))
    add_check(
        "min_negatives_val",
        val_neg >= min_neg_val,
        f"Observed val neg={val_neg}, required>={min_neg_val}",
    )
    add_check(
        "min_negatives_test",
        test_neg >= min_neg_test,
        f"Observed test neg={test_neg}, required>={min_neg_test}",
    )

    threshold = stage_metrics.get("threshold")
    threshold_ok = threshold is not None and thr_min <= float(threshold) <= thr_max
    add_check(
        "threshold_not_extreme",
        threshold_ok,
        f"Observed threshold={threshold}, allowed_range=[{thr_min}, {thr_max}]",
    )

    threshold_status = str(stage_metrics.get("threshold_selection_status", "unknown"))
    valid_threshold_status = {
        "ok",
        "fallback_no_labels",
        "fallback_no_positives",
        "fallback_no_negatives",
    }
    add_check(
        "threshold_selection_status",
        threshold_status in valid_threshold_status,
        f"status={threshold_status}, allowed={sorted(valid_threshold_status)}",
    )

    # If LSPO metrics are present, apply loose stage-aware sanity bounds.
    lspo_f1 = stage_metrics.get("lspo_pairwise_f1")
    if lspo_f1 is not None:
        add_check("lspo_pairwise_f1_sanity", lspo_f1 >= f1_min, f"Observed F1={lspo_f1:.4f}, required>={f1_min:.4f}")

    passed = all(c["passed"] for c in checks)
    blockers = [c["name"] for c in checks if not c["passed"]]
    return {"go": passed, "checks": checks, "blockers": blockers}


def write_go_no_go_report(result: Dict, output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return p


def summarize_block_distribution(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    # pandas compatibility: older versions do not support reset_index(names=...)
    dist = df["block_key"].value_counts().rename_axis("block_key").reset_index(name="mention_count")
    return dist.head(top_k)
