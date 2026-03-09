from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from author_name_disambiguation.common.config import load_resolved_config


def load_gate_config(path: str | Path = "configs/gates.yaml") -> Dict:
    return load_resolved_config(path)


def _default_gate_config() -> Dict:
    return {
        "defaults": {
            "threshold_bounds": {"min": -0.95, "max": 0.95},
            "mention_coverage_min": 1.0,
            "uid_uniqueness_max": 1,
            "cluster_quality": {
                "singleton_ratio_max": 0.45,
                "split_high_sim_rate_probe_max": 0.20,
            },
            "split_balance_policy": {
                "infeasible_severity": "blocker",
                "degraded_severity": "warning",
            },
            "eps_policy": {
                "boundary_hit_severity": "warning",
                "range_limited_delta_f1": 0.005,
                "range_limited_severity": "warning",
            },
        },
        "stages": {
            "smoke": {"f1_min": 0.80, "min_neg_val": 20, "min_neg_test": 20, "cluster_quality_severity": "warning"},
            "mini": {"f1_min": 0.88, "min_neg_val": 50, "min_neg_test": 50, "cluster_quality_severity": "warning"},
            "mid": {
                "f1_min": 0.90,
                "min_neg_val": 200,
                "min_neg_test": 200,
                "cluster_quality_severity": "blocker",
                "lspo_block_size_p95_min": 2,
                "lspo_pairs_min": 50000,
                "eps_range_limited_severity": "blocker",
            },
            "full": {
                "f1_min": 0.90,
                "min_neg_val": 500,
                "min_neg_test": 500,
                "cluster_quality_severity": "blocker",
                "lspo_block_size_p95_min": 2,
                "lspo_pairs_min": 300000,
                "eps_range_limited_severity": "blocker",
            },
            "infer_sources": {
                "f1_min": 0.0,
                "min_neg_val": 0,
                "min_neg_test": 0,
                "cluster_quality_severity": "blocker",
                "singleton_ratio_severity": "warning",
                "eps_range_limited_severity": "warning",
            },
        },
    }


def load_stage_metrics(metrics_file: str | Path) -> Dict:
    p = Path(metrics_file)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_severity(value: Any, default: str = "blocker") -> str:
    raw = str(value if value is not None else default).strip().lower()
    if raw in {"block", "blocker", "hard"}:
        return "blocker"
    if raw in {"warn", "warning", "soft"}:
        return "warning"
    return "blocker" if default == "blocker" else "warning"


def evaluate_go_no_go(stage_metrics: Dict, gate_config: Dict | None = None) -> Dict:
    gates = gate_config or _default_gate_config()
    stage = str(stage_metrics.get("stage", "smoke"))
    metric_scope = str(stage_metrics.get("metric_scope", "") or "").strip().lower()
    is_train_scope = metric_scope == "train"
    is_infer_scope = metric_scope == "infer" or stage == "infer_sources"
    scope_key = "infer" if is_infer_scope else "train" if is_train_scope else ""
    scoped = gates.get(scope_key) if scope_key else None
    if isinstance(scoped, dict):
        defaults = dict(gates.get("defaults", {}))
        defaults.update(dict(scoped.get("defaults", {})))
        stage_gates = dict(gates.get("stages", {}).get(stage, {}))
        stage_gates.update(dict(scoped.get("stages", {}).get(stage, {})))
    else:
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
    cluster_defaults = dict(defaults.get("cluster_quality", {}) or {})
    singleton_ratio_max = float(stage_gates.get("singleton_ratio_max", cluster_defaults.get("singleton_ratio_max", 1.0)))
    split_high_sim_rate_probe_max = float(
        stage_gates.get("split_high_sim_rate_probe_max", cluster_defaults.get("split_high_sim_rate_probe_max", 1.0))
    )
    cluster_quality_severity = _normalize_severity(stage_gates.get("cluster_quality_severity", "warning"), default="warning")
    singleton_ratio_severity = _normalize_severity(
        stage_gates.get("singleton_ratio_severity", cluster_quality_severity),
        default=cluster_quality_severity,
    )
    lspo_block_size_p95_min = stage_gates.get("lspo_block_size_p95_min")
    lspo_pairs_min = stage_gates.get("lspo_pairs_min")
    lspo_block_size_p95_severity = _normalize_severity(
        stage_gates.get("lspo_block_size_p95_severity", "blocker"),
        default="blocker",
    )
    lspo_pairs_min_severity = _normalize_severity(stage_gates.get("lspo_pairs_min_severity", "blocker"), default="blocker")

    split_defaults = dict(defaults.get("split_balance_policy", {}) or {})
    split_infeasible_severity = _normalize_severity(
        stage_gates.get("split_balance_infeasible_severity", split_defaults.get("infeasible_severity", "blocker")),
        default="blocker",
    )
    split_degraded_severity = _normalize_severity(
        stage_gates.get("split_balance_degraded_severity", split_defaults.get("degraded_severity", cluster_quality_severity)),
        default=cluster_quality_severity,
    )

    eps_defaults = dict(defaults.get("eps_policy", {}) or {})
    eps_boundary_severity = _normalize_severity(
        stage_gates.get("eps_boundary_severity", eps_defaults.get("boundary_hit_severity", "warning")),
        default="warning",
    )
    eps_range_limited_severity = _normalize_severity(
        stage_gates.get("eps_range_limited_severity", eps_defaults.get("range_limited_severity", "warning")),
        default="warning",
    )
    eps_range_limited_delta_f1 = float(
        stage_gates.get("eps_range_limited_delta_f1", eps_defaults.get("range_limited_delta_f1", 0.005))
    )
    split_feasibility_severity = _normalize_severity(stage_gates.get("split_feasibility_severity", "blocker"), default="blocker")

    checks: List[Dict] = []

    def add_check(name: str, passed: bool, detail: str, severity: str = "blocker") -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "detail": detail,
                "severity": _normalize_severity(severity, default="blocker"),
            }
        )

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

    if is_train_scope:
        add_check("mention_coverage", True, "not applicable for train scope")
        add_check("uid_uniqueness_max", True, "not applicable for train scope")
        add_check("uid_local_to_global_valid", True, "not applicable for train scope")
    else:
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
        uid_local_to_global_valid = stage_metrics.get("uid_local_to_global_valid")
        uid_local_to_global_max_nunique = stage_metrics.get("uid_local_to_global_max_nunique")
        if uid_local_to_global_valid is None:
            add_check("uid_local_to_global_valid", True, "not available")
        else:
            add_check(
                "uid_local_to_global_valid",
                bool(uid_local_to_global_valid),
                (
                    "Observed uid_local_to_global_valid="
                    f"{bool(uid_local_to_global_valid)} "
                    f"(max_nunique={uid_local_to_global_max_nunique})"
                ),
                severity="blocker",
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
        "bundle_manifest",
    }
    add_check(
        "threshold_selection_status",
        threshold_status in valid_threshold_status,
        f"status={threshold_status}, allowed={sorted(valid_threshold_status)}",
    )

    split_status = str(stage_metrics.get("split_balance_status", "") or "").strip().lower()
    if is_infer_scope:
        add_check("split_balance_status", True, "not applicable for infer scope")
    elif split_status in {"", "unknown"}:
        add_check("split_balance_status", True, "status unavailable (legacy or missing split metadata)")
    elif split_status == "split_balance_infeasible":
        add_check(
            "split_balance_status",
            False,
            f"status={split_status}",
            severity=split_infeasible_severity,
        )
    elif split_status == "split_balance_degraded":
        add_check(
            "split_balance_status",
            False,
            f"status={split_status}",
            severity=split_degraded_severity,
        )
    else:
        add_check("split_balance_status", True, f"status={split_status}")

    max_possible_neg_total = stage_metrics.get("max_possible_neg_total")
    required_neg_total = stage_metrics.get("required_neg_total")
    if is_infer_scope:
        add_check("split_neg_feasible", True, "not applicable for infer scope")
    elif max_possible_neg_total is None or required_neg_total is None:
        add_check("split_neg_feasible", True, "not available")
    else:
        max_possible_neg_total = int(max_possible_neg_total)
        required_neg_total = int(required_neg_total)
        add_check(
            "split_neg_feasible",
            max_possible_neg_total >= required_neg_total,
            (
                f"Observed max_possible_neg_total={max_possible_neg_total}, "
                f"required_neg_total={required_neg_total}"
            ),
            severity=split_feasibility_severity,
        )

    pair_score_range_ok = stage_metrics.get("pair_score_range_ok")
    memory_feasible = stage_metrics.get("memory_feasible")
    if memory_feasible is None:
        add_check("memory_feasible", True, "not available")
    else:
        add_check(
            "memory_feasible",
            bool(memory_feasible),
            f"Observed memory_feasible={bool(memory_feasible)}",
            severity="blocker",
        )

    if pair_score_range_ok is None:
        add_check("pair_score_range_ok", True, "not available")
    else:
        add_check(
            "pair_score_range_ok",
            bool(pair_score_range_ok),
            f"Observed pair_score_range_ok={bool(pair_score_range_ok)}",
            severity=cluster_quality_severity,
        )

    singleton_ratio = stage_metrics.get("singleton_ratio")
    if singleton_ratio is None:
        add_check("singleton_ratio", True, "not available")
    else:
        singleton_ratio = float(singleton_ratio)
        add_check(
            "singleton_ratio",
            singleton_ratio <= singleton_ratio_max,
            f"Observed singleton_ratio={singleton_ratio:.4f}, required<={singleton_ratio_max:.4f}",
            severity=singleton_ratio_severity,
        )

    split_high_sim_rate_probe = stage_metrics.get("split_high_sim_rate_probe")
    if split_high_sim_rate_probe is None:
        add_check("split_high_sim_rate_probe", True, "not available")
    else:
        split_high_sim_rate_probe = float(split_high_sim_rate_probe)
        add_check(
            "split_high_sim_rate_probe",
            split_high_sim_rate_probe <= split_high_sim_rate_probe_max,
            (
                f"Observed split_high_sim_rate_probe={split_high_sim_rate_probe:.4f}, "
                f"required<={split_high_sim_rate_probe_max:.4f}"
            ),
            severity=cluster_quality_severity,
        )

    lspo_block_size_p95 = stage_metrics.get("lspo_block_size_p95")
    if lspo_block_size_p95_min is not None:
        if lspo_block_size_p95 is None:
            add_check("lspo_block_size_p95", False, "not available", severity=lspo_block_size_p95_severity)
        else:
            lspo_block_size_p95 = float(lspo_block_size_p95)
            lspo_block_size_p95_min = float(lspo_block_size_p95_min)
            add_check(
                "lspo_block_size_p95",
                lspo_block_size_p95 >= lspo_block_size_p95_min,
                f"Observed lspo_block_size_p95={lspo_block_size_p95:.4f}, required>={lspo_block_size_p95_min:.4f}",
                severity=lspo_block_size_p95_severity,
            )

    lspo_pairs = stage_metrics.get("lspo_pairs")
    if lspo_pairs_min is not None:
        if lspo_pairs is None:
            add_check("lspo_pairs", False, "not available", severity=lspo_pairs_min_severity)
        else:
            lspo_pairs = int(lspo_pairs)
            lspo_pairs_min = int(lspo_pairs_min)
            add_check(
                "lspo_pairs",
                lspo_pairs >= lspo_pairs_min,
                f"Observed lspo_pairs={lspo_pairs}, required>={lspo_pairs_min}",
                severity=lspo_pairs_min_severity,
            )

    eps_boundary_hit = stage_metrics.get("eps_boundary_hit")
    if eps_boundary_hit is None:
        add_check("eps_boundary_hit", True, "not available")
    elif bool(eps_boundary_hit):
        boundary_side = stage_metrics.get("eps_boundary_side")
        add_check(
            "eps_boundary_hit",
            False,
            f"Observed boundary_hit=true (side={boundary_side})",
            severity=eps_boundary_severity,
        )
    else:
        add_check("eps_boundary_hit", True, "Observed boundary_hit=false")

    eps_range_limited = stage_metrics.get("eps_range_limited")
    eps_diag_delta_f1 = stage_metrics.get("eps_diag_delta_f1")
    if eps_range_limited is None:
        add_check("eps_range_limited", True, "not available")
    elif bool(eps_range_limited):
        detail = f"Observed range_limited=true (diag_delta_f1={eps_diag_delta_f1}, threshold>={eps_range_limited_delta_f1:.4f})"
        add_check(
            "eps_range_limited",
            False,
            detail,
            severity=eps_range_limited_severity,
        )
    else:
        add_check("eps_range_limited", True, "Observed range_limited=false")

    # If LSPO metrics are present, apply loose stage-aware sanity bounds.
    lspo_f1 = stage_metrics.get("lspo_pairwise_f1")
    if lspo_f1 is not None:
        add_check("lspo_pairwise_f1_sanity", lspo_f1 >= f1_min, f"Observed F1={lspo_f1:.4f}, required>={f1_min:.4f}")

    blockers = [c["name"] for c in checks if (not c["passed"]) and c["severity"] == "blocker"]
    warnings = [c["name"] for c in checks if (not c["passed"]) and c["severity"] == "warning"]
    passed = len(blockers) == 0
    return {"go": passed, "checks": checks, "blockers": blockers, "warnings": warnings}


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
