from src.common.run_report import evaluate_go_no_go


def _base_metrics():
    return {
        "run_id": "smoke_abc",
        "stage": "smoke",
        "schema_valid": True,
        "determinism_valid": True,
        "uid_uniqueness_valid": True,
        "uid_uniqueness_max": 1,
        "run_id_consistent": True,
        "mention_coverage": 1.0,
        "lspo_pairwise_f1": 0.90,
        "threshold": 0.35,
        "threshold_selection_status": "ok",
        "val_class_counts": {"pos": 12, "neg": 25},
        "test_class_counts": {"pos": 10, "neg": 22},
        "split_balance_status": "ok",
        "pair_score_range_ok": True,
        "singleton_ratio": 0.10,
        "split_high_sim_rate_probe": 0.05,
        "eps_boundary_hit": False,
        "lspo_block_size_p95": 3.5,
        "lspo_pairs": 1000,
        "max_possible_neg_total": 1000,
        "required_neg_total": 40,
    }


def _gate_cfg():
    return {
        "defaults": {
            "threshold_bounds": {"min": -0.95, "max": 0.95},
            "mention_coverage_min": 1.0,
            "uid_uniqueness_max": 1,
            "cluster_quality": {
                "singleton_ratio_max": 0.30,
                "split_high_sim_rate_probe_max": 0.15,
            },
            "split_balance_policy": {
                "infeasible_severity": "blocker",
                "degraded_severity": "warning",
            },
            "eps_policy": {
                "boundary_hit_severity": "warning",
            },
        },
        "stages": {
            "smoke": {
                "f1_min": 0.80,
                "min_neg_val": 0,
                "min_neg_test": 0,
                "cluster_quality_severity": "warning",
                "split_balance_degraded_severity": "warning",
            },
            "mid": {
                "f1_min": 0.80,
                "min_neg_val": 0,
                "min_neg_test": 0,
                "cluster_quality_severity": "blocker",
                "split_balance_degraded_severity": "blocker",
                "lspo_block_size_p95_min": 2.0,
                "lspo_pairs_min": 500,
            },
        },
    }


def test_go_no_go_passes_with_sufficient_metrics():
    go = evaluate_go_no_go(_base_metrics(), gate_config=_gate_cfg())
    assert go["go"] is True
    assert go["blockers"] == []
    assert go["warnings"] == []


def test_go_no_go_fails_when_negatives_missing():
    metrics = _base_metrics()
    metrics["val_class_counts"] = {"pos": 10, "neg": 0}
    metrics["test_class_counts"] = {"pos": 10, "neg": 0}

    go = evaluate_go_no_go(metrics)
    assert go["go"] is False
    assert "min_negatives_val" in go["blockers"]
    assert "min_negatives_test" in go["blockers"]


def test_go_no_go_warns_for_cluster_quality_on_smoke():
    metrics = _base_metrics()
    metrics["singleton_ratio"] = 0.95

    go = evaluate_go_no_go(metrics, gate_config=_gate_cfg())
    assert go["go"] is True
    assert "singleton_ratio" not in go["blockers"]
    assert "singleton_ratio" in go["warnings"]


def test_go_no_go_blocks_for_cluster_quality_on_mid():
    metrics = _base_metrics()
    metrics["stage"] = "mid"
    metrics["singleton_ratio"] = 0.95

    go = evaluate_go_no_go(metrics, gate_config=_gate_cfg())
    assert go["go"] is False
    assert "singleton_ratio" in go["blockers"]


def test_go_no_go_split_infeasible_is_always_blocker():
    metrics = _base_metrics()
    metrics["split_balance_status"] = "split_balance_infeasible"

    go = evaluate_go_no_go(metrics, gate_config=_gate_cfg())
    assert go["go"] is False
    assert "split_balance_status" in go["blockers"]


def test_go_no_go_eps_boundary_is_warning():
    metrics = _base_metrics()
    metrics["eps_boundary_hit"] = True
    metrics["eps_boundary_side"] = "max"

    go = evaluate_go_no_go(metrics, gate_config=_gate_cfg())
    assert go["go"] is True
    assert "eps_boundary_hit" in go["warnings"]


def test_go_no_go_blocks_when_split_feasibility_fails():
    metrics = _base_metrics()
    metrics["max_possible_neg_total"] = 10
    metrics["required_neg_total"] = 40

    go = evaluate_go_no_go(metrics, gate_config=_gate_cfg())
    assert go["go"] is False
    assert "split_neg_feasible" in go["blockers"]
