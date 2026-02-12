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
        "val_class_counts": {"pos": 12, "neg": 15},
        "test_class_counts": {"pos": 10, "neg": 12},
    }


def test_go_no_go_passes_with_sufficient_metrics():
    go = evaluate_go_no_go(_base_metrics())
    assert go["go"] is True
    assert go["blockers"] == []


def test_go_no_go_fails_when_negatives_missing():
    metrics = _base_metrics()
    metrics["val_class_counts"] = {"pos": 10, "neg": 0}
    metrics["test_class_counts"] = {"pos": 10, "neg": 0}

    go = evaluate_go_no_go(metrics)
    assert go["go"] is False
    assert "min_negatives_val" in go["blockers"]
    assert "min_negatives_test" in go["blockers"]
