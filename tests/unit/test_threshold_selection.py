import numpy as np

from src.approaches.nand.train import _compute_best_threshold


def test_threshold_fallback_no_positives():
    sim = np.array([0.1, -0.2, 0.4], dtype=np.float32)
    labels = np.array([0, 0, 0], dtype=np.int64)
    thr, stats, status, source = _compute_best_threshold(sim, labels, default_threshold=0.35)

    assert thr == 0.35
    assert status == "fallback_no_positives"
    assert source == "fallback_default"
    assert "f1" in stats


def test_threshold_fallback_no_negatives():
    sim = np.array([0.1, 0.2, 0.9], dtype=np.float32)
    labels = np.array([1, 1, 1], dtype=np.int64)
    thr, stats, status, source = _compute_best_threshold(sim, labels, default_threshold=0.35)

    assert thr == 0.35
    assert status == "fallback_no_negatives"
    assert source == "fallback_default"
    assert "f1" in stats


def test_threshold_tie_breaker_prefers_non_edge_values():
    # Many thresholds in (-0.9, 0.9) yield perfect F1.
    # Tie-break should prefer threshold furthest from edges, i.e. around 0.0.
    sim = np.array([0.9, -0.9], dtype=np.float32)
    labels = np.array([1, 0], dtype=np.int64)
    thr, stats, status, source = _compute_best_threshold(sim, labels, default_threshold=0.35)

    assert abs(thr) < 1e-6
    assert status == "ok"
    assert source == "val_f1_opt"
    assert stats["f1"] == 1.0
