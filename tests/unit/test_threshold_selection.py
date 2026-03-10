import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from author_name_disambiguation.approaches.nand.train import _compute_best_threshold


def _compute_best_threshold_reference(
    sim: np.ndarray,
    labels: np.ndarray,
    default_threshold: float = 0.35,
):
    has_pos = bool((labels == 1).any())
    has_neg = bool((labels == 0).any())

    if not has_pos and not has_neg:
        return float(default_threshold), {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}, "fallback_no_labels", "fallback_default"
    if not has_pos:
        pred = (sim >= default_threshold).astype(int)
        stats = {
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
            "accuracy": float(accuracy_score(labels, pred)),
        }
        return float(default_threshold), stats, "fallback_no_positives", "fallback_default"
    if not has_neg:
        pred = (sim >= default_threshold).astype(int)
        stats = {
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
            "accuracy": float(accuracy_score(labels, pred)),
        }
        return float(default_threshold), stats, "fallback_no_negatives", "fallback_default"

    thresholds = np.linspace(-1.0, 1.0, num=2001)
    best_key = (-1.0, -1.0)
    best_thr = float(default_threshold)
    best_stats = {}
    for thr in thresholds:
        pred = (sim >= thr).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        edge_margin = min(float(thr + 1.0), float(1.0 - thr))
        key = (float(f1), edge_margin)
        if key > best_key:
            best_key = key
            best_thr = float(thr)
            best_stats = {
                "f1": float(f1),
                "precision": float(precision_score(labels, pred, zero_division=0)),
                "recall": float(recall_score(labels, pred, zero_division=0)),
                "accuracy": float(accuracy_score(labels, pred)),
            }
    return best_thr, best_stats, "ok", "val_f1_opt"


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


def test_threshold_fast_path_matches_reference_grid_sweep():
    rng = np.random.default_rng(0)
    sim = rng.uniform(-1.0, 1.0, size=512).astype(np.float32)
    labels = rng.integers(0, 2, size=512, dtype=np.int64)

    thr, stats, status, source = _compute_best_threshold(sim, labels, default_threshold=0.35)
    ref_thr, ref_stats, ref_status, ref_source = _compute_best_threshold_reference(sim, labels, default_threshold=0.35)

    assert thr == ref_thr
    assert status == ref_status
    assert source == ref_source
    assert stats["f1"] == pytest.approx(ref_stats["f1"], rel=1e-12, abs=1e-12)
    assert stats["precision"] == pytest.approx(ref_stats["precision"], rel=1e-12, abs=1e-12)
    assert stats["recall"] == pytest.approx(ref_stats["recall"], rel=1e-12, abs=1e-12)
    assert stats["accuracy"] == pytest.approx(ref_stats["accuracy"], rel=1e-12, abs=1e-12)
