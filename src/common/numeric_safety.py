from __future__ import annotations

from typing import Any

import numpy as np


def _safe_min_max(arr: np.ndarray) -> tuple[float | None, float | None]:
    if arr.size == 0:
        return None, None
    return float(np.min(arr)), float(np.max(arr))


def clamp_cosine_sim(sim: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(sim, dtype=np.float32).copy()

    min_before, max_before = _safe_min_max(arr)
    non_finite_mask = ~np.isfinite(arr)
    non_finite_count = int(non_finite_mask.sum())
    if non_finite_count:
        arr[non_finite_mask] = 0.0

    below_min_count = int((arr < -1.0).sum())
    above_max_count = int((arr > 1.0).sum())
    np.clip(arr, -1.0, 1.0, out=arr)
    min_after, max_after = _safe_min_max(arr)

    meta = {
        "non_finite_count": non_finite_count,
        "below_min_count": below_min_count,
        "above_max_count": above_max_count,
        "min_before": min_before,
        "max_before": max_before,
        "min_after": min_after,
        "max_after": max_after,
        "clamped": bool(non_finite_count or below_min_count or above_max_count),
    }
    return arr, meta


def compute_safe_distance_from_cosine(sim: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    sim_arr = np.asarray(sim, dtype=np.float32)
    dist = (1.0 - sim_arr).astype(np.float32)

    min_before, max_before = _safe_min_max(dist)
    non_finite_mask = ~np.isfinite(dist)
    non_finite_count = int(non_finite_mask.sum())
    if non_finite_count:
        dist[non_finite_mask] = 1.0

    below_min_count = int((dist < 0.0).sum())
    above_max_count = int((dist > 2.0).sum())
    np.clip(dist, 0.0, 2.0, out=dist)
    min_after, max_after = _safe_min_max(dist)

    meta = {
        "non_finite_count": non_finite_count,
        "below_min_count": below_min_count,
        "above_max_count": above_max_count,
        "min_before": min_before,
        "max_before": max_before,
        "min_after": min_after,
        "max_after": max_after,
        "clamped": bool(non_finite_count or below_min_count or above_max_count),
    }
    return dist, meta


def sanitize_precomputed_distance_matrix(dist: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(dist, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected a square matrix for precomputed distances, got {arr.shape}.")

    min_before, max_before = _safe_min_max(arr)
    non_finite_mask = ~np.isfinite(arr)
    non_finite_count = int(non_finite_mask.sum())
    if non_finite_count:
        arr[non_finite_mask] = 1.0

    negative_count = int((arr < 0.0).sum())
    above_max_count = int((arr > 2.0).sum())
    np.clip(arr, 0.0, 2.0, out=arr)

    asymmetry_delta = np.abs(arr - arr.T)
    asymmetry_pairs = int(np.count_nonzero(asymmetry_delta > 1e-6))
    arr = ((arr + arr.T) * 0.5).astype(np.float32)

    diag_before = np.diag(arr).copy()
    diag_reset_count = int(np.count_nonzero(np.abs(diag_before) > 1e-12))
    np.fill_diagonal(arr, 0.0)

    min_after, max_after = _safe_min_max(arr)
    meta = {
        "non_finite_count": non_finite_count,
        "negative_count": negative_count,
        "above_max_count": above_max_count,
        "asymmetry_pairs": asymmetry_pairs,
        "diag_reset_count": diag_reset_count,
        "min_before": min_before,
        "max_before": max_before,
        "min_after": min_after,
        "max_after": max_after,
        "corrected": bool(
            non_finite_count
            or negative_count
            or above_max_count
            or asymmetry_pairs
            or diag_reset_count
        ),
    }
    return arr, meta
