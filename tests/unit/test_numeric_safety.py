import numpy as np

from author_name_disambiguation.common.numeric_safety import (
    clamp_cosine_sim,
    compute_safe_distance_from_cosine,
    sanitize_precomputed_distance_matrix,
)


def test_clamp_cosine_sim_clips_out_of_range_and_non_finite_values():
    raw = np.array([1.0000001, -1.0000001, np.nan, np.inf, -np.inf, 0.25], dtype=np.float32)

    clipped, meta = clamp_cosine_sim(raw)

    assert np.isfinite(clipped).all()
    assert float(clipped.min()) >= -1.0
    assert float(clipped.max()) <= 1.0
    assert meta["non_finite_count"] == 3
    assert meta["below_min_count"] == 1
    assert meta["above_max_count"] == 1
    assert meta["clamped"] is True


def test_compute_safe_distance_from_cosine_clips_to_valid_range():
    sim = np.array([1.0, 1.0000001, -1.0, -1.2, np.nan], dtype=np.float32)

    dist, meta = compute_safe_distance_from_cosine(sim)

    assert np.isfinite(dist).all()
    assert float(dist.min()) >= 0.0
    assert float(dist.max()) <= 2.0
    assert meta["non_finite_count"] == 1
    assert meta["below_min_count"] == 1
    assert meta["above_max_count"] == 1
    assert meta["clamped"] is True


def test_sanitize_precomputed_distance_matrix_repairs_numeric_issues():
    raw = np.array(
        [
            [0.0, -1e-7, np.nan],
            [0.2, 0.0, 3.0],
            [0.1, np.inf, 0.5],
        ],
        dtype=np.float32,
    )

    fixed, meta = sanitize_precomputed_distance_matrix(raw)

    assert fixed.shape == (3, 3)
    assert np.isfinite(fixed).all()
    assert float(fixed.min()) >= 0.0
    assert float(fixed.max()) <= 2.0
    assert np.allclose(fixed, fixed.T)
    assert np.allclose(np.diag(fixed), np.zeros(3, dtype=np.float32))
    assert meta["negative_count"] == 1
    assert meta["non_finite_count"] == 2
    assert meta["above_max_count"] == 1
    assert meta["asymmetry_pairs"] > 0
    assert meta["diag_reset_count"] == 1
    assert meta["corrected"] is True
