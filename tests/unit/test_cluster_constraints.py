import numpy as np
import pandas as pd

from src.approaches.nand.cluster import _apply_constraints, _name_conflict, resolve_dbscan_eps


def test_name_conflict_handles_diacritics_and_initials():
    assert _name_conflict("Allègre, C. J.", "Allegre, CJ") is False
    assert _name_conflict("Smith, John", "Smith, Jane") is True
    assert _name_conflict("Wang, Y.", "Wang, Yong") is False


def test_soft_constraints_do_not_force_distance_to_one():
    dist = np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float32)
    block_mentions = pd.DataFrame(
        [
            {"author_raw": "Smith, John", "year": 1990},
            {"author_raw": "Smith, Jane", "year": 2025},
        ]
    )
    constraints = {
        "enabled": True,
        "constraint_mode": "soft",
        "max_year_gap": 20,
        "enforce_name_conflict": True,
        "name_conflict_min_distance": 0.75,
        "year_gap_min_distance": 0.65,
    }

    out = _apply_constraints(dist, block_mentions, constraints)
    assert np.isclose(out[0, 1], 0.75)
    assert np.isclose(out[1, 0], 0.75)


def test_hard_constraints_force_distance_to_one():
    dist = np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float32)
    block_mentions = pd.DataFrame(
        [
            {"author_raw": "Smith, John", "year": 1990},
            {"author_raw": "Smith, Jane", "year": 2025},
        ]
    )
    constraints = {
        "enabled": True,
        "constraint_mode": "hard",
        "max_year_gap": 20,
        "enforce_name_conflict": True,
    }

    out = _apply_constraints(dist, block_mentions, constraints)
    assert np.isclose(out[0, 1], 1.0)
    assert np.isclose(out[1, 0], 1.0)


def test_resolve_dbscan_eps_from_threshold_is_clamped():
    cfg = {
        "eps_mode": "from_threshold",
        "eps": 0.35,
        "eps_min": 0.15,
        "eps_max": 0.85,
    }
    eps, meta = resolve_dbscan_eps(cfg, cosine_threshold=0.98)
    assert eps == 0.15
    assert meta["source"] == "from_threshold"

    eps2, meta2 = resolve_dbscan_eps(cfg, cosine_threshold=0.05)
    assert eps2 == 0.85
    assert meta2["source"] == "from_threshold"
