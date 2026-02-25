import numpy as np
import pandas as pd
import pytest

from src.approaches.nand.cluster import (
    _apply_constraints,
    _name_conflict,
    cluster_blockwise_dbscan,
    resolve_dbscan_eps,
)


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


def test_name_conflict_can_be_hard_while_year_gap_remains_soft():
    dist = np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float32)
    block_mentions = pd.DataFrame(
        [
            {"author_raw": "Smith, John", "year": 1990},
            {"author_raw": "Smith, Jane", "year": 2025},
        ]
    )
    constraints = {
        "enabled": True,
        "constraint_mode": "soft",  # legacy default
        "name_conflict_mode": "hard",
        "year_gap_mode": "soft",
        "max_year_gap": 20,
        "enforce_name_conflict": True,
        "name_conflict_min_distance": 0.75,
        "year_gap_min_distance": 0.65,
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


def test_resolve_dbscan_eps_val_sweep_prefers_selected_then_fallback():
    cfg = {
        "eps_mode": "val_sweep",
        "eps": 0.35,
        "eps_fallback": 0.31,
        "eps_min": 0.15,
        "eps_max": 0.85,
        "selected_eps": 0.27,
    }
    eps, meta = resolve_dbscan_eps(cfg, cosine_threshold=0.1)
    assert np.isclose(eps, 0.27)
    assert meta["source"] == "val_sweep_selected"

    cfg2 = dict(cfg)
    cfg2.pop("selected_eps")
    eps2, meta2 = resolve_dbscan_eps(cfg2, cosine_threshold=0.1)
    assert np.isclose(eps2, 0.31)
    assert meta2["source"] == "val_sweep_fallback"


def test_cluster_blockwise_dbscan_sanitizes_invalid_precomputed_values():
    mentions = pd.DataFrame(
        [
            {"mention_id": "a1", "block_key": "blk.a", "author_raw": "A, A", "year": 2000},
            {"mention_id": "a2", "block_key": "blk.a", "author_raw": "A, A", "year": 2001},
            {"mention_id": "b1", "block_key": "blk.b", "author_raw": "B, B", "year": 2000},
            {"mention_id": "b2", "block_key": "blk.b", "author_raw": "B, B", "year": 2001},
        ]
    )
    pair_scores = pd.DataFrame(
        [
            {
                "pair_id": "a1__a2",
                "mention_id_1": "a1",
                "mention_id_2": "a2",
                "block_key": "blk.a",
                "distance": -1e-7,
            },
            {
                "pair_id": "b1__b2",
                "mention_id_1": "b1",
                "mention_id_2": "b2",
                "block_key": "blk.b",
                "distance": np.nan,
            },
        ]
    )
    cfg = {"eps": 0.35, "min_samples": 1, "metric": "precomputed", "constraints": {"enabled": False}}

    with pytest.warns(RuntimeWarning, match="Sanitized DBSCAN precomputed distances"):
        clusters = cluster_blockwise_dbscan(mentions=mentions, pair_scores=pair_scores, cluster_config=cfg)

    assert len(clusters) == 4
    assert clusters["mention_id"].nunique() == 4


def test_cluster_blockwise_dbscan_cpu_sharding_matches_sequential():
    mentions = pd.DataFrame(
        [
            {"mention_id": "a1", "block_key": "blk.a", "author_raw": "A, A", "year": 2000},
            {"mention_id": "a2", "block_key": "blk.a", "author_raw": "A, A", "year": 2001},
            {"mention_id": "a3", "block_key": "blk.a", "author_raw": "A, A", "year": 2002},
            {"mention_id": "b1", "block_key": "blk.b", "author_raw": "B, B", "year": 2000},
            {"mention_id": "b2", "block_key": "blk.b", "author_raw": "B, B", "year": 2001},
            {"mention_id": "b3", "block_key": "blk.b", "author_raw": "B, B", "year": 2002},
        ]
    )
    pair_scores = pd.DataFrame(
        [
            {"pair_id": "a1__a2", "mention_id_1": "a1", "mention_id_2": "a2", "block_key": "blk.a", "distance": 0.05},
            {"pair_id": "a1__a3", "mention_id_1": "a1", "mention_id_2": "a3", "block_key": "blk.a", "distance": 0.05},
            {"pair_id": "a2__a3", "mention_id_1": "a2", "mention_id_2": "a3", "block_key": "blk.a", "distance": 0.05},
            {"pair_id": "b1__b2", "mention_id_1": "b1", "mention_id_2": "b2", "block_key": "blk.b", "distance": 0.05},
            {"pair_id": "b1__b3", "mention_id_1": "b1", "mention_id_2": "b3", "block_key": "blk.b", "distance": 0.05},
            {"pair_id": "b2__b3", "mention_id_1": "b2", "mention_id_2": "b3", "block_key": "blk.b", "distance": 0.05},
        ]
    )
    cfg = {"eps": 0.2, "min_samples": 1, "metric": "precomputed", "constraints": {"enabled": False}}

    seq = cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cfg,
        num_workers=1,
        sharding_mode="off",
        backend="sklearn_cpu",
    )
    par = cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cfg,
        num_workers=4,
        sharding_mode="on",
        backend="sklearn_cpu",
    )

    seq_m = seq.sort_values("mention_id").reset_index(drop=True)[["mention_id", "author_uid"]]
    par_m = par.sort_values("mention_id").reset_index(drop=True)[["mention_id", "author_uid"]]
    pd.testing.assert_frame_equal(seq_m, par_m)


def test_cluster_backend_gpu_failure_falls_back_to_cpu(monkeypatch):
    mentions = pd.DataFrame(
        [
            {"mention_id": "a1", "block_key": "blk.a", "author_raw": "A, A", "year": 2000},
            {"mention_id": "a2", "block_key": "blk.a", "author_raw": "A, A", "year": 2001},
        ]
    )
    pair_scores = pd.DataFrame(
        [
            {"pair_id": "a1__a2", "mention_id_1": "a1", "mention_id_2": "a2", "block_key": "blk.a", "distance": 0.05}
        ]
    )
    cfg = {"eps": 0.2, "min_samples": 1, "metric": "precomputed", "constraints": {"enabled": False}}

    monkeypatch.setattr(
        "src.approaches.nand.cluster._resolve_cluster_backend",
        lambda backend, metric: {
            "requested": backend,
            "effective": "cuml_gpu",
            "reason": "forced-test",
            "cuml_available": True,
            "metric": metric,
        },
    )
    monkeypatch.setattr(
        "src.approaches.nand.cluster._run_dbscan_cuml",
        lambda dist, eps, min_samples, metric: (_ for _ in ()).throw(RuntimeError("gpu fail")),
    )

    clusters, meta = cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cfg,
        backend="auto",
        return_meta=True,
    )
    assert len(clusters) == 2
    assert meta["cluster_backend_effective"] == "sklearn_cpu"
