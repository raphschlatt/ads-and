import numpy as np
import pandas as pd
import pytest

from author_name_disambiguation.approaches.nand.cluster import (
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


def _complete_block_pair_scores(block_key: str, mention_ids: list[str], distance: float = 0.05) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(len(mention_ids)):
        for j in range(i + 1, len(mention_ids)):
            a = str(mention_ids[i])
            b = str(mention_ids[j])
            rows.append(
                {
                    "pair_id": f"{a}__{b}",
                    "mention_id_1": a,
                    "mention_id_2": b,
                    "block_key": block_key,
                    "distance": float(distance),
                }
            )
    return rows


def test_cluster_eps_block_policy_disabled_uses_base_eps(monkeypatch):
    mentions = pd.DataFrame(
        [
            {"mention_id": "a1", "block_key": "blk.a", "author_raw": "A, A", "year": 2000},
            {"mention_id": "a2", "block_key": "blk.a", "author_raw": "A, A", "year": 2001},
            {"mention_id": "b1", "block_key": "blk.b", "author_raw": "B, B", "year": 2000},
            {"mention_id": "b2", "block_key": "blk.b", "author_raw": "B, B", "year": 2001},
        ]
    )
    pair_scores = pd.DataFrame(
        _complete_block_pair_scores("blk.a", ["a1", "a2"])
        + _complete_block_pair_scores("blk.b", ["b1", "b2"])
    )
    cfg = {
        "eps": 0.35,
        "eps_min": 0.15,
        "eps_max": 0.85,
        "min_samples": 1,
        "metric": "precomputed",
        "constraints": {"enabled": False},
        "eps_block_policy": {
            "enabled": False,
            "strategy": "size_delta",
            "default_delta": 0.0,
            "buckets": [
                {"min_size": 1, "max_size": 10, "delta": 0.03},
            ],
        },
    }

    seen_eps: list[float] = []

    class _FakeDBSCAN:
        def __init__(self, *, eps, min_samples, metric):
            seen_eps.append(float(eps))

        def fit_predict(self, dist):
            return np.zeros(dist.shape[0], dtype=np.int64)

    monkeypatch.setattr("src.approaches.nand.cluster.DBSCAN", _FakeDBSCAN)

    clusters, meta = cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cfg,
        backend="sklearn_cpu",
        return_meta=True,
    )

    assert len(clusters) == 4
    assert seen_eps == [0.35, 0.35]
    assert meta["eps_base"] == 0.35
    assert meta["eps_block_policy_enabled"] is False
    assert meta["eps_block_policy_summary"]["bucket_counts"] == {"default": 2}


def test_cluster_eps_block_policy_applies_buckets_and_clamp(monkeypatch):
    small_ids = [f"s{i}" for i in range(2)]
    mid_ids = [f"m{i}" for i in range(12)]
    big_ids = [f"b{i}" for i in range(66)]

    mentions = pd.DataFrame(
        [
            {"mention_id": m, "block_key": "blk.small", "author_raw": "S, A", "year": 2000}
            for m in small_ids
        ]
        + [
            {"mention_id": m, "block_key": "blk.mid", "author_raw": "M, A", "year": 2001}
            for m in mid_ids
        ]
        + [
            {"mention_id": m, "block_key": "blk.big", "author_raw": "B, A", "year": 2002}
            for m in big_ids
        ]
    )
    pair_scores = pd.DataFrame(
        _complete_block_pair_scores("blk.small", small_ids)
        + _complete_block_pair_scores("blk.mid", mid_ids)
        + _complete_block_pair_scores("blk.big", big_ids)
    )
    cfg = {
        "eps": 0.35,
        "eps_min": 0.15,
        "eps_max": 0.38,
        "min_samples": 1,
        "metric": "precomputed",
        "constraints": {"enabled": False},
        "eps_block_policy": {
            "enabled": True,
            "strategy": "size_delta",
            "default_delta": 0.0,
            "buckets": [
                {"min_size": 1, "max_size": 10, "delta": 0.03},
                {"min_size": 11, "max_size": 65, "delta": 0.0},
                {"min_size": 66, "max_size": None, "delta": -0.05},
            ],
        },
    }

    seen_eps: list[float] = []

    class _FakeDBSCAN:
        def __init__(self, *, eps, min_samples, metric):
            seen_eps.append(float(eps))

        def fit_predict(self, dist):
            return np.zeros(dist.shape[0], dtype=np.int64)

    monkeypatch.setattr("src.approaches.nand.cluster.DBSCAN", _FakeDBSCAN)

    _, meta = cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cfg,
        backend="sklearn_cpu",
        return_meta=True,
    )

    rounded = sorted(round(v, 3) for v in seen_eps)
    assert rounded == [0.3, 0.35, 0.38]
    summary = meta["eps_block_policy_summary"]
    assert summary["bucket_counts"] == {"1-10": 1, "11-65": 1, "66-inf": 1}
    assert summary["effective_eps_min"] == pytest.approx(0.3)
    assert summary["effective_eps_max"] == pytest.approx(0.38)


def test_cluster_eps_block_policy_rejects_overlapping_buckets():
    mentions = pd.DataFrame(
        [
            {"mention_id": "a1", "block_key": "blk.a", "author_raw": "A, A", "year": 2000},
            {"mention_id": "a2", "block_key": "blk.a", "author_raw": "A, A", "year": 2001},
        ]
    )
    pair_scores = pd.DataFrame(_complete_block_pair_scores("blk.a", ["a1", "a2"]))
    cfg = {
        "eps": 0.35,
        "eps_min": 0.15,
        "eps_max": 0.85,
        "min_samples": 1,
        "metric": "precomputed",
        "constraints": {"enabled": False},
        "eps_block_policy": {
            "enabled": True,
            "strategy": "size_delta",
            "default_delta": 0.0,
            "buckets": [
                {"min_size": 1, "max_size": 10, "delta": 0.03},
                {"min_size": 10, "max_size": 20, "delta": 0.0},
            ],
        },
    }

    with pytest.raises(ValueError, match="overlap|overlapping|starts"):
        cluster_blockwise_dbscan(
            mentions=mentions,
            pair_scores=pair_scores,
            cluster_config=cfg,
            backend="sklearn_cpu",
        )


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
        clusters = cluster_blockwise_dbscan(
            mentions=mentions,
            pair_scores=pair_scores,
            cluster_config=cfg,
            backend="sklearn_cpu",
        )

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
