import numpy as np
import pandas as pd
import pytest
from importlib.util import find_spec

from author_name_disambiguation.approaches.nand.cluster import (
    _apply_constraints,
    _name_conflict,
    ExactGraphClusterAccumulator,
    cluster_blockwise_dbscan,
    resolve_dbscan_eps,
)


def _apply_constraints_reference(dist: np.ndarray, block_mentions: pd.DataFrame, constraints: dict) -> np.ndarray:
    if not constraints or not constraints.get("enabled", False):
        return dist

    out = dist.copy()
    max_year_gap = int(constraints.get("max_year_gap", 30))
    enforce_name_conflict = bool(constraints.get("enforce_name_conflict", True))
    constraint_mode = str(constraints.get("constraint_mode", "soft")).lower()
    name_conflict_mode = str(constraints.get("name_conflict_mode", constraint_mode)).lower()
    year_gap_mode = str(constraints.get("year_gap_mode", constraint_mode)).lower()
    name_conflict_min_distance = float(constraints.get("name_conflict_min_distance", 1.0))
    year_gap_min_distance = float(constraints.get("year_gap_min_distance", 1.0))

    authors = block_mentions["author_raw"].fillna("").astype(str).tolist()
    years = block_mentions["year"].tolist()

    n = len(block_mentions)
    for i in range(n):
        for j in range(i + 1, n):
            force_name = False
            force_year = False
            if enforce_name_conflict and _name_conflict(authors[i], authors[j]):
                force_name = True
            yi, yj = years[i], years[j]
            if (yi is not None and yj is not None) and not (pd.isna(yi) or pd.isna(yj)):
                if abs(int(yi) - int(yj)) > max_year_gap:
                    force_year = True

            if not (force_name or force_year):
                continue

            force_hard = (force_name and name_conflict_mode == "hard") or (force_year and year_gap_mode == "hard")
            if force_hard:
                out[i, j] = 1.0
                out[j, i] = 1.0
                continue

            if force_name:
                out[i, j] = max(out[i, j], name_conflict_min_distance)
                out[j, i] = max(out[j, i], name_conflict_min_distance)
            if force_year:
                out[i, j] = max(out[i, j], year_gap_min_distance)
                out[j, i] = max(out[j, i], year_gap_min_distance)

    np.fill_diagonal(out, 0.0)
    return out


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


def test_apply_constraints_matches_reference_loop_for_mixed_block():
    dist = np.array(
        [
            [0.0, 0.2, 0.4, 0.6],
            [0.2, 0.0, 0.3, 0.7],
            [0.4, 0.3, 0.0, 0.5],
            [0.6, 0.7, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    block_mentions = pd.DataFrame(
        [
            {"author_raw": "Smith, John", "year": 1990},
            {"author_raw": "Smith, Jane", "year": 2025},
            {"author_raw": "Wang, Y.", "year": 1992},
            {"author_raw": "Wang, Yong", "year": None},
        ]
    )
    constraints = {
        "enabled": True,
        "constraint_mode": "soft",
        "name_conflict_mode": "hard",
        "year_gap_mode": "soft",
        "max_year_gap": 20,
        "enforce_name_conflict": True,
        "name_conflict_min_distance": 0.75,
        "year_gap_min_distance": 0.65,
    }

    expected = _apply_constraints_reference(dist, block_mentions, constraints)
    out = _apply_constraints(dist, block_mentions, constraints)

    np.testing.assert_allclose(out, expected)


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
            {"mention_id": "a3", "block_key": "blk.a", "author_raw": "A, A", "year": 2002},
            {"mention_id": "b1", "block_key": "blk.b", "author_raw": "B, B", "year": 2000},
            {"mention_id": "b2", "block_key": "blk.b", "author_raw": "B, B", "year": 2001},
            {"mention_id": "b3", "block_key": "blk.b", "author_raw": "B, B", "year": 2002},
        ]
    )
    pair_scores = pd.DataFrame(
        _complete_block_pair_scores("blk.a", ["a1", "a2", "a3"])
        + _complete_block_pair_scores("blk.b", ["b1", "b2", "b3"])
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

    monkeypatch.setattr("author_name_disambiguation.approaches.nand.cluster.DBSCAN", _FakeDBSCAN)

    clusters, meta = cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cfg,
        backend="sklearn_cpu",
        return_meta=True,
    )

    assert len(clusters) == 6
    assert seen_eps == [0.35, 0.35]
    assert meta["eps_base"] == 0.35
    assert meta["eps_block_policy_enabled"] is False
    assert meta["eps_block_policy_summary"]["bucket_counts"] == {"default": 2}


def test_exact_graph_accumulator_uses_numeric_pair_helpers_and_reports_connected_components_meta():
    mentions = pd.DataFrame(
        [
            {"mention_id": "a1", "block_key": "blk.a", "author_raw": "A, A", "year": 2000},
            {"mention_id": "a2", "block_key": "blk.a", "author_raw": "A, A", "year": 2001},
            {"mention_id": "b1", "block_key": "blk.b", "author_raw": "B, B", "year": 2002},
            {"mention_id": "b2", "block_key": "blk.b", "author_raw": "B, B", "year": 2003},
        ]
    )
    accumulator = ExactGraphClusterAccumulator(
        mentions=mentions,
        cluster_config={
            "eps": 0.2,
            "min_samples": 1,
            "metric": "precomputed",
            "constraints": {"enabled": False},
        },
        backend_requested="connected_components_cpu",
    )

    accumulator.consume_score_columns(
        {
            "mention_id_1": np.asarray(["missing-a", "missing-b"], dtype=object),
            "mention_id_2": np.asarray(["missing-c", "missing-d"], dtype=object),
            "mention_idx_1": np.asarray([0, 2], dtype=np.int64),
            "mention_idx_2": np.asarray([1, 3], dtype=np.int64),
            "block_key": np.asarray(["blk.a", "blk.b"], dtype=object),
            "block_idx": np.asarray([0, 1], dtype=np.int64),
            "distance": np.asarray([0.05, 0.05], dtype=np.float32),
        }
    )

    out, meta = accumulator.finalize()

    assert len(out["author_uid"].unique()) == 2
    assert meta["numeric_pair_index_rows"] == 2
    assert meta["string_pair_index_rows"] == 0
    assert meta["dbscan_seconds_total"] == 0.0
    assert meta["connected_components_seconds_total"] >= 0.0
    assert meta["mapping_seconds_total"] >= 0.0
    assert meta["constraint_apply_seconds_total"] >= 0.0
    assert meta["exact_graph_init_tokenize_seconds"] >= 0.0
    assert meta["exact_graph_init_block_index_seconds"] >= 0.0
    assert meta["exact_graph_init_state_seconds"] >= 0.0
    assert meta["score_callback_group_seconds"] >= 0.0
    assert meta["score_callback_index_seconds"] >= 0.0
    assert meta["score_callback_constraint_seconds"] >= 0.0
    assert meta["score_callback_union_seconds"] >= 0.0
    assert meta["union_impl"] in {"python", "numba"}


def test_exact_graph_accumulator_string_fallback_matches_numeric_fast_path():
    mentions = pd.DataFrame(
        [
            {"mention_id": "a1", "block_key": "blk.a", "author_raw": "A, A", "year": 2000},
            {"mention_id": "a2", "block_key": "blk.a", "author_raw": "A, A", "year": 2001},
            {"mention_id": "a3", "block_key": "blk.a", "author_raw": "A, A", "year": 2002},
            {"mention_id": "b1", "block_key": "blk.b", "author_raw": "B, B", "year": 2003},
            {"mention_id": "b2", "block_key": "blk.b", "author_raw": "B, B", "year": 2004},
        ]
    )
    numeric_accumulator = ExactGraphClusterAccumulator(
        mentions=mentions,
        cluster_config={"eps": 0.2, "min_samples": 1, "metric": "precomputed", "constraints": {"enabled": False}},
    )
    string_accumulator = ExactGraphClusterAccumulator(
        mentions=mentions,
        cluster_config={"eps": 0.2, "min_samples": 1, "metric": "precomputed", "constraints": {"enabled": False}},
    )

    score_columns = {
        "mention_id_1": np.asarray(["missing-a", "missing-a", "missing-b"], dtype=object),
        "mention_id_2": np.asarray(["missing-b", "missing-c", "missing-c"], dtype=object),
        "mention_idx_1": np.asarray([0, 0, 1], dtype=np.int64),
        "mention_idx_2": np.asarray([1, 2, 2], dtype=np.int64),
        "block_key": np.asarray(["blk.a", "blk.a", "blk.a"], dtype=object),
        "block_idx": np.asarray([0, 0, 0], dtype=np.int64),
        "distance": np.asarray([0.05, 0.05, 0.05], dtype=np.float32),
    }
    numeric_accumulator.consume_score_columns(score_columns)
    string_accumulator.consume_score_columns(
        {
            "mention_id_1": np.asarray(["a1", "a1", "a2"], dtype=object),
            "mention_id_2": np.asarray(["a2", "a3", "a3"], dtype=object),
            "block_key": np.asarray(["blk.a", "blk.a", "blk.a"], dtype=object),
            "distance": np.asarray([0.05, 0.05, 0.05], dtype=np.float32),
        }
    )

    numeric_out, _ = numeric_accumulator.finalize()
    string_out, _ = string_accumulator.finalize()

    pd.testing.assert_frame_equal(
        numeric_out.sort_values(["block_key", "mention_id"]).reset_index(drop=True),
        string_out.sort_values(["block_key", "mention_id"]).reset_index(drop=True),
    )


def test_exact_graph_accumulator_lazy_union_keeps_edge_free_blocks_as_singletons():
    mentions = pd.DataFrame(
        [
            {"mention_id": "a1", "block_key": "blk.a", "author_raw": "A, A", "year": 2000},
            {"mention_id": "a2", "block_key": "blk.a", "author_raw": "A, A", "year": 2001},
            {"mention_id": "a3", "block_key": "blk.a", "author_raw": "A, A", "year": 2002},
        ]
    )
    accumulator = ExactGraphClusterAccumulator(
        mentions=mentions,
        cluster_config={"eps": 0.2, "min_samples": 1, "metric": "precomputed", "constraints": {"enabled": False}},
    )

    accumulator.consume_score_columns(
        {
            "mention_id_1": np.asarray(["a1", "a1", "a2"], dtype=object),
            "mention_id_2": np.asarray(["a2", "a3", "a3"], dtype=object),
            "mention_idx_1": np.asarray([0, 0, 1], dtype=np.int64),
            "mention_idx_2": np.asarray([1, 2, 2], dtype=np.int64),
            "block_key": np.asarray(["blk.a", "blk.a", "blk.a"], dtype=object),
            "block_idx": np.asarray([0, 0, 0], dtype=np.int64),
            "distance": np.asarray([0.8, 0.9, 0.7], dtype=np.float32),
        }
    )

    out, meta = accumulator.finalize()

    assert list(out.sort_values("mention_id")["author_uid"]) == ["blk.a::0", "blk.a::1", "blk.a::2"]
    assert meta["connected_components_seconds_total"] >= 0.0


@pytest.mark.skipif(find_spec("numba") is None, reason="numba not installed")
def test_exact_graph_numba_union_matches_python_fallback():
    mentions = pd.DataFrame(
        [
            {"mention_id": "a1", "block_key": "blk.a", "author_raw": "A, A", "year": 2000},
            {"mention_id": "a2", "block_key": "blk.a", "author_raw": "A, A", "year": 2001},
            {"mention_id": "a3", "block_key": "blk.a", "author_raw": "A, A", "year": 2002},
            {"mention_id": "a4", "block_key": "blk.a", "author_raw": "A, A", "year": 2003},
        ]
    )
    score_columns = {
        "mention_id_1": np.asarray(["a1", "a2", "a3"], dtype=object),
        "mention_id_2": np.asarray(["a2", "a3", "a4"], dtype=object),
        "mention_idx_1": np.asarray([0, 1, 2], dtype=np.int64),
        "mention_idx_2": np.asarray([1, 2, 3], dtype=np.int64),
        "block_key": np.asarray(["blk.a", "blk.a", "blk.a"], dtype=object),
        "block_idx": np.asarray([0, 0, 0], dtype=np.int64),
        "distance": np.asarray([0.05, 0.05, 0.05], dtype=np.float32),
    }
    python_accumulator = ExactGraphClusterAccumulator(
        mentions=mentions,
        cluster_config={"eps": 0.2, "min_samples": 1, "metric": "precomputed", "constraints": {"enabled": False}},
        union_impl="python",
    )
    numba_accumulator = ExactGraphClusterAccumulator(
        mentions=mentions,
        cluster_config={"eps": 0.2, "min_samples": 1, "metric": "precomputed", "constraints": {"enabled": False}},
        union_impl="numba",
    )

    python_accumulator.consume_score_columns(score_columns)
    numba_accumulator.consume_score_columns(score_columns)

    python_out, python_meta = python_accumulator.finalize()
    numba_out, numba_meta = numba_accumulator.finalize()

    pd.testing.assert_frame_equal(
        python_out.sort_values(["block_key", "mention_id"]).reset_index(drop=True),
        numba_out.sort_values(["block_key", "mention_id"]).reset_index(drop=True),
    )
    assert python_meta["union_impl"] == "python"
    assert numba_meta["union_impl"] == "numba"


def test_cluster_eps_block_policy_applies_buckets_and_clamp(monkeypatch):
    small_ids = [f"s{i}" for i in range(3)]
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

    monkeypatch.setattr("author_name_disambiguation.approaches.nand.cluster.DBSCAN", _FakeDBSCAN)

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


def test_cluster_blockwise_dbscan_cpu_sharding_updates_progress(monkeypatch):
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
    seen = {"updates": 0, "params": None}

    class _FakeTracker:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def update(self, n=1):
            seen["updates"] += int(n)

    class _FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

        def __hash__(self):
            return id(self)

    class _FakeExecutor:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def submit(self, fn, payload):
            return _FakeFuture(fn(payload))

    def _fake_loop_progress(*, total, label, enabled, unit, compact_label=None, **kwargs):
        seen["params"] = {
            "total": int(total),
            "label": str(label),
            "enabled": bool(enabled),
            "unit": str(unit),
            "compact_label": compact_label,
        }
        return _FakeTracker()

    monkeypatch.setattr("author_name_disambiguation.approaches.nand.cluster.loop_progress", _fake_loop_progress)
    monkeypatch.setattr("author_name_disambiguation.approaches.nand.cluster.ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(
        "author_name_disambiguation.approaches.nand.cluster.wait",
        lambda futures, return_when=None: (set(futures), set()),
    )
    monkeypatch.setattr("author_name_disambiguation.approaches.nand.cluster.mp.get_context", lambda _mode: None)

    clusters = cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cfg,
        show_progress=True,
        num_workers=4,
        sharding_mode="on",
        min_pairs_per_worker=1,
        backend="sklearn_cpu",
    )

    assert len(clusters) == len(mentions)
    assert seen["params"] == {
        "total": 2,
        "label": "Cluster blocks",
        "enabled": True,
        "unit": "block",
        "compact_label": "Clustering",
    }
    assert seen["updates"] == 2


def test_cluster_blockwise_dbscan_two_point_fast_path_matches_dbscan():
    mentions = pd.DataFrame(
        [
            {"mention_id": "a1", "block_key": "blk.a", "author_raw": "A, A", "year": 2000},
            {"mention_id": "a2", "block_key": "blk.a", "author_raw": "A, A", "year": 2001},
        ]
    )
    pair_scores = pd.DataFrame(
        [
            {"pair_id": "a1__a2", "mention_id_1": "a1", "mention_id_2": "a2", "block_key": "blk.a", "distance": 0.4}
        ]
    )
    cfg = {"eps": 0.2, "min_samples": 1, "metric": "precomputed", "constraints": {"enabled": False}}

    clusters = cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cfg,
        backend="sklearn_cpu",
    )

    assert list(clusters.sort_values("mention_id")["author_uid"]) == ["blk.a::0", "blk.a::1"]


def test_cluster_backend_auto_prefers_cpu_for_small_workloads(monkeypatch):
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
    seen = {"gpu_called": False}

    monkeypatch.setattr(
        "author_name_disambiguation.approaches.nand.cluster._resolve_cluster_backend",
        lambda backend, metric: {
            "requested": backend,
            "effective": "cuml_gpu",
            "reason": "forced-test",
            "cuml_available": True,
            "metric": metric,
        },
    )

    def _gpu(*args, **kwargs):
        seen["gpu_called"] = True
        return np.asarray([0, 0], dtype=np.int64)

    monkeypatch.setattr("author_name_disambiguation.approaches.nand.cluster._run_dbscan_cuml", _gpu)

    clusters, meta = cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cfg,
        backend="auto",
        return_meta=True,
    )

    assert len(clusters) == 2
    assert meta["cluster_backend_requested"] == "auto"
    assert meta["cluster_backend_effective"] == "sklearn_cpu"
    assert meta["cluster_backend_reason"] == "auto_small_workload_cpu"
    assert seen["gpu_called"] is False


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
        "author_name_disambiguation.approaches.nand.cluster._resolve_cluster_backend",
        lambda backend, metric: {
            "requested": backend,
            "effective": "cuml_gpu",
            "reason": "forced-test",
            "cuml_available": True,
            "metric": metric,
        },
    )
    monkeypatch.setattr(
        "author_name_disambiguation.approaches.nand.cluster._run_dbscan_cuml",
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


def test_cluster_backend_gpu_keeps_small_blocks_on_gpu_in_recovery_mode(monkeypatch):
    small_ids = [f"s{i}" for i in range(3)]
    big_ids = [f"b{i}" for i in range(9)]
    mentions = pd.DataFrame(
        [
            {"mention_id": m, "block_key": "blk.small", "author_raw": "S, A", "year": 2000}
            for m in small_ids
        ]
        + [
            {"mention_id": m, "block_key": "blk.big", "author_raw": "B, A", "year": 2001}
            for m in big_ids
        ]
    )
    pair_scores = pd.DataFrame(
        _complete_block_pair_scores("blk.small", small_ids)
        + _complete_block_pair_scores("blk.big", big_ids)
    )
    cfg = {"eps": 0.2, "min_samples": 1, "metric": "precomputed", "constraints": {"enabled": False}}
    seen = {"cpu_sizes": [], "gpu_sizes": []}

    class _FakeDBSCAN:
        def __init__(self, *, eps, min_samples, metric):
            del eps, min_samples, metric

        def fit_predict(self, dist):
            seen["cpu_sizes"].append(int(dist.shape[0]))
            return np.zeros(dist.shape[0], dtype=np.int64)

    def _fake_gpu(dist, eps, min_samples, metric):
        del eps, min_samples, metric
        seen["gpu_sizes"].append(int(dist.shape[0]))
        return np.zeros(dist.shape[0], dtype=np.int64)

    monkeypatch.setattr("author_name_disambiguation.approaches.nand.cluster.DBSCAN", _FakeDBSCAN)
    monkeypatch.setattr("author_name_disambiguation.approaches.nand.cluster._run_dbscan_cuml", _fake_gpu)
    monkeypatch.setattr(
        "author_name_disambiguation.approaches.nand.cluster._resolve_cluster_backend",
        lambda backend, metric: {
            "requested": backend,
            "effective": "cuml_gpu",
            "reason": "forced-test",
            "cuml_available": True,
            "metric": metric,
        },
    )

    clusters, meta = cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cfg,
        backend="cuml_gpu",
        return_meta=True,
    )

    assert len(clusters) == len(mentions)
    assert seen["cpu_sizes"] == []
    assert seen["gpu_sizes"] == [3, 9]
    assert meta["cluster_backend_effective"] == "cuml_gpu"
    assert meta["backend_block_counts"] == {"cuml_gpu": 2}
    assert meta["block_count_by_bucket"] == {"3-4": 1, "9-16": 1}
    assert "3-4" in meta["total_seconds_by_bucket"]
    assert "9-16" in meta["dbscan_seconds_by_bucket"]
