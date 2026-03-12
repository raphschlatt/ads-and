from contextlib import contextmanager

import pandas as pd

from author_name_disambiguation.approaches.nand import build_pairs as build_pairs_module
from author_name_disambiguation.approaches.nand.build_pairs import assign_lspo_splits, build_pairs_within_blocks


def test_pair_builder_with_labels_and_splits():
    df = pd.DataFrame(
        [
            {"mention_id": "p1::0", "bibcode": "p1", "author_idx": 0, "author_raw": "Smith, John", "title": "t1", "abstract": "a1", "year": 2001, "source_type": "lspo", "block_key": "j.smith", "orcid": "o1"},
            {"mention_id": "p2::0", "bibcode": "p2", "author_idx": 0, "author_raw": "Smith, John", "title": "t2", "abstract": "a2", "year": 2002, "source_type": "lspo", "block_key": "j.smith", "orcid": "o1"},
            {"mention_id": "p3::0", "bibcode": "p3", "author_idx": 0, "author_raw": "Smith, Jane", "title": "t3", "abstract": "a3", "year": 2003, "source_type": "lspo", "block_key": "j.smith", "orcid": "o2"},
        ]
    )
    split_df, meta = assign_lspo_splits(df, seed=11, return_meta=True)
    pairs = build_pairs_within_blocks(split_df, require_same_split=True, labeled_only=False, balance_train=False)
    assert len(pairs) >= 1
    assert "status" in meta
    assert set(["pair_id", "mention_id_1", "mention_id_2", "block_key", "split"]).issubset(set(pairs.columns))


def test_pair_builder_excludes_same_bibcode_pairs_and_reports_meta():
    df = pd.DataFrame(
        [
            {"mention_id": "p1::0", "bibcode": "p1", "author_idx": 0, "author_raw": "Smith, John", "title": "t1", "abstract": "a1", "year": 2001, "source_type": "lspo", "block_key": "j.smith", "split": "train", "orcid": "o1"},
            {"mention_id": "p1::1", "bibcode": "p1", "author_idx": 1, "author_raw": "Smith, John", "title": "t1", "abstract": "a1", "year": 2001, "source_type": "lspo", "block_key": "j.smith", "split": "train", "orcid": "o2"},
            {"mention_id": "p2::0", "bibcode": "p2", "author_idx": 0, "author_raw": "Smith, John", "title": "t2", "abstract": "a2", "year": 2002, "source_type": "lspo", "block_key": "j.smith", "split": "train", "orcid": "o1"},
        ]
    )
    pairs, meta = build_pairs_within_blocks(
        df,
        require_same_split=True,
        labeled_only=False,
        balance_train=False,
        exclude_same_bibcode=True,
        return_meta=True,
    )
    assert len(pairs) == 2
    assert meta["same_publication_pairs_skipped"] == 1


def test_pair_builder_balances_train_pairs_symmetrically():
    df = pd.DataFrame(
        [
            {"mention_id": "a1::0", "bibcode": "a1", "author_idx": 0, "author_raw": "A", "title": "t", "abstract": "a", "year": 2001, "source_type": "lspo", "block_key": "blk.a", "split": "train", "orcid": "oa"},
            {"mention_id": "a2::0", "bibcode": "a2", "author_idx": 0, "author_raw": "A", "title": "t", "abstract": "a", "year": 2002, "source_type": "lspo", "block_key": "blk.a", "split": "train", "orcid": "oa"},
            {"mention_id": "a3::0", "bibcode": "a3", "author_idx": 0, "author_raw": "A", "title": "t", "abstract": "a", "year": 2003, "source_type": "lspo", "block_key": "blk.a", "split": "train", "orcid": "oa"},
            {"mention_id": "a4::0", "bibcode": "a4", "author_idx": 0, "author_raw": "A", "title": "t", "abstract": "a", "year": 2004, "source_type": "lspo", "block_key": "blk.a", "split": "train", "orcid": "oa"},
            {"mention_id": "a5::0", "bibcode": "a5", "author_idx": 0, "author_raw": "A", "title": "t", "abstract": "a", "year": 2005, "source_type": "lspo", "block_key": "blk.a", "split": "train", "orcid": "ob"},
        ]
    )
    pairs, _meta = build_pairs_within_blocks(
        df,
        require_same_split=True,
        labeled_only=False,
        balance_train=True,
        exclude_same_bibcode=False,
        return_meta=True,
    )
    train_pairs = pairs[(pairs["split"] == "train") & pairs["label"].notna()]
    assert int((train_pairs["label"] == 1).sum()) == int((train_pairs["label"] == 0).sum())


def test_pair_builder_can_stream_to_output_without_returning_pairs(tmp_path):
    df = pd.DataFrame(
        [
            {"mention_id": "a1::0", "bibcode": "a1", "author_idx": 0, "author_raw": "A", "title": "t", "abstract": "a", "year": 2001, "source_type": "ads", "block_key": "blk.a", "split": "inference"},
            {"mention_id": "a2::0", "bibcode": "a2", "author_idx": 0, "author_raw": "A", "title": "t", "abstract": "a", "year": 2002, "source_type": "ads", "block_key": "blk.a", "split": "inference"},
            {"mention_id": "a3::0", "bibcode": "a3", "author_idx": 0, "author_raw": "A", "title": "t", "abstract": "a", "year": 2003, "source_type": "ads", "block_key": "blk.a", "split": "inference"},
        ]
    )
    out_path = tmp_path / "pairs.parquet"
    pairs, meta = build_pairs_within_blocks(
        df,
        require_same_split=False,
        labeled_only=False,
        balance_train=False,
        return_pairs=False,
        output_path=out_path,
        chunk_rows=1,
        return_meta=True,
    )
    assert pairs is None
    assert out_path.exists()
    written = pd.read_parquet(out_path)
    assert len(written) == 3
    assert meta["pairs_written"] == 3


def test_pair_builder_sharding_is_deterministic_across_worker_counts():
    rows = []
    for block_idx in range(6):
        block = f"blk.{block_idx}"
        for mention_idx in range(12):
            rows.append(
                {
                    "mention_id": f"{block}::{mention_idx}",
                    "bibcode": f"{block}_{mention_idx}",
                    "author_idx": mention_idx,
                    "author_raw": "Doe, J",
                    "title": f"title {block_idx}",
                    "abstract": "x",
                    "year": 2000 + mention_idx,
                    "source_type": "ads",
                    "block_key": block,
                    "split": "inference",
                }
            )
    df = pd.DataFrame(rows)

    pairs_1, meta_1 = build_pairs_within_blocks(
        df,
        seed=23,
        max_pairs_per_block=15,
        require_same_split=False,
        labeled_only=False,
        balance_train=False,
        exclude_same_bibcode=False,
        num_workers=1,
        sharding_mode="on",
        return_meta=True,
    )
    pairs_4, meta_4 = build_pairs_within_blocks(
        df,
        seed=23,
        max_pairs_per_block=15,
        require_same_split=False,
        labeled_only=False,
        balance_train=False,
        exclude_same_bibcode=False,
        num_workers=4,
        sharding_mode="on",
        return_meta=True,
    )

    key_cols = ["pair_id", "mention_id_1", "mention_id_2", "block_key", "split"]
    lhs = pairs_1[key_cols].sort_values(key_cols).reset_index(drop=True)
    rhs = pairs_4[key_cols].sort_values(key_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(lhs, rhs)
    assert meta_1["pairs_written"] == meta_4["pairs_written"]
    for meta in (meta_1, meta_4):
        assert meta["group_blocks_seconds"] >= 0.0
        assert meta["worker_compute_seconds_total"] >= 0.0
        assert meta["worker_flush_seconds_total"] >= 0.0
        assert isinstance(meta["block_size_histogram"], dict)
        assert isinstance(meta["top_slow_blocks"], list)
        if meta["top_slow_blocks"]:
            wall_seconds = [float(row["wall_seconds"]) for row in meta["top_slow_blocks"]]
            assert wall_seconds == sorted(wall_seconds, reverse=True)


def test_pair_builder_progress_tracks_pair_weights(monkeypatch):
    updates: list[int] = []

    @contextmanager
    def _fake_loop_progress(**kwargs):
        assert kwargs["label"] == "Pair candidates"
        assert kwargs["unit"] == "pair"
        assert kwargs["total"] == 7

        class _Tracker:
            def update(self, n=1):
                updates.append(int(n))

        yield _Tracker()

    monkeypatch.setattr(build_pairs_module, "loop_progress", _fake_loop_progress)

    df = pd.DataFrame(
        [
            {"mention_id": "a1", "bibcode": "a1", "author_idx": 0, "author_raw": "A", "title": "t", "abstract": "a", "year": 2001, "source_type": "ads", "block_key": "blk.a", "split": "inference"},
            {"mention_id": "a2", "bibcode": "a2", "author_idx": 0, "author_raw": "A", "title": "t", "abstract": "a", "year": 2002, "source_type": "ads", "block_key": "blk.a", "split": "inference"},
            {"mention_id": "b1", "bibcode": "b1", "author_idx": 0, "author_raw": "B", "title": "t", "abstract": "a", "year": 2001, "source_type": "ads", "block_key": "blk.b", "split": "inference"},
            {"mention_id": "b2", "bibcode": "b2", "author_idx": 0, "author_raw": "B", "title": "t", "abstract": "a", "year": 2002, "source_type": "ads", "block_key": "blk.b", "split": "inference"},
            {"mention_id": "b3", "bibcode": "b3", "author_idx": 0, "author_raw": "B", "title": "t", "abstract": "a", "year": 2003, "source_type": "ads", "block_key": "blk.b", "split": "inference"},
            {"mention_id": "b4", "bibcode": "b4", "author_idx": 0, "author_raw": "B", "title": "t", "abstract": "a", "year": 2004, "source_type": "ads", "block_key": "blk.b", "split": "inference"},
        ]
    )

    pairs, meta = build_pairs_within_blocks(
        df,
        require_same_split=False,
        labeled_only=False,
        balance_train=False,
        exclude_same_bibcode=False,
        show_progress=True,
        num_workers=1,
        sharding_mode="off",
        return_meta=True,
    )

    assert len(pairs) == 7
    assert updates == [1, 6]
    assert meta["total_pairs_est"] == 7
