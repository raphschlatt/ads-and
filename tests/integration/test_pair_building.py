import pandas as pd

from src.approaches.nand.build_pairs import assign_lspo_splits, build_pairs_within_blocks


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
