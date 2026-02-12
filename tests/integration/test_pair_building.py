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
    split_df = assign_lspo_splits(df, seed=11)
    pairs = build_pairs_within_blocks(split_df, require_same_split=True, labeled_only=False, balance_train=False)
    assert len(pairs) >= 1
    assert set(["pair_id", "mention_id_1", "mention_id_2", "block_key", "split"]).issubset(set(pairs.columns))
