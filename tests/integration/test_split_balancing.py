import pandas as pd

from src.approaches.nand.build_pairs import assign_lspo_splits
from src.common.subset_builder import build_stage_subset


def _mentions(n_orcid: int, reps_per_orcid: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n_orcid):
        for j in range(reps_per_orcid):
            rows.append(
                {
                    "mention_id": f"b{i}_{j}::0",
                    "bibcode": f"b{i}_{j}",
                    "author_idx": 0,
                    "author_raw": f"Smith, A{i}",
                    "title": "t",
                    "abstract": "a",
                    "year": 2000 + j,
                    "source_type": "lspo",
                    "block_key": "a.smith",
                    "orcid": f"o{i}",
                }
            )
    return pd.DataFrame(rows)


def test_assign_lspo_splits_reaches_negative_targets_when_possible():
    df = _mentions(n_orcid=20, reps_per_orcid=2)
    split_df, meta = assign_lspo_splits(
        df,
        seed=11,
        train_ratio=0.4,
        val_ratio=0.3,
        min_neg_val=5,
        min_neg_test=5,
        max_attempts=20,
        return_meta=True,
    )

    assert len(split_df) == len(df)
    assert meta["status"] == "ok"
    assert meta["split_label_counts"]["val"]["neg"] >= 5
    assert meta["split_label_counts"]["test"]["neg"] >= 5


def test_assign_lspo_splits_marks_degraded_when_targets_impossible():
    # With 10 ORCIDs and val_ratio=0.1, val/test usually only receive one ORCID each.
    # Then negatives in val/test are structurally impossible.
    df = _mentions(n_orcid=10, reps_per_orcid=2)
    _, meta = assign_lspo_splits(
        df,
        seed=11,
        train_ratio=0.8,
        val_ratio=0.1,
        min_neg_val=1,
        min_neg_test=1,
        max_attempts=10,
        return_meta=True,
    )

    assert meta["status"] == "split_balance_degraded"


def test_assign_lspo_splits_marks_infeasible_when_total_negatives_too_low():
    rows = []
    for i in range(8):
        for rep in range(2):
            rows.append(
                {
                    "mention_id": f"b{i}_{rep}::0",
                    "bibcode": f"b{i}_{rep}",
                    "author_idx": 0,
                    "author_raw": f"Smith, A{i}",
                    "title": "t",
                    "abstract": "a",
                    "year": 2000 + rep,
                    "source_type": "lspo",
                    "block_key": f"blk{i}",  # no cross-ORCID negatives within block
                    "orcid": f"o{i}",
                }
            )
    df = pd.DataFrame(rows)

    _, meta = assign_lspo_splits(
        df,
        seed=11,
        train_ratio=0.6,
        val_ratio=0.2,
        min_neg_val=1,
        min_neg_test=1,
        max_attempts=100,
        return_meta=True,
    )

    assert meta["status"] == "split_balance_infeasible"
    assert meta["max_possible_neg_total"] == 0
    assert meta["required_neg_total"] == 2


def _rich_block_mentions(n_blocks: int = 400) -> pd.DataFrame:
    rows = []
    for block in range(n_blocks):
        # Each block has 3 ORCIDs repeated twice: enough intra-block positives and negatives.
        for orcid_idx in range(3):
            for rep in range(2):
                mention_idx = len(rows)
                rows.append(
                    {
                        "mention_id": f"b{block}_{orcid_idx}_{rep}::0",
                        "bibcode": f"b{mention_idx}",
                        "author_idx": 0,
                        "author_raw": f"A{orcid_idx}",
                        "title": "t",
                        "abstract": "a",
                        "year": 2000,
                        "source_type": "lspo",
                        "block_key": f"blk{block}",
                        "orcid": f"o{block}_{orcid_idx}",
                    }
                )
    return pd.DataFrame(rows)


def test_pair_rich_subset_sampling_improves_split_balance_feasibility():
    mentions = _rich_block_mentions()
    target = 240  # small-stage regime: target < number of blocks

    subset_mean2 = build_stage_subset(
        mentions,
        stage="smoke",
        seed=11,
        target_mentions=target,
        subset_sampling={"target_mean_block_size": 2},
    )
    _, meta_mean2 = assign_lspo_splits(
        subset_mean2,
        seed=11,
        min_neg_val=10,
        min_neg_test=10,
        max_attempts=200,
        return_meta=True,
    )

    subset_mean4 = build_stage_subset(
        mentions,
        stage="smoke",
        seed=11,
        target_mentions=target,
        subset_sampling={"target_mean_block_size": 4},
    )
    _, meta_mean4 = assign_lspo_splits(
        subset_mean4,
        seed=11,
        min_neg_val=10,
        min_neg_test=10,
        max_attempts=200,
        return_meta=True,
    )

    assert meta_mean2["status"] == "split_balance_degraded"
    assert meta_mean4["status"] == "ok"
    assert meta_mean4["split_label_counts"]["val"]["neg"] > meta_mean2["split_label_counts"]["val"]["neg"]
    assert meta_mean4["split_label_counts"]["test"]["neg"] > meta_mean2["split_label_counts"]["test"]["neg"]
