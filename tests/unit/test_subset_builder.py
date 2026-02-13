import numpy as np
import pandas as pd

from src.common.subset_builder import _allocate_block_quotas, build_stage_subset


def _toy_mentions(n_blocks: int = 120, block_size: int = 8) -> pd.DataFrame:
    rows = []
    for b in range(n_blocks):
        for i in range(block_size):
            idx = b * block_size + i
            rows.append(
                {
                    "mention_id": f"b{b}::{i}",
                    "bibcode": f"b{idx}",
                    "author_idx": i,
                    "author_raw": f"Author {idx}",
                    "title": "t",
                    "abstract": "a",
                    "year": 2000 + (i % 20),
                    "source_type": "toy",
                    "block_key": f"a.block{b}",
                }
            )
    return pd.DataFrame(rows)


def test_subset_deterministic_same_seed():
    mentions = _toy_mentions()
    s1 = build_stage_subset(
        mentions,
        stage="smoke",
        seed=11,
        target_mentions=250,
        subset_sampling={"target_mean_block_size": 4},
    )
    s2 = build_stage_subset(
        mentions,
        stage="smoke",
        seed=11,
        target_mentions=250,
        subset_sampling={"target_mean_block_size": 4},
    )
    assert s1["mention_id"].tolist() == s2["mention_id"].tolist()


def test_subset_target_size_exact():
    mentions = _toy_mentions(n_blocks=90, block_size=7)
    subset = build_stage_subset(
        mentions,
        stage="mini",
        seed=23,
        target_mentions=300,
        subset_sampling={"target_mean_block_size": 4},
    )
    assert len(subset) == 300
    assert (subset["subset_stage"] == "mini").all()
    assert (subset["subset_seed"] == 23).all()


def test_pair_rich_quota_allocation_not_only_two_mentions_per_block():
    counts = pd.Series(
        np.full(300, 8, dtype=int),
        index=[f"a.block{i}" for i in range(300)],
    )
    target = 180  # target < n_blocks -> pair-rich branch
    quotas = _allocate_block_quotas(
        counts=counts,
        target=target,
        seed=11,
        target_mean_block_size=4,
    )

    assert int(quotas.sum()) == target
    assert int((quotas > 0).sum()) < target // 2
    assert int(quotas.max()) > 2


def test_subset_sampling_parameter_changes_selection_shape():
    mentions = _toy_mentions(n_blocks=180, block_size=8)
    target = 120  # keep target < n_blocks so subset_sampling influences allocation
    s_mean2 = build_stage_subset(
        mentions,
        stage="smoke",
        seed=11,
        target_mentions=target,
        subset_sampling={"target_mean_block_size": 2},
    )
    s_mean4 = build_stage_subset(
        mentions,
        stage="smoke",
        seed=11,
        target_mentions=target,
        subset_sampling={"target_mean_block_size": 4},
    )

    # Same target size but different block allocation profile.
    assert len(s_mean2) == len(s_mean4) == target
    assert s_mean2["mention_id"].tolist() != s_mean4["mention_id"].tolist()
