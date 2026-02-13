import numpy as np
import pandas as pd

from src.common.subset_builder import _allocate_block_quotas, build_stage_subset


def _toy_mentions(n=200):
    rows = []
    for i in range(n):
        block = f"a.block{i%20}"
        rows.append(
            {
                "mention_id": f"b{i}::0",
                "bibcode": f"b{i}",
                "author_idx": 0,
                "author_raw": f"Author {i}",
                "title": "t",
                "abstract": "a",
                "year": 2000,
                "source_type": "toy",
                "block_key": block,
            }
        )
    return pd.DataFrame(rows)


def _reference_subset(mentions, stage, seed, target_mentions):
    counts = mentions["block_key"].value_counts().sort_values(ascending=False)
    quotas = _allocate_block_quotas(counts, target_mentions, seed)
    rng = np.random.default_rng(seed)
    parts = []
    for block_key, quota in quotas.items():
        if quota <= 0:
            continue
        block_df = mentions[mentions["block_key"] == block_key]
        if len(block_df) <= quota:
            sampled = block_df
        else:
            sampled = block_df.sample(n=int(quota), random_state=int(rng.integers(0, 2_000_000_000)))
        parts.append(sampled)
    subset = pd.concat(parts, ignore_index=True)
    if len(subset) > target_mentions:
        subset = subset.sample(n=target_mentions, random_state=seed)
    subset = subset.sort_values(["block_key", "bibcode", "author_idx"]).reset_index(drop=True)
    subset["subset_stage"] = stage
    subset["subset_seed"] = seed
    return subset


def test_subset_deterministic_same_seed():
    m = _toy_mentions()
    s1 = build_stage_subset(m, stage="mini", seed=11, target_mentions=50)
    s2 = build_stage_subset(m, stage="mini", seed=11, target_mentions=50)
    assert s1["mention_id"].tolist() == s2["mention_id"].tolist()


def test_subset_matches_reference_behavior():
    # Keep this test lightweight while guarding the sampling semantics.
    m = _toy_mentions(n=1_000)
    stage = "smoke"
    seed = 23
    target = 300

    expected = _reference_subset(m, stage=stage, seed=seed, target_mentions=target)
    actual = build_stage_subset(m, stage=stage, seed=seed, target_mentions=target)
    assert actual["mention_id"].tolist() == expected["mention_id"].tolist()
