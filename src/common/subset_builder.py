from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.common.io_schema import MENTION_REQUIRED_COLUMNS, validate_columns, save_parquet


@dataclass
class SubsetPlan:
    stage: str
    target_mentions: Optional[int]
    seed: int = 11


STAGE_TARGETS = {
    "smoke": 1_000,
    "mini": 10_000,
    "mid": 100_000,
    "full": None,
}


def _allocate_block_quotas(counts: pd.Series, target: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    n_blocks = len(counts)

    if target <= 0:
        return pd.Series(0, index=counts.index)

    quotas = pd.Series(0, index=counts.index, dtype=int)

    if target >= n_blocks:
        quotas[:] = 1
        remaining = target - n_blocks
        if remaining > 0:
            weights = counts.astype(float)
            weights = weights / weights.sum()
            extra = rng.multinomial(remaining, weights.values)
            quotas += pd.Series(extra, index=counts.index)
    else:
        # Ensure small stages still produce pairable blocks by prioritizing ambiguous blocks.
        # Allocate 2 mentions per top block when possible, then spread remaining budget.
        remaining = target
        ambiguous_idx = counts[counts >= 2].index.tolist()
        max_pairable_blocks = remaining // 2
        top_pairable = ambiguous_idx[:max_pairable_blocks]
        for idx in top_pairable:
            if remaining >= 2:
                quotas.loc[idx] = 2
                remaining -= 2

        if remaining > 0:
            not_chosen = [idx for idx in counts.index if quotas.loc[idx] == 0]
            if not_chosen:
                pick_n = min(remaining, len(not_chosen))
                picked = rng.choice(not_chosen, size=pick_n, replace=False)
                quotas.loc[picked] += 1
                remaining -= pick_n

        # If still remaining, distribute by block size weights.
        if remaining > 0:
            weights = counts.astype(float)
            weights = weights / weights.sum()
            extra = rng.multinomial(remaining, weights.values)
            quotas += pd.Series(extra, index=counts.index)

    quotas = quotas.clip(upper=counts)
    return quotas


def build_stage_subset(
    mentions: pd.DataFrame,
    stage: str,
    seed: int = 11,
    target_mentions: Optional[int] = None,
) -> pd.DataFrame:
    validate_columns(mentions, MENTION_REQUIRED_COLUMNS, "mentions")

    if target_mentions is None:
        target_mentions = STAGE_TARGETS.get(stage)

    if target_mentions is None or stage == "full":
        subset = mentions.copy()
        subset = subset.sort_values(["block_key", "bibcode", "author_idx"]).reset_index(drop=True)
        subset["subset_stage"] = stage
        subset["subset_seed"] = seed
        return subset

    counts = mentions["block_key"].value_counts().sort_values(ascending=False)
    quotas = _allocate_block_quotas(counts, target_mentions, seed)

    rng = np.random.default_rng(seed)
    selected_idx_parts = []
    # Cache block row positions once and sample integer positions per block.
    block_positions = mentions.groupby("block_key", sort=False).indices
    for block_key, quota in quotas.items():
        if quota <= 0:
            continue
        block_idx = block_positions.get(block_key)
        if block_idx is None:
            continue
        block_idx = np.asarray(block_idx, dtype=np.int64)
        if block_idx.size <= quota:
            sampled_idx = block_idx
        else:
            # Keep deterministic per-block behavior while avoiding DataFrame materialization.
            sample_seed = int(rng.integers(0, 2_000_000_000))
            take_pos = np.random.RandomState(sample_seed).choice(block_idx.size, size=int(quota), replace=False)
            sampled_idx = block_idx[take_pos]
        selected_idx_parts.append(sampled_idx)

    if not selected_idx_parts:
        raise ValueError("No subset records sampled. Check stage/target settings.")

    selected_idx = np.concatenate(selected_idx_parts).astype(np.int64, copy=False)
    subset = mentions.iloc[selected_idx].copy()
    if len(subset) > target_mentions:
        subset = subset.sample(n=target_mentions, random_state=seed)

    subset = subset.sort_values(["block_key", "bibcode", "author_idx"]).reset_index(drop=True)
    subset["subset_stage"] = stage
    subset["subset_seed"] = seed
    return subset


def write_subset_manifest(subset_df: pd.DataFrame, output_path: str | Path) -> Path:
    cols = [
        "mention_id",
        "bibcode",
        "author_idx",
        "block_key",
        "subset_stage",
        "subset_seed",
    ]
    manifest = subset_df[cols].copy()
    return save_parquet(manifest, output_path, index=False)
