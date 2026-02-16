from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


def _distribute_with_capacity(
    capacity: pd.Series,
    budget: int,
    rng: np.random.Generator,
) -> pd.Series:
    out = pd.Series(0, index=capacity.index, dtype=int)
    remaining = int(max(0, budget))
    avail = capacity.astype(int).clip(lower=0)

    while remaining > 0 and int(avail.sum()) > 0:
        total_avail = int(avail.sum())
        if remaining >= total_avail:
            out += avail
            break

        weights = avail.astype(float)
        weights = weights / weights.sum()
        draw = pd.Series(rng.multinomial(remaining, weights.values), index=avail.index, dtype=int)
        granted = draw.clip(upper=avail)
        out += granted
        remaining -= int(granted.sum())
        avail = (avail - granted).clip(lower=0)

    return out


def _allocate_block_quotas(
    counts: pd.Series,
    target: int,
    seed: int,
    target_mean_block_size: float = 2.0,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    n_blocks = len(counts)

    if target <= 0:
        return pd.Series(0, index=counts.index)

    quotas = pd.Series(0, index=counts.index, dtype=int)

    pairable = counts[counts >= 2]
    if len(pairable) == 0:
        # Degenerate fallback: no pairable blocks available.
        weights = counts.astype(float)
        weights = weights / weights.sum()
        extra = rng.multinomial(min(int(target), int(counts.sum())), weights.values)
        quotas += pd.Series(extra, index=counts.index)
    else:
        mean_block_size = max(2.0, float(target_mean_block_size))
        n_selected_blocks = int(round(target / mean_block_size))
        n_selected_blocks = max(1, min(n_selected_blocks, len(pairable)))

        pairable_idx = pairable.index.to_numpy()
        if len(pairable_idx) == n_selected_blocks:
            selected_idx = pairable_idx
        else:
            weights = pairable.astype(float)
            weights = weights / weights.sum()
            selected_idx = rng.choice(pairable_idx, size=n_selected_blocks, replace=False, p=weights.values)
        selected_set = set(selected_idx.tolist())

        selected_sorted = [idx for idx in counts.index if idx in selected_set]
        quotas.loc[selected_sorted] = 2

        remaining = target - int(quotas.sum())
        if remaining > 0:
            selected_headroom = (counts.loc[selected_sorted] - quotas.loc[selected_sorted]).clip(lower=0).astype(int)
            add_selected = _distribute_with_capacity(selected_headroom, remaining, rng=rng)
            quotas.loc[selected_sorted] += add_selected
            remaining = target - int(quotas.sum())

        if remaining > 0:
            other_sorted = [idx for idx in counts.index if idx not in selected_set]
            if other_sorted:
                other_capacity = (counts.loc[other_sorted] - quotas.loc[other_sorted]).clip(lower=0).astype(int)
                add_others = _distribute_with_capacity(other_capacity, remaining, rng=rng)
                quotas.loc[other_sorted] += add_others
                remaining = target - int(quotas.sum())

        if remaining > 0:
            final_capacity = (counts - quotas).clip(lower=0).astype(int)
            add_final = _distribute_with_capacity(final_capacity, remaining, rng=rng)
            quotas += add_final

    quotas = quotas.clip(upper=counts).astype(int)

    if int(quotas.sum()) < target:
        remaining = int(target - int(quotas.sum()))
        fill_capacity = (counts - quotas).clip(lower=0).astype(int)
        quotas += _distribute_with_capacity(fill_capacity, remaining, rng=rng)

    if int(quotas.sum()) > target:
        overflow = int(int(quotas.sum()) - target)
        removable = quotas.astype(int).clip(lower=0)
        quotas -= _distribute_with_capacity(removable, overflow, rng=rng)

    return quotas


def build_stage_subset(
    mentions: pd.DataFrame,
    stage: str,
    seed: int = 11,
    target_mentions: Optional[int] = None,
    subset_sampling: Optional[dict[str, Any]] = None,
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
    subset_sampling = subset_sampling or {}
    target_mean_block_size = float(subset_sampling.get("target_mean_block_size", 2.0))
    quotas = _allocate_block_quotas(
        counts,
        target_mentions,
        seed,
        target_mean_block_size=target_mean_block_size,
    )

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
    if len(subset) < target_mentions:
        missing = int(target_mentions - len(subset))
        chosen = pd.Index(selected_idx)
        remaining_pool = mentions.loc[~mentions.index.isin(chosen)]
        if len(remaining_pool) > 0:
            top_up = remaining_pool.sample(n=min(missing, len(remaining_pool)), random_state=seed)
            subset = pd.concat([subset, top_up], ignore_index=True)
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
