from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.common.io_schema import PAIR_REQUIRED_COLUMNS, validate_columns, save_parquet


def assign_lspo_splits(
    mentions: pd.DataFrame,
    seed: int = 11,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> pd.DataFrame:
    """Assign split by ORCID groups to avoid identity leakage across splits."""
    out = mentions.copy()
    if "orcid" not in out.columns:
        out["split"] = "inference"
        return out

    known = out["orcid"].notna() & (out["orcid"].astype(str).str.strip() != "")
    unique_orcid = sorted(out.loc[known, "orcid"].astype(str).unique().tolist())

    if not unique_orcid:
        out["split"] = "inference"
        return out

    rng = np.random.default_rng(seed)
    shuffled = unique_orcid.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_set = set(shuffled[:n_train])
    val_set = set(shuffled[n_train : n_train + n_val])
    test_set = set(shuffled[n_train + n_val :])

    def _split(orcid):
        if pd.isna(orcid) or str(orcid).strip() == "":
            return "inference"
        s = str(orcid)
        if s in train_set:
            return "train"
        if s in val_set:
            return "val"
        if s in test_set:
            return "test"
        return "inference"

    out["split"] = out["orcid"].map(_split)
    return out


def _pair_id(m1: str, m2: str) -> str:
    a, b = sorted((m1, m2))
    return f"{a}__{b}"


def build_pairs_within_blocks(
    mentions: pd.DataFrame,
    max_pairs_per_block: Optional[int] = None,
    seed: int = 11,
    require_same_split: bool = True,
    labeled_only: bool = False,
    balance_train: bool = True,
) -> pd.DataFrame:
    """Build mention pairs inside blocks, optionally with labels (LSPO)."""
    if "split" not in mentions.columns:
        mentions = mentions.copy()
        mentions["split"] = "inference"

    rows = []
    rng = np.random.default_rng(seed)

    for block_key, block in mentions.groupby("block_key", sort=False):
        block = block.reset_index(drop=True)
        n = len(block)
        if n < 2:
            continue

        idx_pairs = list(combinations(range(n), 2))
        if max_pairs_per_block is not None and len(idx_pairs) > max_pairs_per_block:
            chosen = rng.choice(len(idx_pairs), size=max_pairs_per_block, replace=False)
            idx_pairs = [idx_pairs[i] for i in np.sort(chosen)]

        for i, j in idx_pairs:
            r1 = block.iloc[i]
            r2 = block.iloc[j]

            split1 = str(r1.get("split", "inference"))
            split2 = str(r2.get("split", "inference"))
            if require_same_split and split1 != split2:
                continue

            split = split1 if split1 == split2 else "mixed"

            label = None
            if "orcid" in block.columns:
                o1, o2 = r1.get("orcid"), r2.get("orcid")
                if pd.notna(o1) and pd.notna(o2) and str(o1).strip() and str(o2).strip():
                    label = int(str(o1) == str(o2))

            if labeled_only and label is None:
                continue

            m1 = str(r1["mention_id"])
            m2 = str(r2["mention_id"])
            rows.append(
                {
                    "pair_id": _pair_id(m1, m2),
                    "mention_id_1": m1,
                    "mention_id_2": m2,
                    "block_key": str(block_key),
                    "split": split,
                    "label": label,
                }
            )

    pairs = pd.DataFrame(rows)
    if len(pairs) == 0:
        return pairs

    if balance_train and "label" in pairs.columns:
        train_mask = (pairs["split"] == "train") & pairs["label"].notna()
        train_df = pairs[train_mask].copy()
        if len(train_df) > 0:
            pos = train_df[train_df["label"] == 1]
            neg = train_df[train_df["label"] == 0]
            if len(pos) > 0 and len(neg) > 0 and len(neg) > len(pos):
                neg = neg.sample(n=len(pos), random_state=seed)
                train_bal = pd.concat([pos, neg], ignore_index=True)
                non_train = pairs[~train_mask]
                pairs = pd.concat([non_train, train_bal], ignore_index=True)

    # Required contract columns.
    validate_columns(pairs, PAIR_REQUIRED_COLUMNS, "pairs")
    pairs = pairs.sort_values(["split", "block_key", "mention_id_1", "mention_id_2"]).reset_index(drop=True)
    return pairs


def write_pairs(pairs: pd.DataFrame, output_path: str | Path) -> Path:
    if len(pairs) == 0:
        raise ValueError("Pair dataframe is empty; nothing to write.")
    return save_parquet(pairs, output_path, index=False)
