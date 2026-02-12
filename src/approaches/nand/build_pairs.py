from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.common.io_schema import PAIR_REQUIRED_COLUMNS, validate_columns, save_parquet


def _split_sets_from_orcid(
    unique_orcid: list[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[set[str], set[str], set[str]]:
    rng = np.random.default_rng(seed)
    shuffled = unique_orcid.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_set = set(shuffled[:n_train])
    val_set = set(shuffled[n_train : n_train + n_val])
    test_set = set(shuffled[n_train + n_val :])
    return train_set, val_set, test_set


def estimate_split_label_counts(mentions: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Estimate positive/negative labeled pair counts by split without explicit pair building."""
    result: Dict[str, Dict[str, int]] = {
        "train": {"pos": 0, "neg": 0, "labeled_pairs": 0},
        "val": {"pos": 0, "neg": 0, "labeled_pairs": 0},
        "test": {"pos": 0, "neg": 0, "labeled_pairs": 0},
    }
    required = {"block_key", "split", "orcid"}
    if not required.issubset(set(mentions.columns)):
        return result

    df = mentions.copy()
    known = df["orcid"].notna() & (df["orcid"].astype(str).str.strip() != "")
    df = df[known]
    if len(df) == 0:
        return result
    df["orcid"] = df["orcid"].astype(str)

    for (block_key, split), grp in df.groupby(["block_key", "split"], sort=False):
        if split not in result:
            continue
        n = len(grp)
        if n < 2:
            continue
        total_pairs = n * (n - 1) // 2
        pos = 0
        for _, c in grp["orcid"].value_counts().items():
            if c > 1:
                pos += int(c * (c - 1) // 2)
        neg = int(total_pairs - pos)
        result[split]["pos"] += int(pos)
        result[split]["neg"] += int(neg)
        result[split]["labeled_pairs"] += int(total_pairs)

    return result


def assign_lspo_splits(
    mentions: pd.DataFrame,
    seed: int = 11,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    min_neg_val: int = 0,
    min_neg_test: int = 0,
    max_attempts: int = 1,
    return_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, Dict[str, object]]:
    """Assign split by ORCID groups to avoid identity leakage across splits."""
    out = mentions.copy()
    if "orcid" not in out.columns:
        out["split"] = "inference"
        meta = {
            "status": "no_orcid_column",
            "attempts": 0,
            "min_neg_val": int(min_neg_val),
            "min_neg_test": int(min_neg_test),
            "split_label_counts": {"train": {"pos": 0, "neg": 0}, "val": {"pos": 0, "neg": 0}, "test": {"pos": 0, "neg": 0}},
        }
        return (out, meta) if return_meta else out

    known = out["orcid"].notna() & (out["orcid"].astype(str).str.strip() != "")
    unique_orcid = sorted(out.loc[known, "orcid"].astype(str).unique().tolist())

    if not unique_orcid:
        out["split"] = "inference"
        meta = {
            "status": "no_known_orcid",
            "attempts": 0,
            "min_neg_val": int(min_neg_val),
            "min_neg_test": int(min_neg_test),
            "split_label_counts": {"train": {"pos": 0, "neg": 0}, "val": {"pos": 0, "neg": 0}, "test": {"pos": 0, "neg": 0}},
        }
        return (out, meta) if return_meta else out

    max_attempts = max(1, int(max_attempts))
    min_neg_val = max(0, int(min_neg_val))
    min_neg_test = max(0, int(min_neg_test))

    best_df: pd.DataFrame | None = None
    best_counts: Dict[str, Dict[str, int]] | None = None
    best_score = (-1.0, -1.0)
    status = "ok"
    attempts_used = 0

    for attempt in range(max_attempts):
        attempts_used = attempt + 1
        train_set, val_set, test_set = _split_sets_from_orcid(
            unique_orcid=unique_orcid,
            seed=seed + attempt,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

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

        candidate = out.copy()
        candidate["split"] = candidate["orcid"].map(_split)
        counts = estimate_split_label_counts(candidate)
        neg_val = counts.get("val", {}).get("neg", 0)
        neg_test = counts.get("test", {}).get("neg", 0)

        val_score = neg_val / max(1, min_neg_val) if min_neg_val > 0 else 1.0
        test_score = neg_test / max(1, min_neg_test) if min_neg_test > 0 else 1.0
        score = (min(val_score, test_score), float(neg_val + neg_test))

        if score > best_score:
            best_score = score
            best_df = candidate
            best_counts = counts

        if neg_val >= min_neg_val and neg_test >= min_neg_test:
            status = "ok"
            best_df = candidate
            best_counts = counts
            break
    else:
        status = "split_balance_degraded"

    assert best_df is not None
    assert best_counts is not None

    meta = {
        "status": status,
        "attempts": attempts_used,
        "min_neg_val": min_neg_val,
        "min_neg_test": min_neg_test,
        "split_label_counts": best_counts,
    }
    return (best_df, meta) if return_meta else best_df


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
