from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any

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
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
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
    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    if train_ratio <= 0.0 or val_ratio < 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"Invalid split ratios: train_ratio={train_ratio}, val_ratio={val_ratio}. "
            "Require train_ratio>0, val_ratio>=0, and train_ratio+val_ratio<1."
        )

    # Structural feasibility check: if the total possible negatives is below the
    # requested val+test budget, retries cannot satisfy the target.
    total_counts_input = out.copy()
    total_counts_input["split"] = "train"
    total_counts = estimate_split_label_counts(total_counts_input).get("train", {})
    total_possible_neg = int(total_counts.get("neg", 0))
    required_neg = int(min_neg_val + min_neg_test)

    if required_neg > 0 and total_possible_neg < required_neg:
        train_set, val_set, test_set = _split_sets_from_orcid(
            unique_orcid=unique_orcid,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        def _split_infeasible(orcid):
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
        candidate["split"] = candidate["orcid"].map(_split_infeasible)
        counts = estimate_split_label_counts(candidate)
        meta = {
            "status": "split_balance_infeasible",
            "attempts": 0,
            "min_neg_val": min_neg_val,
            "min_neg_test": min_neg_test,
            "required_neg_total": required_neg,
            "max_possible_neg_total": total_possible_neg,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": float(1.0 - train_ratio - val_ratio),
            "split_label_counts": counts,
        }
        return (candidate, meta) if return_meta else candidate

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
        "required_neg_total": required_neg,
        "max_possible_neg_total": total_possible_neg,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": float(1.0 - train_ratio - val_ratio),
        "split_label_counts": best_counts,
    }
    return (best_df, meta) if return_meta else best_df


def _pair_id(m1: str, m2: str) -> str:
    a, b = sorted((m1, m2))
    return f"{a}__{b}"


def _pair_count(n: int) -> int:
    return int(n * (n - 1) // 2)


def _pair_from_rank(rank: int, n: int) -> tuple[int, int]:
    i = int(n - 2 - np.floor((np.sqrt(-8.0 * rank + 4.0 * n * (n - 1) - 7.0) - 1.0) / 2.0))
    prev = int(i * (2 * n - i - 1) // 2)
    j = int(rank - prev + i + 1)
    return i, j


def _iter_pair_indices(
    *,
    n: int,
    max_pairs_per_block: Optional[int],
    rng: np.random.Generator,
):
    total_pairs = _pair_count(n)
    if total_pairs <= 0:
        return
    if max_pairs_per_block is not None and total_pairs > int(max_pairs_per_block):
        sampled_ranks = np.sort(rng.choice(total_pairs, size=int(max_pairs_per_block), replace=False))
        for rank in sampled_ranks.tolist():
            yield _pair_from_rank(int(rank), n)
        return

    for i in range(n - 1):
        for j in range(i + 1, n):
            yield i, j


def build_pairs_within_blocks(
    mentions: pd.DataFrame,
    max_pairs_per_block: Optional[int] = None,
    seed: int = 11,
    require_same_split: bool = True,
    labeled_only: bool = False,
    balance_train: bool = True,
    exclude_same_bibcode: bool = True,
    show_progress: bool = False,
    output_path: str | Path | None = None,
    chunk_rows: int = 200_000,
    return_pairs: bool = True,
    return_meta: bool = False,
) -> pd.DataFrame | None | tuple[pd.DataFrame | None, Dict[str, Any]]:
    """Build mention pairs inside blocks, optionally with labels (LSPO)."""
    if "split" not in mentions.columns:
        mentions = mentions.copy()
        mentions["split"] = "inference"

    rows: list[dict[str, Any]] = []
    buffer_rows: list[dict[str, Any]] = []
    same_publication_pairs_skipped = 0
    pairs_written = 0
    rng = np.random.default_rng(seed)

    grouped = mentions.groupby("block_key", sort=False)
    iterator = grouped
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(grouped, total=int(mentions["block_key"].nunique()), desc="Pair blocks", leave=False)
        except Exception:
            pass

    output = Path(output_path) if output_path is not None else None
    writer = None
    writer_schema = None

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()

    def _flush(force: bool = False) -> None:
        nonlocal writer, writer_schema, pairs_written
        if not buffer_rows:
            return
        if not force and len(buffer_rows) < max(1, int(chunk_rows)):
            return

        chunk_df = pd.DataFrame(buffer_rows)
        pairs_written += int(len(chunk_df))
        if return_pairs:
            rows.extend(buffer_rows)

        if output is not None:
            try:
                import pyarrow as pa  # type: ignore
                import pyarrow.parquet as pq  # type: ignore
            except Exception:
                if output.exists():
                    existing = pd.read_parquet(output)
                    chunk_df = pd.concat([existing, chunk_df], ignore_index=True)
                chunk_df.to_parquet(output, index=False)
            else:
                table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                if writer is None:
                    writer_schema = table.schema
                    writer = pq.ParquetWriter(output, writer_schema)
                elif writer_schema is not None and table.schema != writer_schema:
                    table = table.cast(writer_schema)
                writer.write_table(table)
        buffer_rows.clear()

    for block_key, block in iterator:
        block = block.reset_index(drop=True)
        n = len(block)
        if n < 2:
            continue

        for i, j in _iter_pair_indices(
            n=n,
            max_pairs_per_block=max_pairs_per_block,
            rng=rng,
        ):
            r1 = block.iloc[i]
            r2 = block.iloc[j]

            if exclude_same_bibcode and "bibcode" in block.columns:
                b1 = r1.get("bibcode")
                b2 = r2.get("bibcode")
                if pd.notna(b1) and pd.notna(b2) and str(b1).strip() and str(b2).strip() and str(b1) == str(b2):
                    same_publication_pairs_skipped += 1
                    continue

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
            buffer_rows.append(
                {
                    "pair_id": _pair_id(m1, m2),
                    "mention_id_1": m1,
                    "mention_id_2": m2,
                    "block_key": str(block_key),
                    "split": split,
                    "label": label,
                }
            )
            _flush(force=False)

    _flush(force=True)
    if writer is not None:
        writer.close()

    if output is not None and not return_pairs:
        if pairs_written == 0:
            empty = pd.DataFrame(columns=PAIR_REQUIRED_COLUMNS + ["label"])
            save_parquet(empty, output, index=False)
        pairs = None
    elif output is not None and not rows and output.exists():
        pairs = pd.read_parquet(output)
    else:
        pairs = pd.DataFrame(rows)

    meta: Dict[str, Any] = {
        "exclude_same_bibcode": bool(exclude_same_bibcode),
        "same_publication_pairs_skipped": int(same_publication_pairs_skipped),
        "balance_train": bool(balance_train),
        "pairs_written": int(pairs_written if output is not None else len(rows)),
        "chunk_rows": int(chunk_rows),
        "output_path": str(output) if output is not None else None,
    }

    if pairs is None:
        return (None, meta) if return_meta else None

    if len(pairs) == 0:
        return (pairs, meta) if return_meta else pairs

    if balance_train and "label" in pairs.columns:
        train_mask = (pairs["split"] == "train") & pairs["label"].notna()
        train_df = pairs[train_mask].copy()
        if len(train_df) > 0:
            pos = train_df[train_df["label"] == 1]
            neg = train_df[train_df["label"] == 0]
            meta["train_balance_before"] = {"pos": int(len(pos)), "neg": int(len(neg))}
            if len(pos) > 0 and len(neg) > 0 and len(pos) != len(neg):
                target_n = min(len(pos), len(neg))
                if len(pos) > target_n:
                    pos = pos.sample(n=target_n, random_state=seed)
                if len(neg) > target_n:
                    neg = neg.sample(n=target_n, random_state=seed)
                train_bal = pd.concat([pos, neg], ignore_index=True)
                non_train = pairs[~train_mask]
                pairs = pd.concat([non_train, train_bal], ignore_index=True)
            train_bal_counts = pairs[(pairs["split"] == "train") & pairs["label"].notna()]
            meta["train_balance_after"] = {
                "pos": int((train_bal_counts["label"] == 1).sum()),
                "neg": int((train_bal_counts["label"] == 0).sum()),
            }

    # Required contract columns.
    validate_columns(pairs, PAIR_REQUIRED_COLUMNS, "pairs")
    pairs = pairs.sort_values(["split", "block_key", "mention_id_1", "mention_id_2"]).reset_index(drop=True)
    return (pairs, meta) if return_meta else pairs


def write_pairs(pairs: pd.DataFrame, output_path: str | Path) -> Path:
    if len(pairs) == 0:
        raise ValueError("Pair dataframe is empty; nothing to write.")
    return save_parquet(pairs, output_path, index=False)
