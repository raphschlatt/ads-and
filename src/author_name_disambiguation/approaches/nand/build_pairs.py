from __future__ import annotations

from collections import Counter
import hashlib
import multiprocessing as mp
import tempfile
from time import perf_counter
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from author_name_disambiguation.common.cli_ui import loop_progress
from author_name_disambiguation.common.cpu_runtime import (
    cap_workers_by_ram,
    detect_cpu_limit,
    resolve_effective_workers,
    sharding_enabled,
)
from author_name_disambiguation.common.io_schema import PAIR_REQUIRED_COLUMNS, save_parquet, validate_columns


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
            "split_label_counts": {
                "train": {"pos": 0, "neg": 0},
                "val": {"pos": 0, "neg": 0},
                "test": {"pos": 0, "neg": 0},
            },
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
            "split_label_counts": {
                "train": {"pos": 0, "neg": 0},
                "val": {"pos": 0, "neg": 0},
                "test": {"pos": 0, "neg": 0},
            },
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


def _seed_for_block(global_seed: int, block_key: str) -> int:
    blob = f"{int(global_seed)}::{block_key}".encode("utf-8")
    digest = hashlib.sha1(blob).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _block_weight(n: int, max_pairs_per_block: Optional[int]) -> int:
    pairs = _pair_count(n)
    if max_pairs_per_block is not None:
        return int(min(pairs, int(max_pairs_per_block)))
    return int(pairs)


def _new_pair_buffer() -> dict[str, list[Any]]:
    return {
        "pair_id": [],
        "mention_id_1": [],
        "mention_id_2": [],
        "block_key": [],
        "split": [],
        "label": [],
    }


def _buffer_len(buffer_rows: dict[str, list[Any]]) -> int:
    return int(len(buffer_rows["pair_id"]))


def _buffer_clear(buffer_rows: dict[str, list[Any]]) -> None:
    for values in buffer_rows.values():
        values.clear()


def _buffer_extend(dest: dict[str, list[Any]], src: dict[str, list[Any]]) -> None:
    for key, values in src.items():
        dest[key].extend(values)


def _buffer_to_frame(buffer_rows: dict[str, list[Any]]) -> pd.DataFrame:
    return pd.DataFrame(buffer_rows, columns=["pair_id", "mention_id_1", "mention_id_2", "block_key", "split", "label"])


def _append_pair_row(
    buffer_rows: dict[str, list[Any]],
    *,
    pair_id: str,
    mention_id_1: str,
    mention_id_2: str,
    block_key: str,
    split: str,
    label: int | None,
) -> None:
    buffer_rows["pair_id"].append(pair_id)
    buffer_rows["mention_id_1"].append(mention_id_1)
    buffer_rows["mention_id_2"].append(mention_id_2)
    buffer_rows["block_key"].append(block_key)
    buffer_rows["split"].append(split)
    buffer_rows["label"].append(label)


def _prepare_block_arrays(block: pd.DataFrame) -> dict[str, Any]:
    n = int(len(block))
    mention_ids = block["mention_id"].map(str).to_numpy(dtype=object, copy=False)
    if "bibcode" in block.columns:
        bibcodes = block["bibcode"].fillna("").astype(str).str.strip().to_numpy(dtype=object, copy=False)
    else:
        bibcodes = None
    if "split" in block.columns:
        splits = block["split"].map(str).to_numpy(dtype=object, copy=False)
    else:
        splits = np.full(n, "inference", dtype=object)
    orcids = block["orcid"].to_numpy(dtype=object, copy=False) if "orcid" in block.columns else None
    return {
        "mention_ids": mention_ids,
        "bibcodes": bibcodes,
        "splits": splits,
        "orcids": orcids,
    }


def _record_top_slow_block(top_blocks: list[dict[str, Any]], block_meta: dict[str, Any], limit: int = 20) -> None:
    top_blocks.append(block_meta)
    top_blocks.sort(key=lambda row: (-float(row["wall_seconds"]), str(row["block_key"])))
    if len(top_blocks) > int(limit):
        del top_blocks[int(limit) :]


def _flush_pair_buffer(
    *,
    buffer_rows: dict[str, list[Any]],
    output: Path | None,
    writer_holder: dict[str, Any],
    return_rows: dict[str, list[Any]] | None,
) -> int:
    if _buffer_len(buffer_rows) == 0:
        return 0

    chunk_df = _buffer_to_frame(buffer_rows)
    written = int(len(chunk_df))

    if return_rows is not None:
        _buffer_extend(return_rows, buffer_rows)

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
            if writer_holder.get("writer") is None:
                writer_holder["schema"] = table.schema
                writer_holder["writer"] = pq.ParquetWriter(output, writer_holder["schema"])
            elif writer_holder.get("schema") is not None and table.schema != writer_holder["schema"]:
                table = table.cast(writer_holder["schema"])
            writer_holder["writer"].write_table(table)

    _buffer_clear(buffer_rows)
    return written


def _execute_pair_blocks(
    *,
    blocks: list[tuple[str, pd.DataFrame]],
    max_pairs_per_block: Optional[int],
    seed: int,
    require_same_split: bool,
    labeled_only: bool,
    exclude_same_bibcode: bool,
    show_progress: bool,
    output_path: str | Path | None,
    chunk_rows: int,
    return_pairs: bool,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame | None, Dict[str, Any]]:
    output = Path(output_path) if output_path is not None else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()

    rows = _new_pair_buffer() if return_pairs else None
    keep_rows = rows if return_pairs else None
    buffer_rows = _new_pair_buffer()

    same_publication_pairs_skipped = 0
    pairs_written = 0
    writer_holder: dict[str, Any] = {"writer": None, "schema": None}
    flush_seconds_total = 0.0
    worker_compute_seconds_total = 0.0
    top_slow_blocks: list[dict[str, Any]] = []

    for block_key, block in blocks:
        n = len(block)
        if n < 2:
            continue

        block_started_at = perf_counter()
        pair_weight = _block_weight(n, max_pairs_per_block=max_pairs_per_block)
        rng = np.random.default_rng(_seed_for_block(seed, str(block_key)))
        block_arrays = _prepare_block_arrays(block)
        mention_ids = block_arrays["mention_ids"]
        bibcodes = block_arrays["bibcodes"]
        splits = block_arrays["splits"]
        orcids = block_arrays["orcids"]
        block_flush_seconds = 0.0
        block_pairs_emitted = 0
        block_same_publication_pairs_skipped = 0
        for i, j in _iter_pair_indices(n=n, max_pairs_per_block=max_pairs_per_block, rng=rng):
            if exclude_same_bibcode and bibcodes is not None:
                b1 = str(bibcodes[i])
                b2 = str(bibcodes[j])
                if b1 and b2 and b1 == b2:
                    same_publication_pairs_skipped += 1
                    block_same_publication_pairs_skipped += 1
                    continue

            split1 = str(splits[i])
            split2 = str(splits[j])
            if require_same_split and split1 != split2:
                continue

            split = split1 if split1 == split2 else "mixed"
            label = None
            if orcids is not None:
                o1 = orcids[i]
                o2 = orcids[j]
                if pd.notna(o1) and pd.notna(o2) and str(o1).strip() and str(o2).strip():
                    label = int(str(o1) == str(o2))

            if labeled_only and label is None:
                continue

            m1 = str(mention_ids[i])
            m2 = str(mention_ids[j])
            _append_pair_row(
                buffer_rows,
                pair_id=_pair_id(m1, m2),
                mention_id_1=m1,
                mention_id_2=m2,
                block_key=str(block_key),
                split=split,
                label=label,
            )
            block_pairs_emitted += 1

            if _buffer_len(buffer_rows) >= max(1, int(chunk_rows)):
                flush_started_at = perf_counter()
                written = _flush_pair_buffer(
                    buffer_rows=buffer_rows,
                    output=output,
                    writer_holder=writer_holder,
                    return_rows=keep_rows,
                )
                flush_elapsed = perf_counter() - flush_started_at
                pairs_written += written
                flush_seconds_total += flush_elapsed
                block_flush_seconds += flush_elapsed
        block_wall_seconds = perf_counter() - block_started_at
        worker_compute_seconds_total += max(0.0, block_wall_seconds - block_flush_seconds)
        _record_top_slow_block(
            top_slow_blocks,
            {
                "block_key": str(block_key),
                "block_size": int(n),
                "pair_weight": int(pair_weight),
                "pairs_written": int(block_pairs_emitted),
                "same_publication_pairs_skipped": int(block_same_publication_pairs_skipped),
                "wall_seconds": float(block_wall_seconds),
            },
        )
        if progress_callback is not None and pair_weight > 0:
            progress_callback(int(pair_weight))

    final_flush_started_at = perf_counter()
    written = _flush_pair_buffer(
        buffer_rows=buffer_rows,
        output=output,
        writer_holder=writer_holder,
        return_rows=keep_rows,
    )
    final_flush_elapsed = perf_counter() - final_flush_started_at
    pairs_written += written
    flush_seconds_total += final_flush_elapsed

    if writer_holder.get("writer") is not None:
        writer_holder["writer"].close()

    readback_seconds = 0.0
    if output is not None and not return_pairs:
        if pairs_written == 0:
            empty = pd.DataFrame(columns=PAIR_REQUIRED_COLUMNS + ["label"])
            save_parquet(empty, output, index=False)
        pairs = None
    elif not return_pairs:
        pairs = None
    elif output is not None and keep_rows is not None and _buffer_len(keep_rows) == 0 and output.exists():
        readback_started_at = perf_counter()
        pairs = pd.read_parquet(output)
        readback_seconds = perf_counter() - readback_started_at
    else:
        pairs = _buffer_to_frame(keep_rows or _new_pair_buffer())

    meta: Dict[str, Any] = {
        "same_publication_pairs_skipped": int(same_publication_pairs_skipped),
        "pairs_written": int(pairs_written),
        "chunk_rows": int(chunk_rows),
        "output_path": str(output) if output is not None else None,
        "worker_compute_seconds_total": float(worker_compute_seconds_total),
        "worker_flush_seconds_total": float(flush_seconds_total),
        "top_slow_blocks": top_slow_blocks,
        "readback_seconds": float(readback_seconds),
    }
    return pairs, meta


def _partition_block_entries(
    entries: list[dict[str, Any]],
    num_shards: int,
) -> list[list[tuple[str, pd.DataFrame]]]:
    shard_count = int(max(1, num_shards))
    shards: list[list[tuple[str, pd.DataFrame]]] = [[] for _ in range(shard_count)]
    loads: list[int] = [0 for _ in range(shard_count)]

    # Longest-processing-time-first, deterministic tie-break by original order.
    ordered = sorted(
        list(enumerate(entries)),
        key=lambda item: (-int(item[1]["pair_weight"]), int(item[0])),
    )
    for _, entry in ordered:
        shard_idx = min(range(shard_count), key=lambda idx: (loads[idx], idx))
        shards[shard_idx].append((str(entry["block_key"]), entry["block"]))
        loads[shard_idx] += int(entry["pair_weight"])

    return shards


def _merge_parquet_shards(shard_paths: list[Path], output_path: Path) -> int:
    valid_paths = [p for p in shard_paths if p.exists()]
    if not valid_paths:
        empty = pd.DataFrame(columns=PAIR_REQUIRED_COLUMNS + ["label"])
        save_parquet(empty, output_path, index=False)
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_rows = 0
    sorted_paths = sorted(valid_paths, key=lambda p: p.name)

    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        frames = []
        for p in sorted_paths:
            df = pd.read_parquet(p)
            if len(df) > 0:
                frames.append(df)
            total_rows += int(len(df))
        if frames:
            merged = pd.concat(frames, ignore_index=True)
        else:
            merged = pd.DataFrame(columns=PAIR_REQUIRED_COLUMNS + ["label"])
        save_parquet(merged, output_path, index=False)
        return int(len(merged))

    writer = None
    schema = None
    try:
        for path in sorted_paths:
            parquet_file = pq.ParquetFile(path)
            for row_group_idx in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group_idx)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(output_path, schema)
                elif schema is not None and table.schema != schema:
                    table = table.cast(schema)
                writer.write_table(table)
                total_rows += int(table.num_rows)
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0 and not output_path.exists():
        empty = pd.DataFrame(columns=PAIR_REQUIRED_COLUMNS + ["label"])
        save_parquet(empty, output_path, index=False)
    return int(total_rows)


def _pair_worker(payload: dict[str, Any]) -> dict[str, Any]:
    output_path = payload.get("output_path")
    pairs, meta = _execute_pair_blocks(
        blocks=payload["blocks"],
        max_pairs_per_block=payload["max_pairs_per_block"],
        seed=payload["seed"],
        require_same_split=payload["require_same_split"],
        labeled_only=payload["labeled_only"],
        exclude_same_bibcode=payload["exclude_same_bibcode"],
        show_progress=False,
        output_path=output_path,
        chunk_rows=payload["chunk_rows"],
        return_pairs=bool(payload.get("return_pairs", False)),
    )
    out = {
        "meta": meta,
        "output_path": str(output_path) if output_path is not None else None,
    }
    if bool(payload.get("return_pairs", False)):
        out["rows"] = [] if pairs is None else pairs.to_dict(orient="records")
    return out


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
    num_workers: int | None = None,
    sharding_mode: str = "auto",
    min_pairs_per_worker: int = 1_000_000,
    ram_budget_bytes: int | None = None,
) -> pd.DataFrame | None | tuple[pd.DataFrame | None, Dict[str, Any]]:
    """Build mention pairs inside blocks, optionally with labels (LSPO)."""
    if "split" not in mentions.columns:
        mentions = mentions.copy()
        mentions["split"] = "inference"

    group_blocks_started_at = perf_counter()
    grouped = mentions.groupby("block_key", sort=False)
    block_entries: list[dict[str, Any]] = []
    block_size_histogram_counter: Counter[int] = Counter()
    for block_key, block in grouped:
        block = block.reset_index(drop=True)
        n = len(block)
        if n < 2:
            continue
        block_size_histogram_counter[int(n)] += 1
        weight = _block_weight(n, max_pairs_per_block=max_pairs_per_block)
        block_entries.append(
            {
                "block_key": str(block_key),
                "block": block,
                "size": int(n),
                "pair_weight": int(weight),
                "ram_est_bytes": int(max(1, weight) * 160),
            }
        )
    group_blocks_seconds = perf_counter() - group_blocks_started_at

    total_pairs_est = int(sum(int(e["pair_weight"]) for e in block_entries))
    n_blocks = int(len(block_entries))

    cpu_info = detect_cpu_limit()
    worker_info = resolve_effective_workers(
        total_pairs_est=total_pairs_est,
        n_blocks=n_blocks,
        requested_workers=num_workers,
        cpu_limit=int(cpu_info["cpu_limit"]),
        min_pairs_per_worker=int(min_pairs_per_worker),
    )
    requested_repr = worker_info["requested"]
    workers_effective = int(worker_info["effective"])

    largest_block_bytes = int(max((int(e["ram_est_bytes"]) for e in block_entries), default=0))
    workers_effective = cap_workers_by_ram(
        workers=workers_effective,
        ram_budget_bytes=ram_budget_bytes,
        per_worker_bytes=max(1, largest_block_bytes),
    )

    sharding_on = sharding_enabled(
        sharding_mode=sharding_mode,
        effective_workers=workers_effective,
        total_pairs_est=total_pairs_est,
        min_pairs_per_worker=int(min_pairs_per_worker),
    )

    output = Path(output_path) if output_path is not None else None
    oversize_entries: list[dict[str, Any]] = []
    parallel_entries: list[dict[str, Any]] = list(block_entries)

    if sharding_on and ram_budget_bytes is not None and ram_budget_bytes > 0:
        for entry in block_entries:
            if int(entry["ram_est_bytes"]) > int(ram_budget_bytes):
                oversize_entries.append(entry)
        if oversize_entries:
            oversize_keys = [str(e["block_key"]) for e in oversize_entries[:5]]
            if len(oversize_entries) > 5:
                oversize_keys.append("...")
            warnings.warn(
                (
                    "Pair-building detected oversized blocks beyond RAM budget; "
                    f"processing {len(oversize_entries)} block(s) sequentially. "
                    f"examples={oversize_keys}"
                ),
                RuntimeWarning,
            )
            oversize_set = {str(e["block_key"]) for e in oversize_entries}
            parallel_entries = [e for e in block_entries if str(e["block_key"]) not in oversize_set]

    sequential_fallback = (not sharding_on) or workers_effective <= 1 or len(parallel_entries) <= 1
    partition_shards_seconds = 0.0
    oversize_sequential_seconds = 0.0
    worker_submit_seconds = 0.0
    worker_collect_seconds = 0.0
    merge_shards_seconds = 0.0
    final_readback_seconds = 0.0

    with loop_progress(
        total=total_pairs_est,
        label="Pair candidates",
        enabled=show_progress,
        unit="pair",
    ) as tracker:
        progress_update = tracker.update if show_progress else None

        if sequential_fallback:
            blocks = [(str(e["block_key"]), e["block"]) for e in block_entries]
            pairs, pair_meta = _execute_pair_blocks(
                blocks=blocks,
                max_pairs_per_block=max_pairs_per_block,
                seed=seed,
                require_same_split=require_same_split,
                labeled_only=labeled_only,
                exclude_same_bibcode=exclude_same_bibcode,
                show_progress=False,
                output_path=output,
                chunk_rows=chunk_rows,
                return_pairs=return_pairs,
                progress_callback=progress_update,
            )
            final_readback_seconds = float(pair_meta.get("readback_seconds", 0.0))
        else:
            shard_count = int(min(workers_effective, len(parallel_entries)))
            progress_shard_count = int(min(len(parallel_entries), max(shard_count, shard_count * 8)))
            partition_started_at = perf_counter()
            shards = _partition_block_entries(parallel_entries, num_shards=progress_shard_count)
            partition_shards_seconds = perf_counter() - partition_started_at
            collect_in_memory = False

            with tempfile.TemporaryDirectory(prefix="pair_shards_") as tmpdir:
                tmp_root = Path(tmpdir)
                shard_paths: list[Path] = []
                shard_metas: list[dict[str, Any]] = []

                if oversize_entries:
                    oversize_blocks = [(str(e["block_key"]), e["block"]) for e in oversize_entries]
                    oversize_out = None if collect_in_memory else tmp_root / "pairs_oversize.parquet"
                    oversize_started_at = perf_counter()
                    _pairs_oversize, oversize_meta = _execute_pair_blocks(
                        blocks=oversize_blocks,
                        max_pairs_per_block=max_pairs_per_block,
                        seed=seed,
                        require_same_split=require_same_split,
                        labeled_only=labeled_only,
                        exclude_same_bibcode=exclude_same_bibcode,
                        show_progress=False,
                        output_path=oversize_out,
                        chunk_rows=chunk_rows,
                        return_pairs=collect_in_memory,
                        progress_callback=progress_update,
                    )
                    oversize_sequential_seconds += perf_counter() - oversize_started_at
                    if oversize_out is not None:
                        shard_paths.append(oversize_out)
                    shard_metas.append(oversize_meta)

                ctx = mp.get_context("spawn")
                futures = {}
                with ProcessPoolExecutor(max_workers=shard_count, mp_context=ctx) as pool:
                    submit_started_at = perf_counter()
                    for shard_idx, shard_blocks in enumerate(shards):
                        shard_out = None if collect_in_memory else tmp_root / f"pairs_shard_{shard_idx:04d}.parquet"
                        if shard_out is not None:
                            shard_paths.append(shard_out)
                        payload = {
                            "blocks": shard_blocks,
                            "max_pairs_per_block": max_pairs_per_block,
                            "seed": int(seed),
                            "require_same_split": bool(require_same_split),
                            "labeled_only": bool(labeled_only),
                            "exclude_same_bibcode": bool(exclude_same_bibcode),
                            "chunk_rows": int(chunk_rows),
                            "output_path": None if shard_out is None else str(shard_out),
                            "return_pairs": bool(collect_in_memory),
                        }
                        shard_weight = int(
                            sum(_block_weight(len(block), max_pairs_per_block=max_pairs_per_block) for _, block in shard_blocks)
                        )
                        futures[pool.submit(_pair_worker, payload)] = shard_weight
                    worker_submit_seconds += perf_counter() - submit_started_at

                    collect_started_at = perf_counter()
                    for fut in as_completed(futures):
                        result = dict(fut.result())
                        shard_metas.append(dict(result.get("meta", {})))
                        if progress_update is not None:
                            progress_update(int(futures[fut]))
                    worker_collect_seconds += perf_counter() - collect_started_at

                if collect_in_memory:
                    pairs = _pairs_oversize
                    merged_rows = int(0 if pairs is None else len(pairs))
                    final_output = output
                else:
                    if output is not None:
                        final_output = output
                    else:
                        final_output = tmp_root / "pairs_merged.parquet"
                    merge_started_at = perf_counter()
                    merged_rows = _merge_parquet_shards(shard_paths=shard_paths, output_path=final_output)
                    merge_shards_seconds += perf_counter() - merge_started_at

                same_publication_pairs_skipped = int(
                    sum(int(m.get("same_publication_pairs_skipped", 0)) for m in shard_metas)
                )
                worker_compute_seconds_total = float(
                    sum(float(m.get("worker_compute_seconds_total", 0.0) or 0.0) for m in shard_metas)
                )
                worker_flush_seconds_total = float(
                    sum(float(m.get("worker_flush_seconds_total", 0.0) or 0.0) for m in shard_metas)
                )
                top_slow_blocks: list[dict[str, Any]] = []
                for shard_meta in shard_metas:
                    for block_meta in list(shard_meta.get("top_slow_blocks", []) or []):
                        if isinstance(block_meta, dict):
                            _record_top_slow_block(top_slow_blocks, dict(block_meta))
                pair_meta = {
                    "same_publication_pairs_skipped": same_publication_pairs_skipped,
                    "pairs_written": int(merged_rows),
                    "chunk_rows": int(chunk_rows),
                    "output_path": None if final_output is None else str(final_output),
                    "worker_compute_seconds_total": worker_compute_seconds_total,
                    "worker_flush_seconds_total": worker_flush_seconds_total,
                    "top_slow_blocks": top_slow_blocks,
                    "readback_seconds": 0.0,
                }

                if collect_in_memory:
                    pass
                elif output is None and not return_pairs:
                    pairs = None
                elif return_pairs:
                    readback_started_at = perf_counter()
                    pairs = pd.read_parquet(final_output)
                    final_readback_seconds += perf_counter() - readback_started_at
                    pair_meta["readback_seconds"] = float(final_readback_seconds)
                else:
                    pairs = None

    meta: Dict[str, Any] = {
        "exclude_same_bibcode": bool(exclude_same_bibcode),
        "same_publication_pairs_skipped": int(pair_meta.get("same_publication_pairs_skipped", 0)),
        "balance_train": bool(balance_train),
        "pairs_written": int(pair_meta.get("pairs_written", 0)),
        "chunk_rows": int(chunk_rows),
        "output_path": str(output) if output is not None else None,
        "cpu_sharding_mode": str(sharding_mode),
        "cpu_sharding_enabled": bool((not sequential_fallback) and sharding_on),
        "cpu_workers_requested": requested_repr,
        "cpu_workers_effective": int(workers_effective),
        "cpu_limit_detected": int(cpu_info["cpu_limit"]),
        "cpu_limit_source": str(cpu_info["cpu_limit_source"]),
        "cpu_min_pairs_per_worker": int(min_pairs_per_worker),
        "ram_budget_bytes": None if ram_budget_bytes is None else int(ram_budget_bytes),
        "total_pairs_est": int(total_pairs_est),
        "group_blocks_seconds": float(group_blocks_seconds),
        "partition_shards_seconds": float(partition_shards_seconds),
        "oversize_sequential_seconds": float(oversize_sequential_seconds),
        "worker_submit_seconds": float(worker_submit_seconds),
        "worker_collect_seconds": float(worker_collect_seconds),
        "merge_shards_seconds": float(merge_shards_seconds),
        "final_readback_seconds": float(final_readback_seconds),
        "worker_compute_seconds_total": float(pair_meta.get("worker_compute_seconds_total", 0.0) or 0.0),
        "worker_flush_seconds_total": float(pair_meta.get("worker_flush_seconds_total", 0.0) or 0.0),
        "top_slow_blocks": list(pair_meta.get("top_slow_blocks", []) or []),
        "block_size_histogram": {
            str(size): int(count) for size, count in sorted(block_size_histogram_counter.items(), key=lambda item: item[0])
        },
    }

    if output is not None and not return_pairs and int(meta["pairs_written"]) == 0:
        empty = pd.DataFrame(columns=PAIR_REQUIRED_COLUMNS + ["label"])
        save_parquet(empty, output, index=False)

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

    validate_columns(pairs, PAIR_REQUIRED_COLUMNS, "pairs")
    pairs = pairs.sort_values(["split", "block_key", "mention_id_1", "mention_id_2"]).reset_index(drop=True)
    return (pairs, meta) if return_meta else pairs


def write_pairs(pairs: pd.DataFrame, output_path: str | Path) -> Path:
    if len(pairs) == 0:
        raise ValueError("Pair dataframe is empty; nothing to write.")
    return save_parquet(pairs, output_path, index=False)
