from __future__ import annotations

import hashlib
import multiprocessing as mp
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

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


def _flush_pair_buffer(
    *,
    buffer_rows: list[dict[str, Any]],
    output: Path | None,
    writer_holder: dict[str, Any],
    return_rows: list[dict[str, Any]] | None,
) -> int:
    if not buffer_rows:
        return 0

    chunk_df = pd.DataFrame(buffer_rows)
    written = int(len(chunk_df))

    if return_rows is not None:
        return_rows.extend(buffer_rows)

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

    buffer_rows.clear()
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
) -> tuple[pd.DataFrame | None, Dict[str, Any]]:
    output = Path(output_path) if output_path is not None else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()

    rows: list[dict[str, Any]] = [] if return_pairs else []
    keep_rows = rows if return_pairs else None
    buffer_rows: list[dict[str, Any]] = []

    same_publication_pairs_skipped = 0
    pairs_written = 0
    writer_holder: dict[str, Any] = {"writer": None, "schema": None}

    iterator = blocks
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(blocks, total=len(blocks), desc="Pair blocks", leave=False)
        except Exception:
            pass

    for block_key, block in iterator:
        n = len(block)
        if n < 2:
            continue

        rng = np.random.default_rng(_seed_for_block(seed, str(block_key)))
        for i, j in _iter_pair_indices(n=n, max_pairs_per_block=max_pairs_per_block, rng=rng):
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

            if len(buffer_rows) >= max(1, int(chunk_rows)):
                pairs_written += _flush_pair_buffer(
                    buffer_rows=buffer_rows,
                    output=output,
                    writer_holder=writer_holder,
                    return_rows=keep_rows,
                )

    pairs_written += _flush_pair_buffer(
        buffer_rows=buffer_rows,
        output=output,
        writer_holder=writer_holder,
        return_rows=keep_rows,
    )

    if writer_holder.get("writer") is not None:
        writer_holder["writer"].close()

    if output is not None and not return_pairs:
        if pairs_written == 0:
            empty = pd.DataFrame(columns=PAIR_REQUIRED_COLUMNS + ["label"])
            save_parquet(empty, output, index=False)
        pairs = None
    elif not return_pairs:
        pairs = None
    elif output is not None and not rows and output.exists():
        pairs = pd.read_parquet(output)
    else:
        pairs = pd.DataFrame(rows)

    meta: Dict[str, Any] = {
        "same_publication_pairs_skipped": int(same_publication_pairs_skipped),
        "pairs_written": int(pairs_written),
        "chunk_rows": int(chunk_rows),
        "output_path": str(output) if output is not None else None,
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

    grouped = mentions.groupby("block_key", sort=False)
    block_entries: list[dict[str, Any]] = []
    for block_key, block in grouped:
        block = block.reset_index(drop=True)
        n = len(block)
        if n < 2:
            continue
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

    if sequential_fallback:
        blocks = [(str(e["block_key"]), e["block"]) for e in block_entries]
        pairs, pair_meta = _execute_pair_blocks(
            blocks=blocks,
            max_pairs_per_block=max_pairs_per_block,
            seed=seed,
            require_same_split=require_same_split,
            labeled_only=labeled_only,
            exclude_same_bibcode=exclude_same_bibcode,
            show_progress=show_progress,
            output_path=output,
            chunk_rows=chunk_rows,
            return_pairs=return_pairs,
        )
    else:
        shard_count = int(min(workers_effective, len(parallel_entries)))
        shards = _partition_block_entries(parallel_entries, num_shards=shard_count)
        collect_in_memory = output is None and return_pairs

        with tempfile.TemporaryDirectory(prefix="pair_shards_") as tmpdir:
            tmp_root = Path(tmpdir)
            shard_paths: list[Path] = []
            shard_metas: list[dict[str, Any]] = []
            shard_rows: list[dict[str, Any]] = []

            if oversize_entries:
                oversize_blocks = [(str(e["block_key"]), e["block"]) for e in oversize_entries]
                oversize_out = None if collect_in_memory else tmp_root / "pairs_oversize.parquet"
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
                )
                if oversize_out is not None:
                    shard_paths.append(oversize_out)
                if collect_in_memory and _pairs_oversize is not None and len(_pairs_oversize) > 0:
                    shard_rows.extend(_pairs_oversize.to_dict(orient="records"))
                shard_metas.append(oversize_meta)

            ctx = mp.get_context("spawn")
            futures = []
            with ProcessPoolExecutor(max_workers=shard_count, mp_context=ctx) as pool:
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
                    futures.append(pool.submit(_pair_worker, payload))

                for fut in as_completed(futures):
                    result = dict(fut.result())
                    shard_metas.append(dict(result.get("meta", {})))
                    if collect_in_memory:
                        shard_rows.extend(result.get("rows", []))

            if collect_in_memory:
                pairs = pd.DataFrame(shard_rows)
                merged_rows = int(len(pairs))
                final_output = output
            else:
                if output is not None:
                    final_output = output
                else:
                    final_output = tmp_root / "pairs_merged.parquet"
                merged_rows = _merge_parquet_shards(shard_paths=shard_paths, output_path=final_output)

            same_publication_pairs_skipped = int(
                sum(int(m.get("same_publication_pairs_skipped", 0)) for m in shard_metas)
            )
            pair_meta = {
                "same_publication_pairs_skipped": same_publication_pairs_skipped,
                "pairs_written": int(merged_rows),
                "chunk_rows": int(chunk_rows),
                "output_path": None if final_output is None else str(final_output),
            }

            if collect_in_memory:
                pass
            elif output is None and not return_pairs:
                pairs = None
            elif return_pairs:
                pairs = pd.read_parquet(final_output)
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
        "cpu_sharding_enabled": bool(sharding_on),
        "cpu_workers_requested": requested_repr,
        "cpu_workers_effective": int(workers_effective),
        "cpu_limit_detected": int(cpu_info["cpu_limit"]),
        "cpu_limit_source": str(cpu_info["cpu_limit_source"]),
        "cpu_min_pairs_per_worker": int(min_pairs_per_worker),
        "ram_budget_bytes": None if ram_budget_bytes is None else int(ram_budget_bytes),
        "total_pairs_est": int(total_pairs_est),
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
