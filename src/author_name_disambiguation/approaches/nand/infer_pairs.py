from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Any
import warnings

import numpy as np
import pandas as pd

from author_name_disambiguation.approaches.nand.modeling import create_encoder
from author_name_disambiguation.approaches.nand.train import build_feature_matrix as _legacy_build_feature_matrix
from author_name_disambiguation.common.cli_ui import iter_progress
from author_name_disambiguation.common.io_schema import PAIR_SCORE_REQUIRED_COLUMNS, validate_columns, save_parquet
from author_name_disambiguation.common.numeric_safety import clamp_cosine_sim, compute_safe_distance_from_cosine
from author_name_disambiguation.common.torch_runtime import apply_auto_cuda_move_fallback, resolve_torch_device

# Backward-compatible export for tests/legacy monkeypatch points.
build_feature_matrix = _legacy_build_feature_matrix

PAIR_NUMERIC_HELPER_COLUMNS = ("mention_idx_1", "mention_idx_2", "block_idx")


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for NAND inference.") from exc
    return torch


def _build_mention_index(mentions: pd.DataFrame) -> Dict[str, int]:
    return {str(m): i for i, m in enumerate(mentions["mention_id"].tolist())}


def _open_array_view(value: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")


def _init_encoder_runtime_meta(device: str) -> dict[str, Any]:
    return {
        "requested_device": str(device),
        "resolved_device": None,
        "fallback_reason": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "torch_cuda_available": None,
        "cuda_probe_error": None,
        "model_to_cuda_error": None,
        "effective_precision_mode": None,
        "mention_encode_seconds": 0.0,
        "mention_embedding_shape": None,
        "mention_embedding_bytes": 0,
        "mention_norm_bytes": 0,
        "cuda_oom_fallback_used": False,
    }


def _resolve_device(torch, device: str) -> str:
    resolved, _ = resolve_torch_device(torch, device, runtime_label="Pair scoring")
    return resolved


def _resolve_effective_precision_mode(torch, precision_mode: str, device: str) -> str:
    mode = str(precision_mode or "fp32").strip().lower()
    if mode not in {"fp32", "amp_bf16"}:
        warnings.warn(f"Unknown precision_mode={precision_mode!r}; falling back to fp32.", RuntimeWarning)
        return "fp32"
    if mode == "fp32":
        return "fp32"
    if not str(device).startswith("cuda"):
        warnings.warn("precision_mode=amp_bf16 requested on non-CUDA device; falling back to fp32.", RuntimeWarning)
        return "fp32"
    is_supported = True
    try:
        if hasattr(torch.cuda, "is_bf16_supported"):
            is_supported = bool(torch.cuda.is_bf16_supported())
    except Exception:
        is_supported = False
    if not is_supported:
        warnings.warn("CUDA BF16 is not supported in this environment; falling back to fp32.", RuntimeWarning)
        return "fp32"
    return "amp_bf16"


def _autocast_context(torch, precision_mode: str):
    if str(precision_mode) == "amp_bf16":
        if hasattr(torch, "autocast"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()
    return nullcontext()


def _is_cuda_oom_error(torch, exc: Exception) -> bool:
    oom_types: tuple[type[BaseException], ...] = tuple()
    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is not None:
        oom_error = getattr(cuda_module, "OutOfMemoryError", None)
        if isinstance(oom_error, type) and issubclass(oom_error, BaseException):
            oom_types = (oom_error,)
    if oom_types and isinstance(exc, oom_types):
        return True
    return "out of memory" in str(exc).strip().lower()


def _best_effort_clear_cuda_cache(torch) -> None:
    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is None:
        return
    try:
        if hasattr(cuda_module, "is_available") and not bool(cuda_module.is_available()):
            return
        if hasattr(cuda_module, "empty_cache"):
            cuda_module.empty_cache()
    except Exception:
        return


def _init_numeric_clamp_summary() -> dict[str, int | bool]:
    return {
        "clamped": False,
        "events": 0,
        "cosine_non_finite_count": 0,
        "cosine_below_min_count": 0,
        "cosine_above_max_count": 0,
        "distance_non_finite_count": 0,
        "distance_below_min_count": 0,
        "distance_above_max_count": 0,
    }


def _accumulate_numeric_clamp_summary(
    target: dict[str, int | bool],
    *,
    sim_meta: dict[str, Any],
    dist_meta: dict[str, Any],
) -> None:
    if not sim_meta.get("clamped") and not dist_meta.get("clamped"):
        return
    target["clamped"] = True
    target["events"] = int(target.get("events", 0)) + 1
    target["cosine_non_finite_count"] = int(target.get("cosine_non_finite_count", 0)) + int(
        sim_meta.get("non_finite_count", 0)
    )
    target["cosine_below_min_count"] = int(target.get("cosine_below_min_count", 0)) + int(sim_meta.get("below_min_count", 0))
    target["cosine_above_max_count"] = int(target.get("cosine_above_max_count", 0)) + int(sim_meta.get("above_max_count", 0))
    target["distance_non_finite_count"] = int(target.get("distance_non_finite_count", 0)) + int(
        dist_meta.get("non_finite_count", 0)
    )
    target["distance_below_min_count"] = int(target.get("distance_below_min_count", 0)) + int(
        dist_meta.get("below_min_count", 0)
    )
    target["distance_above_max_count"] = int(target.get("distance_above_max_count", 0)) + int(
        dist_meta.get("above_max_count", 0)
    )


def load_checkpoint(checkpoint_path: str | Path, device: str = "auto") -> Dict[str, Any]:
    torch = _require_torch()
    # Load on CPU first to avoid hard failures during CUDA deserialization.
    return torch.load(checkpoint_path, map_location="cpu")


def _encode_mentions(
    *,
    torch,
    model,
    features: np.ndarray,
    batch_size: int,
    device: str,
    precision_mode: str,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=np.float32)

    total = (len(features) + batch_size - 1) // batch_size
    starts = iter_progress(
        range(0, len(features), batch_size),
        total=total,
        label="Encode mentions",
        enabled=show_progress,
        unit="batch",
        compact_visible=False,
        emit_events=False,
    )
    mention_embeddings: np.ndarray | None = None

    with torch.no_grad():
        for start in starts:
            end = min(start + batch_size, len(features))
            batch = torch.from_numpy(np.asarray(features[start:end], dtype=np.float32)).to(device)
            with _autocast_context(torch, precision_mode):
                z = model(batch)
            batch_embeddings = np.asarray(z.detach().cpu().numpy(), dtype=np.float32)
            if mention_embeddings is None:
                if batch_embeddings.ndim != 2:
                    raise ValueError(f"Unexpected encoder output shape: {batch_embeddings.shape}")
                mention_embeddings = np.empty((len(features), batch_embeddings.shape[1]), dtype=np.float32)
            mention_embeddings[start:end] = batch_embeddings

    if mention_embeddings is None:
        mention_embeddings = np.zeros((0, 0), dtype=np.float32)

    mention_norms = (
        np.linalg.norm(mention_embeddings, axis=1).astype(np.float32, copy=False)
        if len(mention_embeddings)
        else np.array([], dtype=np.float32)
    )
    return mention_embeddings, mention_norms


def _pair_index_array(values: np.ndarray, mention_index: Dict[str, int]) -> np.ndarray:
    return np.fromiter((int(mention_index.get(str(value), -1)) for value in values), dtype=np.int64, count=len(values))


def _numeric_index_array(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.size == 0:
        return np.asarray(arr, dtype=np.int64)
    return np.asarray(arr, dtype=np.int64)


def _resolve_numeric_helper_mode(
    *,
    mention_id_1: np.ndarray,
    mention_id_2: np.ndarray,
    mention_idx_1: np.ndarray | None,
    mention_idx_2: np.ndarray | None,
    mention_ids_by_index: np.ndarray | None,
) -> tuple[bool, str]:
    numeric_idx1 = _numeric_index_array(mention_idx_1)
    numeric_idx2 = _numeric_index_array(mention_idx_2)
    if numeric_idx1 is None or numeric_idx2 is None:
        return False, "missing_helper_columns"
    if mention_ids_by_index is None:
        return False, "missing_mention_id_order"
    if len(numeric_idx1) != len(mention_id_1) or len(numeric_idx2) != len(mention_id_2):
        return False, "length_mismatch"
    if len(numeric_idx1) == 0:
        return True, "validated_empty"

    mention_ids = np.asarray(mention_ids_by_index, dtype=object)
    n_mentions = int(len(mention_ids))
    if n_mentions == 0:
        return False, "empty_mentions"

    in_range_1 = (numeric_idx1 >= 0) & (numeric_idx1 < n_mentions)
    in_range_2 = (numeric_idx2 >= 0) & (numeric_idx2 < n_mentions)
    if not bool(np.all(in_range_1)) or not bool(np.all(in_range_2)):
        return False, "out_of_range"

    if not np.array_equal(np.asarray(mention_ids[numeric_idx1], dtype=object), np.asarray(mention_id_1, dtype=object)):
        return False, "mention_id_mismatch"
    if not np.array_equal(np.asarray(mention_ids[numeric_idx2], dtype=object), np.asarray(mention_id_2, dtype=object)):
        return False, "mention_id_mismatch"
    return True, "validated"


def _arrow_string_column(batch, column_name: str) -> np.ndarray:
    field_index = batch.schema.get_field_index(column_name)
    if field_index < 0:
        raise KeyError(column_name)
    return np.asarray(batch.column(field_index).to_pylist(), dtype=object)


def _arrow_numeric_column(batch, column_name: str) -> np.ndarray | None:
    field_index = batch.schema.get_field_index(column_name)
    if field_index < 0:
        return None
    return np.asarray(batch.column(field_index).to_numpy(zero_copy_only=False), dtype=np.int64)


def _extract_pair_batch_columns(batch) -> dict[str, np.ndarray]:
    columns = {
        "pair_id": _arrow_string_column(batch, "pair_id"),
        "mention_id_1": _arrow_string_column(batch, "mention_id_1"),
        "mention_id_2": _arrow_string_column(batch, "mention_id_2"),
        "block_key": _arrow_string_column(batch, "block_key"),
    }
    for column_name in PAIR_NUMERIC_HELPER_COLUMNS:
        values = _arrow_numeric_column(batch, column_name)
        if values is not None:
            columns[column_name] = values
    return columns


def _public_score_columns(score_columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        column: score_columns[column]
        for column in PAIR_SCORE_REQUIRED_COLUMNS
    }


def _build_feature_batch(
    *,
    chars2vec: np.ndarray,
    source_text_embeddings: np.ndarray,
    mention_source_index: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    name_batch = np.asarray(chars2vec[start:end], dtype=np.float32)
    source_indices = np.asarray(mention_source_index[start:end], dtype=np.int64)
    if len(source_indices) != len(name_batch):
        raise ValueError("mention_source_index length must match chars2vec row count.")
    text_batch = np.asarray(source_text_embeddings[source_indices], dtype=np.float32)
    if len(text_batch) != len(name_batch):
        raise ValueError("source_text_embeddings gather length mismatch.")
    return np.concatenate([name_batch, text_batch], axis=1).astype(np.float32, copy=False)


def _load_encoder_for_inference(
    *,
    torch,
    checkpoint_path: str | Path,
    requested_device: str,
) -> tuple[Any, str, dict[str, Any]]:
    runtime_meta = _init_encoder_runtime_meta(requested_device)
    device, device_meta = resolve_torch_device(torch, requested_device, runtime_label="Pair scoring")
    runtime_meta.update(dict(device_meta))

    checkpoint = load_checkpoint(checkpoint_path=checkpoint_path, device=device)
    model = create_encoder(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    try:
        model.to(device)
    except Exception as exc:
        if str(requested_device).strip().lower() == "auto" and str(device).startswith("cuda"):
            device, runtime_meta = apply_auto_cuda_move_fallback(
                requested_device=requested_device,
                runtime_label="Pair scoring",
                runtime_meta=runtime_meta,
                exc=exc,
            )
            model.to(device)
        else:
            raise
    model.eval()
    return model, str(device), runtime_meta


def _write_empty_npy(path: str | Path, array: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, array)
    return target


def encode_mentions_to_memmap(
    *,
    chars2vec: np.ndarray | str | Path,
    source_text_embeddings: np.ndarray | str | Path,
    mention_source_index: np.ndarray | str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    norms_output_path: str | Path | None = None,
    batch_size: int = 8192,
    device: str = "auto",
    precision_mode: str = "fp32",
    show_progress: bool = False,
    return_runtime_meta: bool = False,
) -> Path | tuple[Path, dict[str, Any]]:
    torch = _require_torch()
    batch_size = max(1, int(batch_size))
    requested_device = str(device)
    chars = _open_array_view(chars2vec)
    source_emb = _open_array_view(source_text_embeddings)
    mention_to_source = _open_array_view(mention_source_index)
    output = Path(output_path)
    norms_output = output.with_name(output.stem + "_norms.npy") if norms_output_path is None else Path(norms_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    norms_output.parent.mkdir(parents=True, exist_ok=True)

    runtime_meta = _init_encoder_runtime_meta(requested_device)

    if len(chars) == 0:
        _write_empty_npy(output, np.zeros((0, 0), dtype=np.float32))
        _write_empty_npy(norms_output, np.zeros((0,), dtype=np.float32))
        runtime_meta["resolved_device"] = "cpu"
        runtime_meta["effective_precision_mode"] = "fp32"
        runtime_meta["mention_embedding_shape"] = [0, 0]
        result: Path | tuple[Path, dict[str, Any]] = output
        if return_runtime_meta:
            return output, runtime_meta
        return result

    if len(chars) != len(mention_to_source):
        raise ValueError("chars2vec and mention_source_index lengths must match.")

    model, active_device, runtime_meta = _load_encoder_for_inference(
        torch=torch,
        checkpoint_path=checkpoint_path,
        requested_device=requested_device,
    )

    def _encode(active_device_name: str, active_runtime_meta: dict[str, Any]) -> dict[str, Any]:
        effective_precision_mode = _resolve_effective_precision_mode(
            torch=torch,
            precision_mode=precision_mode,
            device=active_device_name,
        )
        active_runtime_meta["resolved_device"] = str(active_device_name)
        active_runtime_meta["effective_precision_mode"] = effective_precision_mode
        total = (len(chars) + batch_size - 1) // batch_size
        starts = iter_progress(
            range(0, len(chars), batch_size),
            total=total,
            label="Encode mentions",
            enabled=show_progress,
            unit="batch",
            compact_visible=False,
            emit_events=False,
        )
        embeddings_mm = None
        norms_mm = np.lib.format.open_memmap(norms_output, mode="w+", dtype=np.float32, shape=(len(chars),))
        encode_started_at = perf_counter()
        with torch.no_grad():
            for start in starts:
                end = min(start + batch_size, len(chars))
                feature_batch = _build_feature_batch(
                    chars2vec=chars,
                    source_text_embeddings=source_emb,
                    mention_source_index=mention_to_source,
                    start=start,
                    end=end,
                )
                batch_tensor = torch.from_numpy(np.asarray(feature_batch, dtype=np.float32)).to(active_device_name)
                with _autocast_context(torch, effective_precision_mode):
                    z = model(batch_tensor)
                batch_embeddings = np.asarray(z.detach().cpu().numpy(), dtype=np.float32)
                if embeddings_mm is None:
                    if batch_embeddings.ndim != 2:
                        raise ValueError(f"Unexpected encoder output shape: {batch_embeddings.shape}")
                    embeddings_mm = np.lib.format.open_memmap(
                        output,
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(chars), batch_embeddings.shape[1]),
                    )
                embeddings_mm[start:end] = batch_embeddings
                norms_mm[start:end] = np.linalg.norm(batch_embeddings, axis=1).astype(np.float32, copy=False)
        active_runtime_meta["mention_encode_seconds"] = float(perf_counter() - encode_started_at)
        if embeddings_mm is None:
            _write_empty_npy(output, np.zeros((0, 0), dtype=np.float32))
            active_runtime_meta["mention_embedding_shape"] = [0, 0]
            active_runtime_meta["mention_embedding_bytes"] = 0
            active_runtime_meta["mention_norm_bytes"] = int(norms_mm.nbytes)
        else:
            embeddings_mm.flush()
            active_runtime_meta["mention_embedding_shape"] = [int(embeddings_mm.shape[0]), int(embeddings_mm.shape[1])]
            active_runtime_meta["mention_embedding_bytes"] = int(embeddings_mm.nbytes)
            active_runtime_meta["mention_norm_bytes"] = int(norms_mm.nbytes)
        norms_mm.flush()
        return active_runtime_meta

    try:
        runtime_meta = _encode(active_device, runtime_meta)
    except Exception as exc:
        can_retry_on_cpu = (
            str(requested_device).strip().lower() == "auto"
            and str(active_device).startswith("cuda")
            and _is_cuda_oom_error(torch, exc)
        )
        if not can_retry_on_cpu:
            raise
        warnings.warn(
            "Mention encoding hit CUDA OOM; retrying on CPU for this run.",
            RuntimeWarning,
        )
        try:
            model.to("cpu")
        except Exception:
            pass
        _best_effort_clear_cuda_cache(torch)
        retry_meta = dict(runtime_meta)
        retry_meta["resolved_device"] = "cpu"
        retry_meta["fallback_reason"] = "pair_scoring_cuda_oom_retry_cpu"
        retry_meta["cuda_oom_fallback_used"] = True
        retry_meta = _encode("cpu", retry_meta)
        runtime_meta = retry_meta

    if return_runtime_meta:
        return output, runtime_meta
    return output


def score_pairs_from_mention_embeddings(
    *,
    mentions: pd.DataFrame,
    pairs: pd.DataFrame | str | Path,
    mention_embeddings: np.ndarray | str | Path,
    mention_norms: np.ndarray | str | Path | None = None,
    output_path: str | Path | None = None,
    batch_size: int = 8192,
    show_progress: bool = False,
    chunk_rows: int | None = None,
    return_scores: bool = True,
    return_runtime_meta: bool = False,
    score_callback: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    batch_size = max(1, int(batch_size))
    mindex = _build_mention_index(mentions)
    mention_ids_by_index = mentions["mention_id"].astype(str).to_numpy(dtype=object, copy=False)
    embedding_view = _open_array_view(mention_embeddings)
    if mention_norms is None:
        norms_view = (
            np.linalg.norm(np.asarray(embedding_view), axis=1).astype(np.float32, copy=False)
            if len(embedding_view)
            else np.array([], dtype=np.float32)
        )
    else:
        norms_view = _open_array_view(mention_norms)

    runtime_meta: dict[str, Any] = {
        "pair_scoring_strategy": "preencoded_mentions_memmap"
        if not isinstance(mention_embeddings, np.ndarray)
        else "preencoded_mentions",
        "mention_storage_device": "disk_memmap" if not isinstance(mention_embeddings, np.ndarray) else "cpu",
        "cuda_oom_fallback_used": False,
        "score_batch_size": int(batch_size),
        "parquet_read_seconds": 0.0,
        "pandas_conversion_seconds": 0.0,
        "arrow_column_extract_seconds": 0.0,
        "pair_score_seconds": 0.0,
        "parquet_output_table_seconds": 0.0,
        "parquet_write_seconds": 0.0,
        "pairs_total_rows": 0,
        "pairs_valid_rows": 0,
        "numeric_clamping": _init_numeric_clamp_summary(),
        "mention_embedding_shape": (
            [int(embedding_view.shape[0]), int(embedding_view.shape[1])]
            if getattr(embedding_view, "ndim", 0) == 2
            else [int(len(embedding_view))]
        ),
        "mention_embedding_bytes": int(getattr(embedding_view, "nbytes", 0)),
        "mention_norm_bytes": int(getattr(norms_view, "nbytes", 0)),
    }

    def _score_columns_to_output(score_columns: dict[str, np.ndarray], out_rows: list[pd.DataFrame]) -> None:
        if score_callback is not None:
            score_callback(score_columns)
        if return_scores:
            out_rows.append(_scored_pair_arrays_to_frame(score_columns))

    if isinstance(pairs, (str, Path)):
        pair_path = Path(pairs)
        if not pair_path.exists():
            raise FileNotFoundError(pair_path)
        runtime_meta["pair_input_mode"] = "parquet_chunked"
        active_chunk_rows = chunk_rows if chunk_rows is not None else max(10_000, int(batch_size) * 4)
        out_rows: list[pd.DataFrame] = []
        writer = None
        writer_schema = None
        out_path = None if output_path is None else Path(output_path)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                out_path.unlink()

        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:
            raise RuntimeError("Chunked parquet scoring requires pyarrow.") from exc

        parquet = pq.ParquetFile(pair_path)
        batch_iter = parquet.iter_batches(batch_size=int(active_chunk_rows))
        while True:
            read_started_at = perf_counter()
            try:
                batch = next(batch_iter)
            except StopIteration:
                break
            runtime_meta["parquet_read_seconds"] = float(runtime_meta.get("parquet_read_seconds", 0.0)) + float(
                perf_counter() - read_started_at
            )
            extract_started_at = perf_counter()
            pair_columns = _extract_pair_batch_columns(batch)
            score_columns = _build_scored_pair_arrays(
                pair_id=pair_columns["pair_id"],
                mention_id_1=pair_columns["mention_id_1"],
                mention_id_2=pair_columns["mention_id_2"],
                block_key=pair_columns["block_key"],
                mention_idx_1=pair_columns.get("mention_idx_1"),
                mention_idx_2=pair_columns.get("mention_idx_2"),
                block_idx=pair_columns.get("block_idx"),
                mention_index=mindex,
                mention_ids_by_index=mention_ids_by_index,
                mention_embeddings=embedding_view,
                mention_norms=norms_view,
                batch_size=batch_size,
                show_progress=show_progress,
                active_runtime_meta=runtime_meta,
            )
            runtime_meta["arrow_column_extract_seconds"] = float(
                runtime_meta.get("arrow_column_extract_seconds", 0.0)
            ) + float(perf_counter() - extract_started_at)
            _score_columns_to_output(score_columns, out_rows)
            if out_path is not None and len(score_columns["pair_id"]) > 0:
                table_started_at = perf_counter()
                table = pa.Table.from_pydict(_public_score_columns(score_columns))
                runtime_meta["parquet_output_table_seconds"] = float(
                    runtime_meta.get("parquet_output_table_seconds", 0.0)
                ) + float(perf_counter() - table_started_at)
                write_started_at = perf_counter()
                if writer is None:
                    writer_schema = table.schema
                    writer = pq.ParquetWriter(out_path, writer_schema)
                elif writer_schema is not None and table.schema != writer_schema:
                    table = table.cast(writer_schema)
                writer.write_table(table)
                runtime_meta["parquet_write_seconds"] = float(runtime_meta.get("parquet_write_seconds", 0.0)) + float(
                    perf_counter() - write_started_at
                )

        if writer is not None:
            writer.close()
        if out_path is not None and writer is None:
            save_parquet(pd.DataFrame(columns=PAIR_SCORE_REQUIRED_COLUMNS), out_path, index=False)

        if not return_scores:
            result = pd.DataFrame(columns=PAIR_SCORE_REQUIRED_COLUMNS)
        elif not out_rows:
            result = pd.DataFrame(columns=PAIR_SCORE_REQUIRED_COLUMNS)
        else:
            result = pd.concat(out_rows, ignore_index=True)
            validate_columns(result, PAIR_SCORE_REQUIRED_COLUMNS, "pair_scores")
        if return_runtime_meta:
            return result, runtime_meta
        return result

    runtime_meta["pair_input_mode"] = "dataframe"
    score_columns = _build_scored_pair_arrays(
        pair_id=pairs["pair_id"].astype(str).to_numpy(copy=False),
        mention_id_1=pairs["mention_id_1"].astype(str).to_numpy(copy=False),
        mention_id_2=pairs["mention_id_2"].astype(str).to_numpy(copy=False),
        block_key=pairs["block_key"].astype(str).to_numpy(copy=False),
        mention_idx_1=(
            pairs["mention_idx_1"].to_numpy(copy=False)
            if "mention_idx_1" in pairs.columns
            else None
        ),
        mention_idx_2=(
            pairs["mention_idx_2"].to_numpy(copy=False)
            if "mention_idx_2" in pairs.columns
            else None
        ),
        block_idx=(
            pairs["block_idx"].to_numpy(copy=False)
            if "block_idx" in pairs.columns
            else None
        ),
        mention_index=mindex,
        mention_ids_by_index=mention_ids_by_index,
        mention_embeddings=embedding_view,
        mention_norms=norms_view,
        batch_size=batch_size,
        show_progress=show_progress,
        active_runtime_meta=runtime_meta,
    )
    if score_callback is not None:
        score_callback(score_columns)
    result = _scored_pair_arrays_to_frame(score_columns)
    if output_path is not None:
        save_parquet(result, output_path, index=False)
    if return_runtime_meta:
        return result, runtime_meta
    return result


def _build_scored_pair_arrays(
    *,
    pair_id: np.ndarray,
    mention_id_1: np.ndarray,
    mention_id_2: np.ndarray,
    block_key: np.ndarray,
    mention_idx_1: np.ndarray | None,
    mention_idx_2: np.ndarray | None,
    block_idx: np.ndarray | None,
    mention_index: Dict[str, int],
    mention_ids_by_index: np.ndarray | None,
    mention_embeddings: np.ndarray,
    mention_norms: np.ndarray,
    batch_size: int,
    show_progress: bool,
    active_runtime_meta: dict[str, Any],
) -> dict[str, np.ndarray]:
    numeric_idx1 = _numeric_index_array(mention_idx_1)
    numeric_idx2 = _numeric_index_array(mention_idx_2)
    use_numeric_helpers, helper_reason = _resolve_numeric_helper_mode(
        mention_id_1=mention_id_1,
        mention_id_2=mention_id_2,
        mention_idx_1=numeric_idx1,
        mention_idx_2=numeric_idx2,
        mention_ids_by_index=mention_ids_by_index,
    )
    idx1 = numeric_idx1 if use_numeric_helpers else _pair_index_array(mention_id_1, mention_index)
    idx2 = numeric_idx2 if use_numeric_helpers else _pair_index_array(mention_id_2, mention_index)
    use_numeric_block_helpers = use_numeric_helpers and block_idx is not None
    active_runtime_meta["pair_index_mode"] = "numeric_helper_columns" if use_numeric_helpers else "mention_id_lookup"
    active_runtime_meta["pair_index_fallback_reason"] = None if use_numeric_helpers else helper_reason
    active_runtime_meta["block_index_mode"] = "numeric_helper_columns" if use_numeric_block_helpers else "block_key_only"

    valid_mask = (idx1 >= 0) & (idx2 >= 0)
    idx1_valid = idx1[valid_mask]
    idx2_valid = idx2[valid_mask]
    active_runtime_meta["pairs_total_rows"] = int(active_runtime_meta.get("pairs_total_rows", 0)) + int(len(pair_id))
    active_runtime_meta["pairs_valid_rows"] = int(active_runtime_meta.get("pairs_valid_rows", 0)) + int(len(idx1_valid))

    sims = []
    total = (len(idx1_valid) + batch_size - 1) // batch_size
    starts = iter_progress(
        range(0, len(idx1_valid), batch_size),
        total=total,
        label="Score batches",
        enabled=show_progress,
        unit="batch",
        compact_visible=False,
        emit_events=False,
    )
    score_started_at = perf_counter()
    for start in starts:
        end = min(start + batch_size, len(idx1_valid))
        batch_idx1 = idx1_valid[start:end]
        batch_idx2 = idx2_valid[start:end]
        z1 = mention_embeddings[batch_idx1]
        z2 = mention_embeddings[batch_idx2]
        dot = np.einsum("ij,ij->i", z1, z2, optimize=True)
        denom = np.maximum(mention_norms[batch_idx1] * mention_norms[batch_idx2], 1e-8)
        sims.append((dot / denom).astype(np.float32, copy=False))
    active_runtime_meta["pair_score_seconds"] = float(active_runtime_meta.get("pair_score_seconds", 0.0)) + float(
        perf_counter() - score_started_at
    )

    sim_arr = np.concatenate(sims, axis=0) if sims else np.array([], dtype=np.float32)
    sim_arr, sim_meta = clamp_cosine_sim(sim_arr)
    dist_arr, dist_meta = compute_safe_distance_from_cosine(sim_arr)
    _accumulate_numeric_clamp_summary(
        active_runtime_meta.setdefault("numeric_clamping", _init_numeric_clamp_summary()),
        sim_meta=sim_meta,
        dist_meta=dist_meta,
    )

    result = {
        "pair_id": np.asarray(pair_id[valid_mask], dtype=object),
        "mention_id_1": np.asarray(mention_id_1[valid_mask], dtype=object),
        "mention_id_2": np.asarray(mention_id_2[valid_mask], dtype=object),
        "block_key": np.asarray(block_key[valid_mask], dtype=object),
        "cosine_sim": sim_arr.astype(np.float32, copy=False),
        "distance": dist_arr.astype(np.float32, copy=False),
    }
    if use_numeric_helpers:
        result["mention_idx_1"] = idx1_valid.astype(np.int64, copy=False)
        result["mention_idx_2"] = idx2_valid.astype(np.int64, copy=False)
    if use_numeric_block_helpers:
        result["block_idx"] = _numeric_index_array(block_idx[valid_mask]).astype(np.int64, copy=False)
    return result


def _scored_pair_arrays_to_frame(score_columns: dict[str, np.ndarray]) -> pd.DataFrame:
    out = pd.DataFrame(score_columns, columns=PAIR_SCORE_REQUIRED_COLUMNS)
    validate_columns(out, PAIR_SCORE_REQUIRED_COLUMNS, "pair_scores")
    return out


def score_pairs_with_checkpoint(
    mentions: pd.DataFrame,
    pairs: pd.DataFrame | str | Path,
    chars2vec: np.ndarray,
    text_emb: np.ndarray,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    batch_size: int = 8192,
    device: str = "auto",
    precision_mode: str = "fp32",
    show_progress: bool = False,
    chunk_rows: int | None = None,
    return_scores: bool = True,
    return_runtime_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    torch = _require_torch()
    batch_size = max(1, int(batch_size))
    requested_device = device
    device, runtime_meta = resolve_torch_device(torch, device, runtime_label="Pair scoring")

    checkpoint = load_checkpoint(checkpoint_path=checkpoint_path, device=device)
    model = create_encoder(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    try:
        model.to(device)
    except Exception as exc:
        if str(requested_device).strip().lower() == "auto" and str(device).startswith("cuda"):
            device, runtime_meta = apply_auto_cuda_move_fallback(
                requested_device=requested_device,
                runtime_label="Pair scoring",
                runtime_meta=runtime_meta,
                exc=exc,
            )
            model.to(device)
        else:
            raise
    model.eval()
    runtime_meta["pair_scoring_strategy"] = "preencoded_mentions"
    runtime_meta["mention_storage_device"] = "cpu"
    runtime_meta["cuda_oom_fallback_used"] = False
    runtime_meta["score_batch_size"] = int(batch_size)
    runtime_meta["feature_build_seconds"] = 0.0
    runtime_meta["mention_encode_seconds"] = 0.0
    runtime_meta["parquet_read_seconds"] = 0.0
    runtime_meta["pandas_conversion_seconds"] = 0.0
    runtime_meta["arrow_column_extract_seconds"] = 0.0
    runtime_meta["pair_score_seconds"] = 0.0
    runtime_meta["parquet_output_table_seconds"] = 0.0
    runtime_meta["parquet_write_seconds"] = 0.0
    runtime_meta["pairs_total_rows"] = 0
    runtime_meta["pairs_valid_rows"] = 0
    runtime_meta["numeric_clamping"] = _init_numeric_clamp_summary()

    mindex = _build_mention_index(mentions)
    mention_ids_by_index = mentions["mention_id"].astype(str).to_numpy(dtype=object, copy=False)
    feature_started_at = perf_counter()
    features = build_feature_matrix(chars2vec=chars2vec, text_emb=text_emb)
    runtime_meta["feature_build_seconds"] = float(perf_counter() - feature_started_at)
    runtime_meta["feature_matrix_shape"] = [int(features.shape[0]), int(features.shape[1])] if features.ndim == 2 else [int(len(features))]
    runtime_meta["feature_matrix_bytes"] = int(features.nbytes)

    def _score_pairs_frame(
        pairs_df: pd.DataFrame,
        *,
        mention_embeddings: np.ndarray,
        mention_norms: np.ndarray,
        active_runtime_meta: dict[str, Any],
    ) -> pd.DataFrame:
        score_columns = _build_scored_pair_arrays(
            pair_id=pairs_df["pair_id"].astype(str).to_numpy(copy=False),
            mention_id_1=pairs_df["mention_id_1"].astype(str).to_numpy(copy=False),
            mention_id_2=pairs_df["mention_id_2"].astype(str).to_numpy(copy=False),
            block_key=pairs_df["block_key"].astype(str).to_numpy(copy=False),
            mention_idx_1=(
                pairs_df["mention_idx_1"].to_numpy(copy=False)
                if "mention_idx_1" in pairs_df.columns
                else None
            ),
            mention_idx_2=(
                pairs_df["mention_idx_2"].to_numpy(copy=False)
                if "mention_idx_2" in pairs_df.columns
                else None
            ),
            block_idx=(
                pairs_df["block_idx"].to_numpy(copy=False)
                if "block_idx" in pairs_df.columns
                else None
            ),
            mention_index=mindex,
            mention_ids_by_index=mention_ids_by_index,
            mention_embeddings=mention_embeddings,
            mention_norms=mention_norms,
            batch_size=batch_size,
            show_progress=show_progress,
            active_runtime_meta=active_runtime_meta,
        )
        return _scored_pair_arrays_to_frame(score_columns)

    def _run_scoring(active_device: str, active_runtime_meta: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        effective_precision_mode = _resolve_effective_precision_mode(
            torch=torch,
            precision_mode=precision_mode,
            device=active_device,
        )
        active_runtime_meta["resolved_device"] = str(active_device)
        active_runtime_meta["effective_precision_mode"] = effective_precision_mode
        encode_started_at = perf_counter()
        mention_embeddings, mention_norms = _encode_mentions(
            torch=torch,
            model=model,
            features=features,
            batch_size=batch_size,
            device=active_device,
            precision_mode=effective_precision_mode,
            show_progress=show_progress,
        )
        active_runtime_meta["mention_encode_seconds"] = float(perf_counter() - encode_started_at)
        active_runtime_meta["mention_embedding_shape"] = (
            [int(mention_embeddings.shape[0]), int(mention_embeddings.shape[1])]
            if mention_embeddings.ndim == 2
            else [int(len(mention_embeddings))]
        )
        active_runtime_meta["mention_embedding_bytes"] = int(mention_embeddings.nbytes)

        if isinstance(pairs, (str, Path)):
            pair_path = Path(pairs)
            if not pair_path.exists():
                raise FileNotFoundError(pair_path)
            active_runtime_meta["pair_input_mode"] = "parquet_chunked"
            active_chunk_rows = chunk_rows
            if active_chunk_rows is None:
                active_chunk_rows = max(10_000, int(batch_size) * 4)

            out_rows: list[pd.DataFrame] = []
            writer = None
            writer_schema = None
            if output_path is not None:
                out_path = Path(output_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists():
                    out_path.unlink()
            else:
                out_path = None

            try:
                import pyarrow as pa  # type: ignore
                import pyarrow.parquet as pq  # type: ignore
            except Exception as exc:
                raise RuntimeError("Chunked parquet scoring requires pyarrow.") from exc

            parquet = pq.ParquetFile(pair_path)
            batch_iter = parquet.iter_batches(batch_size=int(active_chunk_rows))
            while True:
                read_started_at = perf_counter()
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    break
                active_runtime_meta["parquet_read_seconds"] = float(
                    active_runtime_meta.get("parquet_read_seconds", 0.0)
                ) + float(perf_counter() - read_started_at)
                extract_started_at = perf_counter()
                pair_columns = _extract_pair_batch_columns(batch)
                active_runtime_meta["arrow_column_extract_seconds"] = float(
                    active_runtime_meta.get("arrow_column_extract_seconds", 0.0)
                ) + float(perf_counter() - extract_started_at)
                score_columns = _build_scored_pair_arrays(
                    pair_id=pair_columns["pair_id"],
                    mention_id_1=pair_columns["mention_id_1"],
                    mention_id_2=pair_columns["mention_id_2"],
                    block_key=pair_columns["block_key"],
                    mention_idx_1=pair_columns.get("mention_idx_1"),
                    mention_idx_2=pair_columns.get("mention_idx_2"),
                    block_idx=pair_columns.get("block_idx"),
                    mention_index=mindex,
                    mention_ids_by_index=mention_ids_by_index,
                    mention_embeddings=mention_embeddings,
                    mention_norms=mention_norms,
                    active_runtime_meta=active_runtime_meta,
                    batch_size=batch_size,
                    show_progress=show_progress,
                )
                if return_scores:
                    out_rows.append(_scored_pair_arrays_to_frame(score_columns))
                if out_path is not None and len(score_columns["pair_id"]) > 0:
                    table_started_at = perf_counter()
                    table = pa.Table.from_pydict(_public_score_columns(score_columns))
                    active_runtime_meta["parquet_output_table_seconds"] = float(
                        active_runtime_meta.get("parquet_output_table_seconds", 0.0)
                    ) + float(perf_counter() - table_started_at)
                    write_started_at = perf_counter()
                    if writer is None:
                        writer_schema = table.schema
                        writer = pq.ParquetWriter(out_path, writer_schema)
                    elif writer_schema is not None and table.schema != writer_schema:
                        table = table.cast(writer_schema)
                    writer.write_table(table)
                    active_runtime_meta["parquet_write_seconds"] = float(
                        active_runtime_meta.get("parquet_write_seconds", 0.0)
                    ) + float(perf_counter() - write_started_at)

            if writer is not None:
                writer.close()
            if out_path is not None and writer is None:
                save_parquet(pd.DataFrame(columns=PAIR_SCORE_REQUIRED_COLUMNS), out_path, index=False)

            if return_scores:
                if not out_rows:
                    return pd.DataFrame(columns=PAIR_SCORE_REQUIRED_COLUMNS), active_runtime_meta
                out = pd.concat(out_rows, ignore_index=True)
                validate_columns(out, PAIR_SCORE_REQUIRED_COLUMNS, "pair_scores")
                return out, active_runtime_meta
            return pd.DataFrame(columns=PAIR_SCORE_REQUIRED_COLUMNS), active_runtime_meta

        active_runtime_meta["pair_input_mode"] = "dataframe"
        out = _score_pairs_frame(
            pairs,
            mention_embeddings=mention_embeddings,
            mention_norms=mention_norms,
            active_runtime_meta=active_runtime_meta,
        )
        if output_path is not None:
            save_parquet(out, output_path, index=False)
        return out, active_runtime_meta

    try:
        out, runtime_meta = _run_scoring(device, runtime_meta)
    except Exception as exc:
        can_retry_on_cpu = (
            str(requested_device).strip().lower() == "auto"
            and str(device).startswith("cuda")
            and _is_cuda_oom_error(torch, exc)
        )
        if not can_retry_on_cpu:
            raise
        warnings.warn(
            "Pair scoring hit CUDA OOM; retrying on CPU for this run.",
            RuntimeWarning,
        )
        try:
            model.to("cpu")
        except Exception:
            pass
        _best_effort_clear_cuda_cache(torch)
        retry_runtime_meta = dict(runtime_meta)
        retry_runtime_meta["resolved_device"] = "cpu"
        retry_runtime_meta["fallback_reason"] = "pair_scoring_cuda_oom_retry_cpu"
        retry_runtime_meta["cuda_oom_fallback_used"] = True
        retry_runtime_meta["mention_encode_seconds"] = 0.0
        retry_runtime_meta["pair_score_seconds"] = 0.0
        retry_runtime_meta["parquet_read_seconds"] = 0.0
        retry_runtime_meta["pandas_conversion_seconds"] = 0.0
        retry_runtime_meta["arrow_column_extract_seconds"] = 0.0
        retry_runtime_meta["parquet_output_table_seconds"] = 0.0
        retry_runtime_meta["parquet_write_seconds"] = 0.0
        retry_runtime_meta["pairs_total_rows"] = 0
        retry_runtime_meta["pairs_valid_rows"] = 0
        out, runtime_meta = _run_scoring("cpu", retry_runtime_meta)

    return (out, runtime_meta) if return_runtime_meta else out
