from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Dict, Any
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


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for NAND inference.") from exc
    return torch


def _build_mention_index(mentions: pd.DataFrame) -> Dict[str, int]:
    return {str(m): i for i, m in enumerate(mentions["mention_id"].tolist())}


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
    )
    collect_on_device = str(device).startswith("cuda") and hasattr(torch, "cat")
    tensor_batches: list[Any] = []
    numpy_batches: list[np.ndarray] = []

    with torch.no_grad():
        for start in starts:
            end = min(start + batch_size, len(features))
            batch = torch.from_numpy(np.asarray(features[start:end], dtype=np.float32)).to(device)
            with _autocast_context(torch, precision_mode):
                z = model(batch)
            if collect_on_device:
                tensor_batches.append(z.detach())
            else:
                numpy_batches.append(np.asarray(z.detach().cpu().numpy(), dtype=np.float32))

    if collect_on_device:
        mention_embeddings = (
            torch.cat(tensor_batches, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
            if tensor_batches
            else np.zeros((0, 0), dtype=np.float32)
        )
    else:
        mention_embeddings = (
            np.concatenate(numpy_batches, axis=0).astype(np.float32, copy=False)
            if numpy_batches
            else np.zeros((0, 0), dtype=np.float32)
        )

    mention_norms = (
        np.linalg.norm(mention_embeddings, axis=1).astype(np.float32, copy=False)
        if len(mention_embeddings)
        else np.array([], dtype=np.float32)
    )
    return mention_embeddings, mention_norms


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
    effective_precision_mode = _resolve_effective_precision_mode(torch=torch, precision_mode=precision_mode, device=device)
    runtime_meta["effective_precision_mode"] = effective_precision_mode
    runtime_meta["pair_scoring_strategy"] = "preencoded_mentions"
    runtime_meta["score_batch_size"] = int(batch_size)
    runtime_meta["feature_build_seconds"] = 0.0
    runtime_meta["mention_encode_seconds"] = 0.0
    runtime_meta["parquet_read_seconds"] = 0.0
    runtime_meta["pandas_conversion_seconds"] = 0.0
    runtime_meta["pair_score_seconds"] = 0.0
    runtime_meta["parquet_write_seconds"] = 0.0
    runtime_meta["pairs_total_rows"] = 0
    runtime_meta["pairs_valid_rows"] = 0

    mindex = _build_mention_index(mentions)
    feature_started_at = perf_counter()
    features = build_feature_matrix(chars2vec=chars2vec, text_emb=text_emb)
    runtime_meta["feature_build_seconds"] = float(perf_counter() - feature_started_at)
    runtime_meta["feature_matrix_shape"] = [int(features.shape[0]), int(features.shape[1])] if features.ndim == 2 else [int(len(features))]
    runtime_meta["feature_matrix_bytes"] = int(features.nbytes)

    encode_started_at = perf_counter()
    mention_embeddings, mention_norms = _encode_mentions(
        torch=torch,
        model=model,
        features=features,
        batch_size=batch_size,
        device=device,
        precision_mode=effective_precision_mode,
        show_progress=show_progress,
    )
    runtime_meta["mention_encode_seconds"] = float(perf_counter() - encode_started_at)
    runtime_meta["mention_embedding_shape"] = (
        [int(mention_embeddings.shape[0]), int(mention_embeddings.shape[1])]
        if mention_embeddings.ndim == 2
        else [int(len(mention_embeddings))]
    )
    runtime_meta["mention_embedding_bytes"] = int(mention_embeddings.nbytes)

    def _score_pairs_frame(pairs_df: pd.DataFrame) -> pd.DataFrame:
        idx1 = pairs_df["mention_id_1"].astype(str).map(mindex).values
        idx2 = pairs_df["mention_id_2"].astype(str).map(mindex).values

        valid_mask = ~(pd.isna(idx1) | pd.isna(idx2))
        p = pairs_df.loc[valid_mask].copy().reset_index(drop=True)
        idx1_valid = idx1[valid_mask].astype(int)
        idx2_valid = idx2[valid_mask].astype(int)
        runtime_meta["pairs_total_rows"] = int(runtime_meta.get("pairs_total_rows", 0)) + int(len(pairs_df))
        runtime_meta["pairs_valid_rows"] = int(runtime_meta.get("pairs_valid_rows", 0)) + int(len(p))

        sims = []
        total = (len(p) + batch_size - 1) // batch_size
        starts = iter_progress(
            range(0, len(p), batch_size),
            total=total,
            label="Score batches",
            enabled=show_progress,
            unit="batch",
        )
        score_started_at = perf_counter()
        for start in starts:
            end = min(start + batch_size, len(p))
            batch_idx1 = idx1_valid[start:end]
            batch_idx2 = idx2_valid[start:end]
            z1 = mention_embeddings[batch_idx1]
            z2 = mention_embeddings[batch_idx2]
            dot = np.einsum("ij,ij->i", z1, z2, optimize=True)
            denom = np.maximum(mention_norms[batch_idx1] * mention_norms[batch_idx2], 1e-8)
            sims.append((dot / denom).astype(np.float32, copy=False))
        runtime_meta["pair_score_seconds"] = float(runtime_meta.get("pair_score_seconds", 0.0)) + float(
            perf_counter() - score_started_at
        )

        sim_arr = np.concatenate(sims, axis=0) if sims else np.array([], dtype=np.float32)
        sim_arr, sim_meta = clamp_cosine_sim(sim_arr)
        dist_arr, dist_meta = compute_safe_distance_from_cosine(sim_arr)
        if sim_meta["clamped"] or dist_meta["clamped"]:
            warnings.warn(
                (
                    "Applied numeric clamping to pair scores: "
                    f"cosine_non_finite={sim_meta['non_finite_count']}, "
                    f"cosine_below_min={sim_meta['below_min_count']}, "
                    f"cosine_above_max={sim_meta['above_max_count']}, "
                    f"distance_non_finite={dist_meta['non_finite_count']}, "
                    f"distance_below_min={dist_meta['below_min_count']}, "
                    f"distance_above_max={dist_meta['above_max_count']}."
                ),
                RuntimeWarning,
            )

        out = p[["pair_id", "mention_id_1", "mention_id_2", "block_key"]].copy()
        out["cosine_sim"] = sim_arr.astype(np.float32)
        out["distance"] = dist_arr.astype(np.float32)
        validate_columns(out, PAIR_SCORE_REQUIRED_COLUMNS, "pair_scores")
        return out

    if isinstance(pairs, (str, Path)):
        pair_path = Path(pairs)
        if not pair_path.exists():
            raise FileNotFoundError(pair_path)
        runtime_meta["pair_input_mode"] = "parquet_chunked"
        if chunk_rows is None:
            chunk_rows = max(10_000, int(batch_size) * 4)

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
        batch_iter = parquet.iter_batches(batch_size=int(chunk_rows))
        while True:
            read_started_at = perf_counter()
            try:
                batch = next(batch_iter)
            except StopIteration:
                break
            runtime_meta["parquet_read_seconds"] = float(runtime_meta.get("parquet_read_seconds", 0.0)) + float(
                perf_counter() - read_started_at
            )
            pandas_started_at = perf_counter()
            pairs_chunk = batch.to_pandas()
            runtime_meta["pandas_conversion_seconds"] = float(
                runtime_meta.get("pandas_conversion_seconds", 0.0)
            ) + float(perf_counter() - pandas_started_at)
            scores_chunk = _score_pairs_frame(pairs_chunk)
            if return_scores:
                out_rows.append(scores_chunk)
            if out_path is not None:
                write_started_at = perf_counter()
                table = pa.Table.from_pandas(scores_chunk, preserve_index=False)
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

        if return_scores:
            if not out_rows:
                empty = pd.DataFrame(columns=PAIR_SCORE_REQUIRED_COLUMNS)
                return (empty, runtime_meta) if return_runtime_meta else empty
            out = pd.concat(out_rows, ignore_index=True)
            validate_columns(out, PAIR_SCORE_REQUIRED_COLUMNS, "pair_scores")
            return (out, runtime_meta) if return_runtime_meta else out
        empty = pd.DataFrame(columns=PAIR_SCORE_REQUIRED_COLUMNS)
        return (empty, runtime_meta) if return_runtime_meta else empty

    runtime_meta["pair_input_mode"] = "dataframe"
    out = _score_pairs_frame(pairs)
    if output_path is not None:
        save_parquet(out, output_path, index=False)
    return (out, runtime_meta) if return_runtime_meta else out
