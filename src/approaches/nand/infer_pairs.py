from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Any
import warnings

import numpy as np
import pandas as pd

from src.approaches.nand.modeling import create_encoder
from src.approaches.nand.train import build_feature_matrix as _legacy_build_feature_matrix
from src.common.io_schema import PAIR_SCORE_REQUIRED_COLUMNS, validate_columns, save_parquet
from src.common.numeric_safety import clamp_cosine_sim, compute_safe_distance_from_cosine

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
    if device != "auto":
        return device
    if not torch.cuda.is_available():
        return "cpu"
    try:
        # A real CUDA allocation catches cases where is_available() is true
        # but CUDA init still fails in this process/session.
        _ = torch.cuda.current_device()
        _ = torch.empty(1, device="cuda")
        return "cuda"
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"CUDA appears unavailable in this session ({exc!r}); falling back to CPU.",
            RuntimeWarning,
        )
        return "cpu"


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
) -> pd.DataFrame:
    torch = _require_torch()
    requested_device = device
    device = _resolve_device(torch, device)

    checkpoint = load_checkpoint(checkpoint_path=checkpoint_path, device=device)
    model = create_encoder(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    try:
        model.to(device)
    except Exception as exc:
        if requested_device == "auto" and str(device).startswith("cuda"):
            warnings.warn(
                f"Moving model to CUDA failed ({exc!r}); falling back to CPU.",
                RuntimeWarning,
            )
            device = "cpu"
            model.to(device)
        else:
            raise
    model.eval()
    effective_precision_mode = _resolve_effective_precision_mode(torch=torch, precision_mode=precision_mode, device=device)

    mindex = _build_mention_index(mentions)

    def _score_pairs_frame(pairs_df: pd.DataFrame) -> pd.DataFrame:
        idx1 = pairs_df["mention_id_1"].astype(str).map(mindex).values
        idx2 = pairs_df["mention_id_2"].astype(str).map(mindex).values

        valid_mask = ~(pd.isna(idx1) | pd.isna(idx2))
        p = pairs_df.loc[valid_mask].copy().reset_index(drop=True)
        idx1_valid = idx1[valid_mask].astype(int)
        idx2_valid = idx2[valid_mask].astype(int)

        sims = []
        starts = range(0, len(p), batch_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                total = (len(p) + batch_size - 1) // batch_size
                starts = tqdm(starts, total=total, desc="Score batches", leave=False)
            except Exception:
                pass

        with torch.no_grad():
            for start in starts:
                end = min(start + batch_size, len(p))
                batch_idx1 = idx1_valid[start:end]
                batch_idx2 = idx2_valid[start:end]
                x1_np = np.concatenate(
                    [
                        np.asarray(chars2vec[batch_idx1], dtype=np.float32),
                        np.asarray(text_emb[batch_idx1], dtype=np.float32),
                    ],
                    axis=1,
                )
                x2_np = np.concatenate(
                    [
                        np.asarray(chars2vec[batch_idx2], dtype=np.float32),
                        np.asarray(text_emb[batch_idx2], dtype=np.float32),
                    ],
                    axis=1,
                )
                x1 = torch.from_numpy(x1_np).to(device)
                x2 = torch.from_numpy(x2_np).to(device)
                with _autocast_context(torch, effective_precision_mode):
                    z1 = model(x1)
                    z2 = model(x2)
                    s = torch.nn.functional.cosine_similarity(z1, z2, dim=1)
                sims.append(s.detach().cpu().numpy())

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
        for batch in parquet.iter_batches(batch_size=int(chunk_rows)):
            pairs_chunk = batch.to_pandas()
            scores_chunk = _score_pairs_frame(pairs_chunk)
            if return_scores:
                out_rows.append(scores_chunk)
            if out_path is not None:
                table = pa.Table.from_pandas(scores_chunk, preserve_index=False)
                if writer is None:
                    writer_schema = table.schema
                    writer = pq.ParquetWriter(out_path, writer_schema)
                elif writer_schema is not None and table.schema != writer_schema:
                    table = table.cast(writer_schema)
                writer.write_table(table)

        if writer is not None:
            writer.close()
        if out_path is not None and writer is None:
            save_parquet(pd.DataFrame(columns=PAIR_SCORE_REQUIRED_COLUMNS), out_path, index=False)

        if return_scores:
            if not out_rows:
                return pd.DataFrame(columns=PAIR_SCORE_REQUIRED_COLUMNS)
            out = pd.concat(out_rows, ignore_index=True)
            validate_columns(out, PAIR_SCORE_REQUIRED_COLUMNS, "pair_scores")
            return out
        return pd.DataFrame(columns=PAIR_SCORE_REQUIRED_COLUMNS)

    out = _score_pairs_frame(pairs)
    if output_path is not None:
        save_parquet(out, output_path, index=False)
    return out
