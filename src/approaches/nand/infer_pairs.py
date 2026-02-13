from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import warnings

import numpy as np
import pandas as pd

from src.approaches.nand.modeling import create_encoder
from src.common.io_schema import PAIR_SCORE_REQUIRED_COLUMNS, validate_columns, save_parquet
from src.approaches.nand.train import build_feature_matrix


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


def load_checkpoint(checkpoint_path: str | Path, device: str = "auto") -> Dict[str, Any]:
    torch = _require_torch()
    # Load on CPU first to avoid hard failures during CUDA deserialization.
    return torch.load(checkpoint_path, map_location="cpu")


def score_pairs_with_checkpoint(
    mentions: pd.DataFrame,
    pairs: pd.DataFrame,
    chars2vec: np.ndarray,
    text_emb: np.ndarray,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    batch_size: int = 8192,
    device: str = "auto",
    show_progress: bool = False,
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

    features = build_feature_matrix(chars2vec=chars2vec, text_emb=text_emb)
    mindex = _build_mention_index(mentions)

    idx1 = pairs["mention_id_1"].astype(str).map(mindex).values
    idx2 = pairs["mention_id_2"].astype(str).map(mindex).values

    valid_mask = ~(pd.isna(idx1) | pd.isna(idx2))
    p = pairs.loc[valid_mask].copy().reset_index(drop=True)
    idx1 = idx1[valid_mask].astype(int)
    idx2 = idx2[valid_mask].astype(int)

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
            x1 = torch.from_numpy(features[idx1[start:end]]).to(device)
            x2 = torch.from_numpy(features[idx2[start:end]]).to(device)
            z1 = model(x1)
            z2 = model(x2)
            s = torch.nn.functional.cosine_similarity(z1, z2, dim=1)
            sims.append(s.detach().cpu().numpy())

    sim_arr = np.concatenate(sims, axis=0) if sims else np.array([], dtype=np.float32)

    out = p[["pair_id", "mention_id_1", "mention_id_2", "block_key"]].copy()
    out["cosine_sim"] = sim_arr.astype(np.float32)
    out["distance"] = (1.0 - out["cosine_sim"]).astype(np.float32)

    validate_columns(out, PAIR_SCORE_REQUIRED_COLUMNS, "pair_scores")

    if output_path is not None:
        save_parquet(out, output_path, index=False)

    return out
