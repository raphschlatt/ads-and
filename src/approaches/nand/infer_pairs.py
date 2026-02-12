from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

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


def load_checkpoint(checkpoint_path: str | Path, device: str = "auto") -> Dict[str, Any]:
    torch = _require_torch()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.load(checkpoint_path, map_location=device)


def score_pairs_with_checkpoint(
    mentions: pd.DataFrame,
    pairs: pd.DataFrame,
    chars2vec: np.ndarray,
    text_emb: np.ndarray,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    batch_size: int = 8192,
    device: str = "auto",
) -> pd.DataFrame:
    torch = _require_torch()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = load_checkpoint(checkpoint_path=checkpoint_path, device=device)
    model = create_encoder(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
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
    with torch.no_grad():
        for start in range(0, len(p), batch_size):
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
