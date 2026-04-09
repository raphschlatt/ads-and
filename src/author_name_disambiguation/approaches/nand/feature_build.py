from __future__ import annotations

import numpy as np


def build_feature_matrix(chars2vec: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    if len(chars2vec) != len(text_emb):
        raise ValueError(f"Embedding length mismatch: chars={len(chars2vec)}, text={len(text_emb)}")
    return np.concatenate([chars2vec.astype(np.float32), text_emb.astype(np.float32)], axis=1)
