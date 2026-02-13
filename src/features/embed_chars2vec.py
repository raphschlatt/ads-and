from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _hash_stub_embedding(text: str, dim: int = 50) -> np.ndarray:
    # Deterministic fallback for smoke tests when chars2vec is unavailable.
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(h[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def generate_chars2vec_embeddings(
    names: list[str],
    model_name: str = "eng_50",
    use_stub_if_missing: bool = False,
    quiet_libraries: bool = False,
) -> np.ndarray:
    if not names:
        return np.zeros((0, 50), dtype=np.float32)

    if quiet_libraries:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("ABSL_LOG_LEVEL", "3")

    try:
        import chars2vec  # type: ignore

        model = chars2vec.load_model(model_name)
        emb = model.vectorize_words(names)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != 50:
            raise ValueError(f"Unexpected chars2vec output shape: {emb.shape}")
        return emb
    except Exception as exc:
        if not use_stub_if_missing:
            raise RuntimeError(
                "chars2vec embedding generation failed. Install `chars2vec` or set use_stub_if_missing=True for smoke tests."
            ) from exc

    return np.vstack([_hash_stub_embedding(name, dim=50) for name in names]).astype(np.float32)


def get_or_create_chars2vec_embeddings(
    mentions: pd.DataFrame,
    output_path: str | Path,
    force_recompute: bool = False,
    model_name: str = "eng_50",
    use_stub_if_missing: bool = False,
    quiet_libraries: bool = False,
) -> np.ndarray:
    output = Path(output_path)
    if output.exists() and not force_recompute:
        return np.load(output)

    names = mentions["author_raw"].fillna("").astype(str).tolist()
    emb = generate_chars2vec_embeddings(
        names=names,
        model_name=model_name,
        use_stub_if_missing=use_stub_if_missing,
        quiet_libraries=quiet_libraries,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, emb)
    return emb
