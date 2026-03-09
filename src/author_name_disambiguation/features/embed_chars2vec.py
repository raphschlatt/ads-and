from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from author_name_disambiguation.common.cli_ui import iter_progress
from author_name_disambiguation.common.npy_cache import atomic_save_npy, load_validated_npy


def _hash_stub_embedding(text: str, dim: int = 50) -> np.ndarray:
    # Deterministic fallback for smoke tests when chars2vec is unavailable.
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(h[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _vectorize_words_silently(model, words: list[str], *, batch_size: int, show_progress: bool) -> np.ndarray:
    words = [str(w).lower() for w in words]
    unique_words = np.unique(words)
    new_words = [w for w in unique_words if w not in model.cache]

    if len(new_words) > 0:
        list_of_embeddings = []
        for current_word in new_words:
            if not isinstance(current_word, str):
                raise TypeError("word must be a string")

            current_embedding = []
            for char in current_word:
                if char in model.char_to_ix:
                    vec = np.zeros(model.vocab_size)
                    vec[model.char_to_ix[char]] = 1
                    current_embedding.append(vec)
                else:
                    current_embedding.append(np.zeros(model.vocab_size))

            list_of_embeddings.append(np.asarray(current_embedding))

        embeddings_pad_seq = model.keras.preprocessing.sequence.pad_sequences(list_of_embeddings, maxlen=None)
        total = (len(new_words) + batch_size - 1) // batch_size
        chunks = iter_progress(
            range(0, len(new_words), batch_size),
            total=total,
            label="Chars2Vec batches",
            enabled=show_progress,
            unit="batch",
        )
        vectors = []
        for start in chunks:
            end = min(start + batch_size, len(new_words))
            batch_vectors = model.embedding_model.predict([embeddings_pad_seq[start:end]], verbose=0)
            vectors.append(np.asarray(batch_vectors, dtype=np.float32))
        new_words_vectors = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 50), dtype=np.float32)

        for idx, word in enumerate(new_words):
            model.cache[word] = new_words_vectors[idx]

    return np.asarray([model.cache[current_word] for current_word in words], dtype=np.float32)


def generate_chars2vec_embeddings(
    names: list[str],
    model_name: str = "eng_50",
    use_stub_if_missing: bool = False,
    quiet_libraries: bool = False,
    show_progress: bool = False,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    if not names:
        empty = np.zeros((0, 50), dtype=np.float32)
        meta = {"cache_hit": False, "generation_mode": "empty", "name_count": 0}
        return (empty, meta) if return_meta else empty

    if quiet_libraries:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["ABSL_LOG_LEVEL"] = "3"
        os.environ["KERAS_BACKEND"] = "tensorflow"

    try:
        import chars2vec  # type: ignore

        model = chars2vec.load_model(model_name)
        setattr(model, "keras", chars2vec.keras)
        emb = _vectorize_words_silently(
            model,
            names,
            batch_size=32,
            show_progress=show_progress,
        )
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != 50:
            raise ValueError(f"Unexpected chars2vec output shape: {emb.shape}")
        meta = {
            "cache_hit": False,
            "generation_mode": "chars2vec",
            "name_count": int(len(names)),
        }
        return (emb, meta) if return_meta else emb
    except Exception as exc:
        if not use_stub_if_missing:
            raise RuntimeError(
                "chars2vec embedding generation failed. Install `chars2vec` or set use_stub_if_missing=True for smoke tests."
            ) from exc

    emb = np.vstack([_hash_stub_embedding(name, dim=50) for name in names]).astype(np.float32)
    meta = {
        "cache_hit": False,
        "generation_mode": "stub_only",
        "name_count": int(len(names)),
    }
    return (emb, meta) if return_meta else emb


def get_or_create_chars2vec_embeddings(
    mentions: pd.DataFrame,
    output_path: str | Path,
    force_recompute: bool = False,
    model_name: str = "eng_50",
    use_stub_if_missing: bool = False,
    quiet_libraries: bool = False,
    show_progress: bool = False,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    output = Path(output_path)
    names = mentions["author_raw"].fillna("").astype(str).tolist()
    if output.exists() and not force_recompute:
        cached = load_validated_npy(
            output,
            validator=lambda arr: arr.ndim == 2 and arr.shape == (len(names), 50),
            description="chars2vec embedding cache",
        )
        if cached is not None:
            meta = {"cache_hit": True, "generation_mode": "cache", "name_count": int(len(names))}
            return (cached, meta) if return_meta else cached

    result = generate_chars2vec_embeddings(
        names=names,
        model_name=model_name,
        use_stub_if_missing=use_stub_if_missing,
        quiet_libraries=quiet_libraries,
        show_progress=show_progress,
        return_meta=True,
    )
    emb, meta = result if isinstance(result, tuple) else (result, {"cache_hit": False})

    atomic_save_npy(output, emb)
    return (emb, meta) if return_meta else emb
