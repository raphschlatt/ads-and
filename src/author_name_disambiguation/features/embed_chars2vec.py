from __future__ import annotations

import gc
import hashlib
import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from author_name_disambiguation.common.cli_ui import loop_progress
from author_name_disambiguation.common.npy_cache import atomic_save_npy, load_validated_npy

_CHARS2VEC_DIM = 50
_KNOWN_TF_STDERR_PATTERNS = (
    re.compile(r"^WARNING: All log messages before absl::InitializeLog\(\) is called are written to STDERR\s*$"),
    re.compile(r".*\bgpu_device\.cc:\d+\].*Created device .*", re.IGNORECASE),
    re.compile(r".*\bread_numa_node\b.*", re.IGNORECASE),
    re.compile(r".*\bcpu_feature_guard\b.*", re.IGNORECASE),
)


def _hash_stub_embedding(text: str, dim: int = 50) -> np.ndarray:
    # Deterministic fallback for smoke tests when chars2vec is unavailable.
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(h[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _should_filter_library_stderr_line(line: str) -> bool:
    text = str(line or "").strip()
    if text == "":
        return False
    return any(pattern.match(text) for pattern in _KNOWN_TF_STDERR_PATTERNS)


@contextmanager
def _filter_known_library_stderr(*, enabled: bool) -> None:
    if not enabled:
        yield
        return

    stream = getattr(sys, "stderr", None)
    if stream is None or not hasattr(stream, "fileno"):
        yield
        return

    try:
        stderr_fd = int(stream.fileno())
    except Exception:
        yield
        return

    try:
        stream.flush()
    except Exception:
        pass

    saved_fd = os.dup(stderr_fd)
    read_fd, write_fd = os.pipe()
    buffer = bytearray()

    def _forward(data: bytes) -> None:
        if not data:
            return
        text = data.decode("utf-8", errors="replace")
        if _should_filter_library_stderr_line(text):
            return
        os.write(saved_fd, data)

    def _drain() -> None:
        try:
            while True:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    break
                buffer.extend(chunk)
                while True:
                    newline_index = buffer.find(b"\n")
                    if newline_index < 0:
                        break
                    line = bytes(buffer[: newline_index + 1])
                    del buffer[: newline_index + 1]
                    _forward(line)
        finally:
            if buffer:
                _forward(bytes(buffer))
            os.close(read_fd)

    thread = threading.Thread(target=_drain, daemon=True)
    try:
        os.dup2(write_fd, stderr_fd)
        os.close(write_fd)
        thread.start()
        yield
    finally:
        try:
            stream.flush()
        except Exception:
            pass
        os.dup2(saved_fd, stderr_fd)
        thread.join(timeout=1.0)
        os.close(saved_fd)


def _vectorize_word(model, word: str) -> np.ndarray:
    if not isinstance(word, str):
        raise TypeError("word must be a string")

    current_embedding = []
    for char in word:
        if char in model.char_to_ix:
            vec = np.zeros(model.vocab_size)
            vec[model.char_to_ix[char]] = 1
            current_embedding.append(vec)
        else:
            current_embedding.append(np.zeros(model.vocab_size))
    return np.asarray(current_embedding)


def _pad_sequences(model, words: list[str]) -> np.ndarray:
    list_of_embeddings = [_vectorize_word(model, current_word) for current_word in words]
    return model.keras.preprocessing.sequence.pad_sequences(list_of_embeddings, maxlen=None)


def _configure_tensorflow_memory_growth() -> tuple[bool | None, str | None]:
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        return None, None

    try:
        gpus = list(getattr(tf.config, "list_physical_devices")("GPU"))
    except Exception as exc:
        return False, repr(exc)
    if len(gpus) == 0:
        return None, None

    try:
        for gpu in gpus:
            if hasattr(tf.config, "experimental") and hasattr(tf.config.experimental, "set_memory_growth"):
                tf.config.experimental.set_memory_growth(gpu, True)
            elif hasattr(tf.config, "set_memory_growth"):
                tf.config.set_memory_growth(gpu, True)
        return True, None
    except Exception as exc:
        return False, repr(exc)


def _cleanup_tensorflow_runtime(model) -> str | None:
    error_parts: list[str] = []

    keras_backend = getattr(getattr(model, "keras", None), "backend", None)
    if keras_backend is not None and hasattr(keras_backend, "clear_session"):
        try:
            keras_backend.clear_session()
        except Exception as exc:
            error_parts.append(f"model.keras.backend.clear_session: {exc!r}")

    try:
        import tensorflow as tf  # type: ignore

        tf_backend = getattr(getattr(tf, "keras", None), "backend", None)
        if tf_backend is not None and hasattr(tf_backend, "clear_session"):
            tf_backend.clear_session()
    except Exception as exc:
        error_parts.append(f"tensorflow.keras.backend.clear_session: {exc!r}")

    try:
        del model
    except Exception:
        pass
    gc.collect()
    return "; ".join(error_parts) if error_parts else None


def _predict_word_vectors(model, embeddings_pad_seq: np.ndarray, *, batch_size: int, show_progress: bool) -> np.ndarray:
    total_batches = (len(embeddings_pad_seq) + batch_size - 1) // batch_size
    callbacks = None
    with loop_progress(
        total=total_batches,
        label="Chars2Vec batches",
        enabled=show_progress,
        unit="batch",
    ) as tracker:
        if show_progress:
            callback_base = model.keras.callbacks.Callback

            class _PredictProgressCallback(callback_base):
                def on_predict_batch_end(self, batch, logs=None):
                    del batch, logs
                    tracker.update(1)

            callbacks = [_PredictProgressCallback()]

        vectors = model.embedding_model.predict(
            [embeddings_pad_seq],
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )
    return np.asarray(vectors, dtype=np.float32)


def _vectorize_words_silently(
    model,
    words: list[str],
    *,
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    words = [str(w).lower() for w in words]
    unique_words = np.unique(words)
    new_words = [w for w in unique_words if w not in model.cache]

    if len(new_words) > 0:
        embeddings_pad_seq = _pad_sequences(model, new_words)
        new_words_vectors = _predict_word_vectors(
            model,
            embeddings_pad_seq,
            batch_size=batch_size,
            show_progress=show_progress,
        )

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
        meta = {
            "cache_hit": False,
            "generation_mode": "empty",
            "name_count": 0,
            "tensorflow_memory_growth_enabled": None,
            "tensorflow_memory_growth_error": None,
            "tensorflow_cleanup_attempted": False,
            "tensorflow_cleanup_error": None,
        }
        return (empty, meta) if return_meta else empty

    if quiet_libraries:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["ABSL_LOG_LEVEL"] = "3"
        os.environ["KERAS_BACKEND"] = "tensorflow"

    tf_memory_growth_enabled, tf_memory_growth_error = _configure_tensorflow_memory_growth()
    try:
        import chars2vec  # type: ignore

        with _filter_known_library_stderr(enabled=quiet_libraries):
            model = chars2vec.load_model(model_name)
            setattr(model, "keras", chars2vec.keras)
            emb = _vectorize_words_silently(
                model,
                names,
                batch_size=32,
                show_progress=show_progress,
            )
        cleanup_error = _cleanup_tensorflow_runtime(model)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != _CHARS2VEC_DIM:
            raise ValueError(f"Unexpected chars2vec output shape: {emb.shape}")
        meta = {
            "cache_hit": False,
            "generation_mode": "chars2vec",
            "name_count": int(len(names)),
            "tensorflow_memory_growth_enabled": tf_memory_growth_enabled,
            "tensorflow_memory_growth_error": tf_memory_growth_error,
            "tensorflow_cleanup_attempted": True,
            "tensorflow_cleanup_error": cleanup_error,
        }
        return (emb, meta) if return_meta else emb
    except Exception as exc:
        if not use_stub_if_missing:
            raise RuntimeError(
                "chars2vec embedding generation failed. Install `chars2vec` or set use_stub_if_missing=True for smoke tests."
            ) from exc

    emb = np.vstack([_hash_stub_embedding(name, dim=_CHARS2VEC_DIM) for name in names]).astype(np.float32)
    meta = {
        "cache_hit": False,
        "generation_mode": "stub_only",
        "name_count": int(len(names)),
        "tensorflow_memory_growth_enabled": tf_memory_growth_enabled,
        "tensorflow_memory_growth_error": tf_memory_growth_error,
        "tensorflow_cleanup_attempted": False,
        "tensorflow_cleanup_error": None,
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
            meta = {
                "cache_hit": True,
                "generation_mode": "cache",
                "name_count": int(len(names)),
                "tensorflow_memory_growth_enabled": None,
                "tensorflow_memory_growth_error": None,
                "tensorflow_cleanup_attempted": False,
                "tensorflow_cleanup_error": None,
            }
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
