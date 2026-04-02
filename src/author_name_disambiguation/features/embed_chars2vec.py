from __future__ import annotations

import gc
import hashlib
import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd

from author_name_disambiguation.common.cli_ui import loop_progress
from author_name_disambiguation.common.npy_cache import atomic_save_npy, load_validated_npy

_CHARS2VEC_DIM = 50
_CHARS2VEC_HISTORICAL_BATCH_SIZE = 32
_CHARS2VEC_AUTO_GPU_BATCH_SIZE = 512
_CHARS2VEC_AUTO_CPU_BATCH_SIZE = 128
_CHARS2VEC_MIN_RETRY_BATCH_SIZE = _CHARS2VEC_HISTORICAL_BATCH_SIZE
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


def _tensorflow_gpu_available() -> bool:
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        return False

    try:
        gpus = list(getattr(tf.config, "list_physical_devices")("GPU"))
    except Exception:
        return False
    return len(gpus) > 0


def _normalize_execution_mode(execution_mode: str) -> Literal["predict", "direct_call"]:
    if execution_mode not in {"predict", "direct_call"}:
        raise ValueError("execution_mode must be 'predict' or 'direct_call'")
    return execution_mode


def _resolve_predict_batch_size(batch_size: int | None, *, tensorflow_gpu_available: bool) -> tuple[int | None, int]:
    if batch_size is not None:
        resolved = int(batch_size)
        if resolved < 1:
            raise ValueError("batch_size must be >= 1 when provided")
        return resolved, resolved
    return None, _CHARS2VEC_AUTO_GPU_BATCH_SIZE if tensorflow_gpu_available else _CHARS2VEC_AUTO_CPU_BATCH_SIZE


def _is_tensorflow_gpu_oom_error(exc: Exception) -> bool:
    texts = [
        exc.__class__.__name__,
        str(exc),
    ]
    current = exc.__cause__ or exc.__context__
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        texts.extend([current.__class__.__name__, str(current)])
        current = current.__cause__ or current.__context__

    lowered = " ".join(texts).lower()
    return (
        "resourceexhaustederror" in lowered
        or "out of memory" in lowered
        or "oom" in lowered
        or "failed to allocate memory" in lowered
    )


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


def _predict_word_vectors(
    model,
    embeddings_pad_seq: np.ndarray,
    *,
    batch_size: int,
    show_progress: bool,
    tensorflow_gpu_available: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    effective_batch_size = int(batch_size)
    oom_retry_count = 0

    while True:
        total_batches = (len(embeddings_pad_seq) + effective_batch_size - 1) // effective_batch_size
        callbacks = None
        try:
            with loop_progress(
                total=total_batches,
                label="Chars2Vec batches",
                enabled=show_progress,
                unit="batch",
                compact_label="Name embeddings",
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
                    batch_size=effective_batch_size,
                    verbose=0,
                    callbacks=callbacks,
                )
            return np.asarray(vectors, dtype=np.float32), {
                "effective_batch_size": int(effective_batch_size),
                "predict_batch_count": int(total_batches),
                "oom_retry_count": int(oom_retry_count),
            }
        except Exception as exc:
            if (
                not tensorflow_gpu_available
                or effective_batch_size <= _CHARS2VEC_MIN_RETRY_BATCH_SIZE
                or not _is_tensorflow_gpu_oom_error(exc)
            ):
                raise
            next_batch_size = max(_CHARS2VEC_MIN_RETRY_BATCH_SIZE, effective_batch_size // 2)
            if next_batch_size == effective_batch_size:
                raise
            effective_batch_size = next_batch_size
            oom_retry_count += 1


def _direct_call_word_vectors(model, embeddings_pad_seq: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    vectors = model.embedding_model(embeddings_pad_seq, training=False)
    if hasattr(vectors, "numpy"):
        vectors = vectors.numpy()
    return np.asarray(vectors, dtype=np.float32), {
        "effective_batch_size": int(len(embeddings_pad_seq)),
        "predict_batch_count": 0,
        "oom_retry_count": 0,
    }


def _vectorize_words_silently(
    model,
    words: list[str],
    *,
    batch_size: int | None,
    execution_mode: Literal["predict", "direct_call"],
    show_progress: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    normalize_started_at = perf_counter()
    normalized_words = [str(w).lower() for w in words]
    normalize_seconds = float(perf_counter() - normalize_started_at)

    unique_started_at = perf_counter()
    unique_words = np.unique(normalized_words)
    unique_seconds = float(perf_counter() - unique_started_at)

    pad_seconds = 0.0
    predict_seconds = 0.0
    requested_batch_size = None if batch_size is None else int(batch_size)
    effective_batch_size = None
    predict_batch_count = 0
    oom_retry_count = 0
    missing_words = [str(word) for word in unique_words.tolist() if word not in model.cache]
    if missing_words:
        pad_started_at = perf_counter()
        embeddings_pad_seq = _pad_sequences(model, missing_words)
        pad_seconds = float(perf_counter() - pad_started_at)

        predict_started_at = perf_counter()
        if execution_mode == "predict":
            tensorflow_gpu_available = _tensorflow_gpu_available()
            requested_batch_size, resolved_batch_size = _resolve_predict_batch_size(
                requested_batch_size,
                tensorflow_gpu_available=tensorflow_gpu_available,
            )
            missing_word_vectors, predict_meta = _predict_word_vectors(
                model,
                embeddings_pad_seq,
                batch_size=resolved_batch_size,
                show_progress=show_progress,
                tensorflow_gpu_available=tensorflow_gpu_available,
            )
        else:
            missing_word_vectors, predict_meta = _direct_call_word_vectors(model, embeddings_pad_seq)
        predict_seconds = float(perf_counter() - predict_started_at)
        effective_batch_size = predict_meta.get("effective_batch_size")
        predict_batch_count = int(predict_meta.get("predict_batch_count", 0))
        oom_retry_count = int(predict_meta.get("oom_retry_count", 0))

        for word, vector in zip(missing_words, np.asarray(missing_word_vectors, dtype=np.float32), strict=False):
            model.cache[str(word)] = vector

    materialize_started_at = perf_counter()
    materialized = np.asarray([model.cache[current_word] for current_word in normalized_words], dtype=np.float32)
    materialize_seconds = float(perf_counter() - materialize_started_at)

    meta = {
        "normalize_seconds": normalize_seconds,
        "unique_seconds": unique_seconds,
        "pad_seconds": pad_seconds,
        "predict_seconds": predict_seconds,
        "materialize_seconds": materialize_seconds,
        "unique_name_count": int(len(unique_words)),
        "execution_mode": execution_mode,
        "requested_batch_size": requested_batch_size,
        "effective_batch_size": effective_batch_size,
        "predict_batch_count": predict_batch_count,
        "oom_retry_count": oom_retry_count,
    }
    return materialized, meta


def generate_chars2vec_embeddings(
    names: list[str],
    model_name: str = "eng_50",
    batch_size: int | None = _CHARS2VEC_HISTORICAL_BATCH_SIZE,
    execution_mode: Literal["predict", "direct_call"] = "predict",
    use_stub_if_missing: bool = False,
    quiet_libraries: bool = False,
    show_progress: bool = False,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    execution_mode = _normalize_execution_mode(execution_mode)
    if not names:
        empty = np.zeros((0, 50), dtype=np.float32)
        meta = {
            "cache_hit": False,
            "generation_mode": "empty",
            "name_count": 0,
            "unique_name_count": 0,
            "wall_seconds": 0.0,
            "generation_seconds": 0.0,
            "model_load_seconds": 0.0,
            "normalize_seconds": 0.0,
            "unique_seconds": 0.0,
            "pad_seconds": 0.0,
            "predict_seconds": 0.0,
            "materialize_seconds": 0.0,
            "cache_load_seconds": 0.0,
            "cache_write_seconds": 0.0,
            "tensorflow_memory_growth_enabled": None,
            "tensorflow_memory_growth_error": None,
            "tensorflow_cleanup_attempted": False,
            "tensorflow_cleanup_error": None,
            "execution_mode": execution_mode,
            "requested_batch_size": None if batch_size is None else int(batch_size),
            "effective_batch_size": None,
            "predict_batch_count": 0,
            "oom_retry_count": 0,
        }
        return (empty, meta) if return_meta else empty

    if quiet_libraries:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["ABSL_LOG_LEVEL"] = "3"
        os.environ["KERAS_BACKEND"] = "tensorflow"

    generation_started_at = perf_counter()
    try:
        with _filter_known_library_stderr(enabled=quiet_libraries):
            tf_memory_growth_enabled, tf_memory_growth_error = _configure_tensorflow_memory_growth()
            import chars2vec  # type: ignore

            model_load_started_at = perf_counter()
            model = chars2vec.load_model(model_name)
            model_load_seconds = float(perf_counter() - model_load_started_at)
            setattr(model, "keras", chars2vec.keras)
            cleanup_error = None
            try:
                emb, vectorize_meta = _vectorize_words_silently(
                    model,
                    names,
                    batch_size=batch_size,
                    execution_mode=execution_mode,
                    show_progress=show_progress if execution_mode == "predict" else False,
                )
            finally:
                cleanup_error = _cleanup_tensorflow_runtime(model)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != _CHARS2VEC_DIM:
            raise ValueError(f"Unexpected chars2vec output shape: {emb.shape}")
        generation_elapsed = float(perf_counter() - generation_started_at)
        meta = {
            "cache_hit": False,
            "generation_mode": "chars2vec",
            "name_count": int(len(names)),
            "unique_name_count": int(vectorize_meta.get("unique_name_count", len(names))),
            "wall_seconds": generation_elapsed,
            "generation_seconds": generation_elapsed,
            "model_load_seconds": model_load_seconds,
            "normalize_seconds": float(vectorize_meta.get("normalize_seconds", 0.0)),
            "unique_seconds": float(vectorize_meta.get("unique_seconds", 0.0)),
            "pad_seconds": float(vectorize_meta.get("pad_seconds", 0.0)),
            "predict_seconds": float(vectorize_meta.get("predict_seconds", 0.0)),
            "materialize_seconds": float(vectorize_meta.get("materialize_seconds", 0.0)),
            "cache_load_seconds": 0.0,
            "cache_write_seconds": 0.0,
            "tensorflow_memory_growth_enabled": tf_memory_growth_enabled,
            "tensorflow_memory_growth_error": tf_memory_growth_error,
            "tensorflow_cleanup_attempted": True,
            "tensorflow_cleanup_error": cleanup_error,
            "execution_mode": str(vectorize_meta.get("execution_mode", execution_mode)),
            "requested_batch_size": vectorize_meta.get("requested_batch_size"),
            "effective_batch_size": vectorize_meta.get("effective_batch_size"),
            "predict_batch_count": int(vectorize_meta.get("predict_batch_count", 0)),
            "oom_retry_count": int(vectorize_meta.get("oom_retry_count", 0)),
        }
        return (emb, meta) if return_meta else emb
    except Exception as exc:
        if not use_stub_if_missing:
            raise RuntimeError(
                "chars2vec embedding generation failed. Install `chars2vec` or set use_stub_if_missing=True for smoke tests."
            ) from exc

    emb = np.vstack([_hash_stub_embedding(name, dim=_CHARS2VEC_DIM) for name in names]).astype(np.float32)
    generation_elapsed = float(perf_counter() - generation_started_at)
    meta = {
        "cache_hit": False,
        "generation_mode": "stub_only",
        "name_count": int(len(names)),
        "unique_name_count": int(len({str(name).lower() for name in names})),
        "wall_seconds": generation_elapsed,
        "generation_seconds": generation_elapsed,
        "model_load_seconds": 0.0,
        "normalize_seconds": 0.0,
        "unique_seconds": 0.0,
        "pad_seconds": 0.0,
        "predict_seconds": 0.0,
        "materialize_seconds": 0.0,
        "cache_load_seconds": 0.0,
        "cache_write_seconds": 0.0,
        "tensorflow_memory_growth_enabled": tf_memory_growth_enabled,
        "tensorflow_memory_growth_error": tf_memory_growth_error,
        "tensorflow_cleanup_attempted": False,
        "tensorflow_cleanup_error": None,
        "execution_mode": execution_mode,
        "requested_batch_size": None if batch_size is None else int(batch_size),
        "effective_batch_size": None,
        "predict_batch_count": 0,
        "oom_retry_count": 0,
    }
    return (emb, meta) if return_meta else emb


def get_or_create_chars2vec_embeddings(
    mentions: pd.DataFrame,
    output_path: str | Path,
    force_recompute: bool = False,
    model_name: str = "eng_50",
    batch_size: int | None = _CHARS2VEC_HISTORICAL_BATCH_SIZE,
    execution_mode: Literal["predict", "direct_call"] = "predict",
    use_stub_if_missing: bool = False,
    quiet_libraries: bool = False,
    show_progress: bool = False,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    execution_mode = _normalize_execution_mode(execution_mode)
    output = Path(output_path)
    names = mentions["author_raw"].fillna("").astype(str).tolist()
    call_started_at = perf_counter()
    if output.exists() and not force_recompute:
        cache_load_started_at = perf_counter()
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
                "unique_name_count": int(len({str(name).lower() for name in names})),
                "wall_seconds": float(perf_counter() - call_started_at),
                "generation_seconds": 0.0,
                "model_load_seconds": 0.0,
                "normalize_seconds": 0.0,
                "unique_seconds": 0.0,
                "pad_seconds": 0.0,
                "predict_seconds": 0.0,
                "materialize_seconds": 0.0,
                "cache_load_seconds": float(perf_counter() - cache_load_started_at),
                "cache_write_seconds": 0.0,
                "tensorflow_memory_growth_enabled": None,
                "tensorflow_memory_growth_error": None,
                "tensorflow_cleanup_attempted": False,
                "tensorflow_cleanup_error": None,
                "execution_mode": execution_mode,
                "requested_batch_size": None if batch_size is None else int(batch_size),
                "effective_batch_size": None,
                "predict_batch_count": 0,
                "oom_retry_count": 0,
            }
            return (cached, meta) if return_meta else cached

    result = generate_chars2vec_embeddings(
        names=names,
        model_name=model_name,
        batch_size=batch_size,
        execution_mode=execution_mode,
        use_stub_if_missing=use_stub_if_missing,
        quiet_libraries=quiet_libraries,
        show_progress=show_progress,
        return_meta=True,
    )
    emb, meta = result if isinstance(result, tuple) else (result, {"cache_hit": False})

    cache_write_started_at = perf_counter()
    atomic_save_npy(output, emb)
    meta = dict(meta)
    meta["cache_write_seconds"] = float(perf_counter() - cache_write_started_at)
    meta["wall_seconds"] = float(perf_counter() - call_started_at)
    return (emb, meta) if return_meta else emb
