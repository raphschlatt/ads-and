from __future__ import annotations

import hashlib
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from author_name_disambiguation.common.npy_cache import atomic_save_npy, load_validated_npy
from author_name_disambiguation.common.torch_runtime import apply_auto_cuda_move_fallback, resolve_torch_device

_SPECTER_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_SPECTER_DIM = 768


def _hash_stub_embedding(text: str, dim: int = 768) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(h[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _to_text(title: str, abstract: str) -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if title and abstract:
        return f"{title} [SEP] {abstract}"
    return title or abstract


def _coerce_precomputed_embedding(item: Any, dim: int = _SPECTER_DIM) -> np.ndarray | None:
    if item is None or isinstance(item, (str, bytes, dict)):
        return None
    try:
        arr = np.asarray(item, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 1 or arr.shape[0] != dim:
        return None
    return arr


def _resolve_device(torch, device: str) -> str:
    resolved, _ = resolve_torch_device(torch, device, runtime_label="SPECTER embeddings")
    return resolved


def summarize_precomputed_embeddings(
    values: Iterable | None,
    *,
    total_count: int,
    dim: int = _SPECTER_DIM,
) -> dict[str, Any]:
    if values is None:
        return {
            "column_present": False,
            "precomputed_embedding_count": 0,
            "recomputed_embedding_count": int(total_count),
            "used_precomputed_embeddings": False,
        }
    items = list(values)
    count = 0
    for item in items:
        if _coerce_precomputed_embedding(item, dim=dim) is not None:
            count += 1
    return {
        "column_present": True,
        "precomputed_embedding_count": int(count),
        "recomputed_embedding_count": int(max(0, total_count - count)),
        "used_precomputed_embeddings": bool(count > 0),
    }


def _configure_hf_noise(quiet_libraries: bool) -> None:
    if not quiet_libraries:
        return

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    try:  # pragma: no cover
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass

    try:  # pragma: no cover
        from huggingface_hub.utils import disable_progress_bars, logging as hf_logging

        disable_progress_bars()
        hf_logging.set_verbosity_error()
    except Exception:
        pass


def _normalize_text_backend(text_backend: str) -> str:
    backend = str(text_backend or "transformers").strip().lower()
    if backend not in {"transformers", "adapters"}:
        warnings.warn(
            f"Unknown text_backend={text_backend!r}; falling back to transformers.",
            RuntimeWarning,
        )
        return "transformers"
    return backend


def _build_model_cache_key(
    *,
    model_name: str,
    text_backend: str,
    text_adapter_name: str | None,
    text_adapter_alias: str,
) -> str:
    return f"{text_backend}::{model_name}::{text_adapter_name or ''}::{text_adapter_alias}"


def _load_specter_components(
    model_name: str,
    reuse_model: bool,
    text_backend: str = "transformers",
    text_adapter_name: str | None = None,
    text_adapter_alias: str = "specter2",
):
    from transformers import AutoTokenizer

    backend = _normalize_text_backend(text_backend)
    cache_key = _build_model_cache_key(
        model_name=model_name,
        text_backend=backend,
        text_adapter_name=text_adapter_name,
        text_adapter_alias=text_adapter_alias,
    )
    if reuse_model and cache_key in _SPECTER_MODEL_CACHE:
        return _SPECTER_MODEL_CACHE[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if backend == "transformers":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name)
    else:
        if not text_adapter_name:
            raise ValueError("text_adapter_name is required when text_backend='adapters'.")
        try:
            from adapters import AutoAdapterModel
        except Exception as exc:
            raise RuntimeError(
                "Adapter backend requires the `adapters` package. Install with `pip install -U adapters`."
            ) from exc
        model = AutoAdapterModel.from_pretrained(model_name)
        load_kwargs: dict[str, Any] = {"source": "hf", "set_active": True}
        if text_adapter_alias:
            load_kwargs["load_as"] = text_adapter_alias
        model.load_adapter(text_adapter_name, **load_kwargs)

    if reuse_model:
        _SPECTER_MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def generate_specter_embeddings(
    mentions: pd.DataFrame,
    model_name: str = "allenai/specter",
    text_backend: str = "transformers",
    text_adapter_name: str | None = None,
    text_adapter_alias: str = "specter2",
    batch_size: int = 16,
    max_length: int = 256,
    device: str = "auto",
    prefer_precomputed: bool = True,
    use_stub_if_missing: bool = False,
    show_progress: bool = False,
    quiet_libraries: bool = False,
    reuse_model: bool = True,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    titles = mentions["title"].fillna("").astype(str).tolist()
    abstracts = mentions["abstract"].fillna("").astype(str).tolist()
    texts = [_to_text(t, a) for t, a in zip(titles, abstracts)]

    precomputed_values = mentions["precomputed_embedding"].tolist() if "precomputed_embedding" in mentions.columns else None
    precomputed_summary = summarize_precomputed_embeddings(
        precomputed_values if prefer_precomputed else None,
        total_count=len(texts),
        dim=_SPECTER_DIM,
    )
    precomputed_vectors = (
        [_coerce_precomputed_embedding(item, dim=_SPECTER_DIM) for item in list(precomputed_values or [])]
        if prefer_precomputed and precomputed_values is not None
        else []
    )
    valid_indices = [idx for idx, item in enumerate(precomputed_vectors) if item is not None]
    valid_index_set = set(valid_indices)
    missing_indices = [idx for idx in range(len(texts)) if idx not in valid_index_set]

    if len(texts) == 0:
        empty = np.zeros((0, _SPECTER_DIM), dtype=np.float32)
        meta = {
            "cache_hit": False,
            "generation_mode": "empty",
            "requested_device": str(device),
            "resolved_device": None,
            "fallback_reason": None,
            "torch_version": None,
            "torch_cuda_version": None,
            "torch_cuda_available": None,
            "cuda_probe_error": None,
            "model_to_cuda_error": None,
            "effective_precision_mode": None,
            **precomputed_summary,
        }
        return (empty, meta) if return_meta else empty

    if precomputed_summary["precomputed_embedding_count"] == len(texts):
        emb = np.vstack([item for item in precomputed_vectors if item is not None]).astype(np.float32)
        meta = {
            "cache_hit": False,
            "generation_mode": "precomputed_only",
            "requested_device": str(device),
            "resolved_device": None,
            "fallback_reason": None,
            "torch_version": None,
            "torch_cuda_version": None,
            "torch_cuda_available": None,
            "cuda_probe_error": None,
            "model_to_cuda_error": None,
            "effective_precision_mode": None,
            **precomputed_summary,
        }
        return (emb, meta) if return_meta else emb

    try:
        import torch
    except Exception as exc:
        if not use_stub_if_missing:
            raise RuntimeError(
                "SPECTER embedding generation requires `torch` and `transformers`, or precomputed 768-dim embeddings."
            ) from exc
        out = np.zeros((len(texts), _SPECTER_DIM), dtype=np.float32)
        for idx, item in enumerate(precomputed_vectors):
            if item is not None:
                out[idx] = item
        for idx in missing_indices:
            out[idx] = _hash_stub_embedding(texts[idx], dim=_SPECTER_DIM)
        meta = {
            "cache_hit": False,
            "generation_mode": "precomputed_plus_stub"
            if precomputed_summary["precomputed_embedding_count"]
            else "stub_only",
            "requested_device": str(device),
            "resolved_device": None,
            "fallback_reason": "torch_missing_stub_fallback",
            "torch_version": None,
            "torch_cuda_version": None,
            "torch_cuda_available": None,
            "cuda_probe_error": repr(exc),
            "model_to_cuda_error": None,
            "effective_precision_mode": None,
            **precomputed_summary,
        }
        return (out, meta) if return_meta else out

    _configure_hf_noise(quiet_libraries)

    requested_device = device
    device, runtime_meta = resolve_torch_device(torch, device, runtime_label="SPECTER embeddings")

    tokenizer, model = _load_specter_components(
        model_name=model_name,
        reuse_model=reuse_model,
        text_backend=text_backend,
        text_adapter_name=text_adapter_name,
        text_adapter_alias=text_adapter_alias,
    )
    try:
        model.to(device)
    except Exception as exc:
        if str(requested_device).strip().lower() == "auto" and str(device).startswith("cuda"):
            device, runtime_meta = apply_auto_cuda_move_fallback(
                requested_device=requested_device,
                runtime_label="SPECTER embeddings",
                runtime_meta=runtime_meta,
                exc=exc,
            )
            model.to(device)
        else:
            raise
    model.eval()

    out = np.zeros((len(texts), _SPECTER_DIM), dtype=np.float32)
    for idx, item in enumerate(precomputed_vectors):
        if item is not None:
            out[idx] = item

    vectors_for_indices = missing_indices if precomputed_summary["precomputed_embedding_count"] else list(range(len(texts)))
    starts = range(0, len(vectors_for_indices), batch_size)
    if show_progress:
        try:
            from tqdm.auto import tqdm

            total = (len(vectors_for_indices) + batch_size - 1) // batch_size
            starts = tqdm(starts, total=total, desc="SPECTER batches", leave=False)
        except Exception:
            pass
    with torch.no_grad():
        for start in starts:
            batch_indices = vectors_for_indices[start : start + batch_size]
            chunk = [texts[idx] for idx in batch_indices]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            batch_out = model(**enc)
            cls = batch_out.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
            out[np.asarray(batch_indices, dtype=np.int64)] = cls

    meta = {
        **runtime_meta,
        "cache_hit": False,
        "generation_mode": "precomputed_plus_model"
        if precomputed_summary["precomputed_embedding_count"]
        else "model_only",
        **precomputed_summary,
    }
    return (out.astype(np.float32), meta) if return_meta else out.astype(np.float32)


def get_or_create_specter_embeddings(
    mentions: pd.DataFrame,
    output_path: str | Path,
    force_recompute: bool = False,
    model_name: str = "allenai/specter",
    text_backend: str = "transformers",
    text_adapter_name: str | None = None,
    text_adapter_alias: str = "specter2",
    batch_size: int = 16,
    max_length: int = 256,
    device: str = "auto",
    prefer_precomputed: bool = True,
    use_stub_if_missing: bool = False,
    show_progress: bool = False,
    quiet_libraries: bool = False,
    reuse_model: bool = True,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    output = Path(output_path)
    precomputed_values = mentions["precomputed_embedding"].tolist() if "precomputed_embedding" in mentions.columns else None
    precomputed_summary = summarize_precomputed_embeddings(
        precomputed_values if prefer_precomputed else None,
        total_count=int(len(mentions)),
        dim=_SPECTER_DIM,
    )
    if output.exists() and not force_recompute:
        cached = load_validated_npy(
            output,
            validator=lambda arr: arr.ndim == 2 and arr.shape == (len(mentions), _SPECTER_DIM),
            description="SPECTER embedding cache",
        )
        if cached is not None:
            meta = {
                "cache_hit": True,
                "generation_mode": "cache",
                "requested_device": str(device),
                "resolved_device": None,
                "fallback_reason": None,
                "torch_version": None,
                "torch_cuda_version": None,
                "torch_cuda_available": None,
                "cuda_probe_error": None,
                "model_to_cuda_error": None,
                "effective_precision_mode": None,
                **precomputed_summary,
            }
            return (cached, meta) if return_meta else cached

    emb_result = generate_specter_embeddings(
        mentions=mentions,
        model_name=model_name,
        text_backend=text_backend,
        text_adapter_name=text_adapter_name,
        text_adapter_alias=text_adapter_alias,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        prefer_precomputed=prefer_precomputed,
        use_stub_if_missing=use_stub_if_missing,
        show_progress=show_progress,
        quiet_libraries=quiet_libraries,
        reuse_model=reuse_model,
        return_meta=True,
    )
    emb, meta = emb_result
    atomic_save_npy(output, emb)
    return (emb, meta) if return_meta else emb
