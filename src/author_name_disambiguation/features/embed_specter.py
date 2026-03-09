from __future__ import annotations

import hashlib
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_SPECTER_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


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


def _has_valid_precomputed(values: Iterable) -> bool:
    for item in values:
        if isinstance(item, list) and len(item) == 768:
            return True
    return False


def _stack_precomputed(values: Iterable, texts: list[str]) -> np.ndarray:
    out = []
    for idx, item in enumerate(values):
        if isinstance(item, list) and len(item) == 768:
            out.append(np.asarray(item, dtype=np.float32))
        else:
            out.append(_hash_stub_embedding(texts[idx], dim=768))
    return np.vstack(out).astype(np.float32)


def _resolve_device(torch, device: str) -> str:
    if device != "auto":
        return device
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _ = torch.cuda.current_device()
        _ = torch.empty(1, device="cuda")
        return "cuda"
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"CUDA appears unavailable in this session ({exc!r}); falling back to CPU.",
            RuntimeWarning,
        )
        return "cpu"


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
) -> np.ndarray:
    titles = mentions["title"].fillna("").astype(str).tolist()
    abstracts = mentions["abstract"].fillna("").astype(str).tolist()
    texts = [_to_text(t, a) for t, a in zip(titles, abstracts)]

    if prefer_precomputed and "precomputed_embedding" in mentions.columns:
        pre = mentions["precomputed_embedding"].tolist()
        if _has_valid_precomputed(pre):
            return _stack_precomputed(pre, texts)

    try:
        import torch
    except Exception as exc:
        if not use_stub_if_missing:
            raise RuntimeError(
                "SPECTER embedding generation requires `torch` and `transformers`, or precomputed 768-dim embeddings."
            ) from exc
        return np.vstack([_hash_stub_embedding(t, dim=768) for t in texts]).astype(np.float32)

    _configure_hf_noise(quiet_libraries)

    requested_device = device
    device = _resolve_device(torch, device)

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
        if requested_device == "auto" and str(device).startswith("cuda"):
            warnings.warn(
                f"Moving SPECTER model to CUDA failed ({exc!r}); falling back to CPU.",
                RuntimeWarning,
            )
            device = "cpu"
            model.to(device)
        else:
            raise
    model.eval()

    vectors: list[np.ndarray] = []
    starts = range(0, len(texts), batch_size)
    if show_progress:
        try:
            from tqdm.auto import tqdm

            total = (len(texts) + batch_size - 1) // batch_size
            starts = tqdm(starts, total=total, desc="SPECTER batches", leave=False)
        except Exception:
            pass
    with torch.no_grad():
        for start in starts:
            chunk = texts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
            vectors.append(cls)

    return np.vstack(vectors).astype(np.float32)


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
) -> np.ndarray:
    output = Path(output_path)
    if output.exists() and not force_recompute:
        return np.load(output)

    emb = generate_specter_embeddings(
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
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, emb)
    return emb
