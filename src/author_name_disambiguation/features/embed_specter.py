from __future__ import annotations

from contextlib import nullcontext
import hashlib
import logging
import os
from time import perf_counter
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from author_name_disambiguation.common.cli_ui import loop_progress
from author_name_disambiguation.embedding_contract import TEXT_EMBEDDING_DIM, build_source_text
from author_name_disambiguation.common.npy_cache import atomic_save_npy, load_validated_npy
from author_name_disambiguation.common.torch_runtime import apply_auto_cuda_move_fallback, resolve_torch_device
from author_name_disambiguation.features.specter_runtime import (
    build_onnx_cache_path,
    build_onnx_session,
    compute_token_length_order,
    export_specter_onnx,
    load_tokenizer_prefer_fast,
    normalize_runtime_backend,
    resolve_cpu_batch_size,
    resolve_cpu_thread_count,
    temporary_torch_cpu_thread_policy,
)

_SPECTER_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_SPECTER_DIM = TEXT_EMBEDDING_DIM


def _hash_stub_embedding(text: str, dim: int = 768) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(h[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _to_text(title: str, abstract: str) -> str:
    return build_source_text(title, abstract)


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


def _resolve_effective_precision_mode(torch, precision_mode: str, device: str) -> str:
    mode = str(precision_mode or "auto").strip().lower()
    if mode not in {"auto", "fp32", "amp_bf16"}:
        warnings.warn(
            f"Unknown precision_mode={precision_mode!r}; falling back to auto.",
            RuntimeWarning,
        )
        mode = "auto"
    if mode == "fp32":
        return "fp32"
    if not str(device).startswith("cuda"):
        if mode == "amp_bf16":
            warnings.warn("precision_mode=amp_bf16 requested on non-CUDA device; falling back to fp32.", RuntimeWarning)
        return "fp32"

    is_supported = True
    try:
        if hasattr(torch.cuda, "is_bf16_supported"):
            is_supported = bool(torch.cuda.is_bf16_supported())
    except Exception:
        is_supported = False

    if mode == "auto":
        return "amp_bf16" if is_supported else "fp32"
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


def _resolve_cuda_total_memory_bytes(torch, device: str) -> int | None:
    if not str(device).startswith("cuda"):
        return None
    try:
        device_obj = torch.device(device)
        device_index = device_obj.index
        if device_index is None and hasattr(torch.cuda, "current_device"):
            device_index = int(torch.cuda.current_device())
        if device_index is None:
            device_index = 0
        props = torch.cuda.get_device_properties(device_index)
        total_memory = getattr(props, "total_memory", None)
        if total_memory is None:
            return None
        return int(total_memory)
    except Exception:
        return None


def _resolve_specter_batch_size(torch, batch_size: int | None, device: str) -> tuple[int | None, int]:
    requested = None if batch_size is None else max(1, int(batch_size))
    if requested is not None:
        return requested, requested
    if not str(device).startswith("cuda"):
        return resolve_cpu_batch_size(batch_size)

    total_memory = _resolve_cuda_total_memory_bytes(torch, device)
    if total_memory is None:
        return None, 32
    if total_memory >= 70 * 1024**3:
        return None, 384
    if total_memory >= 24 * 1024**3:
        return None, 160
    if total_memory >= 12 * 1024**3:
        return None, 80
    return None, 32


def _resolve_device_to_host_flush_batch_count(
    torch,
    *,
    device: str,
    effective_batch_size: int | None,
) -> int:
    if not str(device).startswith("cuda"):
        return 1

    batch_size = 0 if effective_batch_size is None else int(effective_batch_size)
    total_memory = _resolve_cuda_total_memory_bytes(torch, device)
    if total_memory is not None and total_memory >= 70 * 1024**3 and batch_size >= 256:
        return 8
    if batch_size >= 128:
        return 6
    return 4


def _is_cuda_oom(torch, exc: BaseException) -> bool:
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    text = str(exc).lower()
    return "out of memory" in text and "cuda" in text


def _best_effort_clear_cuda_cache(torch) -> None:
    try:
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
    except Exception:
        pass


def _base_runtime_meta(
    *,
    requested_device: str,
    resolved_device: str | None,
    effective_precision_mode: str | None,
    requested_batch_size: int | None,
    effective_batch_size: int | None,
    fallback_reason: str | None,
    torch_version: str | None,
    torch_cuda_version: str | None,
    torch_cuda_available: bool | None,
    cuda_probe_error: str | None,
    model_to_cuda_error: str | None,
) -> dict[str, Any]:
    return {
        "cache_hit": False,
        "requested_device": str(requested_device),
        "resolved_device": resolved_device,
        "fallback_reason": fallback_reason,
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "torch_cuda_available": torch_cuda_available,
        "cuda_probe_error": cuda_probe_error,
        "model_to_cuda_error": model_to_cuda_error,
        "effective_precision_mode": effective_precision_mode,
        "requested_batch_size": None if requested_batch_size is None else int(requested_batch_size),
        "effective_batch_size": None if effective_batch_size is None else int(effective_batch_size),
        "oom_retry_count": 0,
        "batches_total": 0,
        "tokenize_seconds_total": 0.0,
        "host_to_device_seconds_total": 0.0,
        "forward_seconds_total": 0.0,
        "device_to_host_seconds_total": 0.0,
        "token_count_total": 0,
        "max_sequence_length_observed": 0,
        "mean_sequence_length_observed": 0.0,
        "device_to_host_flushes": 0,
        "device_to_host_flush_batch_count": 0,
    }


def _safe_pin_memory(tensor: Any) -> Any:
    pin = getattr(tensor, "pin_memory", None)
    if pin is None or not callable(pin):
        return tensor
    try:
        return pin()
    except Exception:
        return tensor


def _move_tensor_to_device(tensor: Any, device: str, *, non_blocking: bool) -> Any:
    mover = getattr(tensor, "to", None)
    if mover is None or not callable(mover):
        return tensor
    try:
        return mover(device, non_blocking=non_blocking)
    except TypeError:
        return mover(device)


def _encoding_to_device(enc: dict[str, Any], device: str) -> dict[str, Any]:
    on_cuda = str(device).startswith("cuda")
    moved: dict[str, Any] = {}
    for key, value in enc.items():
        tensor = value
        if on_cuda:
            tensor = _safe_pin_memory(tensor)
        moved[key] = _move_tensor_to_device(tensor, device, non_blocking=on_cuda)
    return moved


def _observed_token_stats(enc: dict[str, Any]) -> tuple[int, int, int]:
    mask = enc.get("attention_mask")
    if mask is not None and hasattr(mask, "sum") and hasattr(mask, "shape"):
        lengths = mask.sum(dim=1)
        token_total = int(lengths.sum().item())
        max_sequence_length = int(mask.shape[1]) if len(mask.shape) >= 2 else int(lengths.max().item())
        observed_count = int(mask.shape[0]) if len(mask.shape) >= 1 else int(lengths.numel())
        return token_total, max_sequence_length, observed_count

    input_ids = enc.get("input_ids")
    if input_ids is not None and hasattr(input_ids, "shape") and len(input_ids.shape) >= 2:
        batch_size = int(input_ids.shape[0])
        seq_len = int(input_ids.shape[1])
        return batch_size * seq_len, seq_len, batch_size
    return 0, 0, 0


def _flush_pending_cls_batches(
    *,
    torch_module: Any,
    pending_tensors: list[Any],
    pending_indices: list[np.ndarray],
    out: np.ndarray,
    meta: dict[str, Any],
) -> None:
    if not pending_tensors:
        return
    device_to_host_started_at = perf_counter()
    if len(pending_tensors) == 1:
        cls_tensor = pending_tensors[0]
        batch_indices = pending_indices[0]
    else:
        cls_tensor = torch_module.cat(pending_tensors, dim=0)
        batch_indices = np.concatenate(pending_indices, axis=0)
    cls = cls_tensor.to("cpu", dtype=torch_module.float32).numpy()
    meta["device_to_host_seconds_total"] += perf_counter() - device_to_host_started_at
    meta["device_to_host_flushes"] += 1
    out[batch_indices] = cls.astype(np.float32, copy=False)
    pending_tensors.clear()
    pending_indices.clear()


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

    tokenizer = load_tokenizer_prefer_fast(model_name)
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
    runtime_backend: str = "transformers",
    batch_size: int | None = None,
    max_length: int = 256,
    device: str = "auto",
    precision_mode: str = "auto",
    prefer_precomputed: bool = True,
    use_stub_if_missing: bool = False,
    show_progress: bool = False,
    quiet_libraries: bool = False,
    reuse_model: bool = True,
    onnx_cache_path: str | Path | None = None,
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
            "generation_mode": "empty",
            **_base_runtime_meta(
                requested_device=str(device),
                resolved_device=None,
                effective_precision_mode=None,
                requested_batch_size=None if batch_size is None else int(batch_size),
                effective_batch_size=None,
                fallback_reason=None,
                torch_version=None,
                torch_cuda_version=None,
                torch_cuda_available=None,
                cuda_probe_error=None,
                model_to_cuda_error=None,
            ),
            **precomputed_summary,
        }
        return (empty, meta) if return_meta else empty

    if precomputed_summary["precomputed_embedding_count"] == len(texts):
        emb = np.vstack([item for item in precomputed_vectors if item is not None]).astype(np.float32)
        meta = {
            "generation_mode": "precomputed_only",
            **_base_runtime_meta(
                requested_device=str(device),
                resolved_device=None,
                effective_precision_mode=None,
                requested_batch_size=None if batch_size is None else int(batch_size),
                effective_batch_size=None,
                fallback_reason=None,
                torch_version=None,
                torch_cuda_version=None,
                torch_cuda_available=None,
                cuda_probe_error=None,
                model_to_cuda_error=None,
            ),
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
            "generation_mode": "precomputed_plus_stub"
            if precomputed_summary["precomputed_embedding_count"]
            else "stub_only",
            **_base_runtime_meta(
                requested_device=str(device),
                resolved_device=None,
                effective_precision_mode=None,
                requested_batch_size=None if batch_size is None else int(batch_size),
                effective_batch_size=None,
                fallback_reason="torch_missing_stub_fallback",
                torch_version=None,
                torch_cuda_version=None,
                torch_cuda_available=None,
                cuda_probe_error=repr(exc),
                model_to_cuda_error=None,
            ),
            **precomputed_summary,
        }
        return (out, meta) if return_meta else out

    _configure_hf_noise(quiet_libraries)

    requested_device = str(device)
    device, runtime_meta = resolve_torch_device(torch, device, runtime_label="SPECTER embeddings")
    runtime_backend_clean = str(runtime_backend or "transformers").strip().lower() or "transformers"
    if runtime_backend_clean == "onnx_fp32":
        if requested_device.strip().lower() not in {"auto", "cpu"}:
            raise ValueError("specter_runtime_backend='onnx_fp32' requires device='cpu' or device='auto'.")
        device = "cpu"
    runtime_backend_clean = normalize_runtime_backend(runtime_backend_clean, device=device)
    requested_batch_size, effective_batch_size = _resolve_specter_batch_size(torch, batch_size, device)

    tokenizer, model = _load_specter_components(
        model_name=model_name,
        reuse_model=reuse_model,
        text_backend=text_backend,
        text_adapter_name=text_adapter_name,
        text_adapter_alias=text_adapter_alias,
    )
    onnx_session = None
    onnx_path: Path | None = None
    cpu_thread_count = resolve_cpu_thread_count() if str(device).startswith("cpu") else None
    if runtime_backend_clean == "transformers":
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
                requested_batch_size, effective_batch_size = _resolve_specter_batch_size(torch, batch_size, device)
                model.to(device)
            else:
                raise
        model.eval()
    else:
        if _normalize_text_backend(text_backend) != "transformers":
            raise ValueError("specter_runtime_backend='onnx_fp32' requires text_backend='transformers'.")
        model.to("cpu")
        model.eval()
        onnx_path = build_onnx_cache_path(output_path=onnx_cache_path, model_name=model_name, max_length=max_length)
        sample_text = next((str(texts[idx]) for idx in missing_indices if str(texts[idx]).strip()), "")
        if not sample_text:
            sample_text = next((str(text) for text in texts if str(text).strip()), "SPECTER export sample text")
        onnx_path = export_specter_onnx(
            tokenizer=tokenizer,
            model=model,
            torch_module=torch,
            export_path=onnx_path,
            sample_text=sample_text,
            max_length=max_length,
        )
        onnx_session = build_onnx_session(
            onnx_path=onnx_path,
            num_threads=resolve_cpu_thread_count(cpu_thread_count),
        )
    effective_precision_mode = (
        "fp32"
        if runtime_backend_clean == "onnx_fp32"
        else _resolve_effective_precision_mode(torch=torch, precision_mode=precision_mode, device=device)
    )
    runtime_meta["effective_precision_mode"] = effective_precision_mode

    out = np.zeros((len(texts), _SPECTER_DIM), dtype=np.float32)
    for idx, item in enumerate(precomputed_vectors):
        if item is not None:
            out[idx] = item

    vectors_for_indices = missing_indices if precomputed_summary["precomputed_embedding_count"] else list(range(len(texts)))
    if str(device).startswith("cpu") and vectors_for_indices:
        ordered = compute_token_length_order(
            [texts[idx] for idx in vectors_for_indices],
            tokenizer=tokenizer,
            max_length=max_length,
        )
        vectors_for_indices = [vectors_for_indices[int(pos)] for pos in ordered.tolist()]
    else:
        vectors_for_indices = sorted(vectors_for_indices, key=lambda idx: (len(texts[idx]), idx))
    meta = {
        **_base_runtime_meta(
            requested_device=requested_device,
            resolved_device=device,
            effective_precision_mode=effective_precision_mode,
            requested_batch_size=requested_batch_size,
            effective_batch_size=effective_batch_size,
            fallback_reason=runtime_meta.get("fallback_reason"),
            torch_version=runtime_meta.get("torch_version"),
            torch_cuda_version=runtime_meta.get("torch_cuda_version"),
            torch_cuda_available=runtime_meta.get("torch_cuda_available"),
            cuda_probe_error=runtime_meta.get("cuda_probe_error"),
            model_to_cuda_error=runtime_meta.get("model_to_cuda_error"),
        ),
    }
    meta["runtime_backend"] = str(runtime_backend_clean)
    if onnx_path is not None:
        meta["onnx_path"] = str(onnx_path)
    observed_sequence_count = 0
    device_to_host_flush_batch_count = _resolve_device_to_host_flush_batch_count(
        torch,
        device=device,
        effective_batch_size=meta.get("effective_batch_size"),
    )
    meta["device_to_host_flush_batch_count"] = int(device_to_host_flush_batch_count)
    if len(vectors_for_indices) > 0:
        start = 0
        pending_cls_tensors: list[Any] = []
        pending_index_batches: list[np.ndarray] = []
        cpu_thread_policy = (
            temporary_torch_cpu_thread_policy(torch, intra_op_threads=cpu_thread_count)
            if str(device).startswith("cpu")
            else nullcontext({})
        )
        with loop_progress(
            total=len(vectors_for_indices),
            label="SPECTER texts",
            enabled=show_progress,
            unit="text",
        ) as tracker:
            with cpu_thread_policy as cpu_thread_meta:
                if isinstance(cpu_thread_meta, dict) and cpu_thread_meta:
                    meta.update(cpu_thread_meta)
                if runtime_backend_clean == "onnx_fp32":
                    session_input_names = [item.name for item in onnx_session.get_inputs()]
                    while start < len(vectors_for_indices):
                        current_batch_size = min(int(meta["effective_batch_size"]), len(vectors_for_indices) - start)
                        batch_indices = vectors_for_indices[start : start + current_batch_size]
                        chunk = [texts[idx] for idx in batch_indices]
                        tokenize_started_at = perf_counter()
                        enc = tokenizer(
                            chunk,
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="np",
                        )
                        meta["tokenize_seconds_total"] += perf_counter() - tokenize_started_at
                        if "attention_mask" in enc:
                            lengths = np.asarray(enc["attention_mask"], dtype=np.int64).sum(axis=1)
                            token_count = int(lengths.sum())
                            max_sequence_length = int(lengths.max()) if lengths.size else 0
                            observed_count = int(lengths.size)
                        else:
                            input_ids = np.asarray(enc["input_ids"], dtype=np.int64)
                            token_count = int(input_ids.size)
                            max_sequence_length = int(input_ids.shape[1]) if input_ids.ndim >= 2 else 0
                            observed_count = int(input_ids.shape[0]) if input_ids.ndim >= 1 else 0
                        meta["token_count_total"] += int(token_count)
                        meta["max_sequence_length_observed"] = max(
                            int(meta["max_sequence_length_observed"]),
                            int(max_sequence_length),
                        )
                        observed_sequence_count += int(observed_count)
                        forward_started_at = perf_counter()
                        ort_inputs = {
                            name: np.asarray(enc[name], dtype=np.int64)
                            for name in session_input_names
                            if name in enc
                        }
                        last_hidden_state = onnx_session.run(None, ort_inputs)[0]
                        cls = np.asarray(last_hidden_state[:, 0, :], dtype=np.float32)
                        meta["forward_seconds_total"] += perf_counter() - forward_started_at
                        out[np.asarray(batch_indices, dtype=np.int64)] = cls
                        meta["batches_total"] += 1
                        start += current_batch_size
                        tracker.update(current_batch_size)
                else:
                    with torch.inference_mode():
                        while start < len(vectors_for_indices):
                            while True:
                                current_batch_size = min(int(meta["effective_batch_size"]), len(vectors_for_indices) - start)
                                batch_indices = vectors_for_indices[start : start + current_batch_size]
                                chunk = [texts[idx] for idx in batch_indices]
                                try:
                                    tokenize_started_at = perf_counter()
                                    enc = tokenizer(
                                        chunk,
                                        padding=True,
                                        truncation=True,
                                        max_length=max_length,
                                        return_tensors="pt",
                                    )
                                    meta["tokenize_seconds_total"] += perf_counter() - tokenize_started_at
                                    token_count, max_sequence_length, observed_count = _observed_token_stats(enc)
                                    meta["token_count_total"] += int(token_count)
                                    meta["max_sequence_length_observed"] = max(
                                        int(meta["max_sequence_length_observed"]),
                                        int(max_sequence_length),
                                    )
                                    observed_sequence_count += int(observed_count)

                                    host_to_device_started_at = perf_counter()
                                    enc = _encoding_to_device(enc, device)
                                    meta["host_to_device_seconds_total"] += perf_counter() - host_to_device_started_at

                                    forward_started_at = perf_counter()
                                    with _autocast_context(torch, effective_precision_mode):
                                        batch_out = model(**enc)
                                    cls_tensor = batch_out.last_hidden_state[:, 0, :].detach()
                                    meta["forward_seconds_total"] += perf_counter() - forward_started_at

                                    pending_cls_tensors.append(cls_tensor)
                                    pending_index_batches.append(np.asarray(batch_indices, dtype=np.int64))
                                    if len(pending_cls_tensors) >= int(device_to_host_flush_batch_count):
                                        _flush_pending_cls_batches(
                                            torch_module=torch,
                                            pending_tensors=pending_cls_tensors,
                                            pending_indices=pending_index_batches,
                                            out=out,
                                            meta=meta,
                                        )
                                    meta["batches_total"] += 1
                                    start += current_batch_size
                                    tracker.update(current_batch_size)
                                    break
                                except Exception as exc:
                                    if not str(device).startswith("cuda") or not _is_cuda_oom(torch, exc):
                                        raise
                                    _flush_pending_cls_batches(
                                        torch_module=torch,
                                        pending_tensors=pending_cls_tensors,
                                        pending_indices=pending_index_batches,
                                        out=out,
                                        meta=meta,
                                    )
                                    _best_effort_clear_cuda_cache(torch)
                                    if int(meta["effective_batch_size"]) > 16:
                                        meta["oom_retry_count"] += 1
                                        meta["effective_batch_size"] = max(16, int(meta["effective_batch_size"]) // 2)
                                        device_to_host_flush_batch_count = _resolve_device_to_host_flush_batch_count(
                                            torch,
                                            device=device,
                                            effective_batch_size=int(meta["effective_batch_size"]),
                                        )
                                        meta["device_to_host_flush_batch_count"] = int(device_to_host_flush_batch_count)
                                        continue
                                    if requested_device.strip().lower() == "auto":
                                        meta["oom_retry_count"] += 1
                                        meta["fallback_reason"] = "cuda_oom_cpu_fallback"
                                        device = "cpu"
                                        model.to(device)
                                        _best_effort_clear_cuda_cache(torch)
                                        effective_precision_mode = _resolve_effective_precision_mode(
                                            torch=torch,
                                            precision_mode=precision_mode,
                                            device=device,
                                        )
                                        device_to_host_flush_batch_count = _resolve_device_to_host_flush_batch_count(
                                            torch,
                                            device=device,
                                            effective_batch_size=16,
                                        )
                                        meta["resolved_device"] = device
                                        meta["effective_precision_mode"] = effective_precision_mode
                                        meta["effective_batch_size"] = 16
                                        meta["device_to_host_flush_batch_count"] = int(device_to_host_flush_batch_count)
                                        continue
                                    raise
                        _flush_pending_cls_batches(
                            torch_module=torch,
                            pending_tensors=pending_cls_tensors,
                            pending_indices=pending_index_batches,
                            out=out,
                            meta=meta,
                        )

    if observed_sequence_count > 0:
        meta["mean_sequence_length_observed"] = float(meta["token_count_total"]) / float(observed_sequence_count)

    meta.update(
        {
            "generation_mode": "precomputed_plus_model"
            if precomputed_summary["precomputed_embedding_count"]
            else "model_only",
            **precomputed_summary,
        }
    )
    return (out.astype(np.float32), meta) if return_meta else out.astype(np.float32)


def get_or_create_specter_embeddings(
    mentions: pd.DataFrame,
    output_path: str | Path,
    force_recompute: bool = False,
    model_name: str = "allenai/specter",
    text_backend: str = "transformers",
    text_adapter_name: str | None = None,
    text_adapter_alias: str = "specter2",
    runtime_backend: str = "transformers",
    batch_size: int | None = None,
    max_length: int = 256,
    device: str = "auto",
    precision_mode: str = "auto",
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
                "generation_mode": "cache",
                **_base_runtime_meta(
                    requested_device=str(device),
                    resolved_device=None,
                    effective_precision_mode=None,
                    requested_batch_size=None if batch_size is None else int(batch_size),
                    effective_batch_size=None,
                    fallback_reason=None,
                    torch_version=None,
                    torch_cuda_version=None,
                    torch_cuda_available=None,
                    cuda_probe_error=None,
                    model_to_cuda_error=None,
                ),
                **precomputed_summary,
            }
            meta["cache_hit"] = True
            return (cached, meta) if return_meta else cached

    emb_result = generate_specter_embeddings(
        mentions=mentions,
        model_name=model_name,
        text_backend=text_backend,
        text_adapter_name=text_adapter_name,
        text_adapter_alias=text_adapter_alias,
        runtime_backend=runtime_backend,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        precision_mode=precision_mode,
        prefer_precomputed=prefer_precomputed,
        use_stub_if_missing=use_stub_if_missing,
        show_progress=show_progress,
        quiet_libraries=quiet_libraries,
        reuse_model=reuse_model,
        onnx_cache_path=build_onnx_cache_path(output_path=output_path, model_name=model_name, max_length=max_length),
        return_meta=True,
    )
    emb, meta = emb_result
    atomic_save_npy(output, emb)
    return (emb, meta) if return_meta else emb
