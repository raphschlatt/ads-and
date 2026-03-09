from __future__ import annotations

import os
import warnings
from typing import Any


def _safe_torch_cuda_available(torch) -> tuple[bool, str | None]:
    try:
        return bool(torch.cuda.is_available()), None
    except Exception as exc:  # pragma: no cover
        return False, repr(exc)


def _base_runtime_meta(torch, requested_device: str) -> dict[str, Any]:
    cuda_available, availability_error = _safe_torch_cuda_available(torch)
    meta = {
        "requested_device": str(requested_device),
        "resolved_device": None,
        "fallback_reason": None,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "torch_cuda_available": cuda_available,
        "cuda_probe_error": availability_error,
        "model_to_cuda_error": None,
        "effective_precision_mode": None,
    }
    return meta


def _format_fallback_message(runtime_label: str, runtime_meta: dict[str, Any]) -> str:
    torch_version = runtime_meta.get("torch_version")
    torch_cuda_version = runtime_meta.get("torch_cuda_version")
    cuda_available = runtime_meta.get("torch_cuda_available")
    reason = runtime_meta.get("fallback_reason")
    probe_error = runtime_meta.get("cuda_probe_error")
    move_error = runtime_meta.get("model_to_cuda_error")
    return (
        f"{runtime_label}: device=auto requested but PyTorch CUDA is unavailable; falling back to CPU. "
        f"torch={torch_version!s} torch.version.cuda={torch_cuda_version!s} "
        f"torch.cuda.is_available()={cuda_available!s} reason={reason!s} "
        f"cuda_probe_error={probe_error!s} model_to_cuda_error={move_error!s} "
        f"pid={os.getpid()}"
    )


def resolve_torch_device(torch, requested_device: str, *, runtime_label: str) -> tuple[str, dict[str, Any]]:
    requested = str(requested_device or "auto").strip().lower()
    meta = _base_runtime_meta(torch, requested)
    if requested != "auto":
        meta["resolved_device"] = requested
        return requested, meta

    if not bool(meta["torch_cuda_available"]):
        meta["resolved_device"] = "cpu"
        meta["fallback_reason"] = "torch_cuda_unavailable"
        warnings.warn(_format_fallback_message(runtime_label, meta), RuntimeWarning)
        return "cpu", meta

    try:
        _ = torch.cuda.current_device()
        _ = torch.empty(1, device="cuda")
        meta["resolved_device"] = "cuda"
        meta["cuda_probe_error"] = None
        return "cuda", meta
    except Exception as exc:  # pragma: no cover
        meta["resolved_device"] = "cpu"
        meta["fallback_reason"] = "cuda_probe_failed"
        meta["cuda_probe_error"] = repr(exc)
        warnings.warn(_format_fallback_message(runtime_label, meta), RuntimeWarning)
        return "cpu", meta


def apply_auto_cuda_move_fallback(
    *,
    requested_device: str,
    runtime_label: str,
    runtime_meta: dict[str, Any],
    exc: Exception,
) -> tuple[str, dict[str, Any]]:
    requested = str(requested_device or "auto").strip().lower()
    if requested == "auto" and str(runtime_meta.get("resolved_device") or "").startswith("cuda"):
        runtime_meta["resolved_device"] = "cpu"
        runtime_meta["fallback_reason"] = "cuda_model_move_failed"
        runtime_meta["model_to_cuda_error"] = repr(exc)
        warnings.warn(_format_fallback_message(runtime_label, runtime_meta), RuntimeWarning)
        return "cpu", runtime_meta
    raise exc
