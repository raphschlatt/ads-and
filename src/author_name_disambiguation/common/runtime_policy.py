from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from author_name_disambiguation.common.cpu_runtime import detect_available_ram_bytes, detect_cpu_limit
from author_name_disambiguation.common.io_schema import available_disk_bytes
from author_name_disambiguation.common.tensorflow_runtime import probe_tensorflow_runtime

_PAIR_SCORE_BATCH_BYTES_PER_ROW = (50 + 768) * 4 * 2


def _largest_power_of_two_leq(value: int) -> int:
    if int(value) <= 1:
        return 1
    return 1 << (int(value).bit_length() - 1)


def _probe_onnx_cpu_backend() -> tuple[bool, str | None]:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception as exc:
        return False, f"missing_onnx_cpu_runtime:{exc.__class__.__name__}"
    return True, None


def _probe_cuml_gpu_backend() -> tuple[bool, str | None]:
    try:
        import cupy
        from cuml.cluster import DBSCAN as _  # noqa: F401
    except Exception as exc:
        return False, f"missing_cuml_or_cupy:{exc.__class__.__name__}"
    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return False, f"cupy_runtime_error:{exc.__class__.__name__}"
    if device_count < 1:
        return False, "no_cuda_device"
    return True, None


def _probe_torch_host(bootstrap_runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "torch_version": None,
        "torch_cuda_version": None,
        "torch_cuda_available": False,
        "resolved_device": "cpu",
        "gpu_name": None,
        "gpu_total_memory_bytes": None,
        "fallback_reason": None,
        "cuda_probe_error": None,
    }
    if bootstrap_runtime:
        payload.update(
            {
                "torch_version": bootstrap_runtime.get("torch_version"),
                "torch_cuda_version": bootstrap_runtime.get("torch_cuda_version"),
                "torch_cuda_available": bool(bootstrap_runtime.get("torch_cuda_available")),
                "resolved_device": bootstrap_runtime.get("resolved_device") or "cpu",
                "gpu_name": bootstrap_runtime.get("gpu_name"),
                "fallback_reason": bootstrap_runtime.get("fallback_reason"),
                "cuda_probe_error": bootstrap_runtime.get("cuda_probe_error"),
            }
        )
    try:
        import torch
    except Exception as exc:
        payload["fallback_reason"] = payload.get("fallback_reason") or "torch_import_failed"
        payload["cuda_probe_error"] = payload.get("cuda_probe_error") or repr(exc)
        return payload

    if payload.get("torch_version") is None:
        payload["torch_version"] = getattr(torch, "__version__", None)
    if payload.get("torch_cuda_version") is None:
        payload["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        payload["torch_cuda_available"] = False
        payload["resolved_device"] = "cpu"
        payload["fallback_reason"] = payload.get("fallback_reason") or "cuda_probe_failed"
        payload["cuda_probe_error"] = payload.get("cuda_probe_error") or repr(exc)
        return payload

    payload["torch_cuda_available"] = cuda_available
    if not cuda_available:
        payload["resolved_device"] = "cpu"
        payload["fallback_reason"] = payload.get("fallback_reason") or "torch_cuda_unavailable"
        return payload

    try:
        device_index = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device_index)
        payload["resolved_device"] = f"cuda:{device_index}"
        payload["gpu_name"] = payload.get("gpu_name") or torch.cuda.get_device_name(device_index)
        total_memory = getattr(props, "total_memory", None)
        payload["gpu_total_memory_bytes"] = None if total_memory is None else int(total_memory)
        payload["cuda_probe_error"] = None
        payload["fallback_reason"] = None
        return payload
    except Exception as exc:
        payload["resolved_device"] = "cpu"
        payload["fallback_reason"] = payload.get("fallback_reason") or "cuda_probe_failed"
        payload["cuda_probe_error"] = payload.get("cuda_probe_error") or repr(exc)
        return payload


def _resolve_runtime_mode(
    *,
    runtime_mode_requested: str | None,
    specter_runtime_backend_requested: str | None,
    requested_device: str,
    torch_host: Mapping[str, Any],
) -> str:
    requested_mode = None if runtime_mode_requested is None else str(runtime_mode_requested).strip().lower() or None
    if requested_mode is not None:
        return requested_mode
    backend = str(specter_runtime_backend_requested or "transformers").strip().lower() or "transformers"
    device = str(requested_device or "auto").strip().lower() or "auto"
    if backend == "onnx_fp32" or device.startswith("cpu"):
        return "cpu"
    if str(torch_host.get("resolved_device") or "").startswith("cuda"):
        return "gpu"
    return "cpu"


def _resolve_requested_device_for_runtime_mode(*, runtime_mode: str, requested_device: str) -> str:
    device = str(requested_device or "auto").strip().lower() or "auto"
    if runtime_mode == "gpu":
        return "cuda" if device == "auto" else device
    if runtime_mode == "cpu":
        return "cpu" if device == "auto" else device
    return device


def _resolve_chars2vec_cpu_batch_size(available_ram_bytes: int | None) -> int:
    if available_ram_bytes is None:
        return 128
    if int(available_ram_bytes) < 6 * 1024**3:
        return 32
    if int(available_ram_bytes) < 12 * 1024**3:
        return 64
    return 128


def _clamp_score_batch_size_for_cpu(score_batch_size: int, available_ram_bytes: int | None) -> tuple[int, dict[str, Any] | None]:
    requested = int(max(1, score_batch_size))
    if available_ram_bytes is None:
        return requested, None
    budget_bytes = int(max(1, int(available_ram_bytes) * 0.10))
    estimated_peak = int(requested * _PAIR_SCORE_BATCH_BYTES_PER_ROW)
    if estimated_peak <= budget_bytes:
        return requested, None
    allowed_rows = int(max(1, budget_bytes // _PAIR_SCORE_BATCH_BYTES_PER_ROW))
    clamped = int(max(1024, _largest_power_of_two_leq(allowed_rows)))
    clamped = int(min(requested, clamped))
    if clamped >= requested:
        return requested, None
    return clamped, {
        "requested_score_batch_size": requested,
        "effective_score_batch_size": clamped,
        "estimated_peak_bytes": estimated_peak,
        "budget_bytes": budget_bytes,
        "reason": "cpu_score_batch_peak_exceeds_10pct_available_ram",
    }


def _append_policy_fallback(
    target: list[dict[str, Any]],
    *,
    component: str,
    reason: str,
    action: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "component": str(component),
        "reason": str(reason),
        "action": str(action),
    }
    if details:
        payload["details"] = dict(details)
    if payload not in target:
        target.append(payload)


def resolve_infer_runtime_policy(
    *,
    requested_device: str,
    runtime_mode_requested: str | None,
    specter_runtime_backend_requested: str | None,
    cluster_backend_requested: str,
    score_batch_size: int,
    scratch_dir: str | Path,
    bootstrap_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    torch_host = _probe_torch_host(bootstrap_runtime=bootstrap_runtime)
    cpu_info = detect_cpu_limit()
    available_ram_bytes = detect_available_ram_bytes()
    scratch_path = Path(scratch_dir).expanduser().resolve()
    scratch_free_bytes = available_disk_bytes(scratch_path)
    onnx_available, onnx_reason = _probe_onnx_cpu_backend()
    cuml_available, cuml_reason = _probe_cuml_gpu_backend()
    tensorflow_runtime = probe_tensorflow_runtime(force_cpu=True)

    host_profile = {
        "requested_device": str(requested_device or "auto"),
        "cpu": dict(cpu_info),
        "available_ram_bytes": None if available_ram_bytes is None else int(available_ram_bytes),
        "scratch_dir": str(scratch_path),
        "scratch_free_bytes": None if scratch_free_bytes is None else int(scratch_free_bytes),
        "torch": dict(torch_host),
        "tensorflow_runtime": dict(tensorflow_runtime),
        "onnx_cpu_backend": {
            "available": bool(onnx_available),
            "reason": onnx_reason,
        },
        "cuml_gpu_backend": {
            "available": bool(cuml_available),
            "reason": cuml_reason,
        },
    }

    runtime_mode_effective = _resolve_runtime_mode(
        runtime_mode_requested=runtime_mode_requested,
        specter_runtime_backend_requested=specter_runtime_backend_requested,
        requested_device=requested_device,
        torch_host=torch_host,
    )
    effective_request_device = _resolve_requested_device_for_runtime_mode(
        runtime_mode=runtime_mode_effective,
        requested_device=requested_device,
    )
    if runtime_mode_effective == "cpu":
        requested_backend = (
            str(specter_runtime_backend_requested).strip().lower()
            if specter_runtime_backend_requested is not None
            else ""
        )
        specter_runtime_backend_effective = requested_backend or "cpu_auto"
    elif runtime_mode_effective == "gpu":
        if str(specter_runtime_backend_requested or "").strip().lower() == "onnx_fp32":
            raise ValueError("runtime_mode='gpu' is incompatible with specter_runtime_backend='onnx_fp32'.")
        specter_runtime_backend_effective = "transformers"
    else:
        specter_runtime_backend_effective = str(specter_runtime_backend_requested or "transformers").strip().lower() or "transformers"

    chars2vec_batch_size = _resolve_chars2vec_cpu_batch_size(available_ram_bytes)
    effective_score_batch_size = int(max(1, score_batch_size))
    score_batch_clamp_meta = None
    if runtime_mode_effective == "cpu":
        effective_score_batch_size, score_batch_clamp_meta = _clamp_score_batch_size_for_cpu(
            effective_score_batch_size,
            available_ram_bytes,
        )

    cluster_backend_clean = str(cluster_backend_requested or "auto").strip().lower() or "auto"
    cluster_backend_effective = "sklearn_cpu" if cluster_backend_clean == "auto" else cluster_backend_clean

    safety_fallbacks: list[dict[str, Any]] = []
    if str(requested_device or "auto").strip().lower() == "auto" and runtime_mode_effective == "cpu":
        reason = str(torch_host.get("fallback_reason") or "device_auto_cpu_policy")
        _append_policy_fallback(
            safety_fallbacks,
            component="runtime_mode",
            reason=reason,
            action="cpu_runtime_selected",
        )
    if score_batch_clamp_meta is not None:
        _append_policy_fallback(
            safety_fallbacks,
            component="pair_scoring",
            reason=str(score_batch_clamp_meta["reason"]),
            action="score_batch_size_clamped",
            details=score_batch_clamp_meta,
        )
    if chars2vec_batch_size != 128:
        _append_policy_fallback(
            safety_fallbacks,
            component="chars2vec",
            reason="low_available_ram",
            action="cpu_batch_size_reduced",
            details={
                "effective_batch_size": int(chars2vec_batch_size),
                "available_ram_bytes": None if available_ram_bytes is None else int(available_ram_bytes),
            },
        )
    if cluster_backend_clean == "auto":
        _append_policy_fallback(
            safety_fallbacks,
            component="clustering",
            reason="auto_cpu_only_policy",
            action="cluster_backend_resolved_to_sklearn_cpu",
        )
    if runtime_mode_effective == "cpu" and not onnx_available:
        _append_policy_fallback(
            safety_fallbacks,
            component="specter",
            reason=str(onnx_reason or "onnx_cpu_unavailable"),
            action="cpu_auto_transformers_fallback_if_needed",
        )

    resolved_runtime_policy = {
        "runtime_mode_requested": None if runtime_mode_requested is None else str(runtime_mode_requested),
        "runtime_mode_effective": str(runtime_mode_effective),
        "requested_device": str(requested_device or "auto"),
        "effective_request_device": str(effective_request_device),
        "specter_runtime_backend_requested": None
        if specter_runtime_backend_requested is None
        else str(specter_runtime_backend_requested),
        "specter_runtime_backend_effective": str(specter_runtime_backend_effective),
        "chars2vec_force_cpu": True,
        "chars2vec_batch_size": int(chars2vec_batch_size),
        "score_batch_size_requested": int(score_batch_size),
        "score_batch_size_effective": int(effective_score_batch_size),
        "cluster_backend_requested": str(cluster_backend_clean),
        "cluster_backend_effective": str(cluster_backend_effective),
        "exact_graph_union_impl": "python",
        "numba_auto_enabled": False,
        "onnx_cpu_available": bool(onnx_available),
        "cuml_gpu_available": bool(cuml_available),
    }

    return {
        "host_profile": host_profile,
        "resolved_runtime_policy": resolved_runtime_policy,
        "safety_fallbacks": safety_fallbacks,
    }
