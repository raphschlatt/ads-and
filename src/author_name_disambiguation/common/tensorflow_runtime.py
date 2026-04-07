from __future__ import annotations

import importlib.metadata
import os
import re
import sys
from typing import Any, Mapping

_SUPPORTED_TF_CUDA_TAG = "12"
_CUDA_TAG_PATTERN = re.compile(r"-cu(\d+)$")
_VENDOR_PACKAGE_NAMES = (
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvcc-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-cufile-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cusparselt-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvshmem-cu12",
    "nvidia-nvtx-cu12",
    "nvidia-cublas-cu13",
    "nvidia-cuda-cupti-cu13",
    "nvidia-cuda-nvcc-cu13",
    "nvidia-cuda-nvrtc-cu13",
    "nvidia-cuda-runtime-cu13",
    "nvidia-cudnn-cu13",
    "nvidia-cufft-cu13",
    "nvidia-cufile-cu13",
    "nvidia-curand-cu13",
    "nvidia-cusolver-cu13",
    "nvidia-cusparse-cu13",
    "nvidia-cusparselt-cu13",
    "nvidia-nccl-cu13",
    "nvidia-nvjitlink-cu13",
    "nvidia-nvshmem-cu13",
    "nvidia-nvtx-cu13",
)


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _python_environment() -> tuple[str, str | None]:
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if isinstance(virtual_env, str) and virtual_env.strip():
        return sys.executable, virtual_env
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    if sys.prefix != base_prefix:
        return sys.executable, sys.prefix
    return sys.executable, None


def _collect_vendor_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in _VENDOR_PACKAGE_NAMES:
        version = _package_version(package_name)
        if version is not None:
            versions[package_name] = version
    return versions


def _detect_vendor_cuda_tags(package_versions: Mapping[str, str]) -> list[str]:
    tags: set[str] = set()
    for package_name in package_versions:
        match = _CUDA_TAG_PATTERN.search(str(package_name))
        if match:
            tags.add(str(match.group(1)))
    return sorted(tags)


def _probe_torch_runtime() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch_version": None,
        "torch_cuda_version": None,
        "torch_cuda_available": None,
        "torch_resolved_device": None,
        "torch_gpu_name": None,
        "torch_import_error": None,
        "torch_cuda_probe_error": None,
    }
    try:
        import torch
    except Exception as exc:
        payload["torch_import_error"] = repr(exc)
        payload["torch_cuda_available"] = False
        payload["torch_resolved_device"] = "cpu"
        return payload

    payload["torch_version"] = getattr(torch, "__version__", None)
    payload["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        payload["torch_cuda_available"] = False
        payload["torch_resolved_device"] = "cpu"
        payload["torch_cuda_probe_error"] = repr(exc)
        return payload

    payload["torch_cuda_available"] = cuda_available
    if not cuda_available:
        payload["torch_resolved_device"] = "cpu"
        return payload

    try:
        device_index = int(torch.cuda.current_device())
        payload["torch_resolved_device"] = f"cuda:{device_index}"
        payload["torch_gpu_name"] = torch.cuda.get_device_name(device_index)
    except Exception as exc:
        payload["torch_resolved_device"] = "cpu"
        payload["torch_cuda_probe_error"] = repr(exc)
    return payload


def _probe_tensorflow_runtime() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tensorflow_version": None,
        "tensorflow_built_with_cuda": None,
        "tensorflow_cuda_version": None,
        "tensorflow_cudnn_version": None,
        "tensorflow_visible_gpu_count": 0,
        "tensorflow_visible_gpus": [],
        "tensorflow_import_error": None,
        "tensorflow_gpu_probe_error": None,
    }
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        payload["tensorflow_import_error"] = repr(exc)
        return payload

    payload["tensorflow_version"] = getattr(tf, "__version__", None)
    sysconfig = getattr(tf, "sysconfig", None)
    build_info: Mapping[str, Any] | None = None
    if sysconfig is not None and hasattr(sysconfig, "get_build_info"):
        try:
            raw = sysconfig.get_build_info()
        except Exception:
            raw = None
        if isinstance(raw, Mapping):
            build_info = raw
    if build_info:
        payload["tensorflow_built_with_cuda"] = build_info.get("is_cuda_build")
        payload["tensorflow_cuda_version"] = build_info.get("cuda_version")
        payload["tensorflow_cudnn_version"] = build_info.get("cudnn_version")

    try:
        visible_gpus = list(getattr(tf.config, "list_physical_devices")("GPU"))
    except Exception as exc:
        payload["tensorflow_gpu_probe_error"] = repr(exc)
        return payload

    payload["tensorflow_visible_gpu_count"] = int(len(visible_gpus))
    payload["tensorflow_visible_gpus"] = [str(gpu) for gpu in visible_gpus]
    return payload


def _classify_runtime(report: Mapping[str, Any]) -> tuple[str, str | None]:
    tensorflow_import_error = report.get("tensorflow_import_error")
    if tensorflow_import_error:
        return "unavailable", "tensorflow_import_failed"

    visible_gpu_count = int(report.get("tensorflow_visible_gpu_count") or 0)
    if visible_gpu_count > 0:
        return "ok", None

    torch_cuda_available = report.get("torch_cuda_available") is True
    tensorflow_built_with_cuda = report.get("tensorflow_built_with_cuda")
    tensorflow_cuda_version = report.get("tensorflow_cuda_version")
    detected_tags = {str(tag) for tag in report.get("detected_vendor_cuda_tags", []) or []}
    expected_tag = str(tensorflow_cuda_version).split(".", 1)[0] if tensorflow_cuda_version else _SUPPORTED_TF_CUDA_TAG

    if tensorflow_built_with_cuda is False:
        if torch_cuda_available:
            return "mismatch", "tensorflow_not_built_with_cuda"
        return "cpu_fallback", "tensorflow_cpu_build"

    if torch_cuda_available:
        if expected_tag not in detected_tags and len(detected_tags) > 0:
            detected_tag = sorted(detected_tags)[0]
            return "mismatch", f"tensorflow_expected_cu{expected_tag}_but_detected_cu{detected_tag}_stack"
        if report.get("tensorflow_gpu_probe_error"):
            return "mismatch", "tensorflow_gpu_probe_failed"
        return "mismatch", "tensorflow_gpu_unavailable_while_torch_cuda_available"

    return "cpu_fallback", "torch_cuda_unavailable"


def tensorflow_runtime_backend_label(runtime: Mapping[str, Any] | None) -> str:
    if runtime and str(runtime.get("status")) == "ok":
        return "tensorflow-gpu"
    return "tensorflow-cpu"


def probe_tensorflow_runtime() -> dict[str, Any]:
    python_executable, virtual_env = _python_environment()
    vendor_package_versions = _collect_vendor_package_versions()
    report: dict[str, Any] = {
        "python_executable": python_executable,
        "virtual_env": virtual_env,
        "vendor_package_versions": vendor_package_versions,
        "detected_vendor_cuda_tags": _detect_vendor_cuda_tags(vendor_package_versions),
        "supported_cuda_tag": _SUPPORTED_TF_CUDA_TAG,
    }
    report.update(_probe_torch_runtime())
    report.update(_probe_tensorflow_runtime())
    status, reason = _classify_runtime(report)
    report["status"] = status
    report["reason"] = reason
    report["runtime_backend"] = tensorflow_runtime_backend_label(report)
    return report


def tensorflow_runtime_needs_warning(runtime: Mapping[str, Any] | None) -> bool:
    if not runtime:
        return False
    return runtime.get("torch_cuda_available") is True and str(runtime.get("status")) != "ok"


def format_tensorflow_runtime_warning(
    runtime: Mapping[str, Any] | None,
    *,
    repair_hint: str = "Repair the repo venv with the documented cu126/cu12 uv pip workflow and rerun the GPU doctor.",
) -> str:
    if not runtime:
        return "chars2vec TensorFlow GPU unavailable; falling back to CPU."

    status = str(runtime.get("status") or "unknown")
    reason = str(runtime.get("reason") or "unknown")
    tensorflow_cuda_version = runtime.get("tensorflow_cuda_version") or "n/a"
    detected_tags = ",".join(str(tag) for tag in runtime.get("detected_vendor_cuda_tags", []) or []) or "none"
    virtual_env = runtime.get("virtual_env") or runtime.get("python_executable") or "n/a"
    return (
        "chars2vec TensorFlow GPU unavailable; falling back to CPU "
        f"(status={status}, reason={reason}, expected_cuda={tensorflow_cuda_version}, "
        f"detected_vendor_tags={detected_tags}, venv={virtual_env}). {repair_hint}"
    )
