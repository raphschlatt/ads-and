from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from author_name_disambiguation.common import tensorflow_runtime


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, *, cuda_available: bool) -> None:
    class _Cuda:
        @staticmethod
        def is_available():
            return cuda_available

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_name(_index):
            return "Mock GPU"

    fake_torch = SimpleNamespace(
        __version__="2.6.0+cu124",
        version=SimpleNamespace(cuda="12.4"),
        cuda=_Cuda(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _install_fake_tensorflow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    visible_gpu_count: int,
    physical_gpu_count: int | None = None,
    built_with_cuda: bool = True,
    cuda_version: str = "12.5.1",
) -> None:
    resolved_physical_gpu_count = visible_gpu_count if physical_gpu_count is None else int(physical_gpu_count)

    class _Config:
        @staticmethod
        def list_physical_devices(kind):
            if kind != "GPU":
                return []
            return [f"GPU:{idx}" for idx in range(resolved_physical_gpu_count)]

        @staticmethod
        def get_visible_devices(kind):
            if kind != "GPU":
                return []
            return [f"GPU:{idx}" for idx in range(visible_gpu_count)]

    class _SysConfig:
        @staticmethod
        def get_build_info():
            return {
                "is_cuda_build": built_with_cuda,
                "cuda_version": cuda_version,
                "cudnn_version": "9",
            }

    fake_tf = SimpleNamespace(
        __version__="2.20.0",
        config=_Config(),
        sysconfig=_SysConfig(),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)


def test_probe_tensorflow_runtime_classifies_cu13_mismatch(monkeypatch: pytest.MonkeyPatch):
    _install_fake_torch(monkeypatch, cuda_available=True)
    _install_fake_tensorflow(monkeypatch, visible_gpu_count=0)
    monkeypatch.setattr(
        tensorflow_runtime,
        "_collect_vendor_package_versions",
        lambda: {"nvidia-cudnn-cu13": "9.19.0.56"},
    )

    report = tensorflow_runtime.probe_tensorflow_runtime()

    assert report["status"] == "mismatch"
    assert report["reason"] == "tensorflow_expected_cu12_but_detected_cu13_stack"
    assert report["runtime_backend"] == "tensorflow-cpu"
    assert report["torch_cuda_available"] is True
    assert report["tensorflow_visible_gpu_count"] == 0


def test_probe_tensorflow_runtime_marks_ok_when_tensorflow_sees_gpu(monkeypatch: pytest.MonkeyPatch):
    _install_fake_torch(monkeypatch, cuda_available=True)
    _install_fake_tensorflow(monkeypatch, visible_gpu_count=1)
    monkeypatch.setattr(
        tensorflow_runtime,
        "_collect_vendor_package_versions",
        lambda: {"nvidia-cudnn-cu12": "9.10.2.21"},
    )

    report = tensorflow_runtime.probe_tensorflow_runtime()

    assert report["status"] == "ok"
    assert report["reason"] is None
    assert report["runtime_backend"] == "tensorflow-gpu"
    assert report["tensorflow_visible_gpu_count"] == 1


def test_probe_tensorflow_runtime_classifies_forced_cpu_without_warning(monkeypatch: pytest.MonkeyPatch):
    _install_fake_torch(monkeypatch, cuda_available=True)
    _install_fake_tensorflow(monkeypatch, visible_gpu_count=0, physical_gpu_count=1)
    monkeypatch.setattr(
        tensorflow_runtime,
        "_collect_vendor_package_versions",
        lambda: {"nvidia-cudnn-cu12": "9.10.2.21"},
    )

    report = tensorflow_runtime.probe_tensorflow_runtime(force_cpu=True)

    assert report["status"] == "cpu_fallback"
    assert report["reason"] == "forced_cpu"
    assert report["runtime_backend"] == "tensorflow-cpu"
    assert report["tensorflow_force_cpu_requested"] is True
    assert report["tensorflow_physical_gpu_count"] == 1
    assert report["tensorflow_visible_gpu_count"] == 0
    assert tensorflow_runtime.tensorflow_runtime_needs_warning(report) is False
