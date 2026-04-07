from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_doctor_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "ops"
        / "gpu_env_doctor.py"
    )
    spec = importlib.util.spec_from_file_location("gpu_env_doctor", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Failed to load gpu_env_doctor module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_env_doctor_reports_mismatch_and_repair_steps(monkeypatch):
    module = _load_doctor_module()
    monkeypatch.setattr(module, "_run_pip_check", lambda: (True, []))
    monkeypatch.setattr(
        module,
        "probe_tensorflow_runtime",
        lambda: {
            "python_executable": "/tmp/python",
            "virtual_env": "/tmp/venv",
            "torch_version": "2.10.0+cu126",
            "torch_cuda_version": "12.6",
            "torch_cuda_available": True,
            "tensorflow_version": "2.20.0",
            "tensorflow_built_with_cuda": True,
            "tensorflow_cuda_version": "12.5.1",
            "tensorflow_cudnn_version": "9",
            "tensorflow_visible_gpu_count": 0,
            "status": "mismatch",
            "reason": "tensorflow_expected_cu12_but_detected_cu13_stack",
            "runtime_backend": "tensorflow-cpu",
            "detected_vendor_cuda_tags": ["13"],
            "vendor_package_versions": {"nvidia-cudnn-cu13": "9.19.0.56"},
        },
    )

    report = module.build_gpu_env_report()
    message = module._format_report(report)

    assert report["runtime"]["status"] == "mismatch"
    assert module._report_exit_code(report, require_tensorflow_gpu=False) == 1
    assert "GPU environment doctor" in message
    assert "tensorflow_expected_cu12_but_detected_cu13_stack" in message
    assert "requirements-gpu-cu126.txt" in message


def test_gpu_env_doctor_require_gpu_fails_on_cpu_fallback(monkeypatch):
    module = _load_doctor_module()
    monkeypatch.setattr(module, "_run_pip_check", lambda: (True, []))
    monkeypatch.setattr(
        module,
        "probe_tensorflow_runtime",
        lambda: {
            "status": "cpu_fallback",
            "reason": "torch_cuda_unavailable",
            "runtime_backend": "tensorflow-cpu",
            "torch_cuda_available": False,
            "detected_vendor_cuda_tags": [],
        },
    )

    report = module.build_gpu_env_report()

    assert module._report_exit_code(report, require_tensorflow_gpu=False) == 0
    assert module._report_exit_code(report, require_tensorflow_gpu=True) == 1
