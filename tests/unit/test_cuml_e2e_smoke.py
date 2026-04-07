from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "benchmarks"
        / "cuml_e2e_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("cuml_e2e_smoke", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Failed to load cuml_e2e_smoke module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_gpu_env_integrity_reports_version_drift(monkeypatch):
    module = _load_script_module()
    monkeypatch.setattr(
        module,
        "_run_pip_check",
        lambda: (
            False,
            [
                "cupy-cuda12x 14.0.1 has requirement cuda-pathfinder==1.*,>=1.3.3, but you have cuda-pathfinder 1.2.2.",
                "cuda-python 12.9.5 has requirement cuda-bindings~=12.9.5, but you have cuda-bindings 12.9.4.",
            ],
        ),
    )
    monkeypatch.setattr(
        module,
        "_probe_module_import",
        lambda module_name: {
            "ok": False,
            "error_type": "AttributeError",
            "error": f"{module_name} missing found_via",
        },
    )
    monkeypatch.setattr(
        module,
        "_collect_package_versions",
        lambda: {
            "torch": "2.10.0+cu126",
            "cupy-cuda12x": "14.0.1",
            "cuml-cu12": "26.2.0",
            "cuda-python": "12.9.5",
            "cuda-bindings": "12.9.4",
            "cuda-pathfinder": "1.2.2",
            "rmm-cu12": "26.2.0",
            "pylibraft-cu12": "26.2.0",
        },
    )

    report = module.check_gpu_env_integrity()

    assert report["ok"] is False
    assert any("cuda-pathfinder" in failure for failure in report["failures"])
    assert any("cuda-bindings" in failure for failure in report["failures"])
    message = module._format_env_integrity_failure(report)
    assert "GPU environment integrity failed." in message
    assert "cuda-pathfinder=1.2.2" in message
    assert "dedicated RAPIDS/cuML environment" in message
