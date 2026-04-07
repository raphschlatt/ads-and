#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from author_name_disambiguation.common.tensorflow_runtime import (  # noqa: E402
    format_tensorflow_runtime_warning,
    probe_tensorflow_runtime,
)


@contextmanager
def _quiet_stderr() -> None:
    stream = getattr(sys, "stderr", None)
    if stream is None or not hasattr(stream, "fileno"):
        yield
        return
    try:
        stderr_fd = int(stream.fileno())
    except Exception:
        yield
        return

    saved_fd = os.dup(stderr_fd)
    read_fd, write_fd = os.pipe()

    def _drain() -> None:
        try:
            while True:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    break
        finally:
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


def _run_pip_check() -> tuple[bool, list[str]]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [
        line.strip()
        for line in (completed.stdout.splitlines() + completed.stderr.splitlines())
        if line.strip()
    ]
    if completed.returncode == 0:
        lines = [line for line in lines if line != "No broken requirements found."]
    return completed.returncode == 0, lines


def build_gpu_env_report() -> dict[str, Any]:
    pip_check_ok, pip_check_issues = _run_pip_check()
    with _quiet_stderr():
        runtime = probe_tensorflow_runtime()
    return {
        "runtime": runtime,
        "pip_check": {
            "ok": pip_check_ok,
            "issues": pip_check_issues,
        },
        "repair_commands": [
            "source /home/ubuntu/Author_Name_Disambiguation/.venv/bin/activate",
            'uv pip install --python /home/ubuntu/Author_Name_Disambiguation/.venv/bin/python --editable ".[dev]" --torch-backend cu126',
            "uv pip install --python /home/ubuntu/Author_Name_Disambiguation/.venv/bin/python --reinstall --no-deps -r requirements-gpu-cu126.txt",
            "python -m pip check",
            "python scripts/ops/gpu_env_doctor.py --json",
        ],
    }


def _format_report(report: dict[str, Any]) -> str:
    runtime = dict(report.get("runtime") or {})
    lines = [
        "GPU environment doctor",
        f"- python_executable={runtime.get('python_executable') or 'n/a'}",
        f"- virtual_env={runtime.get('virtual_env') or 'n/a'}",
        f"- torch_version={runtime.get('torch_version') or 'n/a'}",
        f"- torch_cuda_version={runtime.get('torch_cuda_version') or 'n/a'}",
        f"- torch_cuda_available={runtime.get('torch_cuda_available')}",
        f"- tensorflow_version={runtime.get('tensorflow_version') or 'n/a'}",
        f"- tensorflow_built_with_cuda={runtime.get('tensorflow_built_with_cuda')}",
        f"- tensorflow_cuda_version={runtime.get('tensorflow_cuda_version') or 'n/a'}",
        f"- tensorflow_cudnn_version={runtime.get('tensorflow_cudnn_version') or 'n/a'}",
        f"- tensorflow_visible_gpu_count={runtime.get('tensorflow_visible_gpu_count')}",
        f"- status={runtime.get('status') or 'n/a'}",
        f"- reason={runtime.get('reason') or 'n/a'}",
        f"- runtime_backend={runtime.get('runtime_backend') or 'n/a'}",
        f"- detected_vendor_cuda_tags={','.join(runtime.get('detected_vendor_cuda_tags') or []) or 'none'}",
    ]
    vendor_packages = dict(runtime.get("vendor_package_versions") or {})
    if vendor_packages:
        lines.append("Vendor packages:")
        for package_name in sorted(vendor_packages):
            lines.append(f"- {package_name}={vendor_packages[package_name]}")
    pip_check = dict(report.get("pip_check") or {})
    lines.append(f"Pip check: ok={bool(pip_check.get('ok'))}")
    for issue in pip_check.get("issues", []) or []:
        lines.append(f"- {issue}")
    lines.append(format_tensorflow_runtime_warning(runtime))
    lines.append("Repair commands:")
    for cmd in report.get("repair_commands", []) or []:
        lines.append(f"- {cmd}")
    return "\n".join(lines)


def _report_exit_code(report: dict[str, Any], *, require_tensorflow_gpu: bool) -> int:
    runtime = dict(report.get("runtime") or {})
    status = str(runtime.get("status") or "unknown")
    if require_tensorflow_gpu and status != "ok":
        return 1
    if status in {"mismatch", "unavailable"}:
        return 1
    if not bool((report.get("pip_check") or {}).get("ok")):
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose the repo chars2vec TensorFlow GPU runtime.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--require-tensorflow-gpu",
        action="store_true",
        help="Fail unless TensorFlow can see a GPU in the active repo venv.",
    )
    args = parser.parse_args()

    report = build_gpu_env_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_format_report(report))
    return _report_exit_code(report, require_tensorflow_gpu=bool(args.require_tensorflow_gpu))


if __name__ == "__main__":
    raise SystemExit(main())
