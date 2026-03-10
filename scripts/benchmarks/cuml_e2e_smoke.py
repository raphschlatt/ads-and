#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import pandas as pd

from author_name_disambiguation.approaches.nand import cluster as cluster_mod


GPU_ENV_PACKAGES = (
    "torch",
    "cupy-cuda12x",
    "cuml-cu12",
    "cuda-python",
    "cuda-bindings",
    "cuda-pathfinder",
    "rmm-cu12",
    "pylibraft-cu12",
)


def _toy_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    mentions = pd.DataFrame(
        [
            {"mention_id": "a1", "block_key": "blk.a", "author_raw": "A, A", "year": 2000},
            {"mention_id": "a2", "block_key": "blk.a", "author_raw": "A, A", "year": 2001},
            {"mention_id": "b1", "block_key": "blk.b", "author_raw": "B, B", "year": 2000},
            {"mention_id": "b2", "block_key": "blk.b", "author_raw": "B, B", "year": 2001},
        ]
    )
    pair_scores = pd.DataFrame(
        [
            {"pair_id": "a1__a2", "mention_id_1": "a1", "mention_id_2": "a2", "block_key": "blk.a", "distance": 0.05},
            {"pair_id": "b1__b2", "mention_id_1": "b1", "mention_id_2": "b2", "block_key": "blk.b", "distance": 0.05},
        ]
    )
    cluster_config = {"eps": 0.2, "min_samples": 1, "metric": "precomputed", "constraints": {"enabled": False}}
    return mentions, pair_scores, cluster_config


def _collect_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in GPU_ENV_PACKAGES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


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


def _probe_module_import(module_name: str) -> dict[str, Any]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return {
            "ok": False,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
    return {
        "ok": True,
        "error_type": None,
        "error": None,
    }


def check_gpu_env_integrity() -> dict[str, Any]:
    pip_check_ok, pip_check_issues = _run_pip_check()
    imports = {
        module_name: _probe_module_import(module_name)
        for module_name in ("cupy", "cuml")
    }
    failures: list[str] = []
    if not pip_check_ok:
        failures.extend(f"pip_check:{line}" for line in pip_check_issues)
    for module_name, payload in imports.items():
        if not payload["ok"]:
            failures.append(
                f"import_{module_name}:{payload['error_type']}:{payload['error']}"
            )
    return {
        "ok": not failures,
        "python_executable": sys.executable,
        "package_versions": _collect_package_versions(),
        "pip_check": {
            "ok": pip_check_ok,
            "issues": pip_check_issues,
        },
        "imports": imports,
        "failures": failures,
    }


def _format_env_integrity_failure(report: dict[str, Any]) -> str:
    lines = ["GPU environment integrity failed."]
    for failure in report.get("failures", []):
        lines.append(f"- {failure}")
    versions = report.get("package_versions", {}) or {}
    if versions:
        lines.append("Package versions:")
        for package_name in GPU_ENV_PACKAGES:
            lines.append(f"- {package_name}={versions.get(package_name)}")
    lines.append(
        "Repair the venv with `uv pip --no-deps -r requirements-gpu-cu126.txt`; "
        "do not mix pip and uv installs."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test cuML clustering backend resolution and fallback behavior.")
    parser.add_argument(
        "--require-gpu-backend",
        action="store_true",
        help="Fail unless `cluster_backend=auto` resolves to `cuml_gpu`.",
    )
    args = parser.parse_args()

    env_report = check_gpu_env_integrity()
    if not env_report["ok"]:
        raise SystemExit(_format_env_integrity_failure(env_report))

    mentions, pair_scores, cluster_config = _toy_inputs()

    # Case A: real runtime resolution path.
    _clusters_auto, meta_auto = cluster_mod.cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cluster_config,
        backend="auto",
        return_meta=True,
    )

    # Case B: deterministic GPU-failure fallback path.
    with patch.object(
        cluster_mod,
        "_resolve_cluster_backend",
        lambda backend, metric: {
            "requested": str(backend),
            "effective": "cuml_gpu",
            "reason": "forced-test",
            "cuml_available": True,
            "metric": str(metric),
        },
    ), patch.object(
        cluster_mod,
        "_run_dbscan_cuml",
        lambda dist, eps, min_samples, metric: (_ for _ in ()).throw(RuntimeError("forced gpu failure")),
    ):
        _clusters_fallback, meta_fallback = cluster_mod.cluster_blockwise_dbscan(
            mentions=mentions,
            pair_scores=pair_scores,
            cluster_config=cluster_config,
            backend="auto",
            return_meta=True,
        )

    if args.require_gpu_backend and meta_auto.get("cluster_backend_effective") != "cuml_gpu":
        raise SystemExit(
            "Expected auto backend to resolve to cuml_gpu, "
            f"got {meta_auto.get('cluster_backend_effective')!r} "
            f"(reason={meta_auto.get('cluster_backend_reason')!r})."
        )

    if meta_fallback.get("cluster_backend_effective") != "sklearn_cpu":
        raise SystemExit(
            "Expected forced GPU failure fallback to resolve to sklearn_cpu, "
            f"got {meta_fallback.get('cluster_backend_effective')!r}."
        )

    print(
        json.dumps(
            {
                "env_report": env_report,
                "auto_meta": meta_auto,
                "fallback_meta": meta_fallback,
                "require_gpu_backend": bool(args.require_gpu_backend),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
