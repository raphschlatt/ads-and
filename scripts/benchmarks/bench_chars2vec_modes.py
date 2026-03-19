#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from author_name_disambiguation.features.embed_chars2vec import generate_chars2vec_embeddings

_EXACT32_BATCH_SIZE = 32
_LAB_MATRIX = "lab_modes"
_TRACK_A_MATRIX = "exact32_device_compare"
_DATASET_AS_IS = "as_is"
_DATASET_UNIQUE_ONLY = "unique_only"
_DEVICE_GPU = "gpu"
_DEVICE_CPU = "cpu"
_RUN_MEDIAN_FIELDS = ("wall_seconds",)
_META_MEDIAN_FIELDS = (
    "predict_seconds",
    "pad_seconds",
    "model_load_seconds",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_output_path(repo_root: Path, *, matrix: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "artifacts" / "benchmarks" / f"chars2vec_{matrix}_{timestamp}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark chars2vec execution modes and exact-path device options.")
    parser.add_argument("--names-file", required=True, help="Text file with one author name per line.")
    parser.add_argument("--model-name", default="eng_50", help="chars2vec model name to load. Default: eng_50")
    parser.add_argument(
        "--matrix",
        choices=[_LAB_MATRIX, _TRACK_A_MATRIX],
        default=_LAB_MATRIX,
        help=f"Benchmark matrix to run. Default: {_LAB_MATRIX}",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warm-up runs per exact32 CPU/GPU case. Default: 1",
    )
    parser.add_argument(
        "--measure-runs",
        type=int,
        default=3,
        help="Measured runs per exact32 CPU/GPU case. Default: 3",
    )
    parser.add_argument(
        "--quiet-libraries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Silence noisy TensorFlow/chars2vec startup logs. Default: true",
    )
    parser.add_argument(
        "--pretty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pretty-print output JSON. Default: true",
    )
    parser.add_argument("--internal-single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device-target", choices=[_DEVICE_GPU, _DEVICE_CPU], default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dataset-shape",
        choices=[_DATASET_AS_IS, _DATASET_UNIQUE_ONLY],
        default=_DATASET_AS_IS,
        help=argparse.SUPPRESS,
    )
    return parser


def _load_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").splitlines()


def _build_dataset_variants(names: list[str]) -> dict[str, list[str]]:
    unique_only = sorted({str(name).lower() for name in names})
    return {
        _DATASET_AS_IS: list(names),
        _DATASET_UNIQUE_ONLY: unique_only,
    }


def _dataset_meta(names: list[str]) -> dict[str, int]:
    return {
        "name_count": int(len(names)),
        "unique_name_count": int(len({str(name).lower() for name in names})),
    }


def _run_lab_case(
    *,
    label: str,
    names: list[str],
    model_name: str,
    batch_size: int | None,
    execution_mode: str,
    quiet_libraries: bool,
) -> dict[str, Any]:
    started_at = perf_counter()
    embeddings, meta = generate_chars2vec_embeddings(
        names=names,
        model_name=model_name,
        batch_size=batch_size,
        execution_mode=execution_mode,
        quiet_libraries=quiet_libraries,
        show_progress=False,
        return_meta=True,
    )
    wall_seconds = float(perf_counter() - started_at)
    return {
        "label": label,
        "execution_mode": execution_mode,
        "requested_batch_size": batch_size,
        "wall_seconds": wall_seconds,
        "embedding_shape": list(embeddings.shape),
        "meta": meta,
    }


def _run_exact32_single_case(
    *,
    names: list[str],
    model_name: str,
    quiet_libraries: bool,
    device_target: str,
    dataset_shape: str,
) -> dict[str, Any]:
    started_at = perf_counter()
    embeddings, meta = generate_chars2vec_embeddings(
        names=names,
        model_name=model_name,
        batch_size=_EXACT32_BATCH_SIZE,
        execution_mode="predict",
        quiet_libraries=quiet_libraries,
        show_progress=False,
        return_meta=True,
    )
    meta = dict(meta)
    meta["device_target"] = str(device_target)
    meta["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    wall_seconds = float(perf_counter() - started_at)
    return {
        "label": f"exact32_{device_target}_{dataset_shape}",
        "matrix": _TRACK_A_MATRIX,
        "device_target": device_target,
        "dataset_shape": dataset_shape,
        "requested_batch_size": _EXACT32_BATCH_SIZE,
        "execution_mode": "predict",
        "wall_seconds": wall_seconds,
        "embedding_shape": list(embeddings.shape),
        "meta": meta,
        "dataset": _dataset_meta(names),
    }


def _device_env(*, device_target: str) -> dict[str, str]:
    env = dict(os.environ)
    if device_target == _DEVICE_CPU:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _single_run_command(
    *,
    script_path: Path,
    names_file: Path,
    model_name: str,
    quiet_libraries: bool,
    device_target: str,
    dataset_shape: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--names-file",
        str(names_file),
        "--model-name",
        str(model_name),
        "--matrix",
        _TRACK_A_MATRIX,
        "--internal-single-run",
        "--device-target",
        str(device_target),
        "--dataset-shape",
        str(dataset_shape),
    ]
    cmd.append("--quiet-libraries" if quiet_libraries else "--no-quiet-libraries")
    return cmd


def _parse_single_run_stdout(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in str(stdout).splitlines() if line.strip()]
    if not lines:
        raise ValueError("Expected JSON output from internal single-run subprocess, got empty stdout.")
    return json.loads(lines[-1])


def _run_exact32_subprocess(
    *,
    script_path: Path,
    names_file: Path,
    model_name: str,
    quiet_libraries: bool,
    device_target: str,
    dataset_shape: str,
) -> dict[str, Any]:
    cmd = _single_run_command(
        script_path=script_path,
        names_file=names_file,
        model_name=model_name,
        quiet_libraries=quiet_libraries,
        device_target=device_target,
        dataset_shape=dataset_shape,
    )
    completed = subprocess.run(
        cmd,
        env=_device_env(device_target=device_target),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Exact32 chars2vec subprocess failed "
            f"(device={device_target}, dataset_shape={dataset_shape}, returncode={completed.returncode}).\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    payload = _parse_single_run_stdout(completed.stdout)
    payload["subprocess_returncode"] = int(completed.returncode)
    return payload


def _median_numeric(values: list[Any]) -> float | int | None:
    normalized = [float(value) for value in values if value is not None]
    if not normalized:
        return None
    median_value = statistics.median(normalized)
    if all(float(value).is_integer() for value in normalized):
        return int(round(median_value))
    return float(median_value)


def _median_runtime_meta(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for run in runs for key in (run.get("meta") or {}).keys()})
    summary: dict[str, Any] = {}
    for key in keys:
        values = [run.get("meta", {}).get(key) for run in runs]
        present = [value for value in values if value is not None]
        if not present:
            continue
        if all(isinstance(value, bool) for value in present):
            if all(value == present[0] for value in present):
                summary[key] = bool(present[0])
            continue
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in present):
            summary[key] = _median_numeric(present)
            continue
        if all(value == present[0] for value in present):
            summary[key] = present[0]
    return summary


def _median_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for field in _RUN_MEDIAN_FIELDS:
        values = [run.get(field) for run in runs]
        summary[field] = _median_numeric(values)
    runtime_meta = _median_runtime_meta(runs)
    for field in _META_MEDIAN_FIELDS:
        summary[field] = runtime_meta.get(field)
    summary["execution_mode"] = runtime_meta.get("execution_mode")
    summary["requested_batch_size"] = runtime_meta.get("requested_batch_size")
    summary["effective_batch_size"] = runtime_meta.get("effective_batch_size")
    summary["predict_batch_count"] = runtime_meta.get("predict_batch_count")
    summary["oom_retry_count"] = runtime_meta.get("oom_retry_count")
    summary["device_target"] = runtime_meta.get("device_target")
    summary["cuda_visible_devices"] = runtime_meta.get("cuda_visible_devices")
    return summary


def _build_case_result(
    *,
    device_target: str,
    dataset_shape: str,
    dataset_meta: dict[str, int],
    warmup_runs: list[dict[str, Any]],
    measured_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    case_id = f"exact32_{device_target}_{dataset_shape}"
    median_runtime_meta = _median_runtime_meta(measured_runs)
    median_runtime_meta.setdefault("device_target", device_target)
    median_runtime_meta.setdefault("cuda_visible_devices", "" if device_target == _DEVICE_CPU else None)
    median = _median_summary(measured_runs)
    median["device_target"] = median_runtime_meta.get("device_target")
    median["cuda_visible_devices"] = median_runtime_meta.get("cuda_visible_devices")
    return {
        "case_id": case_id,
        "label": case_id,
        "device_target": device_target,
        "dataset_shape": dataset_shape,
        "dataset": dataset_meta,
        "warmup_run_count": int(len(warmup_runs)),
        "measure_run_count": int(len(measured_runs)),
        "measured_runs": measured_runs,
        "median": median,
        "median_runtime_meta": median_runtime_meta,
    }


def _decide_reference_device(cases: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {(case["dataset_shape"], case["device_target"]): case for case in cases}
    comparisons: list[dict[str, Any]] = []
    cpu_candidate = True
    for dataset_shape in (_DATASET_AS_IS, _DATASET_UNIQUE_ONLY):
        cpu_case = indexed[(dataset_shape, _DEVICE_CPU)]
        gpu_case = indexed[(dataset_shape, _DEVICE_GPU)]
        cpu_wall = float(cpu_case["median"]["wall_seconds"])
        gpu_wall = float(gpu_case["median"]["wall_seconds"])
        cpu_faster_by_5pct = cpu_wall <= (gpu_wall * 0.95)
        within_five_percent = abs(cpu_wall - gpu_wall) <= (gpu_wall * 0.05)
        comparisons.append(
            {
                "dataset_shape": dataset_shape,
                "cpu_wall_seconds_median": cpu_wall,
                "gpu_wall_seconds_median": gpu_wall,
                "cpu_faster_by_5pct": bool(cpu_faster_by_5pct),
                "within_five_percent": bool(within_five_percent),
            }
        )
        cpu_candidate = cpu_candidate and cpu_faster_by_5pct

    if cpu_candidate:
        return {
            "preferred_reference_device": _DEVICE_CPU,
            "cpu_candidate_for_gate1": True,
            "reason": "cpu_faster_by_5pct_on_both_datasets",
            "comparisons": comparisons,
        }

    return {
        "preferred_reference_device": _DEVICE_GPU,
        "cpu_candidate_for_gate1": False,
        "reason": "cpu_not_consistently_faster_by_5pct",
        "comparisons": comparisons,
    }


def _write_names_file(path: Path, names: list[str]) -> None:
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def _run_lab_matrix(*, args: argparse.Namespace, names: list[str]) -> dict[str, Any]:
    cases = [
        {"label": "predict_32", "execution_mode": "predict", "batch_size": 32},
        {"label": "predict_auto", "execution_mode": "predict", "batch_size": None},
        {"label": "direct_call", "execution_mode": "direct_call", "batch_size": None},
    ]
    results = [
        _run_lab_case(
            label=str(case["label"]),
            names=names,
            model_name=str(args.model_name),
            batch_size=case["batch_size"],
            execution_mode=str(case["execution_mode"]),
            quiet_libraries=bool(args.quiet_libraries),
        )
        for case in cases
    ]
    return {
        "matrix": _LAB_MATRIX,
        "cases": results,
    }


def _run_track_a_matrix(*, args: argparse.Namespace, names_file: Path, names: list[str]) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    dataset_variants = _build_dataset_variants(names)
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="chars2vec_exact32_") as tmpdir:
        tmp_root = Path(tmpdir)
        dataset_files = {
            _DATASET_AS_IS: names_file,
            _DATASET_UNIQUE_ONLY: tmp_root / f"{names_file.stem}__unique_only.txt",
        }
        _write_names_file(dataset_files[_DATASET_UNIQUE_ONLY], dataset_variants[_DATASET_UNIQUE_ONLY])

        for dataset_shape in (_DATASET_AS_IS, _DATASET_UNIQUE_ONLY):
            warmup_runs: list[dict[str, Any]]
            measured_runs: list[dict[str, Any]]
            for device_target in (_DEVICE_GPU, _DEVICE_CPU):
                warmup_runs = [
                    _run_exact32_subprocess(
                        script_path=script_path,
                        names_file=dataset_files[dataset_shape],
                        model_name=str(args.model_name),
                        quiet_libraries=bool(args.quiet_libraries),
                        device_target=device_target,
                        dataset_shape=dataset_shape,
                    )
                    for _ in range(int(args.warmup_runs))
                ]
                measured_runs = [
                    _run_exact32_subprocess(
                        script_path=script_path,
                        names_file=dataset_files[dataset_shape],
                        model_name=str(args.model_name),
                        quiet_libraries=bool(args.quiet_libraries),
                        device_target=device_target,
                        dataset_shape=dataset_shape,
                    )
                    for _ in range(int(args.measure_runs))
                ]
                cases.append(
                    _build_case_result(
                        device_target=device_target,
                        dataset_shape=dataset_shape,
                        dataset_meta=_dataset_meta(dataset_variants[dataset_shape]),
                        warmup_runs=warmup_runs,
                        measured_runs=measured_runs,
                    )
                )

    return {
        "matrix": _TRACK_A_MATRIX,
        "warmup_runs": int(args.warmup_runs),
        "measure_runs": int(args.measure_runs),
        "decision": _decide_reference_device(cases),
        "cases": cases,
    }


def _write_output(*, path: Path, payload: dict[str, Any], pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2 if pretty else None, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _run_internal_single(args: argparse.Namespace) -> int:
    names = _build_dataset_variants(_load_names(Path(args.names_file)))[str(args.dataset_shape)]
    payload = _run_exact32_single_case(
        names=names,
        model_name=str(args.model_name),
        quiet_libraries=bool(args.quiet_libraries),
        device_target=str(args.device_target),
        dataset_shape=str(args.dataset_shape),
    )
    print(json.dumps(payload, ensure_ascii=True))
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    if args.internal_single_run:
        return _run_internal_single(args)

    names_file = Path(args.names_file)
    names = _load_names(names_file)
    if not names:
        raise ValueError("--names-file must contain at least one line.")

    base_payload = {
        "created_at_utc": _utc_now(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "names_file": str(names_file),
        "model_name": str(args.model_name),
        "name_count": int(len(names)),
        "unique_name_count": int(len({str(name).lower() for name in names})),
        "quiet_libraries": bool(args.quiet_libraries),
    }

    if args.matrix == _LAB_MATRIX:
        payload = {**base_payload, **_run_lab_matrix(args=args, names=names)}
    else:
        payload = {**base_payload, **_run_track_a_matrix(args=args, names_file=names_file, names=names)}

    output_path = Path(args.output) if args.output else _default_output_path(REPO_ROOT, matrix=str(args.matrix))
    _write_output(path=output_path, payload=payload, pretty=bool(args.pretty))
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
