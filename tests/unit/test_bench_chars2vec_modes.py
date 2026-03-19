from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_benchmark_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "benchmarks"
        / "bench_chars2vec_modes.py"
    )
    spec = importlib.util.spec_from_file_location("bench_chars2vec_modes", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load benchmark module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_dataset_variants_derives_normalized_unique_only():
    module = _load_benchmark_module()

    variants = module._build_dataset_variants(["Beta", "beta", "Alpha", "ALPHA", "Gamma"])

    assert variants["as_is"] == ["Beta", "beta", "Alpha", "ALPHA", "Gamma"]
    assert variants["unique_only"] == ["alpha", "beta", "gamma"]


def test_device_env_hides_gpu_for_cpu_target():
    module = _load_benchmark_module()

    env = module._device_env(device_target="cpu")

    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_decide_reference_device_prefers_cpu_only_if_faster_on_both():
    module = _load_benchmark_module()

    decision = module._decide_reference_device(
        [
            {"dataset_shape": "as_is", "device_target": "gpu", "median": {"wall_seconds": 10.0}},
            {"dataset_shape": "as_is", "device_target": "cpu", "median": {"wall_seconds": 9.4}},
            {"dataset_shape": "unique_only", "device_target": "gpu", "median": {"wall_seconds": 8.0}},
            {"dataset_shape": "unique_only", "device_target": "cpu", "median": {"wall_seconds": 7.5}},
        ]
    )

    assert decision["preferred_reference_device"] == "cpu"
    assert decision["cpu_candidate_for_gate1"] is True
    assert decision["reason"] == "cpu_faster_by_5pct_on_both_datasets"


def test_decide_reference_device_prefers_gpu_when_cpu_not_consistently_faster():
    module = _load_benchmark_module()

    decision = module._decide_reference_device(
        [
            {"dataset_shape": "as_is", "device_target": "gpu", "median": {"wall_seconds": 10.0}},
            {"dataset_shape": "as_is", "device_target": "cpu", "median": {"wall_seconds": 9.4}},
            {"dataset_shape": "unique_only", "device_target": "gpu", "median": {"wall_seconds": 8.0}},
            {"dataset_shape": "unique_only", "device_target": "cpu", "median": {"wall_seconds": 7.9}},
        ]
    )

    assert decision["preferred_reference_device"] == "gpu"
    assert decision["cpu_candidate_for_gate1"] is False
    assert decision["reason"] == "cpu_not_consistently_faster_by_5pct"


def test_run_exact32_single_case_uses_historical_predict32(monkeypatch: pytest.MonkeyPatch):
    module = _load_benchmark_module()
    observed: dict[str, object] = {}

    def _fake_generate_chars2vec_embeddings(**kwargs):
        observed.update(kwargs)
        return np.zeros((1, 50), dtype=np.float32), {
            "execution_mode": kwargs["execution_mode"],
            "requested_batch_size": kwargs["batch_size"],
            "effective_batch_size": kwargs["batch_size"],
            "predict_batch_count": 1,
            "oom_retry_count": 0,
            "pad_seconds": 0.1,
            "predict_seconds": 0.2,
            "model_load_seconds": 0.3,
        }

    monkeypatch.setattr(module, "generate_chars2vec_embeddings", _fake_generate_chars2vec_embeddings)

    payload = module._run_exact32_single_case(
        names=["Doe J"],
        model_name="eng_50",
        quiet_libraries=True,
        device_target="gpu",
        dataset_shape="as_is",
    )

    assert observed["batch_size"] == 32
    assert observed["execution_mode"] == "predict"
    assert payload["label"] == "exact32_gpu_as_is"
    assert payload["requested_batch_size"] == 32
    assert payload["meta"]["device_target"] == "gpu"


def test_median_summary_reads_nested_runtime_meta():
    module = _load_benchmark_module()

    runs = [
        {
            "wall_seconds": 11.0,
            "meta": {
                "predict_seconds": 7.0,
                "pad_seconds": 1.0,
                "model_load_seconds": 0.5,
                "execution_mode": "predict",
                "requested_batch_size": 32,
                "effective_batch_size": 32,
                "predict_batch_count": 100,
                "oom_retry_count": 0,
                "device_target": "cpu",
                "cuda_visible_devices": "",
            },
        },
        {
            "wall_seconds": 13.0,
            "meta": {
                "predict_seconds": 9.0,
                "pad_seconds": 3.0,
                "model_load_seconds": 0.7,
                "execution_mode": "predict",
                "requested_batch_size": 32,
                "effective_batch_size": 32,
                "predict_batch_count": 100,
                "oom_retry_count": 0,
                "device_target": "cpu",
                "cuda_visible_devices": "",
            },
        },
        {
            "wall_seconds": 12.0,
            "meta": {
                "predict_seconds": 8.0,
                "pad_seconds": 2.0,
                "model_load_seconds": 0.6,
                "execution_mode": "predict",
                "requested_batch_size": 32,
                "effective_batch_size": 32,
                "predict_batch_count": 100,
                "oom_retry_count": 0,
                "device_target": "cpu",
                "cuda_visible_devices": "",
            },
        },
    ]

    summary = module._median_summary(runs)
    runtime_meta = module._median_runtime_meta(runs)

    assert summary["wall_seconds"] == 12.0
    assert summary["predict_seconds"] == 8.0
    assert summary["pad_seconds"] == 2.0
    assert summary["model_load_seconds"] == 0.6
    assert summary["device_target"] == "cpu"
    assert summary["cuda_visible_devices"] == ""
    assert runtime_meta["predict_seconds"] == 8.0
    assert runtime_meta["device_target"] == "cpu"


def test_build_case_result_sets_case_id_and_median_runtime_meta():
    module = _load_benchmark_module()

    measured_runs = [
        {
            "label": "exact32_cpu_as_is",
            "wall_seconds": 12.0,
            "meta": {
                "predict_seconds": 8.0,
                "pad_seconds": 2.0,
                "model_load_seconds": 0.6,
                "execution_mode": "predict",
                "requested_batch_size": 32,
                "effective_batch_size": 32,
                "predict_batch_count": 100,
                "oom_retry_count": 0,
                "device_target": "cpu",
                "cuda_visible_devices": "",
            },
        }
    ]

    case = module._build_case_result(
        device_target="cpu",
        dataset_shape="as_is",
        dataset_meta={"name_count": 1, "unique_name_count": 1},
        warmup_runs=[],
        measured_runs=measured_runs,
    )

    assert case["case_id"] == "exact32_cpu_as_is"
    assert case["median_runtime_meta"]["predict_seconds"] == 8.0
    assert case["median_runtime_meta"]["device_target"] == "cpu"
    assert case["median_runtime_meta"]["cuda_visible_devices"] == ""
    assert case["median"]["requested_batch_size"] == 32
