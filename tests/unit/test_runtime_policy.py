from __future__ import annotations

from pathlib import Path

from author_name_disambiguation.common import runtime_policy


def test_runtime_policy_cpu_only_host_prefers_cpu_safe_defaults(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        runtime_policy,
        "_probe_torch_host",
        lambda bootstrap_runtime=None: {
            "torch_version": "2.6.0+cpu",
            "torch_cuda_version": None,
            "torch_cuda_available": False,
            "resolved_device": "cpu",
            "gpu_name": None,
            "gpu_total_memory_bytes": None,
            "fallback_reason": "torch_cuda_unavailable",
            "cuda_probe_error": None,
        },
    )
    monkeypatch.setattr(runtime_policy, "detect_cpu_limit", lambda: {"cpu_limit": 8, "cpu_limit_source": "test"})
    monkeypatch.setattr(runtime_policy, "detect_available_ram_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(runtime_policy, "available_disk_bytes", lambda _path: 100 * 1024**3)
    monkeypatch.setattr(runtime_policy, "_probe_onnx_cpu_backend", lambda: (True, None))
    monkeypatch.setattr(runtime_policy, "_probe_cuml_gpu_backend", lambda: (False, "missing"))
    monkeypatch.setattr(runtime_policy, "probe_tensorflow_runtime", lambda force_cpu=False: {"status": "mismatch", "reason": "test"})

    policy = runtime_policy.resolve_infer_runtime_policy(
        requested_device="auto",
        runtime_mode_requested=None,
        specter_runtime_backend_requested=None,
        cluster_backend_requested="auto",
        score_batch_size=8192,
        scratch_dir=tmp_path / "scratch",
    )

    assert policy["resolved_runtime_policy"]["runtime_mode_effective"] == "cpu"
    assert policy["resolved_runtime_policy"]["effective_request_device"] == "cpu"
    assert policy["resolved_runtime_policy"]["specter_runtime_backend_effective"] == "cpu_auto"
    assert policy["resolved_runtime_policy"]["chars2vec_force_cpu"] is True
    assert policy["resolved_runtime_policy"]["chars2vec_batch_size"] == 128
    assert policy["resolved_runtime_policy"]["cluster_backend_effective"] == "sklearn_cpu"
    assert policy["resolved_runtime_policy"]["exact_graph_union_impl"] == "python"
    assert policy["host_profile"]["tensorflow_runtime"]["status"] == "mismatch"
    assert any(item["component"] == "runtime_mode" for item in policy["safety_fallbacks"])
    assert any(item["component"] == "clustering" for item in policy["safety_fallbacks"])


def test_runtime_policy_cuda_host_keeps_gpu_specter_but_cpu_chars(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        runtime_policy,
        "_probe_torch_host",
        lambda bootstrap_runtime=None: {
            "torch_version": "2.6.0+cu124",
            "torch_cuda_version": "12.4",
            "torch_cuda_available": True,
            "resolved_device": "cuda:0",
            "gpu_name": "RTX A6000",
            "gpu_total_memory_bytes": 48 * 1024**3,
            "fallback_reason": None,
            "cuda_probe_error": None,
        },
    )
    monkeypatch.setattr(runtime_policy, "detect_cpu_limit", lambda: {"cpu_limit": 16, "cpu_limit_source": "test"})
    monkeypatch.setattr(runtime_policy, "detect_available_ram_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr(runtime_policy, "available_disk_bytes", lambda _path: 100 * 1024**3)
    monkeypatch.setattr(runtime_policy, "_probe_onnx_cpu_backend", lambda: (True, None))
    monkeypatch.setattr(runtime_policy, "_probe_cuml_gpu_backend", lambda: (True, None))
    monkeypatch.setattr(runtime_policy, "probe_tensorflow_runtime", lambda force_cpu=False: {"status": "ok", "reason": None})

    policy = runtime_policy.resolve_infer_runtime_policy(
        requested_device="auto",
        runtime_mode_requested=None,
        specter_runtime_backend_requested="transformers",
        cluster_backend_requested="auto",
        score_batch_size=8192,
        scratch_dir=tmp_path / "scratch",
    )

    assert policy["resolved_runtime_policy"]["runtime_mode_effective"] == "gpu"
    assert policy["resolved_runtime_policy"]["effective_request_device"] == "cuda"
    assert policy["resolved_runtime_policy"]["specter_runtime_backend_effective"] == "transformers"
    assert policy["resolved_runtime_policy"]["chars2vec_batch_size"] == 128
    assert policy["resolved_runtime_policy"]["cluster_backend_effective"] == "sklearn_cpu"
    assert policy["resolved_runtime_policy"]["numba_auto_enabled"] is False
    assert policy["host_profile"]["torch"]["gpu_total_memory_bytes"] == 48 * 1024**3
    assert any(item["component"] == "clustering" for item in policy["safety_fallbacks"])


def test_runtime_policy_low_ram_reduces_chars_batch_and_clamps_cpu_score_batch(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        runtime_policy,
        "_probe_torch_host",
        lambda bootstrap_runtime=None: {
            "torch_version": "2.6.0+cpu",
            "torch_cuda_version": None,
            "torch_cuda_available": False,
            "resolved_device": "cpu",
            "gpu_name": None,
            "gpu_total_memory_bytes": None,
            "fallback_reason": "torch_cuda_unavailable",
            "cuda_probe_error": None,
        },
    )
    monkeypatch.setattr(runtime_policy, "detect_cpu_limit", lambda: {"cpu_limit": 2, "cpu_limit_source": "test"})
    monkeypatch.setattr(runtime_policy, "detect_available_ram_bytes", lambda: 32 * 1024**2)
    monkeypatch.setattr(runtime_policy, "available_disk_bytes", lambda _path: 100 * 1024**3)
    monkeypatch.setattr(runtime_policy, "_probe_onnx_cpu_backend", lambda: (False, "missing_onnx"))
    monkeypatch.setattr(runtime_policy, "_probe_cuml_gpu_backend", lambda: (False, "missing"))
    monkeypatch.setattr(runtime_policy, "probe_tensorflow_runtime", lambda force_cpu=False: {"status": "cpu_fallback", "reason": "forced_cpu"})

    policy = runtime_policy.resolve_infer_runtime_policy(
        requested_device="auto",
        runtime_mode_requested="cpu",
        specter_runtime_backend_requested=None,
        cluster_backend_requested="auto",
        score_batch_size=8192,
        scratch_dir=tmp_path / "scratch",
    )

    assert policy["resolved_runtime_policy"]["chars2vec_batch_size"] == 32
    assert policy["resolved_runtime_policy"]["score_batch_size_effective"] == 1024
    assert any(item["component"] == "pair_scoring" for item in policy["safety_fallbacks"])
    assert any(item["component"] == "specter" for item in policy["safety_fallbacks"])


def test_runtime_policy_keeps_python_union_even_if_numba_could_exist(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        runtime_policy,
        "_probe_torch_host",
        lambda bootstrap_runtime=None: {
            "torch_version": "2.6.0+cu124",
            "torch_cuda_version": "12.4",
            "torch_cuda_available": True,
            "resolved_device": "cuda:0",
            "gpu_name": "RTX A6000",
            "gpu_total_memory_bytes": 48 * 1024**3,
            "fallback_reason": None,
            "cuda_probe_error": None,
        },
    )
    monkeypatch.setattr(runtime_policy, "detect_cpu_limit", lambda: {"cpu_limit": 8, "cpu_limit_source": "test"})
    monkeypatch.setattr(runtime_policy, "detect_available_ram_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(runtime_policy, "available_disk_bytes", lambda _path: 100 * 1024**3)
    monkeypatch.setattr(runtime_policy, "_probe_onnx_cpu_backend", lambda: (True, None))
    monkeypatch.setattr(runtime_policy, "_probe_cuml_gpu_backend", lambda: (False, "missing"))
    monkeypatch.setattr(runtime_policy, "probe_tensorflow_runtime", lambda force_cpu=False: {"status": "ok", "reason": None})

    policy = runtime_policy.resolve_infer_runtime_policy(
        requested_device="auto",
        runtime_mode_requested=None,
        specter_runtime_backend_requested="transformers",
        cluster_backend_requested="sklearn_cpu",
        score_batch_size=8192,
        scratch_dir=tmp_path / "scratch",
    )

    assert policy["resolved_runtime_policy"]["exact_graph_union_impl"] == "python"
    assert policy["resolved_runtime_policy"]["numba_auto_enabled"] is False
