from __future__ import annotations

import sys

import pytest

from author_name_disambiguation.common import cpu_runtime


def test_normalize_workers_request_accepts_auto_and_ints():
    assert cpu_runtime.normalize_workers_request(None) is None
    assert cpu_runtime.normalize_workers_request("auto") is None
    assert cpu_runtime.normalize_workers_request("  AUTO ") is None
    assert cpu_runtime.normalize_workers_request("4") == 4
    assert cpu_runtime.normalize_workers_request(3) == 3


@pytest.mark.parametrize("value", ["0", 0, -1, "x"])
def test_normalize_workers_request_rejects_invalid(value):
    with pytest.raises(ValueError):
        cpu_runtime.normalize_workers_request(value)


def test_resolve_effective_workers_auto_uses_pairs_and_cpu_limit():
    out = cpu_runtime.resolve_effective_workers(
        total_pairs_est=4_500_000,
        n_blocks=10,
        requested_workers=None,
        cpu_limit=8,
        min_pairs_per_worker=1_000_000,
    )
    assert out["requested"] == "auto"
    assert out["effective"] == 4


def test_sharding_enabled_modes():
    assert cpu_runtime.sharding_enabled(
        sharding_mode="off",
        effective_workers=8,
        total_pairs_est=10_000_000,
        min_pairs_per_worker=1_000_000,
    ) is False
    assert cpu_runtime.sharding_enabled(
        sharding_mode="on",
        effective_workers=2,
        total_pairs_est=10,
        min_pairs_per_worker=1_000_000,
    ) is True
    assert cpu_runtime.sharding_enabled(
        sharding_mode="auto",
        effective_workers=2,
        total_pairs_est=500_000,
        min_pairs_per_worker=1_000_000,
    ) is False


def test_cap_workers_by_ram_applies_limit():
    capped = cpu_runtime.cap_workers_by_ram(workers=8, ram_budget_bytes=3_000, per_worker_bytes=1_000)
    assert capped == 3


def test_detect_available_ram_bytes_knows_the_host_memory():
    # psutil is a declared dependency, so the amount is known on every platform.
    assert cpu_runtime.detect_available_ram_bytes() > 0


def test_detect_available_ram_bytes_is_none_when_memory_is_unknown(monkeypatch):
    def _unsupported(_name):
        raise OSError("unsupported")

    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(cpu_runtime, "_read_text", lambda _path: None)
    monkeypatch.setattr(cpu_runtime.os, "sysconf", _unsupported, raising=False)

    assert cpu_runtime.detect_available_ram_bytes() is None
    assert cpu_runtime.compute_ram_budget_bytes(target_fraction=0.6) is None


def test_compute_ram_budget_bytes_scales_known_memory():
    budget = cpu_runtime.compute_ram_budget_bytes(target_fraction=0.5, available_ram_bytes=8 * 1024**3)

    assert budget == 4 * 1024**3


def test_detect_cpu_limit_prefers_smallest_source(monkeypatch):
    monkeypatch.setattr(cpu_runtime.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(cpu_runtime.os, "sched_getaffinity", lambda _pid: set(range(12)), raising=False)
    monkeypatch.setattr(cpu_runtime, "detect_cgroup_quota_cpus", lambda: (6.0, "mock"))

    out = cpu_runtime.detect_cpu_limit()
    assert out["cpu_limit"] == 6
    assert out["cpu_limit_source"].startswith("cgroup")
