from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any


def _read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _cgroup_v2_quota_cpus() -> tuple[float | None, str | None]:
    text = _read_text("/sys/fs/cgroup/cpu.max")
    if not text:
        return None, None
    parts = text.split()
    if len(parts) < 2:
        return None, None
    quota, period = parts[0].strip(), parts[1].strip()
    if quota == "max":
        return None, "/sys/fs/cgroup/cpu.max"
    try:
        quota_us = float(quota)
        period_us = float(period)
    except Exception:
        return None, "/sys/fs/cgroup/cpu.max"
    if quota_us <= 0 or period_us <= 0:
        return None, "/sys/fs/cgroup/cpu.max"
    return float(quota_us / period_us), "/sys/fs/cgroup/cpu.max"


def _cgroup_v1_quota_cpus() -> tuple[float | None, str | None]:
    quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
    quota_text = _read_text(quota_path)
    period_text = _read_text(period_path)
    if quota_text is None or period_text is None:
        return None, None
    try:
        quota_us = float(quota_text)
        period_us = float(period_text)
    except Exception:
        return None, f"{quota_path},{period_path}"
    if quota_us <= 0 or period_us <= 0:
        return None, f"{quota_path},{period_path}"
    return float(quota_us / period_us), f"{quota_path},{period_path}"


def detect_cgroup_quota_cpus() -> tuple[float | None, str | None]:
    v2_quota, v2_src = _cgroup_v2_quota_cpus()
    if v2_src is not None:
        return v2_quota, v2_src
    return _cgroup_v1_quota_cpus()


def detect_cpu_limit() -> dict[str, Any]:
    os_count = int(os.cpu_count() or 1)
    affinity_count: int | None = None
    try:
        affinity_count = int(len(os.sched_getaffinity(0)))
    except Exception:
        affinity_count = None

    cgroup_quota_cpus, cgroup_source = detect_cgroup_quota_cpus()

    candidates: list[tuple[int, str]] = [(max(1, os_count), "os.cpu_count")]
    if affinity_count is not None and affinity_count > 0:
        candidates.append((int(affinity_count), "sched_getaffinity"))
    if cgroup_quota_cpus is not None and cgroup_quota_cpus > 0:
        quota_limit = int(max(1, math.floor(float(cgroup_quota_cpus))))
        candidates.append((quota_limit, f"cgroup:{cgroup_source or 'unknown'}"))

    cpu_limit, source = min(candidates, key=lambda x: x[0])
    return {
        "os_cpu_count": int(os_count),
        "affinity_cpu_count": None if affinity_count is None else int(affinity_count),
        "cgroup_quota_cpus": None if cgroup_quota_cpus is None else float(cgroup_quota_cpus),
        "cgroup_source": cgroup_source,
        "cpu_limit": int(max(1, cpu_limit)),
        "cpu_limit_source": source,
    }


def detect_available_ram_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    text = _read_text("/proc/meminfo")
    if text:
        for line in text.splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return int(parts[1]) * 1024
                    except Exception:
                        return None

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except Exception:
        return None


def compute_ram_budget_bytes(
    *,
    target_fraction: float,
    available_ram_bytes: int | None = None,
) -> int | None:
    fraction = float(target_fraction)
    if fraction <= 0:
        return None
    if available_ram_bytes is None:
        available_ram_bytes = detect_available_ram_bytes()
    if available_ram_bytes is None:
        return None
    return int(max(1, available_ram_bytes * fraction))


def normalize_workers_request(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "auto"}:
            return None
        try:
            parsed = int(cleaned)
        except Exception as exc:
            raise ValueError(f"Invalid cpu_workers value={value!r}; expected 'auto' or positive integer.") from exc
    else:
        parsed = int(value)

    if parsed < 1:
        raise ValueError(f"Invalid cpu_workers value={value!r}; expected >=1 or 'auto'.")
    return int(parsed)


def resolve_effective_workers(
    *,
    total_pairs_est: int,
    n_blocks: int,
    requested_workers: int | None,
    cpu_limit: int,
    min_pairs_per_worker: int,
) -> dict[str, Any]:
    total_pairs = int(max(0, total_pairs_est))
    blocks = int(max(0, n_blocks))
    cpu_cap = int(max(1, cpu_limit))
    min_pairs = int(max(1, min_pairs_per_worker))

    if requested_workers is None:
        by_pairs = int(math.floor(total_pairs / float(min_pairs))) if total_pairs > 0 else 1
        by_pairs = max(1, by_pairs)
        by_blocks = max(1, blocks)
        effective = min(cpu_cap, by_pairs, by_blocks)
        requested_repr: int | str = "auto"
    else:
        requested_repr = int(max(1, requested_workers))
        by_blocks = max(1, blocks)
        effective = min(int(requested_repr), cpu_cap, by_blocks)

    return {
        "requested": requested_repr,
        "effective": int(max(1, effective)),
        "cpu_limit": cpu_cap,
        "n_blocks": blocks,
        "total_pairs_est": total_pairs,
        "min_pairs_per_worker": min_pairs,
    }


def sharding_enabled(
    *,
    sharding_mode: str,
    effective_workers: int,
    total_pairs_est: int,
    min_pairs_per_worker: int,
) -> bool:
    mode = str(sharding_mode).strip().lower()
    if mode not in {"auto", "on", "off"}:
        raise ValueError(f"Invalid sharding_mode={sharding_mode!r}; expected one of auto/on/off.")
    workers = int(max(1, effective_workers))
    if mode == "off":
        return False
    if workers <= 1:
        return False
    if mode == "on":
        return True
    return int(total_pairs_est) >= int(max(1, min_pairs_per_worker))


def cap_workers_by_ram(
    *,
    workers: int,
    ram_budget_bytes: int | None,
    per_worker_bytes: int,
) -> int:
    base_workers = int(max(1, workers))
    if ram_budget_bytes is None:
        return base_workers
    if per_worker_bytes <= 0:
        return base_workers
    cap = int(max(1, ram_budget_bytes // per_worker_bytes))
    return int(max(1, min(base_workers, cap)))
