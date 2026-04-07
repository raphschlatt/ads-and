from __future__ import annotations

import multiprocessing as mp
import re
import unicodedata
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from author_name_disambiguation.common.cli_ui import iter_progress, loop_progress
from author_name_disambiguation.common.cpu_runtime import (
    cap_workers_by_ram,
    detect_cpu_limit,
    resolve_effective_workers,
    sharding_enabled,
)
from author_name_disambiguation.common.io_schema import CLUSTER_REQUIRED_COLUMNS, save_parquet, validate_columns
from author_name_disambiguation.common.numeric_safety import sanitize_precomputed_distance_matrix

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SURNAME_PARTICLES = {
    "da",
    "de",
    "del",
    "della",
    "der",
    "di",
    "du",
    "la",
    "le",
    "van",
    "von",
}
_BLOCK_SIZE_HIST_BUCKETS = [
    ("1", 1, 1),
    ("2", 2, 2),
    ("3-4", 3, 4),
    ("5-8", 5, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65+", 65, None),
]
_LAST_CUML_TIMINGS = {"gpu_transfer_seconds": 0.0, "dbscan_seconds": 0.0}


def _ascii_fold(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text or "")
    return norm.encode("ascii", "ignore").decode("ascii").lower().strip()


def _clean_token(token: str) -> str:
    return _NON_ALNUM.sub("", token.lower())


def _is_initial_token(token: str) -> bool:
    t = _clean_token(token)
    return 0 < len(t) <= 2


def _split_name_parts(author_raw: str) -> tuple[list[str], list[str]]:
    raw = _ascii_fold(author_raw)
    if not raw:
        return [], []

    if "," in raw:
        left, right = raw.split(",", 1)
        surname_tokens = [_clean_token(t) for t in left.split() if _clean_token(t)]
        given_tokens = [_clean_token(t) for t in right.split() if _clean_token(t)]
        return surname_tokens, given_tokens

    parts = [_clean_token(t) for t in raw.split() if _clean_token(t)]
    if len(parts) <= 1:
        return parts, parts

    trailing_initial_count = 0
    for token in reversed(parts):
        if _is_initial_token(token):
            trailing_initial_count += 1
        else:
            break

    if 0 < trailing_initial_count < len(parts):
        surname_tokens = parts[:-trailing_initial_count]
        given_tokens = parts[-trailing_initial_count:]
        return surname_tokens, given_tokens

    # Fallback "Firstname Lastname".
    return [parts[-1]], [parts[0]]


def _extract_name_tokens(author_raw: str) -> tuple[str, str]:
    surname_tokens, given_tokens = _split_name_parts(author_raw)
    given = given_tokens[0] if given_tokens else ""
    if not surname_tokens:
        return given, ""
    if len(surname_tokens) > 1 and surname_tokens[0] in _SURNAME_PARTICLES:
        surname_tokens = surname_tokens[1:]
    surname = surname_tokens[-1] if surname_tokens else ""
    return given, surname


def _given_name_token(author_raw: str) -> str:
    given, _ = _extract_name_tokens(author_raw)
    return given


def _surname_token(author_raw: str) -> str:
    _, surname = _extract_name_tokens(author_raw)
    return surname


def _name_conflict_tokens(ga: str, gb: str, sa: str, sb: str) -> bool:
    if sa and sb and sa != sb and len(sa) > 2 and len(sb) > 2:
        return True

    if not ga or not gb:
        return False

    if ga == gb:
        return False

    if _is_initial_token(ga) or _is_initial_token(gb):
        return ga[0] != gb[0]

    if ga[0] != gb[0]:
        return True

    if ga.startswith(gb) or gb.startswith(ga):
        return False

    return ga != gb


def _name_conflict(a: str, b: str) -> bool:
    ga, sa = _extract_name_tokens(a)
    gb, sb = _extract_name_tokens(b)
    return _name_conflict_tokens(ga, gb, sa, sb)


def _build_distance_matrix(block_mentions: pd.DataFrame, block_scores: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    mention_ids = block_mentions["mention_id"].astype(str).tolist()
    n = len(mention_ids)
    idx = {m: i for i, m in enumerate(mention_ids)}

    dist = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(dist, 0.0)

    id1 = block_scores["mention_id_1"].astype(str).map(idx)
    id2 = block_scores["mention_id_2"].astype(str).map(idx)
    valid = id1.notna() & id2.notna()
    i_arr = id1[valid].astype(np.int64).to_numpy()
    j_arr = id2[valid].astype(np.int64).to_numpy()
    d_arr = block_scores.loc[valid, "distance"].astype(np.float32).to_numpy()
    dist[i_arr, j_arr] = d_arr
    dist[j_arr, i_arr] = d_arr

    return dist, idx


def _apply_constraints(dist: np.ndarray, block_mentions: pd.DataFrame, constraints: Dict) -> np.ndarray:
    if not constraints or not constraints.get("enabled", False):
        return dist

    out = dist.copy()
    max_year_gap = int(constraints.get("max_year_gap", 30))
    enforce_name_conflict = bool(constraints.get("enforce_name_conflict", True))
    constraint_mode = str(constraints.get("constraint_mode", "soft")).lower()
    name_conflict_mode = str(constraints.get("name_conflict_mode", constraint_mode)).lower()
    year_gap_mode = str(constraints.get("year_gap_mode", constraint_mode)).lower()
    name_conflict_min_distance = float(constraints.get("name_conflict_min_distance", 1.0))
    year_gap_min_distance = float(constraints.get("year_gap_min_distance", 1.0))

    authors = block_mentions["author_raw"].fillna("").astype(str).tolist()
    years = block_mentions["year"].tolist()

    n = len(block_mentions)
    if n < 2:
        np.fill_diagonal(out, 0.0)
        return out

    given_tokens = np.empty(n, dtype=object)
    surname_tokens = np.empty(n, dtype=object)
    for idx, author in enumerate(authors):
        given, surname = _extract_name_tokens(author)
        given_tokens[idx] = given
        surname_tokens[idx] = surname

    name_conflict_mask = np.zeros((n, n), dtype=bool)
    if enforce_name_conflict:
        for i in range(n):
            gi = str(given_tokens[i])
            si = str(surname_tokens[i])
            for j in range(i + 1, n):
                if _name_conflict_tokens(gi, str(given_tokens[j]), si, str(surname_tokens[j])):
                    name_conflict_mask[i, j] = True
                    name_conflict_mask[j, i] = True

    years_arr = pd.to_numeric(pd.Series(years), errors="coerce").to_numpy(dtype=np.float64)
    valid_years = ~np.isnan(years_arr)
    year_gap_mask = valid_years[:, None] & valid_years[None, :]
    year_gap_mask &= np.abs(years_arr[:, None] - years_arr[None, :]) > max_year_gap

    upper_i, upper_j = np.triu_indices(n, k=1)
    force_name = name_conflict_mask[upper_i, upper_j]
    force_year = year_gap_mask[upper_i, upper_j]
    force_hard = (force_name & (name_conflict_mode == "hard")) | (force_year & (year_gap_mode == "hard"))

    if bool(force_hard.any()):
        hard_i = upper_i[force_hard]
        hard_j = upper_j[force_hard]
        out[hard_i, hard_j] = 1.0
        out[hard_j, hard_i] = 1.0

    soft_name = force_name & ~force_hard
    if bool(soft_name.any()):
        soft_name_i = upper_i[soft_name]
        soft_name_j = upper_j[soft_name]
        soft_name_values = np.maximum(out[soft_name_i, soft_name_j], name_conflict_min_distance)
        out[soft_name_i, soft_name_j] = soft_name_values
        out[soft_name_j, soft_name_i] = soft_name_values

    soft_year = force_year & ~force_hard
    if bool(soft_year.any()):
        soft_year_i = upper_i[soft_year]
        soft_year_j = upper_j[soft_year]
        soft_year_values = np.maximum(out[soft_year_i, soft_year_j], year_gap_min_distance)
        out[soft_year_i, soft_year_j] = soft_year_values
        out[soft_year_j, soft_year_i] = soft_year_values

    np.fill_diagonal(out, 0.0)
    return out


def resolve_dbscan_eps(cluster_config: Dict[str, Any], cosine_threshold: float | None = None) -> tuple[float, Dict[str, Any]]:
    eps_mode = str(cluster_config.get("eps_mode", "fixed")).lower()
    fixed_eps = float(cluster_config.get("eps", 0.35))
    eps_fallback = float(cluster_config.get("eps_fallback", fixed_eps))
    eps_min = float(cluster_config.get("eps_min", 0.0))
    eps_max = float(cluster_config.get("eps_max", 1.0))
    selected_eps = cluster_config.get("selected_eps")

    if eps_mode == "from_threshold" and cosine_threshold is not None:
        raw_eps = 1.0 - float(cosine_threshold)
        source = "from_threshold"
    elif eps_mode == "val_sweep" and selected_eps is not None:
        raw_eps = float(selected_eps)
        source = "val_sweep_selected"
    elif eps_mode == "val_sweep":
        raw_eps = eps_fallback
        source = "val_sweep_fallback"
    else:
        raw_eps = fixed_eps
        source = "fixed"

    resolved = float(np.clip(raw_eps, eps_min, eps_max))
    meta = {
        "eps_mode": eps_mode,
        "source": source,
        "cosine_threshold": cosine_threshold,
        "raw_eps": float(raw_eps),
        "selected_eps": None if selected_eps is None else float(selected_eps),
        "eps_fallback": float(eps_fallback),
        "eps_sweep_min": cluster_config.get("eps_sweep_min"),
        "eps_sweep_max": cluster_config.get("eps_sweep_max"),
        "eps_sweep_step": cluster_config.get("eps_sweep_step"),
        "eps_min": eps_min,
        "eps_max": eps_max,
        "resolved_eps": resolved,
    }
    return resolved, meta


def _eps_bucket_label(bucket: dict[str, Any]) -> str:
    max_size = bucket.get("max_size")
    suffix = "inf" if max_size is None else str(int(max_size))
    return f"{int(bucket['min_size'])}-{suffix}"


def _normalize_eps_block_policy(cluster_config: Dict[str, Any]) -> dict[str, Any]:
    raw_policy = cluster_config.get("eps_block_policy") or {}
    if not isinstance(raw_policy, dict):
        raise ValueError("eps_block_policy must be an object when provided.")

    enabled = bool(raw_policy.get("enabled", False))
    strategy = str(raw_policy.get("strategy", "size_delta")).strip().lower()
    default_delta = float(raw_policy.get("default_delta", 0.0))
    raw_buckets = raw_policy.get("buckets", [])

    if not enabled:
        return {
            "enabled": False,
            "strategy": "size_delta",
            "default_delta": default_delta,
            "buckets": [],
        }

    if strategy != "size_delta":
        raise ValueError(
            f"Unsupported eps_block_policy.strategy={strategy!r}; expected 'size_delta'."
        )
    if not isinstance(raw_buckets, list) or len(raw_buckets) == 0:
        raise ValueError("eps_block_policy.enabled=true requires a non-empty buckets list.")

    buckets: list[dict[str, Any]] = []
    for idx, raw_bucket in enumerate(raw_buckets):
        if not isinstance(raw_bucket, dict):
            raise ValueError(f"eps_block_policy.buckets[{idx}] must be an object.")
        if "min_size" not in raw_bucket:
            raise ValueError(f"eps_block_policy.buckets[{idx}] missing required key 'min_size'.")
        if "delta" not in raw_bucket:
            raise ValueError(f"eps_block_policy.buckets[{idx}] missing required key 'delta'.")

        min_size = int(raw_bucket["min_size"])
        max_size_raw = raw_bucket.get("max_size")
        max_size = None if max_size_raw is None else int(max_size_raw)
        delta = float(raw_bucket["delta"])

        if min_size < 1:
            raise ValueError(
                f"eps_block_policy.buckets[{idx}].min_size must be >= 1; got {min_size}."
            )
        if max_size is not None and max_size < min_size:
            raise ValueError(
                f"eps_block_policy.buckets[{idx}] invalid range: min_size={min_size}, max_size={max_size}."
            )
        buckets.append({"min_size": min_size, "max_size": max_size, "delta": delta})

    buckets = sorted(
        buckets,
        key=lambda b: (
            int(b["min_size"]),
            float("inf") if b.get("max_size") is None else int(b["max_size"]),
        ),
    )

    prev_max = 0
    open_ended_seen = False
    for idx, bucket in enumerate(buckets):
        min_size = int(bucket["min_size"])
        max_size = bucket.get("max_size")
        if open_ended_seen:
            raise ValueError(
                "eps_block_policy has a bucket after an open-ended bucket (max_size=null), which overlaps."
            )
        if min_size <= prev_max:
            raise ValueError(
                "eps_block_policy buckets overlap or touch ambiguously; "
                f"bucket[{idx}] starts at {min_size} while previous max_size is {prev_max}."
            )
        if max_size is None:
            open_ended_seen = True
        else:
            prev_max = int(max_size)

    return {
        "enabled": True,
        "strategy": "size_delta",
        "default_delta": default_delta,
        "buckets": buckets,
    }


def _resolve_block_eps(
    *,
    block_size: int,
    eps_base: float,
    eps_min: float,
    eps_max: float,
    eps_block_policy: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    delta = float(eps_block_policy.get("default_delta", 0.0))
    bucket_label = "default"

    if bool(eps_block_policy.get("enabled", False)):
        for bucket in eps_block_policy.get("buckets", []):
            min_size = int(bucket["min_size"])
            max_size = bucket.get("max_size")
            if block_size < min_size:
                continue
            if max_size is not None and block_size > int(max_size):
                continue
            delta = float(bucket["delta"])
            bucket_label = _eps_bucket_label(bucket)
            break

    raw_eps = float(eps_base) + float(delta)
    effective_eps = float(np.clip(raw_eps, eps_min, eps_max))
    return effective_eps, {
        "bucket": bucket_label,
        "delta": float(delta),
        "raw_eps": float(raw_eps),
        "effective_eps": float(effective_eps),
    }


def _summarize_block_eps(
    *,
    eps_base: float,
    eps_block_policy: dict[str, Any],
    block_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    bucket_counts: dict[str, int] = {}
    effective_eps_values: list[float] = []
    for row in block_rows:
        label = str(row["bucket"])
        bucket_counts[label] = int(bucket_counts.get(label, 0)) + 1
        effective_eps_values.append(float(row["effective_eps"]))

    if len(effective_eps_values) == 0:
        eff_min = None
        eff_max = None
        eff_mean = None
    else:
        arr = np.asarray(effective_eps_values, dtype=np.float64)
        eff_min = float(arr.min())
        eff_max = float(arr.max())
        eff_mean = float(arr.mean())

    bucket_specs = []
    for bucket in eps_block_policy.get("buckets", []):
        bucket_specs.append(
            {
                "label": _eps_bucket_label(bucket),
                "min_size": int(bucket["min_size"]),
                "max_size": None if bucket.get("max_size") is None else int(bucket["max_size"]),
                "delta": float(bucket["delta"]),
            }
        )

    return {
        "strategy": str(eps_block_policy.get("strategy", "size_delta")),
        "default_delta": float(eps_block_policy.get("default_delta", 0.0)),
        "bucket_specs": bucket_specs,
        "bucket_counts": bucket_counts,
        "effective_eps_min": eff_min,
        "effective_eps_max": eff_max,
        "effective_eps_mean": eff_mean,
        "n_blocks": int(len(block_rows)),
        "eps_base": float(eps_base),
    }


def _resolve_cluster_backend(backend: str, metric: str) -> dict[str, Any]:
    requested = str(backend or "auto").strip().lower()
    if requested not in {"auto", "sklearn_cpu", "cuml_gpu"}:
        raise ValueError(
            f"Invalid cluster backend={backend!r}; expected one of auto/sklearn_cpu/cuml_gpu."
        )

    def _cuml_available() -> tuple[bool, str | None]:
        try:
            import cupy  # noqa: F401
            from cuml.cluster import DBSCAN as _  # noqa: F401

            try:
                device_count = int(cupy.cuda.runtime.getDeviceCount())
            except Exception as exc:
                return False, f"cupy_runtime_error:{exc.__class__.__name__}"
            if device_count < 1:
                return False, "no_cuda_device"
            return True, None
        except Exception as exc:
            return False, f"missing_cuml_or_cupy:{exc.__class__.__name__}"

    metric_clean = str(metric).strip().lower()
    available, reason = _cuml_available()

    if requested == "sklearn_cpu":
        return {
            "requested": requested,
            "effective": "sklearn_cpu",
            "reason": "forced_cpu",
            "cuml_available": bool(available),
            "metric": metric_clean,
        }

    if requested == "cuml_gpu":
        if metric_clean != "precomputed":
            return {
                "requested": requested,
                "effective": "sklearn_cpu",
                "reason": f"unsupported_metric:{metric_clean}",
                "cuml_available": bool(available),
                "metric": metric_clean,
            }
        if available:
            return {
                "requested": requested,
                "effective": "cuml_gpu",
                "reason": "forced_gpu",
                "cuml_available": True,
                "metric": metric_clean,
            }
        return {
            "requested": requested,
            "effective": "sklearn_cpu",
            "reason": reason or "cuml_unavailable",
            "cuml_available": False,
            "metric": metric_clean,
        }

    # auto
    if metric_clean == "precomputed" and available:
        return {
            "requested": requested,
            "effective": "cuml_gpu",
            "reason": "auto_gpu",
            "cuml_available": True,
            "metric": metric_clean,
        }
    return {
        "requested": requested,
        "effective": "sklearn_cpu",
        "reason": reason or f"auto_cpu_metric_{metric_clean}",
        "cuml_available": bool(available),
        "metric": metric_clean,
    }


def _run_dbscan_cuml(dist: np.ndarray, eps: float, min_samples: int, metric: str) -> np.ndarray:
    import cupy as cp  # type: ignore
    from cuml.cluster import DBSCAN as CuMLDBSCAN  # type: ignore

    global _LAST_CUML_TIMINGS
    transfer_started_at = perf_counter()
    x_gpu = cp.asarray(dist)
    transfer_seconds = perf_counter() - transfer_started_at
    model = CuMLDBSCAN(eps=float(eps), min_samples=int(min_samples), metric=str(metric))
    fit_started_at = perf_counter()
    labels_gpu = model.fit_predict(x_gpu)
    labels = cp.asnumpy(labels_gpu)
    fit_seconds = perf_counter() - fit_started_at
    _LAST_CUML_TIMINGS = {
        "gpu_transfer_seconds": float(transfer_seconds),
        "dbscan_seconds": float(fit_seconds),
    }
    return np.asarray(labels, dtype=np.int64)


def _cluster_two_point_block(dist: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    edge_distance = float(dist[0, 1])
    if edge_distance <= float(eps):
        if int(min_samples) <= 2:
            return np.asarray([0, 0], dtype=np.int64)
        return np.asarray([-1, -1], dtype=np.int64)
    if int(min_samples) <= 1:
        return np.asarray([0, 1], dtype=np.int64)
    return np.asarray([-1, -1], dtype=np.int64)


def _build_block_size_histogram(entries: list[dict[str, Any]]) -> dict[str, int]:
    hist = {label: 0 for label, _, _ in _BLOCK_SIZE_HIST_BUCKETS}
    for entry in entries:
        label = _block_size_bucket_label(int(entry["size"]))
        if label is None:
            continue
        hist[label] += 1
    return {label: int(count) for label, count in hist.items() if count > 0}


def _block_size_bucket_label(size: int) -> str | None:
    for label, min_size, max_size in _BLOCK_SIZE_HIST_BUCKETS:
        if size < int(min_size):
            continue
        if max_size is not None and size > int(max_size):
            continue
        return str(label)
    return None


def _resolve_block_backend(
    *,
    requested_backend: str,
    block_size: int,
) -> tuple[str, str | None]:
    del block_size
    return str(requested_backend).strip().lower(), None


def _sanitize_pair_distance_array(distances: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    arr = np.asarray(distances, dtype=np.float32).copy()
    non_finite_mask = ~np.isfinite(arr)
    negative_mask = arr < 0.0
    above_max_mask = arr > 2.0
    if bool(non_finite_mask.any()):
        arr[non_finite_mask] = 1.0
    if bool(negative_mask.any()):
        arr[negative_mask] = 0.0
    if bool(above_max_mask.any()):
        arr[above_max_mask] = 2.0
    return arr, {
        "corrected_blocks": int(bool(non_finite_mask.any() or negative_mask.any() or above_max_mask.any())),
        "non_finite_count": int(non_finite_mask.sum()),
        "negative_count": int(negative_mask.sum()),
        "above_max_count": int(above_max_mask.sum()),
        "asymmetry_pairs": 0,
        "diag_reset_count": 0,
    }


def _prepare_constraint_state(block_mentions: pd.DataFrame) -> dict[str, Any]:
    authors = block_mentions["author_raw"].fillna("").astype(str).tolist()
    years_raw = pd.to_numeric(block_mentions["year"], errors="coerce").to_numpy(dtype=np.float64, copy=False)

    given_tokens: list[str] = []
    surname_tokens: list[str] = []
    first_chars: list[str] = []
    given_initial_flags: list[bool] = []
    surname_long_flags: list[bool] = []
    for author in authors:
        given, surname = _extract_name_tokens(author)
        given = str(given or "")
        surname = str(surname or "")
        given_tokens.append(given)
        surname_tokens.append(surname)
        first_chars.append(given[:1])
        given_initial_flags.append(bool(_is_initial_token(given)))
        surname_long_flags.append(bool(len(surname) > 2))

    return {
        "given": np.asarray(given_tokens, dtype=str),
        "surname": np.asarray(surname_tokens, dtype=str),
        "first_char": np.asarray(first_chars, dtype=str),
        "given_is_initial": np.asarray(given_initial_flags, dtype=bool),
        "surname_long": np.asarray(surname_long_flags, dtype=bool),
        "years": years_raw,
        "years_valid": ~np.isnan(years_raw),
    }


def _apply_constraints_to_pair_distances(
    *,
    distances: np.ndarray,
    idx1: np.ndarray,
    idx2: np.ndarray,
    constraint_state: dict[str, Any],
    constraints: Dict[str, Any],
) -> np.ndarray:
    if not constraints or not constraints.get("enabled", False) or len(distances) == 0:
        return np.asarray(distances, dtype=np.float32, copy=False)

    out = np.asarray(distances, dtype=np.float32).copy()
    max_year_gap = int(constraints.get("max_year_gap", 30))
    enforce_name_conflict = bool(constraints.get("enforce_name_conflict", True))
    constraint_mode = str(constraints.get("constraint_mode", "soft")).lower()
    name_conflict_mode = str(constraints.get("name_conflict_mode", constraint_mode)).lower()
    year_gap_mode = str(constraints.get("year_gap_mode", constraint_mode)).lower()
    name_conflict_min_distance = float(constraints.get("name_conflict_min_distance", 1.0))
    year_gap_min_distance = float(constraints.get("year_gap_min_distance", 1.0))

    force_name = np.zeros(len(out), dtype=bool)
    if enforce_name_conflict:
        given = constraint_state["given"]
        surname = constraint_state["surname"]
        first_char = constraint_state["first_char"]
        given_is_initial = constraint_state["given_is_initial"]
        surname_long = constraint_state["surname_long"]

        ga = given[idx1]
        gb = given[idx2]
        sa = surname[idx1]
        sb = surname[idx2]
        first_diff = first_char[idx1] != first_char[idx2]
        missing_given = (ga == "") | (gb == "")
        same_given = ga == gb
        surname_conflict = surname_long[idx1] & surname_long[idx2] & (sa != "") & (sb != "") & (sa != sb)
        initial_case = given_is_initial[idx1] | given_is_initial[idx2]
        prefix_case = np.char.startswith(ga, gb) | np.char.startswith(gb, ga)
        non_initial_conflict = (~missing_given) & (~same_given) & (~initial_case) & (first_diff | (~prefix_case))
        initial_conflict = (~missing_given) & (~same_given) & initial_case & first_diff
        force_name = surname_conflict | initial_conflict | non_initial_conflict

    years = constraint_state["years"]
    years_valid = constraint_state["years_valid"]
    force_year = (
        years_valid[idx1]
        & years_valid[idx2]
        & (np.abs(years[idx1] - years[idx2]) > float(max_year_gap))
    )

    hard_mask = (force_name & (name_conflict_mode == "hard")) | (force_year & (year_gap_mode == "hard"))
    if bool(hard_mask.any()):
        out[hard_mask] = 1.0

    soft_name_mask = force_name & ~hard_mask
    if bool(soft_name_mask.any()):
        out[soft_name_mask] = np.maximum(out[soft_name_mask], np.float32(name_conflict_min_distance))

    soft_year_mask = force_year & ~hard_mask
    if bool(soft_year_mask.any()):
        out[soft_year_mask] = np.maximum(out[soft_year_mask], np.float32(year_gap_min_distance))

    return out


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(int(n), dtype=np.int64)
        self.rank = np.zeros(int(n), dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        root = int(x)
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[x]) != x:
            next_x = int(parent[x])
            parent[x] = root
            x = next_x
        return root

    def union(self, a: int, b: int) -> None:
        ra = self.find(int(a))
        rb = self.find(int(b))
        if ra == rb:
            return
        rank = self.rank
        parent = self.parent
        if int(rank[ra]) < int(rank[rb]):
            parent[ra] = rb
            return
        if int(rank[ra]) > int(rank[rb]):
            parent[rb] = ra
            return
        parent[rb] = ra
        rank[ra] = np.int8(int(rank[ra]) + 1)


class ExactGraphClusterAccumulator:
    def __init__(
        self,
        *,
        mentions: pd.DataFrame,
        cluster_config: Dict[str, Any],
        backend_requested: str = "connected_components_cpu",
    ):
        self.mentions = mentions
        self.cluster_config = dict(cluster_config or {})
        self.eps = float(self.cluster_config.get("eps", 0.35))
        self.eps_min = float(self.cluster_config.get("eps_min", 0.0))
        self.eps_max = float(self.cluster_config.get("eps_max", 1.0))
        self.constraints = dict(self.cluster_config.get("constraints", {}) or {})
        self.eps_block_policy = _normalize_eps_block_policy(self.cluster_config)
        self.backend_requested = str(backend_requested)
        self.sanitize_totals = {
            "corrected_blocks": 0,
            "non_finite_count": 0,
            "negative_count": 0,
            "above_max_count": 0,
            "asymmetry_pairs": 0,
            "diag_reset_count": 0,
        }
        self.mapping_seconds_total = 0.0
        self.constraint_seconds_total = 0.0
        self.union_seconds_total = 0.0
        self.processed_pair_rows = 0
        self.numeric_pair_index_rows = 0
        self.string_pair_index_rows = 0

        self.block_states: dict[str, dict[str, Any]] = {}
        self.block_states_by_idx: dict[int, dict[str, Any]] = {}
        eps_rows: list[dict[str, Any]] = []
        mentions_with_indices = mentions.reset_index(drop=True).copy()
        mentions_with_indices["__exact_graph_mention_idx"] = np.arange(len(mentions_with_indices), dtype=np.int64)
        block_codes, _ = pd.factorize(mentions_with_indices["block_key"].astype(str), sort=False)
        mentions_with_indices["__exact_graph_block_idx"] = np.asarray(block_codes, dtype=np.int64)
        for block_key, raw_block_mentions in mentions_with_indices.groupby("block_key", sort=False):
            global_mention_indices = raw_block_mentions["__exact_graph_mention_idx"].to_numpy(dtype=np.int64, copy=False)
            block_idx = int(raw_block_mentions["__exact_graph_block_idx"].iloc[0]) if len(raw_block_mentions) else 0
            block_mentions = raw_block_mentions.drop(
                columns=["__exact_graph_mention_idx", "__exact_graph_block_idx"],
                errors="ignore",
            ).reset_index(drop=True)
            n = int(len(block_mentions))
            effective_eps, eps_row = _resolve_block_eps(
                block_size=n,
                eps_base=float(self.eps),
                eps_min=float(self.eps_min),
                eps_max=float(self.eps_max),
                eps_block_policy=self.eps_block_policy,
            )
            eps_rows.append(eps_row)
            mention_ids = block_mentions["mention_id"].astype(str).tolist()
            state = {
                "block_key": str(block_key),
                "block_idx": int(block_idx),
                "mentions": block_mentions,
                "mention_to_local": {mention_id: idx for idx, mention_id in enumerate(mention_ids)},
                "mention_global_to_local": {
                    int(global_idx): idx for idx, global_idx in enumerate(global_mention_indices.tolist())
                },
                "mention_ids": mention_ids,
                "uf": _UnionFind(n),
                "size": n,
                "pair_est": int(n * (n - 1) // 2),
                "effective_eps": float(effective_eps),
                "eps_bucket": str(eps_row["bucket"]),
                "constraint_state": _prepare_constraint_state(block_mentions),
            }
            self.block_states[str(block_key)] = state
            self.block_states_by_idx[int(block_idx)] = state
        self.eps_block_policy_summary = _summarize_block_eps(
            eps_base=float(self.eps),
            eps_block_policy=self.eps_block_policy,
            block_rows=eps_rows,
        )

    def consume_score_columns(self, score_columns: dict[str, np.ndarray]) -> None:
        block_keys = np.asarray(score_columns.get("block_key", []), dtype=object)
        if len(block_keys) == 0:
            return
        mention_id_1 = np.asarray(score_columns["mention_id_1"], dtype=object)
        mention_id_2 = np.asarray(score_columns["mention_id_2"], dtype=object)
        mention_idx_1 = (
            np.asarray(score_columns["mention_idx_1"], dtype=np.int64)
            if "mention_idx_1" in score_columns
            else None
        )
        mention_idx_2 = (
            np.asarray(score_columns["mention_idx_2"], dtype=np.int64)
            if "mention_idx_2" in score_columns
            else None
        )
        block_idx = (
            np.asarray(score_columns["block_idx"], dtype=np.int64)
            if "block_idx" in score_columns
            else None
        )
        distances, sanitize_meta = _sanitize_pair_distance_array(score_columns["distance"])
        for key, value in sanitize_meta.items():
            self.sanitize_totals[key] = int(self.sanitize_totals.get(key, 0)) + int(value)

        self.processed_pair_rows += int(len(block_keys))
        group_source = block_keys if block_idx is None else block_idx
        unique_block_keys, starts = np.unique(group_source, return_index=True)
        order = np.argsort(starts)
        block_order = unique_block_keys[order]
        block_starts = starts[order]
        block_ends = list(block_starts[1:]) + [len(block_keys)]

        for raw_block_key, start, end in zip(block_order.tolist(), block_starts.tolist(), block_ends):
            if block_idx is None:
                state = self.block_states.get(str(raw_block_key))
            else:
                state = self.block_states_by_idx.get(int(raw_block_key))
            if state is None:
                continue
            map_started_at = perf_counter()
            if mention_idx_1 is not None and mention_idx_2 is not None:
                local_index = state["mention_global_to_local"]
                idx1 = np.fromiter(
                    (int(local_index.get(int(value), -1)) for value in mention_idx_1[start:end]),
                    dtype=np.int64,
                    count=int(end - start),
                )
                idx2 = np.fromiter(
                    (int(local_index.get(int(value), -1)) for value in mention_idx_2[start:end]),
                    dtype=np.int64,
                    count=int(end - start),
                )
                self.numeric_pair_index_rows += int(end - start)
            else:
                local_index = state["mention_to_local"]
                idx1 = np.fromiter(
                    (int(local_index.get(str(value), -1)) for value in mention_id_1[start:end]),
                    dtype=np.int64,
                    count=int(end - start),
                )
                idx2 = np.fromiter(
                    (int(local_index.get(str(value), -1)) for value in mention_id_2[start:end]),
                    dtype=np.int64,
                    count=int(end - start),
                )
                self.string_pair_index_rows += int(end - start)
            self.mapping_seconds_total += float(perf_counter() - map_started_at)

            valid_mask = (idx1 >= 0) & (idx2 >= 0)
            if not bool(valid_mask.any()):
                continue
            idx1_valid = idx1[valid_mask]
            idx2_valid = idx2[valid_mask]
            distance_block = distances[start:end][valid_mask]

            constraint_started_at = perf_counter()
            effective = _apply_constraints_to_pair_distances(
                distances=distance_block,
                idx1=idx1_valid,
                idx2=idx2_valid,
                constraint_state=state["constraint_state"],
                constraints=self.constraints,
            )
            self.constraint_seconds_total += float(perf_counter() - constraint_started_at)

            union_started_at = perf_counter()
            edge_mask = effective <= float(state["effective_eps"])
            if bool(edge_mask.any()):
                uf = state["uf"]
                for i, j in zip(idx1_valid[edge_mask].tolist(), idx2_valid[edge_mask].tolist()):
                    uf.union(int(i), int(j))
            self.union_seconds_total += float(perf_counter() - union_started_at)

    def finalize(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        rows: list[dict[str, str]] = []
        block_count_by_bucket: dict[str, int] = {}
        for state in self.block_states.values():
            block_count_by_bucket[str(_block_size_bucket_label(int(state["size"])) or "unknown")] = (
                int(block_count_by_bucket.get(str(_block_size_bucket_label(int(state["size"])) or "unknown"), 0)) + 1
            )
            root_to_label: dict[int, int] = {}
            next_label = 0
            uf = state["uf"]
            for local_idx, mention_id in enumerate(state["mention_ids"]):
                root = uf.find(int(local_idx))
                if root not in root_to_label:
                    root_to_label[root] = int(next_label)
                    next_label += 1
                rows.append(
                    {
                        "mention_id": str(mention_id),
                        "block_key": str(state["block_key"]),
                        "author_uid": f"{state['block_key']}::{int(root_to_label[root])}",
                    }
                )

        out = pd.DataFrame(rows, columns=CLUSTER_REQUIRED_COLUMNS)
        validate_columns(out, CLUSTER_REQUIRED_COLUMNS, "clusters")

        block_sizes = np.asarray([int(state["size"]) for state in self.block_states.values()], dtype=np.int64)
        meta = {
            "cluster_backend_requested": self.backend_requested,
            "cluster_backend_effective": "connected_components_cpu",
            "cluster_backend_reason": "exact_min_samples_1_graph",
            "cluster_strategy": "connected_components_exact",
            "eps_base": float(self.eps),
            "eps_block_policy_enabled": bool(self.eps_block_policy.get("enabled", False)),
            "eps_block_policy_summary": self.eps_block_policy_summary,
            "backend_block_counts": {"connected_components_cpu": int(len(self.block_states))},
            "cpu_sharding_mode": "off",
            "cpu_sharding_enabled": False,
            "cpu_workers_requested": 1,
            "cpu_workers_effective": 1,
            "cpu_limit_detected": 1,
            "cpu_limit_source": "graph_exact_single_process",
            "cpu_min_pairs_per_worker": 0,
            "ram_budget_bytes": None,
            "total_pairs_est": int(sum(int(state["pair_est"]) for state in self.block_states.values())),
            "block_p95": float(np.percentile(block_sizes, 95)) if len(block_sizes) else 0.0,
            "block_size_histogram": _build_block_size_histogram(list(self.block_states.values())),
            "block_count_by_bucket": {str(key): int(value) for key, value in sorted(block_count_by_bucket.items())},
            "total_seconds_by_bucket": {},
            "dbscan_seconds_by_bucket": {},
            "build_entries_seconds": 0.0,
            "distance_matrix_seconds_total": 0.0,
            "constraints_seconds_total": float(self.constraint_seconds_total),
            "constraint_apply_seconds_total": float(self.constraint_seconds_total),
            "sanitize_seconds_total": 0.0,
            "dbscan_seconds_total": 0.0,
            "connected_components_seconds_total": float(self.union_seconds_total),
            "gpu_transfer_seconds_total": 0.0,
            "top_slow_blocks": [],
            "sanitize_totals": dict(self.sanitize_totals),
            "processed_pair_rows": int(self.processed_pair_rows),
            "pair_index_seconds_total": float(self.mapping_seconds_total),
            "mapping_seconds_total": float(self.mapping_seconds_total),
            "numeric_pair_index_rows": int(self.numeric_pair_index_rows),
            "string_pair_index_rows": int(self.string_pair_index_rows),
        }
        return out, meta


def _cluster_single_block(
    *,
    block_key: str,
    block_mentions: pd.DataFrame,
    block_scores: pd.DataFrame,
    eps: float,
    min_samples: int,
    metric: str,
    constraints: Dict[str, Any],
    backend: str,
) -> tuple[list[dict[str, str]], dict[str, int], dict[str, Any]]:
    block_backend, backend_reason = _resolve_block_backend(
        requested_backend=backend,
        block_size=int(len(block_mentions)),
    )
    sanitize_totals = {
        "corrected_blocks": 0,
        "non_finite_count": 0,
        "negative_count": 0,
        "above_max_count": 0,
        "asymmetry_pairs": 0,
        "diag_reset_count": 0,
    }
    timing = {
        "block_key": str(block_key),
        "block_size": int(len(block_mentions)),
        "pair_rows": int(len(block_scores)),
        "distance_matrix_seconds": 0.0,
        "constraints_seconds": 0.0,
        "sanitize_seconds": 0.0,
        "dbscan_seconds": 0.0,
        "gpu_transfer_seconds": 0.0,
        "total_seconds": 0.0,
        "backend_requested": str(backend),
        "backend": str(block_backend),
        "backend_reason": backend_reason,
    }
    block_started_at = perf_counter()

    block_mentions = block_mentions.reset_index(drop=True)
    if len(block_mentions) == 1:
        m = str(block_mentions.iloc[0]["mention_id"])
        rows = [{"mention_id": m, "block_key": str(block_key), "author_uid": f"{block_key}::0"}]
        timing["total_seconds"] = float(perf_counter() - block_started_at)
        return rows, sanitize_totals, timing

    distance_started_at = perf_counter()
    dist, _ = _build_distance_matrix(block_mentions, block_scores)
    timing["distance_matrix_seconds"] = float(perf_counter() - distance_started_at)
    constraints_started_at = perf_counter()
    dist = _apply_constraints(dist, block_mentions, constraints)
    timing["constraints_seconds"] = float(perf_counter() - constraints_started_at)
    sanitize_started_at = perf_counter()
    dist, sanitize_meta = sanitize_precomputed_distance_matrix(dist)
    timing["sanitize_seconds"] = float(perf_counter() - sanitize_started_at)
    if sanitize_meta["corrected"]:
        sanitize_totals["corrected_blocks"] += 1
    sanitize_totals["non_finite_count"] += int(sanitize_meta["non_finite_count"])
    sanitize_totals["negative_count"] += int(sanitize_meta["negative_count"])
    sanitize_totals["above_max_count"] += int(sanitize_meta["above_max_count"])
    sanitize_totals["asymmetry_pairs"] += int(sanitize_meta["asymmetry_pairs"])
    sanitize_totals["diag_reset_count"] += int(sanitize_meta["diag_reset_count"])

    dbscan_started_at = perf_counter()
    if len(block_mentions) == 2:
        labels = _cluster_two_point_block(dist=dist, eps=eps, min_samples=min_samples)
        timing["dbscan_seconds"] = float(perf_counter() - dbscan_started_at)
    elif str(block_backend) == "cuml_gpu":
        global _LAST_CUML_TIMINGS
        _LAST_CUML_TIMINGS = {"gpu_transfer_seconds": 0.0, "dbscan_seconds": 0.0}
        labels = _run_dbscan_cuml(dist=dist, eps=eps, min_samples=min_samples, metric=metric)
        timing["gpu_transfer_seconds"] = float(_LAST_CUML_TIMINGS.get("gpu_transfer_seconds", 0.0))
        timing["dbscan_seconds"] = float(_LAST_CUML_TIMINGS.get("dbscan_seconds", 0.0))
    else:
        model = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric=str(metric))
        labels = model.fit_predict(dist)
        timing["dbscan_seconds"] = float(perf_counter() - dbscan_started_at)

    rows: list[dict[str, str]] = []
    for mention_id, label in zip(block_mentions["mention_id"].astype(str).tolist(), np.asarray(labels).tolist()):
        uid = f"{block_key}::{int(label)}"
        rows.append({"mention_id": mention_id, "block_key": str(block_key), "author_uid": uid})

    timing["total_seconds"] = float(perf_counter() - block_started_at)
    return rows, sanitize_totals, timing


def _cluster_worker(payload: dict[str, Any]) -> dict[str, Any]:
    rows, sanitize, timing = _cluster_single_block(
        block_key=payload["block_key"],
        block_mentions=payload["block_mentions"],
        block_scores=payload["block_scores"],
        eps=payload["eps"],
        min_samples=payload["min_samples"],
        metric=payload["metric"],
        constraints=payload["constraints"],
        backend="sklearn_cpu",
    )
    return {
        "rows": rows,
        "sanitize": sanitize,
        "timing": timing,
        "block_key": payload["block_key"],
        "matrix_bytes": int(payload["matrix_bytes"]),
    }


def _build_block_entries(
    mentions: pd.DataFrame,
    pair_scores: pd.DataFrame,
    *,
    eps_base: float,
    eps_min: float,
    eps_max: float,
    eps_block_policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = mentions.groupby("block_key", sort=False)
    if len(pair_scores) > 0 and "block_key" in pair_scores.columns:
        score_indices = pair_scores.groupby("block_key", sort=False).indices
    else:
        score_indices = {}

    entries: list[dict[str, Any]] = []
    eps_rows: list[dict[str, Any]] = []
    for block_key, block_mentions in grouped:
        block_mentions = block_mentions.reset_index(drop=True)
        block_idx = score_indices.get(block_key)
        if block_idx is None:
            block_scores = pair_scores.iloc[0:0]
        else:
            block_scores = pair_scores.iloc[np.asarray(block_idx, dtype=np.int64)]

        n = int(len(block_mentions))
        pair_est = int(n * (n - 1) // 2)
        effective_eps, eps_row = _resolve_block_eps(
            block_size=n,
            eps_base=float(eps_base),
            eps_min=float(eps_min),
            eps_max=float(eps_max),
            eps_block_policy=eps_block_policy,
        )
        eps_rows.append(eps_row)
        entries.append(
            {
                "block_key": str(block_key),
                "mentions": block_mentions,
                "scores": block_scores,
                "size": n,
                "pair_est": pair_est,
                "matrix_bytes": int(max(1, n * n) * 4),
                "effective_eps": float(effective_eps),
                "eps_bucket": str(eps_row["bucket"]),
            }
        )
    summary = _summarize_block_eps(
        eps_base=float(eps_base),
        eps_block_policy=eps_block_policy,
        block_rows=eps_rows,
    )
    return entries, summary


def _aggregate_sanitize_totals(rows: list[dict[str, int]]) -> dict[str, int]:
    out = {
        "corrected_blocks": 0,
        "non_finite_count": 0,
        "negative_count": 0,
        "above_max_count": 0,
        "asymmetry_pairs": 0,
        "diag_reset_count": 0,
    }
    for row in rows:
        out["corrected_blocks"] += int(row.get("corrected_blocks", 0))
        out["non_finite_count"] += int(row.get("non_finite_count", 0))
        out["negative_count"] += int(row.get("negative_count", 0))
        out["above_max_count"] += int(row.get("above_max_count", 0))
        out["asymmetry_pairs"] += int(row.get("asymmetry_pairs", 0))
        out["diag_reset_count"] += int(row.get("diag_reset_count", 0))
    return out


def _cluster_entries_sequential(
    *,
    entries: list[dict[str, Any]],
    eps: float,
    min_samples: int,
    metric: str,
    constraints: Dict[str, Any],
    backend: str,
    show_progress: bool,
) -> tuple[list[dict[str, str]], dict[str, int], list[dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    sanitize_rows: list[dict[str, int]] = []
    timing_rows: list[dict[str, Any]] = []

    iterator = iter_progress(
        entries,
        total=len(entries),
        label="Cluster blocks",
        enabled=show_progress,
        unit="block",
        compact_label="Clustering",
    )

    for entry in iterator:
        block_rows, sanitize, timing = _cluster_single_block(
            block_key=entry["block_key"],
            block_mentions=entry["mentions"],
            block_scores=entry["scores"],
            eps=float(entry.get("effective_eps", eps)),
            min_samples=min_samples,
            metric=metric,
            constraints=constraints,
            backend=backend,
        )
        rows.extend(block_rows)
        sanitize_rows.append(sanitize)
        timing_rows.append(timing)

    return rows, _aggregate_sanitize_totals(sanitize_rows), timing_rows


def cluster_blockwise_dbscan(
    mentions: pd.DataFrame,
    pair_scores: pd.DataFrame | str | Path,
    cluster_config: Dict,
    output_path: str | Path | None = None,
    show_progress: bool = False,
    num_workers: int | None = None,
    sharding_mode: str = "auto",
    min_pairs_per_worker: int = 1_000_000,
    ram_budget_bytes: int | None = None,
    backend: str = "auto",
    return_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, Dict[str, Any]]:
    eps = float(cluster_config.get("eps", 0.35))
    eps_min = float(cluster_config.get("eps_min", 0.0))
    eps_max = float(cluster_config.get("eps_max", 1.0))
    min_samples = int(cluster_config.get("min_samples", 1))
    metric = str(cluster_config.get("metric", "precomputed"))
    constraints = cluster_config.get("constraints", {})
    eps_block_policy = _normalize_eps_block_policy(cluster_config)

    if isinstance(pair_scores, (str, Path)) and metric == "precomputed" and int(min_samples) == 1:
        accumulator = ExactGraphClusterAccumulator(
            mentions=mentions,
            cluster_config=cluster_config,
            backend_requested=str(backend),
        )
        pair_path = Path(pair_scores)
        if not pair_path.exists():
            raise FileNotFoundError(pair_path)
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:
            raise RuntimeError("Exact graph clustering from parquet inputs requires pyarrow.") from exc

        iterator = pq.ParquetFile(pair_path).iter_batches(batch_size=200_000)
        for batch in iterator:
            accumulator.consume_score_columns(
                {
                    "mention_id_1": np.asarray(batch.column(batch.schema.get_field_index("mention_id_1")).to_pylist(), dtype=object),
                    "mention_id_2": np.asarray(batch.column(batch.schema.get_field_index("mention_id_2")).to_pylist(), dtype=object),
                    "block_key": np.asarray(batch.column(batch.schema.get_field_index("block_key")).to_pylist(), dtype=object),
                    "distance": np.asarray(batch.column(batch.schema.get_field_index("distance")).to_pylist(), dtype=np.float32),
                }
            )
        out, meta = accumulator.finalize()
        if output_path is not None:
            save_parquet(out, output_path, index=False)
        out.attrs["cluster_meta"] = meta
        if return_meta:
            return out, meta
        return out
    if isinstance(pair_scores, (str, Path)):
        pair_scores = pd.read_parquet(pair_scores)

    build_entries_started_at = perf_counter()
    entries, eps_block_policy_summary = _build_block_entries(
        mentions=mentions,
        pair_scores=pair_scores,
        eps_base=eps,
        eps_min=eps_min,
        eps_max=eps_max,
        eps_block_policy=eps_block_policy,
    )
    build_entries_seconds = perf_counter() - build_entries_started_at
    total_pairs_est = int(sum(int(e["pair_est"]) for e in entries))
    n_blocks = int(len(entries))
    block_sizes = np.asarray([int(e["size"]) for e in entries], dtype=np.int64)
    block_p95 = float(np.percentile(block_sizes, 95)) if len(block_sizes) else 0.0

    backend_info = _resolve_cluster_backend(backend=backend, metric=metric)
    backend_requested = str(backend_info["requested"])
    backend_effective = str(backend_info["effective"])
    if backend_requested == "auto" and backend_effective == "cuml_gpu":
        if total_pairs_est < 1_000_000 or block_p95 <= 4.0:
            backend_effective = "sklearn_cpu"
            backend_info = dict(backend_info)
            backend_info["effective"] = "sklearn_cpu"
            backend_info["reason"] = "auto_small_workload_cpu"

    if backend_requested == "cuml_gpu" and backend_effective != "cuml_gpu":
        warnings.warn(
            (
                "Requested cluster backend 'cuml_gpu' is unavailable; "
                f"falling back to 'sklearn_cpu' ({backend_info.get('reason')})."
            ),
            RuntimeWarning,
        )

    cpu_info = detect_cpu_limit()
    worker_info = resolve_effective_workers(
        total_pairs_est=total_pairs_est,
        n_blocks=n_blocks,
        requested_workers=num_workers,
        cpu_limit=int(cpu_info["cpu_limit"]),
        min_pairs_per_worker=int(min_pairs_per_worker),
    )
    workers_effective = int(worker_info["effective"])

    largest_matrix_bytes = int(max((int(e["matrix_bytes"]) for e in entries), default=0))
    workers_effective = cap_workers_by_ram(
        workers=workers_effective,
        ram_budget_bytes=ram_budget_bytes,
        per_worker_bytes=max(1, largest_matrix_bytes),
    )

    sharding_on = sharding_enabled(
        sharding_mode=sharding_mode,
        effective_workers=workers_effective,
        total_pairs_est=total_pairs_est,
        min_pairs_per_worker=int(min_pairs_per_worker),
    )

    rows: list[dict[str, str]]
    sanitize_totals: dict[str, int]
    timing_rows: list[dict[str, Any]] = []

    # GPU backend runs in-process (single-device path).
    if backend_effective == "cuml_gpu":
        try:
            rows, sanitize_totals, timing_rows = _cluster_entries_sequential(
                entries=entries,
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                constraints=constraints,
                backend="cuml_gpu",
                show_progress=show_progress,
            )
        except Exception as exc:
            warnings.warn(
                (
                    "GPU clustering backend failed; falling back to sklearn CPU backend. "
                    f"reason={exc.__class__.__name__}: {exc}"
                ),
                RuntimeWarning,
            )
            backend_effective = "sklearn_cpu"
            rows, sanitize_totals, timing_rows = _cluster_entries_sequential(
                entries=entries,
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                constraints=constraints,
                backend="sklearn_cpu",
                show_progress=show_progress,
            )
    else:
        if (not sharding_on) or workers_effective <= 1 or len(entries) <= 1:
            rows, sanitize_totals, timing_rows = _cluster_entries_sequential(
                entries=entries,
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                constraints=constraints,
                backend="sklearn_cpu",
                show_progress=show_progress,
            )
        else:
            oversize_entries: list[dict[str, Any]] = []
            parallel_entries: list[dict[str, Any]] = list(entries)
            if ram_budget_bytes is not None and ram_budget_bytes > 0:
                for entry in entries:
                    if int(entry["matrix_bytes"]) > int(ram_budget_bytes):
                        oversize_entries.append(entry)
                if oversize_entries:
                    oversize_keys = [str(e["block_key"]) for e in oversize_entries[:5]]
                    if len(oversize_entries) > 5:
                        oversize_keys.append("...")
                    warnings.warn(
                        (
                            "Clustering detected oversized blocks beyond RAM budget; "
                            f"processing {len(oversize_entries)} block(s) sequentially. "
                            f"examples={oversize_keys}"
                        ),
                        RuntimeWarning,
                    )
                    oversize_set = {str(e["block_key"]) for e in oversize_entries}
                    parallel_entries = [e for e in entries if str(e["block_key"]) not in oversize_set]

            rows = []
            sanitize_rows: list[dict[str, int]] = []
            timing_rows = []

            with loop_progress(
                total=len(entries),
                label="Cluster blocks",
                enabled=show_progress,
                unit="block",
                compact_label="Clustering",
            ) as cluster_progress:
                if oversize_entries:
                    seq_rows, seq_sanitize, seq_timing_rows = _cluster_entries_sequential(
                        entries=oversize_entries,
                        eps=eps,
                        min_samples=min_samples,
                        metric=metric,
                        constraints=constraints,
                        backend="sklearn_cpu",
                        show_progress=False,
                    )
                    rows.extend(seq_rows)
                    sanitize_rows.append(seq_sanitize)
                    timing_rows.extend(seq_timing_rows)
                    cluster_progress.update(len(seq_timing_rows))

                if parallel_entries:
                    # Largest matrices first to improve tail latency and make memory pressure explicit.
                    pending = sorted(parallel_entries, key=lambda e: (-int(e["matrix_bytes"]), str(e["block_key"])))

                    inflight: dict[Any, int] = {}
                    ctx = mp.get_context("spawn")
                    with ProcessPoolExecutor(
                        max_workers=int(min(workers_effective, len(parallel_entries))),
                        mp_context=ctx,
                    ) as pool:
                        while pending or inflight:
                            submitted = False
                            while pending and len(inflight) < int(workers_effective):
                                candidate = pending[0]
                                cand_bytes = int(candidate["matrix_bytes"])
                                inflight_bytes = int(sum(inflight.values()))
                                if (
                                    ram_budget_bytes is not None
                                    and len(inflight) > 0
                                    and inflight_bytes + cand_bytes > int(ram_budget_bytes)
                                ):
                                    break
                                pending.pop(0)
                                payload = {
                                    "block_key": candidate["block_key"],
                                    "block_mentions": candidate["mentions"],
                                    "block_scores": candidate["scores"],
                                    "eps": float(candidate.get("effective_eps", eps)),
                                    "min_samples": int(min_samples),
                                    "metric": str(metric),
                                    "constraints": dict(constraints or {}),
                                    "matrix_bytes": int(cand_bytes),
                                }
                                fut = pool.submit(_cluster_worker, payload)
                                inflight[fut] = int(cand_bytes)
                                submitted = True

                            if not inflight and pending and not submitted:
                                candidate = pending.pop(0)
                                payload = {
                                    "block_key": candidate["block_key"],
                                    "block_mentions": candidate["mentions"],
                                    "block_scores": candidate["scores"],
                                    "eps": float(candidate.get("effective_eps", eps)),
                                    "min_samples": int(min_samples),
                                    "metric": str(metric),
                                    "constraints": dict(constraints or {}),
                                    "matrix_bytes": int(candidate["matrix_bytes"]),
                                }
                                fut = pool.submit(_cluster_worker, payload)
                                inflight[fut] = int(candidate["matrix_bytes"])

                            if inflight:
                                done, _ = wait(set(inflight.keys()), return_when=FIRST_COMPLETED)
                                for fut in done:
                                    inflight.pop(fut, None)
                                    result = dict(fut.result())
                                    rows.extend(result.get("rows", []))
                                    sanitize_rows.append(dict(result.get("sanitize", {})))
                                    timing_rows.append(dict(result.get("timing", {})))
                                    cluster_progress.update(1)

            sanitize_totals = _aggregate_sanitize_totals(sanitize_rows)

    out = pd.DataFrame(rows)
    validate_columns(out, CLUSTER_REQUIRED_COLUMNS, "clusters")

    if len(out) > 0:
        mention_order = {
            str(m): int(i)
            for i, m in enumerate(mentions["mention_id"].astype(str).tolist())
        }
        out["__order"] = out["mention_id"].map(mention_order).fillna(10**18).astype(np.int64)
        out = out.sort_values(["__order", "block_key", "mention_id"]).drop(columns=["__order"]).reset_index(drop=True)

    if sanitize_totals["corrected_blocks"] > 0:
        warnings.warn(
            (
                "Sanitized DBSCAN precomputed distances: "
                f"corrected_blocks={sanitize_totals['corrected_blocks']}, "
                f"non_finite_count={sanitize_totals['non_finite_count']}, "
                f"negative_count={sanitize_totals['negative_count']}, "
                f"above_max_count={sanitize_totals['above_max_count']}, "
                f"asymmetry_pairs={sanitize_totals['asymmetry_pairs']}, "
                f"diag_reset_count={sanitize_totals['diag_reset_count']}."
            ),
            RuntimeWarning,
        )

    if output_path is not None:
        save_parquet(out, output_path, index=False)

    timing_rows = [row for row in timing_rows if row]
    backend_block_counts: dict[str, int] = {}
    block_count_by_bucket: dict[str, int] = {}
    total_seconds_by_bucket: dict[str, float] = {}
    dbscan_seconds_by_bucket: dict[str, float] = {}
    for row in timing_rows:
        backend_label = str(row.get("backend", "") or "unknown")
        backend_block_counts[backend_label] = int(backend_block_counts.get(backend_label, 0)) + 1
        bucket_label = _block_size_bucket_label(int(row.get("block_size", 0)))
        if bucket_label is None:
            continue
        block_count_by_bucket[bucket_label] = int(block_count_by_bucket.get(bucket_label, 0)) + 1
        total_seconds_by_bucket[bucket_label] = float(
            total_seconds_by_bucket.get(bucket_label, 0.0) + float(row.get("total_seconds", 0.0))
        )
        dbscan_seconds_by_bucket[bucket_label] = float(
            dbscan_seconds_by_bucket.get(bucket_label, 0.0) + float(row.get("dbscan_seconds", 0.0))
        )
    top_slow_blocks = sorted(
        timing_rows,
        key=lambda row: (-float(row.get("total_seconds", 0.0)), str(row.get("block_key", ""))),
    )[:10]
    distance_matrix_seconds_total = float(sum(float(row.get("distance_matrix_seconds", 0.0)) for row in timing_rows))
    constraints_seconds_total = float(sum(float(row.get("constraints_seconds", 0.0)) for row in timing_rows))
    sanitize_seconds_total = float(sum(float(row.get("sanitize_seconds", 0.0)) for row in timing_rows))
    dbscan_seconds_total = float(sum(float(row.get("dbscan_seconds", 0.0)) for row in timing_rows))
    gpu_transfer_seconds_total = float(sum(float(row.get("gpu_transfer_seconds", 0.0)) for row in timing_rows))

    meta = {
        "eps_base": float(eps),
        "eps_block_policy_enabled": bool(eps_block_policy.get("enabled", False)),
        "eps_block_policy_summary": eps_block_policy_summary,
        "cluster_backend_requested": backend_requested,
        "cluster_backend_effective": backend_effective,
        "cluster_backend_reason": backend_info.get("reason"),
        "backend_block_counts": {str(key): int(value) for key, value in sorted(backend_block_counts.items())},
        "cpu_sharding_mode": str(sharding_mode),
        "cpu_sharding_enabled": bool(sharding_on and backend_effective == "sklearn_cpu"),
        "cpu_workers_requested": worker_info.get("requested"),
        "cpu_workers_effective": int(workers_effective),
        "cpu_limit_detected": int(cpu_info["cpu_limit"]),
        "cpu_limit_source": str(cpu_info["cpu_limit_source"]),
        "cpu_min_pairs_per_worker": int(min_pairs_per_worker),
        "ram_budget_bytes": None if ram_budget_bytes is None else int(ram_budget_bytes),
        "total_pairs_est": int(total_pairs_est),
        "block_p95": float(block_p95),
        "block_size_histogram": _build_block_size_histogram(entries),
        "block_count_by_bucket": {str(key): int(value) for key, value in sorted(block_count_by_bucket.items())},
        "total_seconds_by_bucket": {str(key): float(value) for key, value in sorted(total_seconds_by_bucket.items())},
        "dbscan_seconds_by_bucket": {str(key): float(value) for key, value in sorted(dbscan_seconds_by_bucket.items())},
        "build_entries_seconds": float(build_entries_seconds),
        "distance_matrix_seconds_total": distance_matrix_seconds_total,
        "constraints_seconds_total": constraints_seconds_total,
        "sanitize_seconds_total": sanitize_seconds_total,
        "dbscan_seconds_total": dbscan_seconds_total,
        "gpu_transfer_seconds_total": gpu_transfer_seconds_total,
        "top_slow_blocks": top_slow_blocks,
        "sanitize_totals": sanitize_totals,
    }
    out.attrs["cluster_meta"] = meta

    if return_meta:
        return out, meta
    return out
