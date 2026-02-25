from __future__ import annotations

import multiprocessing as mp
import re
import unicodedata
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from src.common.cpu_runtime import (
    cap_workers_by_ram,
    detect_cpu_limit,
    resolve_effective_workers,
    sharding_enabled,
)
from src.common.io_schema import CLUSTER_REQUIRED_COLUMNS, save_parquet, validate_columns
from src.common.numeric_safety import sanitize_precomputed_distance_matrix

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


def _given_name_token(author_raw: str) -> str:
    _, given_tokens = _split_name_parts(author_raw)
    return given_tokens[0] if given_tokens else ""


def _surname_token(author_raw: str) -> str:
    surname_tokens, _ = _split_name_parts(author_raw)
    if not surname_tokens:
        return ""
    if len(surname_tokens) > 1 and surname_tokens[0] in _SURNAME_PARTICLES:
        surname_tokens = surname_tokens[1:]
    return surname_tokens[-1] if surname_tokens else ""


def _name_conflict(a: str, b: str) -> bool:
    ga = _given_name_token(a)
    gb = _given_name_token(b)
    sa = _surname_token(a)
    sb = _surname_token(b)

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


def _build_distance_matrix(block_mentions: pd.DataFrame, block_scores: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    mention_ids = block_mentions["mention_id"].astype(str).tolist()
    n = len(mention_ids)
    idx = {m: i for i, m in enumerate(mention_ids)}

    dist = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(dist, 0.0)

    for row in block_scores.itertuples(index=False):
        i = idx.get(str(row.mention_id_1))
        j = idx.get(str(row.mention_id_2))
        if i is None or j is None:
            continue
        d = float(row.distance)
        dist[i, j] = d
        dist[j, i] = d

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
    for i in range(n):
        for j in range(i + 1, n):
            force_name = False
            force_year = False
            if enforce_name_conflict and _name_conflict(authors[i], authors[j]):
                force_name = True
            yi, yj = years[i], years[j]
            if (yi is not None and yj is not None) and not (pd.isna(yi) or pd.isna(yj)):
                if abs(int(yi) - int(yj)) > max_year_gap:
                    force_year = True

            if not (force_name or force_year):
                continue

            force_hard = (force_name and name_conflict_mode == "hard") or (force_year and year_gap_mode == "hard")
            if force_hard:
                out[i, j] = 1.0
                out[j, i] = 1.0
                continue

            if force_name:
                out[i, j] = max(out[i, j], name_conflict_min_distance)
                out[j, i] = max(out[j, i], name_conflict_min_distance)
            if force_year:
                out[i, j] = max(out[i, j], year_gap_min_distance)
                out[j, i] = max(out[j, i], year_gap_min_distance)

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

    x_gpu = cp.asarray(dist)
    model = CuMLDBSCAN(eps=float(eps), min_samples=int(min_samples), metric=str(metric))
    labels_gpu = model.fit_predict(x_gpu)
    labels = cp.asnumpy(labels_gpu)
    return np.asarray(labels, dtype=np.int64)


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
) -> tuple[list[dict[str, str]], dict[str, int]]:
    sanitize_totals = {
        "corrected_blocks": 0,
        "non_finite_count": 0,
        "negative_count": 0,
        "above_max_count": 0,
        "asymmetry_pairs": 0,
        "diag_reset_count": 0,
    }

    block_mentions = block_mentions.reset_index(drop=True)
    if len(block_mentions) == 1:
        m = str(block_mentions.iloc[0]["mention_id"])
        rows = [{"mention_id": m, "block_key": str(block_key), "author_uid": f"{block_key}::0"}]
        return rows, sanitize_totals

    dist, _ = _build_distance_matrix(block_mentions, block_scores)
    dist = _apply_constraints(dist, block_mentions, constraints)
    dist, sanitize_meta = sanitize_precomputed_distance_matrix(dist)
    if sanitize_meta["corrected"]:
        sanitize_totals["corrected_blocks"] += 1
    sanitize_totals["non_finite_count"] += int(sanitize_meta["non_finite_count"])
    sanitize_totals["negative_count"] += int(sanitize_meta["negative_count"])
    sanitize_totals["above_max_count"] += int(sanitize_meta["above_max_count"])
    sanitize_totals["asymmetry_pairs"] += int(sanitize_meta["asymmetry_pairs"])
    sanitize_totals["diag_reset_count"] += int(sanitize_meta["diag_reset_count"])

    backend_clean = str(backend).strip().lower()
    if backend_clean == "cuml_gpu":
        labels = _run_dbscan_cuml(dist=dist, eps=eps, min_samples=min_samples, metric=metric)
    else:
        model = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric=str(metric))
        labels = model.fit_predict(dist)

    rows: list[dict[str, str]] = []
    for mention_id, label in zip(block_mentions["mention_id"].astype(str).tolist(), np.asarray(labels).tolist()):
        uid = f"{block_key}::{int(label)}"
        rows.append({"mention_id": mention_id, "block_key": str(block_key), "author_uid": uid})

    return rows, sanitize_totals


def _cluster_worker(payload: dict[str, Any]) -> dict[str, Any]:
    rows, sanitize = _cluster_single_block(
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
        "block_key": payload["block_key"],
        "matrix_bytes": int(payload["matrix_bytes"]),
    }


def _build_block_entries(
    mentions: pd.DataFrame,
    pair_scores: pd.DataFrame,
) -> list[dict[str, Any]]:
    grouped = mentions.groupby("block_key", sort=False)
    if len(pair_scores) > 0 and "block_key" in pair_scores.columns:
        score_indices = pair_scores.groupby("block_key", sort=False).indices
    else:
        score_indices = {}

    entries: list[dict[str, Any]] = []
    for block_key, block_mentions in grouped:
        block_mentions = block_mentions.reset_index(drop=True)
        block_idx = score_indices.get(block_key)
        if block_idx is None:
            block_scores = pair_scores.iloc[0:0]
        else:
            block_scores = pair_scores.iloc[np.asarray(block_idx, dtype=np.int64)]

        n = int(len(block_mentions))
        pair_est = int(n * (n - 1) // 2)
        entries.append(
            {
                "block_key": str(block_key),
                "mentions": block_mentions,
                "scores": block_scores,
                "size": n,
                "pair_est": pair_est,
                "matrix_bytes": int(max(1, n * n) * 4),
            }
        )
    return entries


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
) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    sanitize_rows: list[dict[str, int]] = []

    iterator = entries
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(entries, total=len(entries), desc="Cluster blocks", leave=False)
        except Exception:
            pass

    for entry in iterator:
        block_rows, sanitize = _cluster_single_block(
            block_key=entry["block_key"],
            block_mentions=entry["mentions"],
            block_scores=entry["scores"],
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            constraints=constraints,
            backend=backend,
        )
        rows.extend(block_rows)
        sanitize_rows.append(sanitize)

    return rows, _aggregate_sanitize_totals(sanitize_rows)


def cluster_blockwise_dbscan(
    mentions: pd.DataFrame,
    pair_scores: pd.DataFrame,
    cluster_config: Dict,
    output_path: str | Path | None = None,
    show_progress: bool = False,
    num_workers: int | None = None,
    sharding_mode: str = "auto",
    min_pairs_per_worker: int = 1_000_000,
    ram_budget_bytes: int | None = None,
    backend: str = "sklearn_cpu",
    return_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, Dict[str, Any]]:
    eps = float(cluster_config.get("eps", 0.35))
    min_samples = int(cluster_config.get("min_samples", 1))
    metric = str(cluster_config.get("metric", "precomputed"))
    constraints = cluster_config.get("constraints", {})

    entries = _build_block_entries(mentions=mentions, pair_scores=pair_scores)
    total_pairs_est = int(sum(int(e["pair_est"]) for e in entries))
    n_blocks = int(len(entries))

    backend_info = _resolve_cluster_backend(backend=backend, metric=metric)
    backend_requested = str(backend_info["requested"])
    backend_effective = str(backend_info["effective"])

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

    # GPU backend runs in-process (single-device path).
    if backend_effective == "cuml_gpu":
        try:
            rows, sanitize_totals = _cluster_entries_sequential(
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
            rows, sanitize_totals = _cluster_entries_sequential(
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
            rows, sanitize_totals = _cluster_entries_sequential(
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

            if oversize_entries:
                seq_rows, seq_sanitize = _cluster_entries_sequential(
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

            if parallel_entries:
                # Largest matrices first to improve tail latency and make memory pressure explicit.
                pending = sorted(parallel_entries, key=lambda e: (-int(e["matrix_bytes"]), str(e["block_key"])))

                inflight: dict[Any, int] = {}
                ctx = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=int(min(workers_effective, len(parallel_entries))), mp_context=ctx) as pool:
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
                                "eps": float(eps),
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
                                "eps": float(eps),
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

    meta = {
        "cluster_backend_requested": backend_requested,
        "cluster_backend_effective": backend_effective,
        "cluster_backend_reason": backend_info.get("reason"),
        "cpu_sharding_mode": str(sharding_mode),
        "cpu_sharding_enabled": bool(sharding_on and backend_effective == "sklearn_cpu"),
        "cpu_workers_requested": worker_info.get("requested"),
        "cpu_workers_effective": int(workers_effective),
        "cpu_limit_detected": int(cpu_info["cpu_limit"]),
        "cpu_limit_source": str(cpu_info["cpu_limit_source"]),
        "cpu_min_pairs_per_worker": int(min_pairs_per_worker),
        "ram_budget_bytes": None if ram_budget_bytes is None else int(ram_budget_bytes),
        "total_pairs_est": int(total_pairs_est),
        "sanitize_totals": sanitize_totals,
    }
    out.attrs["cluster_meta"] = meta

    if return_meta:
        return out, meta
    return out
