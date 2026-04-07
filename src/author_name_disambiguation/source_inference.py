from __future__ import annotations

import gc
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from author_name_disambiguation.approaches.nand.build_pairs import build_pairs_within_blocks
from author_name_disambiguation.approaches.nand.cluster import ExactGraphClusterAccumulator, cluster_blockwise_dbscan
from author_name_disambiguation.approaches.nand.export import (
    build_author_entities,
    build_source_author_assignments,
    export_source_mirrored_outputs,
)
from author_name_disambiguation.approaches.nand.infer_pairs import (
    encode_mentions_to_memmap,
    score_pairs_from_mention_embeddings,
    score_pairs_with_checkpoint,
)
from author_name_disambiguation.common.cli_ui import CliProgressHandler, get_active_ui
from author_name_disambiguation.common.cpu_runtime import compute_ram_budget_bytes, normalize_workers_request
from author_name_disambiguation.common.io_schema import (
    PAIR_REQUIRED_COLUMNS,
    PAIR_SCORE_REQUIRED_COLUMNS,
    available_disk_bytes,
    save_parquet,
    sort_parquet_file,
    write_parquet_block_manifest,
)
from author_name_disambiguation.common.package_resources import load_yaml_like, load_yaml_resource
from author_name_disambiguation.common.pipeline_reports import (
    build_cluster_qc,
    build_infer_stage_metrics,
    build_pairs_qc,
    default_run_id,
    load_json,
    write_json,
)
from author_name_disambiguation.common.run_report import evaluate_go_no_go, write_go_no_go_report
from author_name_disambiguation.common.subset_builder import build_stage_subset
from author_name_disambiguation.common.uid_registry import assign_registry_uids, load_uid_registry, save_uid_registry
from author_name_disambiguation.data.prepare_ads import prepare_ads_source_data
from author_name_disambiguation.embedding_contract import build_bundle_embedding_contract
from author_name_disambiguation.features.embed_chars2vec import get_or_create_chars2vec_embeddings
from author_name_disambiguation.features.embed_specter import get_or_create_specter_embeddings, summarize_precomputed_embeddings
from author_name_disambiguation.progress import ProgressReporter, activate_progress_reporter

if TYPE_CHECKING:
    from author_name_disambiguation.infer_sources import InferSourcesRequest, InferSourcesResult


MODEL_BUNDLE_SCHEMA_VERSION = "v1"
UID_SCOPE_VALUES = {"dataset", "local", "registry"}
INFER_STAGE_VALUES = {"smoke", "mini", "mid", "full", "incremental"}
INFER_STAGE_ALIASES = {"incremental": "full"}
RUNTIME_MODE_VALUES = {"gpu", "cpu", "hf"}
INFER_PROGRESS_STAGES: dict[str, tuple[int, str]] = {
    "bootstrap": (1, "Bootstrap"),
    "load_inputs": (2, "Load inputs"),
    "preflight": (3, "Preflight"),
    "name_embeddings": (4, "Name embeddings"),
    "text_embeddings": (5, "Text embeddings"),
    "pair_inference": (6, "Pair inference"),
    "clustering": (7, "Clustering"),
    "export": (8, "Export and reports"),
}


def _ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_consistency(output_path: Path, *, run_id: str, stage: str, extras: dict[str, Any] | None = None) -> Path:
    payload = {
        "run_id": str(run_id),
        "run_stage": str(stage),
    }
    if extras:
        payload.update(extras)
    return write_json(payload, output_path)


def _summarize_precomputed_frame(frame: pd.DataFrame) -> dict[str, Any]:
    values = frame["precomputed_embedding"].tolist() if "precomputed_embedding" in frame.columns else None
    return summarize_precomputed_embeddings(values, total_count=int(len(frame)))


def _select_specter_source_records(*, canonical_records: pd.DataFrame, mentions: pd.DataFrame) -> pd.DataFrame:
    if "canonical_record_id" not in canonical_records.columns:
        raise ValueError("canonical_records missing required column: canonical_record_id")
    if "canonical_record_id" not in mentions.columns:
        raise ValueError("mentions missing required column: canonical_record_id")

    mention_ids = pd.to_numeric(mentions["canonical_record_id"], errors="coerce")
    if mention_ids.isna().any():
        raise ValueError("mentions contain null canonical_record_id values.")
    selected_ids = pd.Index(pd.unique(mention_ids.astype("int64")))
    out = canonical_records[canonical_records["canonical_record_id"].isin(selected_ids)].copy()
    if len(out) != len(selected_ids):
        missing_ids = sorted(set(selected_ids.tolist()) - set(out["canonical_record_id"].astype("int64").tolist()))
        raise RuntimeError(f"Missing canonical records for mention fanout: {missing_ids[:5]!r}")
    return out.reset_index(drop=True)


def _fanout_specter_embeddings_to_mentions(
    *,
    specter_source_records: pd.DataFrame,
    mentions: pd.DataFrame,
    source_embeddings: np.ndarray,
) -> np.ndarray:
    if "canonical_record_id" not in specter_source_records.columns:
        raise ValueError("specter_source_records missing required column: canonical_record_id")
    if "canonical_record_id" not in mentions.columns:
        raise ValueError("mentions missing required column: canonical_record_id")
    if source_embeddings.ndim != 2:
        raise ValueError("source_embeddings must be a 2D array.")
    if len(source_embeddings) != len(specter_source_records):
        raise ValueError("source_embeddings row count must match specter_source_records.")

    source_ids = pd.to_numeric(specter_source_records["canonical_record_id"], errors="coerce")
    mention_ids = pd.to_numeric(mentions["canonical_record_id"], errors="coerce")
    if source_ids.isna().any():
        raise ValueError("specter_source_records contain null canonical_record_id values.")
    if mention_ids.isna().any():
        raise ValueError("mentions contain null canonical_record_id values.")

    source_ids = source_ids.astype("int64")
    mention_ids = mention_ids.astype("int64")
    if source_ids.duplicated().any():
        raise ValueError("specter_source_records canonical_record_id values must be unique.")

    source_positions = pd.Series(np.arange(len(source_ids), dtype=np.int64), index=source_ids.to_numpy())
    try:
        mention_positions = source_positions.loc[mention_ids.to_numpy()].to_numpy(dtype=np.int64, copy=False)
    except KeyError as exc:
        raise RuntimeError("Missing source embeddings for one or more mention canonical_record_id values.") from exc
    return np.asarray(source_embeddings[mention_positions], dtype=np.float32)


def _build_mention_source_index(
    *,
    specter_source_records: pd.DataFrame,
    mentions: pd.DataFrame,
) -> np.ndarray:
    if "canonical_record_id" not in specter_source_records.columns:
        raise ValueError("specter_source_records missing required column: canonical_record_id")
    if "canonical_record_id" not in mentions.columns:
        raise ValueError("mentions missing required column: canonical_record_id")

    source_ids = pd.to_numeric(specter_source_records["canonical_record_id"], errors="coerce")
    mention_ids = pd.to_numeric(mentions["canonical_record_id"], errors="coerce")
    if source_ids.isna().any():
        raise ValueError("specter_source_records contain null canonical_record_id values.")
    if mention_ids.isna().any():
        raise ValueError("mentions contain null canonical_record_id values.")

    source_ids = source_ids.astype("int64")
    mention_ids = mention_ids.astype("int64")
    if source_ids.duplicated().any():
        raise ValueError("specter_source_records canonical_record_id values must be unique.")

    source_positions = pd.Series(np.arange(len(source_ids), dtype=np.int64), index=source_ids.to_numpy())
    try:
        mention_positions = source_positions.loc[mention_ids.to_numpy()].to_numpy(dtype=np.int64, copy=False)
    except KeyError as exc:
        raise RuntimeError("Missing source embeddings for one or more mention canonical_record_id values.") from exc
    return np.asarray(mention_positions, dtype=np.int64)


def _resolve_scratch_dir(output_root: Path, scratch_dir: str | Path | None) -> Path:
    if scratch_dir is None:
        return _ensure_dir(output_root / "scratch")
    return _ensure_dir(Path(scratch_dir).expanduser().resolve())


def _estimate_exact_scratch_bytes(
    *,
    n_mentions: int,
    pair_upper_bound: int,
    mention_embedding_dim: int,
) -> int:
    mention_source_index_bytes = int(n_mentions) * 8
    mention_embedding_bytes = int(n_mentions) * int(mention_embedding_dim) * 4
    mention_norm_bytes = int(n_mentions) * 4
    pairs_bytes = int(pair_upper_bound) * 384
    pair_scores_bytes = int(pair_upper_bound) * 320
    manifest_bytes = int(max(1, n_mentions)) * 64
    return int(
        mention_source_index_bytes
        + mention_embedding_bytes
        + mention_norm_bytes
        + pairs_bytes
        + pair_scores_bytes
        + manifest_bytes
    )


def _default_runtime_meta(*, requested_device: str, effective_precision_mode: str | None = None) -> dict[str, Any]:
    return {
        "requested_device": str(requested_device),
        "resolved_device": None,
        "fallback_reason": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "torch_cuda_available": None,
        "cuda_probe_error": None,
        "model_to_cuda_error": None,
        "effective_precision_mode": effective_precision_mode,
    }


def _normalize_runtime_meta(
    meta: Mapping[str, Any] | None,
    *,
    requested_device: str,
    effective_precision_mode: str | None = None,
    skipped: bool = False,
) -> dict[str, Any]:
    out = _default_runtime_meta(requested_device=requested_device, effective_precision_mode=effective_precision_mode)
    if meta:
        out.update(dict(meta))
    if skipped:
        out["skipped"] = True
    return out


def _compact_specter_runtime_meta(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    if not meta:
        return {}
    drop_keys = {
        "requested_device",
        "runtime_backend_requested",
        "legacy_runtime_overrides",
    }
    return {str(key): value for key, value in dict(meta).items() if key not in drop_keys and value is not None}


def _normalize_runtime_mode(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == "":
        return None
    if normalized not in RUNTIME_MODE_VALUES:
        raise ValueError(f"Unsupported runtime_mode={value!r}. Expected one of {sorted(RUNTIME_MODE_VALUES)!r}.")
    return normalized


def _normalize_infer_stage(value: str | None) -> tuple[str, str]:
    requested = str(value or "full").strip().lower() or "full"
    if requested not in INFER_STAGE_VALUES:
        raise ValueError(f"Unsupported infer_stage={value!r}.")
    return requested, INFER_STAGE_ALIASES.get(requested, requested)


def _requested_device_for_runtime_mode(*, runtime_mode: str | None, requested_device: str) -> str:
    device = str(requested_device or "auto").strip().lower() or "auto"
    if runtime_mode == "gpu":
        return "cuda" if device == "auto" else device
    if runtime_mode in {"cpu", "hf"}:
        return "cpu" if device == "auto" else device
    return device


def _infer_runtime_mode(*, runtime_mode: str | None, specter_runtime_backend: str | None, requested_device: str) -> str:
    if runtime_mode is not None:
        return runtime_mode
    backend = str(specter_runtime_backend or "transformers").strip().lower() or "transformers"
    device = str(requested_device or "auto").strip().lower() or "auto"
    if backend == "hf_endpoint":
        return "hf"
    if backend == "onnx_fp32" or device.startswith("cpu"):
        return "cpu"
    return "gpu"


def _validate_request(request: InferSourcesRequest) -> None:
    if str(request.dataset_id).strip() == "":
        raise ValueError("dataset_id must be non-empty.")
    _normalize_infer_stage(getattr(request, "infer_stage", None))
    if str(request.uid_scope).strip().lower() not in UID_SCOPE_VALUES:
        raise ValueError(f"Unsupported uid_scope={request.uid_scope!r}.")
    _normalize_runtime_mode(getattr(request, "runtime_mode", None))

    pubs = Path(request.publications_path).expanduser()
    if not pubs.exists():
        raise FileNotFoundError(f"publications_path not found: {pubs}")

    refs = None if request.references_path is None else Path(request.references_path).expanduser()
    if refs is not None and not refs.exists():
        raise FileNotFoundError(f"references_path not found: {refs}")

    if request.model_bundle is None:
        raise ValueError("model_bundle must be provided or resolved before source inference runs.")
    bundle = Path(request.model_bundle).expanduser()
    if not bundle.exists():
        raise FileNotFoundError(f"model_bundle not found: {bundle}")


def _resolve_uid_namespace(*, uid_scope: str, uid_namespace: str | None, dataset_id: str) -> str | None:
    if uid_scope == "local":
        return None
    candidate = str(dataset_id if uid_namespace is None else uid_namespace).strip()
    if candidate == "":
        raise ValueError(f"uid_namespace must be non-empty when uid_scope={uid_scope!r}.")
    if "::" in candidate:
        raise ValueError("uid_namespace must not contain '::'.")
    return candidate


def _apply_uid_scope_to_clusters(
    *,
    clusters: pd.DataFrame,
    uid_scope: str,
    uid_namespace: str | None,
) -> pd.DataFrame:
    out = clusters.copy()
    if "author_uid" not in out.columns:
        raise ValueError("clusters missing required column: author_uid")

    if "author_uid_local" not in out.columns:
        out["author_uid_local"] = out["author_uid"].astype(str)
    else:
        out["author_uid_local"] = out["author_uid_local"].astype(str)

    if uid_scope == "local":
        out["author_uid"] = out["author_uid_local"].astype(str)
        return out

    if uid_namespace is None:
        raise ValueError(f"uid_namespace is required when uid_scope={uid_scope!r}.")

    out["author_uid"] = out["author_uid_local"].map(lambda value: f"{uid_namespace}::{value}")
    return out


def _apply_uid_mode_to_clusters(
    *,
    clusters: pd.DataFrame,
    uid_scope: str,
    uid_namespace: str | None,
    uid_registry_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if uid_scope != "registry":
        return _apply_uid_scope_to_clusters(
            clusters=clusters,
            uid_scope=uid_scope,
            uid_namespace=uid_namespace,
        ), None

    if uid_namespace is None:
        raise ValueError("uid_namespace is required when uid_scope='registry'.")
    if uid_registry_path is None:
        raise ValueError("uid_registry_path is required when uid_scope='registry'.")

    registry = load_uid_registry(uid_registry_path, namespace=uid_namespace)
    scoped, registry_out, assign_meta = assign_registry_uids(
        clusters=clusters,
        registry=registry,
        uid_namespace=uid_namespace,
    )
    save_uid_registry(uid_registry_path, registry_out)
    return scoped, {
        "uid_scope": "registry",
        "uid_namespace": uid_namespace,
        "uid_registry_path": str(uid_registry_path),
        "uid_registry_size_after": int(assign_meta.registry_size_after),
        "uid_registry_next_id_after": int(assign_meta.next_id_after),
        "uid_local_to_global_max_nunique": int(assign_meta.local_to_global_max_nunique),
        "uid_global_to_local_max_nunique": int(assign_meta.global_to_local_max_nunique),
        "uid_local_to_global_valid": bool(assign_meta.local_to_global_valid),
    }


def _resolve_model_bundle(model_bundle: str | Path) -> dict[str, Any]:
    raw = Path(model_bundle).expanduser()
    bundle_dir = raw.parent.resolve() if raw.is_file() else raw.resolve()
    manifest_path = bundle_dir / "bundle_manifest.json"
    checkpoint_path = bundle_dir / "checkpoint.pt"
    model_cfg_path = bundle_dir / "model_config.yaml"
    clustering_path = bundle_dir / "clustering_resolved.json"

    for path in [manifest_path, checkpoint_path, model_cfg_path, clustering_path]:
        if not path.exists():
            raise FileNotFoundError(f"Model bundle is missing required file: {path}")

    manifest = load_json(manifest_path)
    schema_version = str(manifest.get("bundle_schema_version", "")).strip()
    if schema_version != MODEL_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model bundle schema version: {schema_version!r}. Expected {MODEL_BUNDLE_SCHEMA_VERSION!r}."
        )

    selected_eps = manifest.get("selected_eps")
    best_threshold = manifest.get("best_threshold")
    if selected_eps is None or best_threshold is None:
        raise ValueError("bundle_manifest.json must contain selected_eps and best_threshold.")

    model_cfg = load_yaml_like(model_cfg_path, default_resource="resources/infer_runs/full.yaml", param_name="model_config")
    clustering_payload = load_json(clustering_path)
    eps_resolution = dict(clustering_payload.get("eps_resolution", {}) or {})
    if eps_resolution.get("selected_eps") is None:
        eps_resolution["selected_eps"] = float(selected_eps)

    run_cfg = {
        "max_pairs_per_block": manifest.get("max_pairs_per_block"),
        "pair_building": dict(manifest.get("pair_building", {}) or {}),
    }
    embedding_contract = dict(manifest.get("embedding_contract") or build_bundle_embedding_contract(model_cfg))
    return {
        "bundle_dir": bundle_dir,
        "manifest": manifest,
        "checkpoint_path": checkpoint_path,
        "model_cfg": model_cfg,
        "best_threshold": float(best_threshold),
        "selected_eps": float(selected_eps),
        "eps_resolution": eps_resolution,
        "run_cfg": run_cfg,
        "embedding_contract": embedding_contract,
        "source_model_run_id": str(manifest.get("source_model_run_id", "bundle")),
    }


def _resolve_infer_run_cfg(infer_stage: str) -> dict[str, Any]:
    cfg = load_yaml_resource(f"resources/infer_runs/{infer_stage}.yaml")
    cfg["stage"] = str(infer_stage)
    cfg["seed"] = int(cfg.get("seed", 11))
    cfg["subset_sampling"] = dict(cfg.get("subset_sampling", {}) or {})
    cfg["infer_overrides"] = dict(cfg.get("infer_overrides", {}) or {})
    return cfg


def _resolve_source_output_path(*, input_path: Path, output_root: Path, base_name: str) -> Path:
    if input_path.suffix.lower() == ".parquet":
        return output_root / f"{base_name}.parquet"
    return output_root / f"{base_name}.jsonl"


def _estimate_ram_total_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def _best_effort_release_runtime_memory() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available() and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
    except Exception:
        return


def _estimate_pair_upper_bound(mentions: pd.DataFrame, max_pairs_per_block: int | None) -> int:
    if len(mentions) == 0 or "block_key" not in mentions.columns:
        return 0
    block_sizes = mentions.groupby("block_key").size().to_numpy(dtype=np.int64, copy=False)
    pair_counts = (block_sizes * (block_sizes - 1)) // 2
    if max_pairs_per_block is not None:
        pair_counts = np.minimum(pair_counts, int(max_pairs_per_block))
    return int(pair_counts.sum())


def _build_infer_preflight(
    *,
    mentions: pd.DataFrame,
    max_pairs_per_block: int | None,
    score_batch_size: int,
    max_ram_fraction: float,
    mention_embedding_dim: int,
    scratch_dir: Path,
) -> dict[str, Any]:
    n_mentions = int(len(mentions))
    block_sizes = mentions.groupby("block_key").size() if len(mentions) and "block_key" in mentions.columns else pd.Series(dtype=int)
    n_blocks = int(len(block_sizes))
    block_p95 = float(block_sizes.quantile(0.95)) if len(block_sizes) else 0.0
    block_max = int(block_sizes.max()) if len(block_sizes) else 0

    pair_upper_bound = _estimate_pair_upper_bound(mentions, max_pairs_per_block=max_pairs_per_block)
    emb_chars_bytes = n_mentions * 50 * 4
    emb_text_bytes = n_mentions * 768 * 4
    mention_embedding_bytes = n_mentions * int(mention_embedding_dim) * 4
    mention_source_index_bytes = n_mentions * 8
    mention_norm_bytes = n_mentions * 4
    pairs_bytes = pair_upper_bound * 384
    pair_scores_bytes = pair_upper_bound * 320
    qc_bytes = pair_upper_bound * 48
    score_batch_rows = max(1, min(pair_upper_bound, int(score_batch_size)))
    score_batch_bytes = score_batch_rows * (50 + 768) * 4 * 2
    estimate_total = int(
        emb_chars_bytes
        + emb_text_bytes
        + mention_embedding_bytes
        + mention_source_index_bytes
        + mention_norm_bytes
        + pairs_bytes
        + pair_scores_bytes
        + qc_bytes
        + score_batch_bytes
    )
    ram_total = _estimate_ram_total_bytes()
    ram_budget = int(ram_total * float(max_ram_fraction)) if ram_total is not None else None
    memory_feasible = None if ram_budget is None else bool(estimate_total <= ram_budget)
    scratch_free_bytes = available_disk_bytes(scratch_dir)
    projected_scratch_bytes = _estimate_exact_scratch_bytes(
        n_mentions=n_mentions,
        pair_upper_bound=pair_upper_bound,
        mention_embedding_dim=mention_embedding_dim,
    )
    exact_infeasible_reason = None
    if scratch_free_bytes is not None and projected_scratch_bytes > int(scratch_free_bytes * 0.95):
        exact_infeasible_reason = (
            "projected_scratch_bytes_exceeds_available_free_space:"
            f"{projected_scratch_bytes}>{int(scratch_free_bytes * 0.95)}"
        )

    return {
        "n_mentions": n_mentions,
        "n_blocks": n_blocks,
        "block_p95": block_p95,
        "block_max": block_max,
        "pair_upper_bound": int(pair_upper_bound),
        "max_pairs_per_block": None if max_pairs_per_block is None else int(max_pairs_per_block),
        "estimate_bytes": {
            "emb_chars": int(emb_chars_bytes),
            "emb_text": int(emb_text_bytes),
            "mention_embeddings": int(mention_embedding_bytes),
            "mention_source_index": int(mention_source_index_bytes),
            "mention_norms": int(mention_norm_bytes),
            "pairs": int(pairs_bytes),
            "pair_scores": int(pair_scores_bytes),
            "cluster_qc": int(qc_bytes),
            "score_batch_peak": int(score_batch_bytes),
            "total_upper_bound": int(estimate_total),
        },
        "ram_total_bytes": ram_total,
        "ram_budget_bytes": ram_budget,
        "max_ram_fraction": float(max_ram_fraction),
        "memory_feasible": memory_feasible,
        "storage_mode": "out_of_core_exact",
        "scratch_dir": str(scratch_dir),
        "scratch_free_bytes": None if scratch_free_bytes is None else int(scratch_free_bytes),
        "projected_scratch_bytes": int(projected_scratch_bytes),
        "exact_infeasible_reason": exact_infeasible_reason,
    }


def _build_output_dirs(output_root: Path) -> dict[str, Path]:
    return {
        "root": _ensure_dir(output_root),
        "interim": _ensure_dir(output_root / "interim"),
        "processed": _ensure_dir(output_root / "processed"),
        "embeddings": _ensure_dir(output_root / "artifacts" / "embeddings"),
        "pair_scores": _ensure_dir(output_root / "artifacts" / "pair_scores"),
    }


def _required_outputs_exist(output_root: Path, references_present: bool) -> bool:
    required = [
        output_root / "mention_clusters.parquet",
        output_root / "source_author_assignments.parquet",
        output_root / "author_entities.parquet",
        output_root / "05_stage_metrics_infer_sources.json",
        output_root / "05_go_no_go_infer_sources.json",
    ]
    if references_present:
        required.append(output_root / "references_disambiguated.parquet")
    return all(path.exists() for path in required)


def _format_count(value: int | float) -> str:
    return f"{int(value):,}"


def _format_elapsed(seconds: float) -> str:
    if seconds >= 120.0:
        return f"{round(seconds):.0f}s"
    return f"{seconds:.1f}s"


def _format_rate(value: int, seconds: float) -> str:
    if seconds <= 0:
        return "n/a"
    return f"{int(round(float(value) / seconds)):,}/s"


def _yes_no(value: bool) -> str:
    return "yes" if bool(value) else "no"


def _format_ram_budget(value: int | None) -> str:
    if value is None:
        return "n/a"
    gib = float(value) / float(1024**3)
    if gib >= 10.0:
        return f"{gib:.0f}GiB"
    return f"{gib:.1f}GiB"


def _format_worker_request(value: object) -> str:
    return "auto" if value is None else str(value)


def _probe_bootstrap_runtime(requested_device: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "requested_device": str(requested_device or "auto"),
        "resolved_device": None,
        "gpu_name": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "torch_cuda_available": None,
        "fallback_reason": None,
        "cuda_probe_error": None,
    }
    try:
        import torch
    except Exception as exc:
        payload["resolved_device"] = "cpu" if str(requested_device or "auto").strip().lower() == "auto" else str(requested_device)
        payload["fallback_reason"] = "torch_import_failed"
        payload["cuda_probe_error"] = repr(exc)
        return payload

    requested = str(requested_device or "auto").strip().lower()
    payload["torch_version"] = getattr(torch, "__version__", None)
    payload["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        payload["torch_cuda_available"] = False
        payload["resolved_device"] = "cpu" if requested == "auto" else requested
        payload["fallback_reason"] = "cuda_probe_failed"
        payload["cuda_probe_error"] = repr(exc)
        return payload

    payload["torch_cuda_available"] = cuda_available
    if requested != "auto":
        payload["resolved_device"] = requested
        if requested.startswith("cuda") and cuda_available:
            try:
                device_index = int(requested.split(":", 1)[1]) if ":" in requested else int(torch.cuda.current_device())
                payload["gpu_name"] = torch.cuda.get_device_name(device_index)
            except Exception:
                payload["gpu_name"] = None
        return payload

    if not cuda_available:
        payload["resolved_device"] = "cpu"
        payload["fallback_reason"] = "torch_cuda_unavailable"
        return payload

    try:
        device_index = int(torch.cuda.current_device())
        torch.empty(1, device=f"cuda:{device_index}")
        payload["resolved_device"] = f"cuda:{device_index}"
        payload["gpu_name"] = torch.cuda.get_device_name(device_index)
    except Exception as exc:
        payload["resolved_device"] = "cpu"
        payload["fallback_reason"] = "cuda_probe_failed"
        payload["cuda_probe_error"] = repr(exc)
    return payload


def run_source_inference(request: InferSourcesRequest) -> InferSourcesResult:
    from author_name_disambiguation.infer_sources import InferSourcesResult

    ui = get_active_ui()
    reporter_handler = request.progress_handler
    if reporter_handler is None and ui is not None:
        reporter_handler = CliProgressHandler(ui)
    reporter = ProgressReporter(handler=reporter_handler)
    progress_enabled = bool(request.progress or reporter.has_handler())

    def _stage_start(stage_key: str) -> None:
        stage_index, stage_label = INFER_PROGRESS_STAGES[stage_key]
        reporter.start_stage(
            stage_index=stage_index,
            stage_total=len(INFER_PROGRESS_STAGES),
            stage_key=stage_key,
            stage_label=stage_label,
        )

    def _stage_info(message: str, *, payload: dict[str, Any] | None = None) -> None:
        reporter.info(message, payload=payload)

    def _stage_warn(message: str, *, payload: dict[str, Any] | None = None) -> None:
        reporter.warn(message, payload=payload)

    def _stage_done(message: str, *, elapsed_seconds: float | None = None, payload: dict[str, Any] | None = None) -> None:
        reporter.done(message, elapsed_seconds=elapsed_seconds, payload=payload)

    def _stage_skip(message: str, *, payload: dict[str, Any] | None = None) -> None:
        reporter.done(message, payload=payload, skipped=True)

    run_started_at = perf_counter()
    summary_payload: dict[str, Any] | None = None

    with activate_progress_reporter(reporter):
        bootstrap_started_at = run_started_at
        _stage_start("bootstrap")
        _stage_info(
            f"dataset={str(request.dataset_id)} | infer_stage={str(request.infer_stage)} | "
            f"output_root={Path(request.output_root).expanduser().resolve()}"
        )
        _validate_request(request)

        infer_stage_requested, infer_stage = _normalize_infer_stage(getattr(request, "infer_stage", None))
        uid_scope = str(request.uid_scope).strip().lower()
        uid_namespace = _resolve_uid_namespace(
            uid_scope=uid_scope,
            uid_namespace=request.uid_namespace,
            dataset_id=str(request.dataset_id),
        )
        runtime_mode_requested = _normalize_runtime_mode(getattr(request, "runtime_mode", None))
        bootstrap_device = _requested_device_for_runtime_mode(
            runtime_mode=runtime_mode_requested,
            requested_device=str(request.device),
        )

        output_root = Path(request.output_root).expanduser().resolve()
        dirs = _build_output_dirs(output_root)
        scratch_dir = _resolve_scratch_dir(output_root, getattr(request, "scratch_dir", None))
        run_id = default_run_id("infer_sources")
        references_present = request.references_path is not None
        bootstrap_runtime = _probe_bootstrap_runtime(str(bootstrap_device))
        _stage_info(
            f"device={str(bootstrap_device)} -> {bootstrap_runtime.get('resolved_device') or 'n/a'} | "
            f"gpu={bootstrap_runtime.get('gpu_name') or 'n/a'} | "
            f"precision={str(request.precision_mode)} | "
            f"torch={bootstrap_runtime.get('torch_version') or 'n/a'}"
        )
        _stage_info(f"run_id={run_id} | uid_scope={uid_scope} | references_present={bool(references_present)}")
        bootstrap_elapsed = perf_counter() - bootstrap_started_at
        _stage_done(f"Bootstrap in {_format_elapsed(bootstrap_elapsed)}", elapsed_seconds=bootstrap_elapsed)

    result_paths = {
        "publications_disambiguated": _resolve_source_output_path(
            input_path=Path(request.publications_path),
            output_root=output_root,
            base_name="publications_disambiguated",
        ),
        "references_disambiguated": None
        if request.references_path is None
        else _resolve_source_output_path(
            input_path=Path(request.references_path),
            output_root=output_root,
            base_name="references_disambiguated",
        ),
        "source_author_assignments": output_root / "source_author_assignments.parquet",
        "author_entities": output_root / "author_entities.parquet",
        "mention_clusters": output_root / "mention_clusters.parquet",
        "stage_metrics": output_root / "05_stage_metrics_infer_sources.json",
        "go_no_go": output_root / "05_go_no_go_infer_sources.json",
    }

    if not request.force and _required_outputs_exist(output_root, references_present=references_present):
        go_payload = load_json(result_paths["go_no_go"])
        _stage_start("load_inputs")
        _stage_skip("Reused existing infer outputs.")
        summary_path = output_root / "summary.json"
        summary_payload = (
            load_json(summary_path)
            if summary_path.exists()
            else {
                "run_id": run_id,
                "dataset_id": str(request.dataset_id),
                "go": go_payload.get("go"),
                "summary_path": str(summary_path),
                "publications_disambiguated_path": str(result_paths["publications_disambiguated"]),
                "references_disambiguated_path": (
                    None
                    if result_paths["references_disambiguated"] is None
                    else str(result_paths["references_disambiguated"])
                ),
                "author_entities_path": str(result_paths["author_entities"]),
                "source_author_assignments_path": str(result_paths["source_author_assignments"]),
                "mention_clusters_path": str(result_paths["mention_clusters"]),
                "stage_metrics_path": str(result_paths["stage_metrics"]),
                "go_no_go_path": str(result_paths["go_no_go"]),
            }
        )
        reporter.run_done(payload=summary_payload, message=f"Run complete | run_id={run_id}")
        return InferSourcesResult(
            run_id=run_id,
            go=go_payload.get("go"),
            output_root=output_root,
            publications_disambiguated_path=result_paths["publications_disambiguated"],
            references_disambiguated_path=result_paths["references_disambiguated"],
            source_author_assignments_path=result_paths["source_author_assignments"],
            author_entities_path=result_paths["author_entities"],
            mention_clusters_path=result_paths["mention_clusters"],
            stage_metrics_path=result_paths["stage_metrics"],
            go_no_go_path=result_paths["go_no_go"],
            summary_path=output_root / "summary.json",
        )

    load_started_at = perf_counter()
    _stage_start("load_inputs")
    _stage_info(
        f"publications={Path(request.publications_path).expanduser().resolve()} | "
        f"references={None if request.references_path is None else Path(request.references_path).expanduser().resolve()}"
    )
    cluster_cfg = load_yaml_like(
        request.cluster_config,
        default_resource="resources/clustering/default.yaml",
        param_name="cluster_config",
    )
    gate_cfg = load_yaml_like(
        request.gates_config,
        default_resource="resources/gates.yaml",
        param_name="gates_config",
    )
    infer_cfg = _resolve_infer_run_cfg(infer_stage)
    model_info = _resolve_model_bundle(request.model_bundle)

    pair_build_cfg = dict(model_info["run_cfg"].get("pair_building", {}) or {})
    infer_overrides = dict(infer_cfg.get("infer_overrides", {}) or {})
    max_pairs_per_block = infer_overrides.get("max_pairs_per_block", model_info["run_cfg"].get("max_pairs_per_block"))
    max_pairs_per_block = None if max_pairs_per_block is None else int(max_pairs_per_block)
    score_batch_size = int(infer_overrides.get("score_batch_size", 8192))
    specter_batch_size_override = infer_overrides.get("specter_batch_size")
    specter_batch_size = None if specter_batch_size_override is None else int(specter_batch_size_override)
    specter_precision_mode = str(infer_overrides.get("specter_precision_mode", "auto")).strip().lower()
    legacy_specter_runtime_backend = (
        str(request.specter_runtime_backend).strip().lower()
        if getattr(request, "specter_runtime_backend", None) is not None
        else str(infer_overrides.get("specter_runtime_backend", "transformers")).strip().lower()
    )
    runtime_mode = runtime_mode_requested
    effective_request_device = _requested_device_for_runtime_mode(
        runtime_mode=runtime_mode,
        requested_device=str(request.device),
    )
    if runtime_mode == "gpu":
        if getattr(request, "specter_runtime_backend", None) == "onnx_fp32":
            raise ValueError("runtime_mode='gpu' is incompatible with specter_runtime_backend='onnx_fp32'.")
        specter_runtime_backend = "transformers"
    elif runtime_mode == "hf":
        if getattr(request, "specter_runtime_backend", None) is not None:
            raise ValueError("runtime_mode='hf' does not accept specter_runtime_backend overrides.")
        specter_runtime_backend = "hf_endpoint"
    elif runtime_mode == "cpu":
        specter_runtime_backend = (
            legacy_specter_runtime_backend if getattr(request, "specter_runtime_backend", None) is not None else "cpu_auto"
        )
    else:
        specter_runtime_backend = legacy_specter_runtime_backend
        runtime_mode = _infer_runtime_mode(
            runtime_mode=runtime_mode,
            specter_runtime_backend=specter_runtime_backend,
            requested_device=effective_request_device,
        )
    score_chunk_rows = int(infer_overrides.get("score_chunk_rows", 200_000))
    pair_chunk_rows = int(infer_overrides.get("pair_chunk_rows", 200_000))
    cpu_workers = normalize_workers_request(infer_overrides.get("cpu_workers"))
    cpu_sharding_mode = str(infer_overrides.get("cpu_sharding_mode", "auto")).strip().lower()
    cpu_min_pairs_per_worker = int(infer_overrides.get("cpu_min_pairs_per_worker", 1_000_000))
    cpu_target_ram_fraction = float(infer_overrides.get("cpu_target_ram_fraction", 0.6))
    cpu_ram_budget_bytes = compute_ram_budget_bytes(target_fraction=cpu_target_ram_fraction)
    cluster_backend = (
        str(infer_overrides.get("cluster_backend", "sklearn_cpu"))
        if request.cluster_backend is None
        else str(request.cluster_backend)
    )
    max_ram_fraction = 0.80
    context_path = output_root / "00_context.json"
    input_summary_path = output_root / "01_input_summary.json"
    preflight_path = output_root / "02_preflight_infer.json"
    pairs_qc_path = output_root / "03_pairs_qc.json"
    cluster_qc_path = output_root / "04_cluster_qc.json"
    source_export_qc_path = output_root / "04_source_export_qc.json"
    cluster_cfg_used_path = output_root / "04_clustering_config_used.json"
    consistency_paths = [
        _write_consistency(output_root / "00_run_consistency.json", run_id=run_id, stage="infer_sources"),
    ]

    context_payload = {
        "run_id": run_id,
        "run_stage": "infer_sources",
        "pipeline_scope": "infer",
        "dataset_id": str(request.dataset_id),
        "publications_path": str(Path(request.publications_path).expanduser().resolve()),
        "references_path": None if request.references_path is None else str(Path(request.references_path).expanduser().resolve()),
        "output_root": str(output_root),
        "scratch_dir": str(scratch_dir),
        "infer_stage": infer_stage,
        "infer_stage_requested": infer_stage_requested,
        "infer_stage_effective": infer_stage,
        "uid_scope": uid_scope,
        "uid_namespace": uid_namespace,
        "model_bundle": str(Path(request.model_bundle).expanduser().resolve()),
        "source_model_run_id": model_info["source_model_run_id"],
        "selected_eps": float(model_info["selected_eps"]),
        "best_threshold": float(model_info["best_threshold"]),
        "embedding_contract": dict(model_info.get("embedding_contract", {}) or {}),
        "runtime_mode": runtime_mode,
        "runtime_backend": specter_runtime_backend,
        "resolved_device": None,
        "generation_mode": None,
        "precision_mode": str(request.precision_mode),
        "cluster_backend": str(cluster_backend),
        "storage_mode": "out_of_core_exact",
        "cpu_runtime_policy": {
            "cpu_workers": _format_worker_request(cpu_workers),
            "cpu_sharding_mode": cpu_sharding_mode,
            "cpu_min_pairs_per_worker": int(cpu_min_pairs_per_worker),
            "cpu_target_ram_fraction": float(cpu_target_ram_fraction),
            "cpu_ram_budget_bytes": None if cpu_ram_budget_bytes is None else int(cpu_ram_budget_bytes),
        },
    }
    write_json(context_payload, context_path)

    prepared = prepare_ads_source_data(
        publications_path=request.publications_path,
        references_path=request.references_path,
        return_raw_sources=False,
        return_runtime_meta=True,
    )
    publications = prepared["publications"]
    references = prepared["references"]
    canonical_records = prepared["canonical_records"]
    full_mentions = prepared["mentions"]
    load_inputs_runtime = dict(prepared.get("runtime", {}) or {})
    context_payload["runtime"] = {"load_inputs": load_inputs_runtime}
    write_json(context_payload, context_path)
    load_elapsed = perf_counter() - load_started_at
    _stage_done(
        "Loaded "
        f"publications={_format_count(len(publications))} "
        f"references={_format_count(len(references))} "
        f"canonical_records={_format_count(len(canonical_records))} "
        f"in {_format_elapsed(load_elapsed)}",
        elapsed_seconds=load_elapsed,
    )
    load_inputs_runtime["wall_seconds"] = float(load_elapsed)

    preflight_started_at = perf_counter()
    _stage_start("preflight")
    save_parquet(full_mentions, dirs["interim"] / "mentions_full.parquet", index=False)

    if infer_stage == "full":
        mentions = full_mentions
    else:
        mentions = build_stage_subset(
            full_mentions,
            stage=infer_stage,
            seed=int(infer_cfg.get("seed", 11)),
            target_mentions=infer_cfg.get("subset_target_mentions"),
            subset_sampling=dict(infer_cfg.get("subset_sampling", {}) or {}),
        )
    save_parquet(mentions, dirs["interim"] / "mentions.parquet", index=False)
    specter_source_records = _select_specter_source_records(canonical_records=canonical_records, mentions=mentions)

    precomputed_embeddings = {
        "publications": _summarize_precomputed_frame(publications),
        "references": _summarize_precomputed_frame(references),
        "canonical_records": _summarize_precomputed_frame(canonical_records),
        "specter_sources": _summarize_precomputed_frame(specter_source_records),
        "mentions_full": _summarize_precomputed_frame(full_mentions),
        "mentions": _summarize_precomputed_frame(mentions),
    }

    write_json(
        {
            "run_id": run_id,
            "run_stage": "infer_sources",
            "dataset_id": str(request.dataset_id),
            "references_present": bool(references_present),
            "publications_rows": int(len(publications)),
            "references_rows": int(len(references)),
            "canonical_records": int(len(canonical_records)),
            "specter_sources": int(len(specter_source_records)),
            "ads_mentions_full": int(len(full_mentions)),
            "ads_mentions": int(len(mentions)),
            "infer_stage": infer_stage,
            "subset_ratio": float(len(mentions) / max(1, len(full_mentions))),
            "precomputed_embeddings": precomputed_embeddings,
        },
        input_summary_path,
    )
    consistency_paths.append(
        _write_consistency(
            output_root / "01_run_consistency.json",
            run_id=run_id,
            stage="infer_sources",
            extras={"ads_mentions": int(len(mentions))},
        )
    )

    preflight = _build_infer_preflight(
        mentions=mentions,
        max_pairs_per_block=max_pairs_per_block,
        score_batch_size=score_batch_size,
        max_ram_fraction=max_ram_fraction,
        mention_embedding_dim=int(model_info["model_cfg"].get("training", {}).get("output_dim", 256)),
        scratch_dir=scratch_dir,
    )
    preflight["run_id"] = run_id
    preflight["run_stage"] = "infer_sources"
    preflight["infer_stage"] = infer_stage
    preflight["precomputed_embeddings"] = precomputed_embeddings
    write_json(preflight, preflight_path)
    consistency_paths.append(
        _write_consistency(
            output_root / "02_run_consistency.json",
            run_id=run_id,
            stage="infer_sources",
            extras={"pair_upper_bound": int(preflight["pair_upper_bound"])},
        )
    )
    _stage_info(
        f"precomputed_embedding column present={_yes_no(precomputed_embeddings['specter_sources']['column_present'])}"
    )
    _stage_info(
        f"specter_sources={_format_count(len(specter_source_records))} | "
        f"specter_recompute={_format_count(precomputed_embeddings['specter_sources']['recomputed_embedding_count'])} | "
        f"pair_upper_bound={_format_count(preflight['pair_upper_bound'])}"
    )
    if preflight.get("exact_infeasible_reason"):
        raise RuntimeError(
            "Exact out-of-core inference is physically infeasible on this scratch volume: "
            f"{preflight['exact_infeasible_reason']}. "
            f"scratch_dir={preflight.get('scratch_dir')} "
            f"scratch_free_bytes={preflight.get('scratch_free_bytes')} "
            f"projected_scratch_bytes={preflight.get('projected_scratch_bytes')}"
        )
    preflight_elapsed = perf_counter() - preflight_started_at
    _stage_done(
        f"Preflight for {_format_count(len(mentions))} mentions in "
        f"{_format_elapsed(preflight_elapsed)}",
        elapsed_seconds=preflight_elapsed,
    )
    preflight["wall_seconds"] = float(preflight_elapsed)

    name_embeddings_started_at = perf_counter()
    _stage_start("name_embeddings")
    chars_path = dirs["embeddings"] / "chars2vec.npy"
    chars_cache_requested = chars_path.exists() and not bool(request.force)
    chars_execution_mode = "predict"
    chars_batch_size = None
    _stage_info(
        f"cache={'reuse-if-valid' if chars_cache_requested else 'miss'} | "
        f"backend=chars2vec/tensorflow | mode={chars_execution_mode} | "
        f"batch_size={'auto' if chars_batch_size is None else _format_count(chars_batch_size)}"
    )
    with activate_progress_reporter(reporter):
        chars_result = get_or_create_chars2vec_embeddings(
            mentions=mentions,
            output_path=chars_path,
            force_recompute=bool(request.force),
            batch_size=chars_batch_size,
            execution_mode=chars_execution_mode,
            use_stub_if_missing=False,
            quiet_libraries=True,
            show_progress=progress_enabled,
            return_meta=True,
        )
    if isinstance(chars_result, tuple) and len(chars_result) == 2:
        chars, chars_meta = chars_result
    else:
        chars = chars_result
        chars_meta = {"cache_hit": False, "generation_mode": "unknown"}
    if not isinstance(chars, np.ndarray):
        chars = np.load(chars_path, mmap_mode="r")
    chars_elapsed = perf_counter() - name_embeddings_started_at
    chars_meta["wall_seconds"] = float(chars_elapsed)
    _stage_done(
        f"{_format_count(len(mentions))} names embedded in {_format_elapsed(chars_elapsed)} "
        f"({_format_rate(len(mentions), chars_elapsed)}) | "
        f"cache={'hit' if chars_meta.get('cache_hit') else 'miss'} | backend=chars2vec/tensorflow",
        elapsed_seconds=chars_elapsed,
    )

    text_embeddings_started_at = perf_counter()
    _stage_start("text_embeddings")
    specter_batch_size_label = "auto" if specter_batch_size is None else _format_count(specter_batch_size)
    preferred_text_backend = "onnx_fp32" if specter_runtime_backend == "cpu_auto" else specter_runtime_backend
    text_cache_name = (
        "specter_sources.npy"
        if preferred_text_backend == "transformers"
        else f"specter_{preferred_text_backend}_sources.npy"
    )
    text_path = dirs["embeddings"] / text_cache_name
    text_cache_requested = text_path.exists() and not bool(request.force)
    _stage_info(
        f"cache={'reuse-if-valid' if text_cache_requested else 'miss'} | "
        f"sources={_format_count(len(specter_source_records))} | "
        f"precomputed={_format_count(precomputed_embeddings['specter_sources']['precomputed_embedding_count'])} | "
        f"batch_size={specter_batch_size_label} | device={str(effective_request_device)} | "
        f"backend={specter_runtime_backend} | "
        f"precision={specter_precision_mode}"
    )
    def _run_text_embeddings(selected_backend: str, selected_output_path: Path):
        with activate_progress_reporter(reporter):
            return get_or_create_specter_embeddings(
                mentions=specter_source_records,
                output_path=selected_output_path,
                force_recompute=bool(request.force),
                model_name=model_info["model_cfg"].get("representation", {}).get("text_model_name", "allenai/specter"),
                text_backend=model_info["model_cfg"].get("representation", {}).get("text_backend", "transformers"),
                text_adapter_name=model_info["model_cfg"].get("representation", {}).get("text_adapter_name"),
                text_adapter_alias=model_info["model_cfg"].get("representation", {}).get("text_adapter_alias", "specter2"),
                runtime_backend=selected_backend,
                max_length=int(model_info["model_cfg"].get("representation", {}).get("max_length", 256)),
                batch_size=specter_batch_size,
                device=str(effective_request_device),
                precision_mode=specter_precision_mode,
                prefer_precomputed=True,
                use_stub_if_missing=False,
                show_progress=progress_enabled,
                quiet_libraries=True,
                reuse_model=False,
                return_meta=True,
            )

    resolved_specter_runtime_backend = specter_runtime_backend
    if specter_runtime_backend == "cpu_auto":
        onnx_text_path = dirs["embeddings"] / "specter_onnx_fp32_sources.npy"
        transformers_text_path = dirs["embeddings"] / "specter_sources.npy"
        try:
            text_result = _run_text_embeddings("onnx_fp32", onnx_text_path)
            text_path = onnx_text_path
            resolved_specter_runtime_backend = "onnx_fp32"
        except Exception as exc:
            _stage_warn(
                "ONNX CPU SPECTER unavailable; falling back to transformers "
                f"({type(exc).__name__}: {exc})"
            )
            text_result = _run_text_embeddings("transformers", transformers_text_path)
            text_path = transformers_text_path
            resolved_specter_runtime_backend = "transformers"
            if isinstance(text_result, tuple) and len(text_result) == 2 and isinstance(text_result[1], dict):
                text_result[1]["fallback_reason"] = text_result[1].get("fallback_reason") or "cpu_auto_onnx_fallback"
                text_result[1]["cpu_auto_fallback_error"] = f"{type(exc).__name__}: {exc}"
    else:
        text_result = _run_text_embeddings(specter_runtime_backend, text_path)
        resolved_specter_runtime_backend = specter_runtime_backend
    if isinstance(text_result, tuple) and len(text_result) == 2:
        text_source_embeddings, text_runtime_meta_raw = text_result
    else:
        text_source_embeddings = text_result
        text_runtime_meta_raw = None
    text_runtime_meta = _normalize_runtime_meta(
        text_runtime_meta_raw,
        requested_device=str(effective_request_device),
    )
    text_runtime_meta["runtime_mode"] = runtime_mode
    text_runtime_meta["runtime_backend"] = str(
        text_runtime_meta.get("runtime_backend") or resolved_specter_runtime_backend
    )
    text_runtime_meta = _compact_specter_runtime_meta(text_runtime_meta)
    context_payload["runtime_mode"] = runtime_mode
    context_payload["runtime_backend"] = text_runtime_meta.get("runtime_backend")
    context_payload["resolved_device"] = text_runtime_meta.get("resolved_device")
    context_payload["generation_mode"] = text_runtime_meta.get("generation_mode")
    write_json(context_payload, context_path)
    if not isinstance(text_source_embeddings, np.ndarray):
        text_source_embeddings = np.load(text_path, mmap_mode="r")
    mention_source_index_path = scratch_dir / "mention_source_index.npy"
    mention_source_index = _build_mention_source_index(
        specter_source_records=specter_source_records,
        mentions=mentions,
    )
    np.save(mention_source_index_path, mention_source_index.astype(np.int64, copy=False))
    text_runtime_meta["source_embedding_count"] = int(len(specter_source_records))
    text_runtime_meta["mention_materialization_count"] = 0
    text_runtime_meta["canonical_to_mentions_fanout"] = float(len(mentions) / max(1, len(specter_source_records)))
    text_runtime_meta["mention_source_index_path"] = str(mention_source_index_path)
    text_runtime_meta["mention_source_index_bytes"] = int(mention_source_index.nbytes)
    text_elapsed = perf_counter() - text_embeddings_started_at
    text_runtime_meta["wall_seconds"] = float(text_elapsed)
    resolved_text_device = text_runtime_meta.get("resolved_device")
    if text_runtime_meta.get("cache_hit"):
        resolved_text_device = resolved_text_device or "cache"
    _stage_done(
        f"{_format_count(len(specter_source_records))} source texts prepared for {_format_count(len(mentions))} mentions in "
        f"{_format_elapsed(text_elapsed)} ({_format_rate(len(specter_source_records), text_elapsed)}) | "
        f"cache={'hit' if text_runtime_meta.get('cache_hit') else 'miss'} | "
        f"precomputed={_format_count(text_runtime_meta.get('precomputed_embedding_count') or 0)} | "
        f"device={resolved_text_device or 'n/a'}",
        elapsed_seconds=text_elapsed,
    )

    preflight["runtime"] = {
        "load_inputs": load_inputs_runtime,
        "chars2vec": dict(chars_meta),
        "specter": text_runtime_meta,
    }
    write_json(preflight, preflight_path)
    _best_effort_release_runtime_memory()

    pair_inference_started_at = perf_counter()
    _stage_start("pair_inference")
    _stage_info(
        f"pair_upper_bound={_format_count(preflight['pair_upper_bound'])} | "
        f"score_batch_size={_format_count(score_batch_size)}"
    )
    _stage_info(
        f"cpu_stage=pair_building | cpu_workers={_format_worker_request(cpu_workers)} | "
        f"sharding={cpu_sharding_mode} | ram_budget={_format_ram_budget(cpu_ram_budget_bytes)}"
    )
    pairs_path = dirs["processed"] / "pairs.parquet"
    pairs_manifest_path = dirs["processed"] / "pairs_manifest.json"
    pair_scores_path = dirs["pair_scores"] / "pair_scores.parquet"
    pair_scores_manifest_path = dirs["pair_scores"] / "pair_scores_manifest.json"
    mention_embeddings_path = scratch_dir / "mention_embeddings.npy"
    mention_norms_path = scratch_dir / "mention_embeddings_norms.npy"
    cluster_cfg_used = json.loads(json.dumps(cluster_cfg))
    cluster_cfg_used["eps_mode"] = "fixed"
    cluster_cfg_used["selected_eps"] = float(model_info["selected_eps"])
    cluster_cfg_used["eps"] = float(model_info["selected_eps"])
    use_exact_graph_clustering = (
        str(cluster_cfg_used.get("metric", "precomputed")) == "precomputed"
        and int(cluster_cfg_used.get("min_samples", 1)) == 1
    )
    cluster_accumulator = (
        ExactGraphClusterAccumulator(
            mentions=mentions,
            cluster_config=cluster_cfg_used,
            backend_requested=str(cluster_backend),
        )
        if use_exact_graph_clustering
        else None
    )
    with activate_progress_reporter(reporter):
        _pairs_unused, pair_meta = build_pairs_within_blocks(
            mentions=mentions,
            max_pairs_per_block=max_pairs_per_block,
            seed=int(infer_cfg.get("seed", 11)),
            require_same_split=False,
            labeled_only=False,
            balance_train=False,
            exclude_same_bibcode=bool(pair_build_cfg.get("exclude_same_bibcode", True)),
            show_progress=progress_enabled,
            output_path=pairs_path,
            chunk_rows=pair_chunk_rows,
            return_pairs=False,
            return_meta=True,
            num_workers=cpu_workers,
            sharding_mode=cpu_sharding_mode,
            min_pairs_per_worker=cpu_min_pairs_per_worker,
            ram_budget_bytes=cpu_ram_budget_bytes,
        )
    if not pairs_path.exists() and isinstance(_pairs_unused, pd.DataFrame):
        save_parquet(_pairs_unused, pairs_path)
    sort_parquet_file(pairs_path, order_by=["block_key", "mention_id_1", "mention_id_2", "pair_id"])
    pairs_manifest = write_parquet_block_manifest(pairs_path, pairs_manifest_path)
    pairs_count = int(pairs_manifest["row_count"])
    pair_meta["pairs_written"] = int(pairs_count)
    pair_meta["output_path"] = str(pairs_path)
    pair_meta["manifest_path"] = str(pairs_manifest_path)
    pair_meta["storage_mode"] = "out_of_core_exact"

    empty_mentions = pd.DataFrame(columns=["mention_id", "block_key", "orcid", "split"])
    empty_pairs = pd.DataFrame(columns=PAIR_REQUIRED_COLUMNS + ["label"])
    pairs_qc = build_pairs_qc(
        lspo_mentions=empty_mentions,
        lspo_pairs=empty_pairs,
        ads_pairs=empty_pairs,
        split_meta={"status": "not_applicable"},
        lspo_pair_build_meta={},
        ads_pair_build_meta=pair_meta,
        ads_pairs_count=int(pairs_count),
    )
    write_json(pairs_qc, pairs_qc_path)

    if pairs_count == 0:
        save_parquet(pd.DataFrame(columns=PAIR_SCORE_REQUIRED_COLUMNS), pair_scores_path, index=False)
        pair_scores_manifest = write_parquet_block_manifest(pair_scores_path, pair_scores_manifest_path)
        mention_encode_runtime_meta = _normalize_runtime_meta(
            None,
            requested_device=str(effective_request_device),
            effective_precision_mode=str(request.precision_mode),
            skipped=True,
        )
        mention_encode_runtime_meta["storage_mode"] = "out_of_core_exact"
        pair_score_runtime_meta = _normalize_runtime_meta(
            None,
            requested_device=str(effective_request_device),
            effective_precision_mode=str(request.precision_mode),
            skipped=True,
        )
        pair_score_runtime_meta["pair_scoring_strategy"] = "preencoded_mentions_memmap"
        pair_score_runtime_meta["mention_storage_device"] = "disk_memmap"
        pair_score_runtime_meta["cuda_oom_fallback_used"] = False
        pair_score_runtime_meta["storage_mode"] = "out_of_core_exact"
        pair_score_runtime_meta["output_path"] = str(pair_scores_path)
        pair_score_runtime_meta["manifest_path"] = str(pair_scores_manifest_path)
    else:
        with activate_progress_reporter(reporter):
            _mention_embeddings_path, mention_encode_runtime_meta_raw = encode_mentions_to_memmap(
                chars2vec=chars,
                source_text_embeddings=text_source_embeddings,
                mention_source_index=mention_source_index_path,
                checkpoint_path=model_info["checkpoint_path"],
                output_path=mention_embeddings_path,
                norms_output_path=mention_norms_path,
                batch_size=score_batch_size,
                device=str(effective_request_device),
                precision_mode=str(request.precision_mode),
                show_progress=progress_enabled,
                return_runtime_meta=True,
            )
        mention_encode_runtime_meta = _normalize_runtime_meta(
            mention_encode_runtime_meta_raw,
            requested_device=str(effective_request_device),
            effective_precision_mode=str(request.precision_mode),
        )
        mention_encode_runtime_meta["storage_mode"] = "out_of_core_exact"
        mention_encode_runtime_meta["mention_embeddings_path"] = str(mention_embeddings_path)
        mention_encode_runtime_meta["mention_norms_path"] = str(mention_norms_path)
        with activate_progress_reporter(reporter):
            _pair_scores_unused, pair_score_runtime_meta_raw = score_pairs_from_mention_embeddings(
                mentions=mentions,
                pairs=pairs_path,
                mention_embeddings=mention_embeddings_path,
                mention_norms=mention_norms_path,
                output_path=pair_scores_path,
                batch_size=score_batch_size,
                show_progress=progress_enabled,
                chunk_rows=score_chunk_rows,
                return_scores=False,
                return_runtime_meta=True,
                score_callback=(
                    None if cluster_accumulator is None else cluster_accumulator.consume_score_columns
                ),
            )
        pair_score_runtime_meta = _normalize_runtime_meta(
            pair_score_runtime_meta_raw,
            requested_device=str(effective_request_device),
            effective_precision_mode=str(request.precision_mode),
        )
        pair_scores_manifest = write_parquet_block_manifest(pair_scores_path, pair_scores_manifest_path)
        pair_score_runtime_meta["resolved_device"] = mention_encode_runtime_meta.get("resolved_device")
        pair_score_runtime_meta["effective_precision_mode"] = (
            mention_encode_runtime_meta.get("effective_precision_mode") or str(request.precision_mode)
        )
        pair_score_runtime_meta["fallback_reason"] = mention_encode_runtime_meta.get("fallback_reason")
        pair_score_runtime_meta["cuda_oom_fallback_used"] = bool(mention_encode_runtime_meta.get("cuda_oom_fallback_used"))
        pair_score_runtime_meta["mention_encode_seconds"] = float(mention_encode_runtime_meta.get("mention_encode_seconds") or 0.0)
        pair_score_runtime_meta["mention_embedding_shape"] = mention_encode_runtime_meta.get("mention_embedding_shape")
        pair_score_runtime_meta["mention_embedding_bytes"] = int(mention_encode_runtime_meta.get("mention_embedding_bytes") or 0)
        pair_score_runtime_meta["mention_norm_bytes"] = int(mention_encode_runtime_meta.get("mention_norm_bytes") or 0)
        pair_score_runtime_meta["mention_embeddings_path"] = str(mention_embeddings_path)
        pair_score_runtime_meta["mention_norms_path"] = str(mention_norms_path)
        pair_score_runtime_meta["storage_mode"] = "out_of_core_exact"
        pair_score_runtime_meta["output_path"] = str(pair_scores_path)
        pair_score_runtime_meta["manifest_path"] = str(pair_scores_manifest_path)

    preflight["runtime"] = {
        "load_inputs": load_inputs_runtime,
        "chars2vec": dict(chars_meta),
        "specter": text_runtime_meta,
        "pair_building": pair_meta,
        "mention_encoding": mention_encode_runtime_meta,
        "pair_scoring": pair_score_runtime_meta,
    }
    write_json(preflight, preflight_path)
    clamping_meta = dict(pair_score_runtime_meta.get("numeric_clamping", {}) or {})
    if bool(clamping_meta.get("clamped")):
        _stage_warn(
            "Applied numeric clamping to pair scores: "
            f"events={int(clamping_meta.get('events', 0))} | "
            f"cosine_non_finite={int(clamping_meta.get('cosine_non_finite_count', 0))} | "
            f"cosine_below_min={int(clamping_meta.get('cosine_below_min_count', 0))} | "
            f"cosine_above_max={int(clamping_meta.get('cosine_above_max_count', 0))} | "
            f"distance_non_finite={int(clamping_meta.get('distance_non_finite_count', 0))} | "
            f"distance_below_min={int(clamping_meta.get('distance_below_min_count', 0))} | "
            f"distance_above_max={int(clamping_meta.get('distance_above_max_count', 0))}"
        )
    pair_inference_elapsed = perf_counter() - pair_inference_started_at
    pair_meta["wall_seconds"] = float(pair_inference_elapsed)
    pair_score_runtime_meta["wall_seconds"] = float(pair_inference_elapsed)
    _stage_done(
        f"pair_count={_format_count(pairs_count)} | "
        f"pairs_est={_format_count(int(pair_meta.get('total_pairs_est', pairs_count)))} | "
        f"workers={_format_worker_request(pair_meta.get('cpu_workers_effective'))} | "
        f"sharding={_yes_no(bool(pair_meta.get('cpu_sharding_enabled')))} | "
        f"score_count={_format_count(int(pair_scores_manifest.get('row_count', 0)))} | "
        f"device={pair_score_runtime_meta.get('resolved_device') or 'n/a'} | "
        f"precision={pair_score_runtime_meta.get('effective_precision_mode') or str(request.precision_mode)} "
        f"in {_format_elapsed(pair_inference_elapsed)}",
        elapsed_seconds=pair_inference_elapsed,
    )
    if any(
        float(pair_score_runtime_meta.get(key) or 0.0) > 0.0
        for key in ("parquet_read_seconds", "pandas_conversion_seconds", "pair_score_seconds", "parquet_write_seconds")
    ):
        _stage_info(
            "pair_score_timing "
            f"read={_format_elapsed(float(pair_score_runtime_meta.get('parquet_read_seconds') or 0.0))} | "
            f"to_pandas={_format_elapsed(float(pair_score_runtime_meta.get('pandas_conversion_seconds') or 0.0))} | "
            f"score={_format_elapsed(float(pair_score_runtime_meta.get('pair_score_seconds') or 0.0))} | "
            f"write={_format_elapsed(float(pair_score_runtime_meta.get('parquet_write_seconds') or 0.0))}"
        )

    clustering_started_at = perf_counter()
    _stage_start("clustering")
    _stage_info(
        f"backend={cluster_backend} | cpu_workers={_format_worker_request(cpu_workers)} | "
        f"sharding={cpu_sharding_mode} | ram_budget={_format_ram_budget(cpu_ram_budget_bytes)}"
    )
    if cluster_accumulator is not None:
        clusters, cluster_runtime_meta = cluster_accumulator.finalize()
        if reporter.has_handler():
            block_total = int(max(1, clusters["block_key"].nunique())) if "block_key" in clusters.columns else 1
            reporter.progress(current=block_total, total=block_total, unit="block")
    else:
        with activate_progress_reporter(reporter):
            clusters, cluster_runtime_meta = cluster_blockwise_dbscan(
                mentions=mentions,
                pair_scores=pair_scores_path,
                cluster_config=cluster_cfg_used,
                output_path=None,
                show_progress=progress_enabled,
                num_workers=cpu_workers,
                sharding_mode=cpu_sharding_mode,
                min_pairs_per_worker=cpu_min_pairs_per_worker,
                ram_budget_bytes=cpu_ram_budget_bytes,
                backend=cluster_backend,
                return_meta=True,
            )
    uid_registry_path = None if uid_scope != "registry" else dirs["root"] / "uid_registry" / f"{uid_namespace}.json"
    clusters, uid_mode_meta = _apply_uid_mode_to_clusters(
        clusters=clusters,
        uid_scope=uid_scope,
        uid_namespace=uid_namespace,
        uid_registry_path=uid_registry_path,
    )
    preflight["runtime"] = {
        "load_inputs": load_inputs_runtime,
        "chars2vec": dict(chars_meta),
        "specter": text_runtime_meta,
        "pair_building": pair_meta,
        "mention_encoding": mention_encode_runtime_meta,
        "pair_scoring": pair_score_runtime_meta,
        "clustering": cluster_runtime_meta,
    }
    write_json(preflight, preflight_path)
    clustering_elapsed = perf_counter() - clustering_started_at
    cluster_runtime_meta["wall_seconds"] = float(clustering_elapsed)
    _stage_done(
        f"clusters={_format_count(int(clusters['author_uid'].nunique()))} | "
        f"mentions={_format_count(len(clusters))} | "
        f"backend={cluster_runtime_meta.get('cluster_backend_effective') or cluster_runtime_meta.get('backend_effective')} | "
        f"workers={_format_worker_request(cluster_runtime_meta.get('cpu_workers_effective'))} | "
        f"sharding={_yes_no(bool(cluster_runtime_meta.get('cpu_sharding_enabled')))} "
        f"in {_format_elapsed(clustering_elapsed)}",
        elapsed_seconds=clustering_elapsed,
    )

    export_started_at = perf_counter()
    _stage_start("export")
    _stage_info(f"writing {result_paths['publications_disambiguated'].name}")
    if result_paths["references_disambiguated"] is not None:
        _stage_info(f"writing {result_paths['references_disambiguated'].name}")
    export_runtime_meta: dict[str, Any] = {}
    assignments_result = build_source_author_assignments(
        publications=publications,
        references=references,
        canonical_records=canonical_records,
        clusters=clusters,
        uid_scope=uid_scope,
        uid_namespace=uid_namespace,
        output_path=result_paths["source_author_assignments"],
        return_author_entities=True,
        return_runtime_meta=True,
    )
    if isinstance(assignments_result, tuple) and len(assignments_result) == 3:
        assignments, author_entities, export_build_runtime = assignments_result
        export_runtime_meta.update(dict(export_build_runtime))
    elif isinstance(assignments_result, tuple) and len(assignments_result) == 2:
        assignments, author_entities = assignments_result
    else:
        assignments = assignments_result
        author_entities = build_author_entities(assignments)
        export_runtime_meta["build_assignments_seconds"] = 0.0
        export_runtime_meta["build_author_entities_seconds"] = 0.0
    save_parquet(author_entities, result_paths["author_entities"], index=False)
    clusters = clusters.merge(
        author_entities[["author_uid", "author_display_name"]],
        on="author_uid",
        how="left",
    )
    save_parquet(clusters, result_paths["mention_clusters"], index=False)

    source_export_result = export_source_mirrored_outputs(
        assignments=assignments,
        publications_path=request.publications_path,
        references_path=request.references_path,
        publications_output_path=result_paths["publications_disambiguated"],
        references_output_path=result_paths["references_disambiguated"],
        return_runtime_meta=True,
    )
    if isinstance(source_export_result, tuple) and len(source_export_result) == 2:
        source_export_qc, export_mirror_runtime = source_export_result
        export_runtime_meta.update(dict(export_mirror_runtime))
    else:
        source_export_qc = source_export_result
    export_elapsed = perf_counter() - export_started_at
    export_runtime_meta["wall_seconds"] = float(export_elapsed)
    write_json(source_export_qc, source_export_qc_path)
    preflight["runtime"] = {
        "load_inputs": load_inputs_runtime,
        "chars2vec": dict(chars_meta),
        "specter": text_runtime_meta,
        "pair_building": pair_meta,
        "mention_encoding": mention_encode_runtime_meta,
        "pair_scoring": pair_score_runtime_meta,
        "clustering": cluster_runtime_meta,
        "export": export_runtime_meta,
    }
    write_json(preflight, preflight_path)

    cluster_qc = build_cluster_qc(
        pair_scores=pair_scores_path,
        clusters=clusters,
        threshold=float(model_info["best_threshold"]),
        cluster_uid_col="author_uid_local" if uid_scope == "registry" else "author_uid",
    )
    write_json(cluster_qc, cluster_qc_path)

    eps_meta = dict(model_info["eps_resolution"] or {})
    eps_meta["selected_eps"] = float(model_info["selected_eps"])
    cluster_used_payload = {
        "run_id": run_id,
        "run_stage": "infer_sources",
        "model_bundle": str(Path(request.model_bundle).expanduser().resolve()),
        "source_model_run_id": model_info["source_model_run_id"],
        "uid_scope": uid_scope,
        "uid_namespace": uid_namespace,
        "best_threshold": float(model_info["best_threshold"]),
        "eps_resolution": eps_meta,
        "cluster_config_used": cluster_cfg_used,
        "cluster_runtime": cluster_runtime_meta,
        "runtime": {
            "load_inputs": load_inputs_runtime,
            "chars2vec": dict(chars_meta),
            "specter": text_runtime_meta,
            "pair_building": pair_meta,
            "mention_encoding": mention_encode_runtime_meta,
            "pair_scoring": pair_score_runtime_meta,
            "clustering": cluster_runtime_meta,
            "export": export_runtime_meta,
        },
        "precomputed_embeddings": precomputed_embeddings,
        "uid_resolution": uid_mode_meta,
    }
    write_json(cluster_used_payload, cluster_cfg_used_path)
    context_payload["runtime"] = {
        "load_inputs": load_inputs_runtime,
        "chars2vec": dict(chars_meta),
        "specter": text_runtime_meta,
        "pair_building": pair_meta,
        "mention_encoding": mention_encode_runtime_meta,
        "pair_scoring": pair_score_runtime_meta,
        "clustering": cluster_runtime_meta,
        "export": export_runtime_meta,
    }
    context_payload["precomputed_embeddings"] = precomputed_embeddings
    write_json(context_payload, context_path)
    consistency_paths.append(
        _write_consistency(
            output_root / "04_run_consistency.json",
            run_id=run_id,
            stage="infer_sources",
            extras={"cluster_count": int(cluster_qc.get("cluster_count", 0))},
        )
    )

    determinism_paths = [
        context_path,
        input_summary_path,
        preflight_path,
        pairs_qc_path,
        cluster_qc_path,
        source_export_qc_path,
        cluster_cfg_used_path,
        result_paths["mention_clusters"],
        result_paths["source_author_assignments"],
        result_paths["author_entities"],
        result_paths["publications_disambiguated"],
    ]
    if result_paths["references_disambiguated"] is not None:
        determinism_paths.append(result_paths["references_disambiguated"])

    stage_metrics = build_infer_stage_metrics(
        run_id=run_id,
        run_stage="infer_sources",
        ads_mentions=mentions,
        clusters=clusters,
        consistency_files=consistency_paths,
        determinism_paths=determinism_paths,
        cluster_qc=cluster_qc,
        eps_meta=eps_meta,
        threshold=float(model_info["best_threshold"]),
        threshold_selection_status="bundle_manifest",
        threshold_source="bundle_manifest",
        precision_mode=str(request.precision_mode),
        infer_stage=infer_stage,
        subset_tag=f"infer_{infer_stage}",
        subset_ratio=float(len(mentions) / max(1, len(full_mentions))),
        memory_feasible=preflight.get("memory_feasible"),
        pair_upper_bound=preflight.get("pair_upper_bound"),
        source_export_qc=source_export_qc,
        runtime={
            "load_inputs": load_inputs_runtime,
            "infer_stage_requested": infer_stage_requested,
            "infer_stage_effective": infer_stage,
            "chars2vec": dict(chars_meta),
            "specter": text_runtime_meta,
            "pair_building": pair_meta,
            "mention_encoding": mention_encode_runtime_meta,
            "pair_scoring": pair_score_runtime_meta,
            "clustering": cluster_runtime_meta,
            "export": export_runtime_meta,
        },
        precomputed_embeddings=precomputed_embeddings,
        storage_mode=str(preflight.get("storage_mode") or "out_of_core_exact"),
        scratch_dir=str(preflight.get("scratch_dir")) if preflight.get("scratch_dir") is not None else None,
        scratch_free_bytes=preflight.get("scratch_free_bytes"),
        projected_scratch_bytes=preflight.get("projected_scratch_bytes"),
        exact_infeasible_reason=preflight.get("exact_infeasible_reason"),
    )
    stage_metrics["embedding_contract"] = dict(model_info.get("embedding_contract", {}) or {})
    _stage_info(f"writing {result_paths['stage_metrics'].name}")
    write_json(stage_metrics, result_paths["stage_metrics"])
    consistency_paths.append(
        _write_consistency(
            output_root / "05_run_consistency.json",
            run_id=run_id,
            stage="infer_sources",
            extras={"go_no_go_path": str(result_paths["go_no_go"])},
        )
    )
    go = evaluate_go_no_go(stage_metrics, gate_config=gate_cfg)
    write_go_no_go_report(go, result_paths["go_no_go"])
    total_elapsed = perf_counter() - run_started_at
    summary_payload = {
        "run_id": run_id,
        "dataset_id": str(request.dataset_id),
        "output_root": str(output_root),
        "summary_path": str(output_root / "summary.json"),
        "go": go.get("go"),
        "infer_stage_requested": infer_stage_requested,
        "infer_stage_effective": infer_stage,
        "runtime_mode": runtime_mode,
        "runtime_backend": text_runtime_meta.get("runtime_backend"),
        "resolved_device": text_runtime_meta.get("resolved_device") or bootstrap_runtime.get("resolved_device"),
        "precision_mode": text_runtime_meta.get("effective_precision_mode") or str(request.precision_mode),
        "clustering_backend": cluster_runtime_meta.get("cluster_backend_effective")
        or cluster_runtime_meta.get("backend_effective"),
        "counts": {
            "publications": int(len(publications)),
            "references": int(len(references)),
            "canonical_records": int(len(canonical_records)),
            "specter_sources": int(len(specter_source_records)),
            "mentions": int(len(mentions)),
            "clusters": int(clusters["author_uid"].nunique()),
            "authors_total": int(source_export_qc.get("authors_total", 0)),
            "authors_mapped": int(source_export_qc.get("authors_mapped", 0)),
            "authors_unmapped": int(source_export_qc.get("authors_unmapped", 0)),
        },
        "stage_seconds": {
            "bootstrap": float(bootstrap_elapsed),
            "load_inputs": float(load_elapsed),
            "preflight": float(preflight_elapsed),
            "name_embeddings": float(chars_elapsed),
            "text_embeddings": float(text_elapsed),
            "pair_inference": float(pair_inference_elapsed),
            "clustering": float(clustering_elapsed),
            "export": float(export_elapsed),
            "total": float(total_elapsed),
        },
        "warnings": list(go.get("warnings", []) or []),
        "blockers": list(go.get("blockers", []) or []),
        "publications_disambiguated_path": str(result_paths["publications_disambiguated"]),
        "references_disambiguated_path": (
            None
            if result_paths["references_disambiguated"] is None
            else str(result_paths["references_disambiguated"])
        ),
        "author_entities_path": str(result_paths["author_entities"]),
        "source_author_assignments_path": str(result_paths["source_author_assignments"]),
        "mention_clusters_path": str(result_paths["mention_clusters"]),
        "stage_metrics_path": str(result_paths["stage_metrics"]),
        "go_no_go_path": str(result_paths["go_no_go"]),
        "outputs": {
            "publications_disambiguated_path": str(result_paths["publications_disambiguated"]),
            "references_disambiguated_path": (
                None
                if result_paths["references_disambiguated"] is None
                else str(result_paths["references_disambiguated"])
            ),
            "author_entities_path": str(result_paths["author_entities"]),
            "source_author_assignments_path": str(result_paths["source_author_assignments"]),
            "mention_clusters_path": str(result_paths["mention_clusters"]),
            "stage_metrics_path": str(result_paths["stage_metrics"]),
            "go_no_go_path": str(result_paths["go_no_go"]),
        },
    }
    summary_path = write_json(summary_payload, output_root / "summary.json")
    _stage_done(
        f"GO={go.get('go')} | blockers={len(go.get('blockers', []))} "
        f"in {_format_elapsed(export_elapsed)}",
        elapsed_seconds=export_elapsed,
    )
    reporter.run_done(payload=summary_payload, message=f"Run complete | run_id={run_id}")

    return InferSourcesResult(
        run_id=run_id,
        go=go.get("go"),
        output_root=output_root,
        publications_disambiguated_path=result_paths["publications_disambiguated"],
        references_disambiguated_path=result_paths["references_disambiguated"],
        source_author_assignments_path=result_paths["source_author_assignments"],
        author_entities_path=result_paths["author_entities"],
        mention_clusters_path=result_paths["mention_clusters"],
        stage_metrics_path=result_paths["stage_metrics"],
        go_no_go_path=result_paths["go_no_go"],
        summary_path=summary_path,
    )
