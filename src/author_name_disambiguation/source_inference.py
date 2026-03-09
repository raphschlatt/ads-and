from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from author_name_disambiguation.approaches.nand.build_pairs import build_pairs_within_blocks, write_pairs
from author_name_disambiguation.approaches.nand.cluster import cluster_blockwise_dbscan
from author_name_disambiguation.approaches.nand.export import (
    build_author_entities,
    build_source_author_assignments,
    export_source_mirrored_outputs,
)
from author_name_disambiguation.approaches.nand.infer_pairs import score_pairs_with_checkpoint
from author_name_disambiguation.common.cli_ui import get_active_ui
from author_name_disambiguation.common.cpu_runtime import compute_ram_budget_bytes, normalize_workers_request
from author_name_disambiguation.common.io_schema import PAIR_REQUIRED_COLUMNS, PAIR_SCORE_REQUIRED_COLUMNS, read_parquet, save_parquet
from author_name_disambiguation.common.package_resources import load_yaml_like, load_yaml_resource
from author_name_disambiguation.common.pipeline_reports import (
    build_cluster_qc,
    build_infer_stage_metrics,
    build_pairs_qc,
    write_json,
)
from author_name_disambiguation.common.run_report import evaluate_go_no_go, write_go_no_go_report
from author_name_disambiguation.common.subset_builder import build_stage_subset
from author_name_disambiguation.common.uid_registry import assign_registry_uids, load_uid_registry, save_uid_registry
from author_name_disambiguation.data.prepare_ads import prepare_ads_source_data
from author_name_disambiguation.features.embed_chars2vec import get_or_create_chars2vec_embeddings
from author_name_disambiguation.features.embed_specter import get_or_create_specter_embeddings, summarize_precomputed_embeddings

if TYPE_CHECKING:
    from author_name_disambiguation.infer_sources import InferSourcesRequest, InferSourcesResult


MODEL_BUNDLE_SCHEMA_VERSION = "v1"
UID_SCOPE_VALUES = {"dataset", "local", "registry"}
INFER_STAGE_VALUES = {"smoke", "mini", "mid", "full"}


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_run_id(stage: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stage}_{timestamp}_{uuid.uuid4().hex[:8]}"


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


def _validate_request(request: InferSourcesRequest) -> None:
    if str(request.dataset_id).strip() == "":
        raise ValueError("dataset_id must be non-empty.")
    if str(request.infer_stage).strip().lower() not in INFER_STAGE_VALUES:
        raise ValueError(f"Unsupported infer_stage={request.infer_stage!r}.")
    if str(request.uid_scope).strip().lower() not in UID_SCOPE_VALUES:
        raise ValueError(f"Unsupported uid_scope={request.uid_scope!r}.")

    pubs = Path(request.publications_path).expanduser()
    if not pubs.exists():
        raise FileNotFoundError(f"publications_path not found: {pubs}")

    refs = None if request.references_path is None else Path(request.references_path).expanduser()
    if refs is not None and not refs.exists():
        raise FileNotFoundError(f"references_path not found: {refs}")

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

    manifest = _load_json(manifest_path)
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
    clustering_payload = _load_json(clustering_path)
    eps_resolution = dict(clustering_payload.get("eps_resolution", {}) or {})
    if eps_resolution.get("selected_eps") is None:
        eps_resolution["selected_eps"] = float(selected_eps)

    run_cfg = {
        "max_pairs_per_block": manifest.get("max_pairs_per_block"),
        "pair_building": dict(manifest.get("pair_building", {}) or {}),
    }
    return {
        "bundle_dir": bundle_dir,
        "manifest": manifest,
        "checkpoint_path": checkpoint_path,
        "model_cfg": model_cfg,
        "best_threshold": float(best_threshold),
        "selected_eps": float(selected_eps),
        "eps_resolution": eps_resolution,
        "run_cfg": run_cfg,
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
) -> dict[str, Any]:
    n_mentions = int(len(mentions))
    block_sizes = mentions.groupby("block_key").size() if len(mentions) and "block_key" in mentions.columns else pd.Series(dtype=int)
    n_blocks = int(len(block_sizes))
    block_p95 = float(block_sizes.quantile(0.95)) if len(block_sizes) else 0.0
    block_max = int(block_sizes.max()) if len(block_sizes) else 0

    pair_upper_bound = _estimate_pair_upper_bound(mentions, max_pairs_per_block=max_pairs_per_block)
    emb_chars_bytes = n_mentions * 50 * 4
    emb_text_bytes = n_mentions * 768 * 4
    pairs_bytes = pair_upper_bound * 160
    pair_scores_bytes = pair_upper_bound * 80
    qc_bytes = pair_upper_bound * 48
    score_batch_rows = max(1, min(pair_upper_bound, int(score_batch_size)))
    score_batch_bytes = score_batch_rows * (50 + 768) * 4 * 2
    estimate_total = int(emb_chars_bytes + emb_text_bytes + pairs_bytes + pair_scores_bytes + qc_bytes + score_batch_bytes)
    ram_total = _estimate_ram_total_bytes()
    ram_budget = int(ram_total * float(max_ram_fraction)) if ram_total is not None else None
    memory_feasible = None if ram_budget is None else bool(estimate_total <= ram_budget)

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
    }


def _build_output_dirs(output_root: Path) -> dict[str, Path]:
    return {
        "root": _ensure_dir(output_root),
        "interim": _ensure_dir(output_root / "interim"),
        "processed": _ensure_dir(output_root / "processed"),
        "subset_cache": _ensure_dir(output_root / "subsets" / "cache"),
        "subset_manifests": _ensure_dir(output_root / "subsets" / "manifests"),
        "embeddings": _ensure_dir(output_root / "artifacts" / "embeddings"),
        "checkpoints": _ensure_dir(output_root / "artifacts" / "checkpoints"),
        "pair_scores": _ensure_dir(output_root / "artifacts" / "pair_scores"),
        "clusters": _ensure_dir(output_root / "artifacts" / "clusters"),
        "metrics": _ensure_dir(output_root / "artifacts" / "metrics"),
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

    def _ui_start(label: str) -> None:
        if ui is not None:
            ui.start(label)

    def _ui_info(message: str) -> None:
        if ui is not None:
            ui.info(message)

    def _ui_done(message: str) -> None:
        if ui is not None:
            ui.done(message)

    def _ui_skip(message: str) -> None:
        if ui is not None:
            ui.skip(message)

    bootstrap_started_at = perf_counter()
    _ui_start("Bootstrap")
    _ui_info(
        f"dataset={str(request.dataset_id)} | infer_stage={str(request.infer_stage)} | "
        f"output_root={Path(request.output_root).expanduser().resolve()}"
    )
    _validate_request(request)

    infer_stage = str(request.infer_stage).strip().lower()
    uid_scope = str(request.uid_scope).strip().lower()
    uid_namespace = _resolve_uid_namespace(
        uid_scope=uid_scope,
        uid_namespace=request.uid_namespace,
        dataset_id=str(request.dataset_id),
    )

    output_root = Path(request.output_root).expanduser().resolve()
    dirs = _build_output_dirs(output_root)
    run_id = _default_run_id("infer_sources")
    references_present = request.references_path is not None
    bootstrap_runtime = _probe_bootstrap_runtime(str(request.device))
    _ui_info(
        f"device={str(request.device)} -> {bootstrap_runtime.get('resolved_device') or 'n/a'} | "
        f"gpu={bootstrap_runtime.get('gpu_name') or 'n/a'} | "
        f"precision={str(request.precision_mode)} | "
        f"torch={bootstrap_runtime.get('torch_version') or 'n/a'}"
    )
    _ui_info(f"run_id={run_id} | uid_scope={uid_scope} | references_present={bool(references_present)}")
    _ui_done(f"Bootstrap in {_format_elapsed(perf_counter() - bootstrap_started_at)}")

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
        go_payload = _load_json(result_paths["go_no_go"])
        _ui_start("Load inputs")
        _ui_skip("Reused existing infer outputs.")
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
        )

    load_started_at = perf_counter()
    _ui_start("Load inputs")
    _ui_info(
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
    score_chunk_rows = int(infer_overrides.get("score_chunk_rows", 200_000))
    score_chunked_threshold = int(infer_overrides.get("score_chunked_threshold", 500_000))
    pair_chunk_rows = int(infer_overrides.get("pair_chunk_rows", 200_000))
    cpu_workers = normalize_workers_request(infer_overrides.get("cpu_workers"))
    cpu_sharding_mode = str(infer_overrides.get("cpu_sharding_mode", "auto")).strip().lower()
    cpu_min_pairs_per_worker = int(infer_overrides.get("cpu_min_pairs_per_worker", 1_000_000))
    cpu_target_ram_fraction = float(infer_overrides.get("cpu_target_ram_fraction", 0.6))
    cpu_ram_budget_bytes = compute_ram_budget_bytes(target_fraction=cpu_target_ram_fraction)
    cluster_backend = (
        str(infer_overrides.get("cluster_backend", "auto"))
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
        "infer_stage": infer_stage,
        "uid_scope": uid_scope,
        "uid_namespace": uid_namespace,
        "model_bundle": str(Path(request.model_bundle).expanduser().resolve()),
        "source_model_run_id": model_info["source_model_run_id"],
        "selected_eps": float(model_info["selected_eps"]),
        "best_threshold": float(model_info["best_threshold"]),
        "device": str(request.device),
        "precision_mode": str(request.precision_mode),
        "cluster_backend": str(cluster_backend),
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
    )
    publications = prepared["publications"]
    references = prepared["references"]
    canonical_records = prepared["canonical_records"]
    full_mentions = prepared["mentions"]
    _ui_done(
        "Loaded "
        f"publications={_format_count(len(publications))} "
        f"references={_format_count(len(references))} "
        f"canonical_records={_format_count(len(canonical_records))} "
        f"in {_format_elapsed(perf_counter() - load_started_at)}"
    )

    preflight_started_at = perf_counter()
    _ui_start("Preflight")
    save_parquet(full_mentions, dirs["interim"] / "mentions_full.parquet", index=False)

    if infer_stage == "full":
        mentions = full_mentions.copy()
    else:
        mentions = build_stage_subset(
            full_mentions,
            stage=infer_stage,
            seed=int(infer_cfg.get("seed", 11)),
            target_mentions=infer_cfg.get("subset_target_mentions"),
            subset_sampling=dict(infer_cfg.get("subset_sampling", {}) or {}),
        )
    save_parquet(mentions, dirs["interim"] / "mentions.parquet", index=False)

    precomputed_embeddings = {
        "publications": _summarize_precomputed_frame(publications),
        "references": _summarize_precomputed_frame(references),
        "canonical_records": _summarize_precomputed_frame(canonical_records),
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
    _ui_info(
        f"precomputed_embedding column present={_yes_no(precomputed_embeddings['mentions']['column_present'])}"
    )
    _ui_info(
        f"specter_recompute={_format_count(precomputed_embeddings['mentions']['recomputed_embedding_count'])} | "
        f"pair_upper_bound={_format_count(preflight['pair_upper_bound'])}"
    )
    _ui_done(
        f"Preflight for {_format_count(len(mentions))} mentions in "
        f"{_format_elapsed(perf_counter() - preflight_started_at)}"
    )

    name_embeddings_started_at = perf_counter()
    _ui_start("Name embeddings")
    chars_path = dirs["embeddings"] / "chars2vec.npy"
    text_path = dirs["embeddings"] / "specter.npy"
    chars_cache_requested = chars_path.exists() and not bool(request.force)
    _ui_info(
        f"cache={'reuse-if-valid' if chars_cache_requested else 'miss'} | backend=chars2vec/tensorflow"
    )
    chars_result = get_or_create_chars2vec_embeddings(
        mentions=mentions,
        output_path=chars_path,
        force_recompute=bool(request.force),
        use_stub_if_missing=False,
        quiet_libraries=True,
        show_progress=bool(request.progress),
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
    _ui_done(
        f"{_format_count(len(mentions))} names embedded in {_format_elapsed(chars_elapsed)} "
        f"({_format_rate(len(mentions), chars_elapsed)}) | "
        f"cache={'hit' if chars_meta.get('cache_hit') else 'miss'} | backend=chars2vec/tensorflow"
    )

    text_embeddings_started_at = perf_counter()
    _ui_start("Text embeddings")
    text_cache_requested = text_path.exists() and not bool(request.force)
    _ui_info(
        f"cache={'reuse-if-valid' if text_cache_requested else 'miss'} | "
        f"precomputed={_format_count(precomputed_embeddings['mentions']['precomputed_embedding_count'])} | "
        f"batch_size=32 | device={str(request.device)}"
    )
    text_result = get_or_create_specter_embeddings(
        mentions=mentions,
        output_path=text_path,
        force_recompute=bool(request.force),
        model_name=model_info["model_cfg"].get("representation", {}).get("text_model_name", "allenai/specter"),
        text_backend=model_info["model_cfg"].get("representation", {}).get("text_backend", "transformers"),
        text_adapter_name=model_info["model_cfg"].get("representation", {}).get("text_adapter_name"),
        text_adapter_alias=model_info["model_cfg"].get("representation", {}).get("text_adapter_alias", "specter2"),
        max_length=int(model_info["model_cfg"].get("representation", {}).get("max_length", 256)),
        batch_size=32,
        device=str(request.device),
        prefer_precomputed=True,
        use_stub_if_missing=False,
        show_progress=bool(request.progress),
        quiet_libraries=True,
        reuse_model=True,
        return_meta=True,
    )
    if isinstance(text_result, tuple) and len(text_result) == 2:
        text, text_runtime_meta_raw = text_result
    else:
        text = text_result
        text_runtime_meta_raw = None
    text_runtime_meta = _normalize_runtime_meta(
        text_runtime_meta_raw,
        requested_device=str(request.device),
    )
    if not isinstance(text, np.ndarray):
        text = np.load(text_path, mmap_mode="r")
    text_elapsed = perf_counter() - text_embeddings_started_at
    resolved_text_device = text_runtime_meta.get("resolved_device")
    if text_runtime_meta.get("cache_hit"):
        resolved_text_device = resolved_text_device or "cache"
    _ui_done(
        f"{_format_count(len(mentions))} texts embedded in {_format_elapsed(text_elapsed)} "
        f"({_format_rate(len(mentions), text_elapsed)}) | "
        f"cache={'hit' if text_runtime_meta.get('cache_hit') else 'miss'} | "
        f"precomputed={_format_count(text_runtime_meta.get('precomputed_embedding_count') or 0)} | "
        f"device={resolved_text_device or 'n/a'}"
    )

    preflight["runtime"] = {
        "specter": text_runtime_meta,
    }
    write_json(preflight, preflight_path)

    pair_inference_started_at = perf_counter()
    _ui_start("Pair inference")
    _ui_info(
        f"pair_upper_bound={_format_count(preflight['pair_upper_bound'])} | "
        f"score_batch_size={_format_count(score_batch_size)}"
    )
    _ui_info(
        f"cpu_stage=pair_building | cpu_workers={_format_worker_request(cpu_workers)} | "
        f"sharding={cpu_sharding_mode} | ram_budget={_format_ram_budget(cpu_ram_budget_bytes)}"
    )
    pairs_path = dirs["processed"] / "pairs.parquet"
    pairs, pair_meta = build_pairs_within_blocks(
        mentions=mentions,
        max_pairs_per_block=max_pairs_per_block,
        seed=int(infer_cfg.get("seed", 11)),
        require_same_split=False,
        labeled_only=False,
        balance_train=False,
        exclude_same_bibcode=bool(pair_build_cfg.get("exclude_same_bibcode", True)),
        show_progress=bool(request.progress),
        output_path=None,
        chunk_rows=pair_chunk_rows,
        return_pairs=True,
        return_meta=True,
        num_workers=cpu_workers,
        sharding_mode=cpu_sharding_mode,
        min_pairs_per_worker=cpu_min_pairs_per_worker,
        ram_budget_bytes=cpu_ram_budget_bytes,
    )
    if pairs is None:
        pairs = pd.DataFrame(columns=PAIR_REQUIRED_COLUMNS + ["label"])
    pairs = pairs.copy()
    if "label" not in pairs.columns:
        pairs["label"] = pd.Series(dtype="object")
    if len(pairs) == 0:
        save_parquet(pairs, pairs_path, index=False)
    else:
        write_pairs(pairs, pairs_path)

    empty_mentions = pd.DataFrame(columns=["mention_id", "block_key", "orcid", "split"])
    empty_pairs = pd.DataFrame(columns=PAIR_REQUIRED_COLUMNS + ["label"])
    pairs_qc = build_pairs_qc(
        lspo_mentions=empty_mentions,
        lspo_pairs=empty_pairs,
        ads_pairs=empty_pairs,
        split_meta={"status": "not_applicable"},
        lspo_pair_build_meta={},
        ads_pair_build_meta=pair_meta,
        ads_pairs_count=int(len(pairs)),
    )
    write_json(pairs_qc, pairs_qc_path)

    pair_scores_path = dirs["pair_scores"] / "pair_scores.parquet"
    if len(pairs) == 0:
        pair_scores = pd.DataFrame(columns=PAIR_SCORE_REQUIRED_COLUMNS)
        save_parquet(pair_scores, pair_scores_path, index=False)
        pair_score_runtime_meta = _normalize_runtime_meta(
            None,
            requested_device=str(request.device),
            effective_precision_mode=str(request.precision_mode),
            skipped=True,
        )
    else:
        pairs_input: pd.DataFrame | str | Path = pairs_path if len(pairs) > score_chunked_threshold else pairs
        pair_scores_result = score_pairs_with_checkpoint(
            mentions=mentions,
            pairs=pairs_input,
            chars2vec=chars,
            text_emb=text,
            checkpoint_path=model_info["checkpoint_path"],
            output_path=pair_scores_path,
            batch_size=score_batch_size,
            device=str(request.device),
            precision_mode=str(request.precision_mode),
            show_progress=bool(request.progress),
            chunk_rows=score_chunk_rows,
            return_scores=not isinstance(pairs_input, Path),
            return_runtime_meta=True,
        )
        if isinstance(pair_scores_result, tuple) and len(pair_scores_result) == 2:
            pair_scores, pair_score_runtime_meta_raw = pair_scores_result
        else:
            pair_scores = pair_scores_result
            pair_score_runtime_meta_raw = None
        pair_score_runtime_meta = _normalize_runtime_meta(
            pair_score_runtime_meta_raw,
            requested_device=str(request.device),
            effective_precision_mode=str(request.precision_mode),
        )
        if not isinstance(pair_scores, pd.DataFrame):
            pair_scores = read_parquet(pair_scores_path)

    preflight["runtime"] = {
        "specter": text_runtime_meta,
        "pair_building": pair_meta,
        "pair_scoring": pair_score_runtime_meta,
    }
    write_json(preflight, preflight_path)
    _ui_done(
        f"pair_count={_format_count(len(pairs))} | "
        f"pairs_est={_format_count(int(pair_meta.get('total_pairs_est', len(pairs))))} | "
        f"workers={_format_worker_request(pair_meta.get('cpu_workers_effective'))} | "
        f"sharding={_yes_no(bool(pair_meta.get('cpu_sharding_enabled')))} | "
        f"score_count={_format_count(len(pair_scores))} | "
        f"device={pair_score_runtime_meta.get('resolved_device') or 'n/a'} | "
        f"precision={pair_score_runtime_meta.get('effective_precision_mode') or str(request.precision_mode)} "
        f"in {_format_elapsed(perf_counter() - pair_inference_started_at)}"
    )

    clustering_started_at = perf_counter()
    _ui_start("Clustering")
    _ui_info(
        f"backend={cluster_backend} | cpu_workers={_format_worker_request(cpu_workers)} | "
        f"sharding={cpu_sharding_mode} | ram_budget={_format_ram_budget(cpu_ram_budget_bytes)}"
    )
    cluster_cfg_used = json.loads(json.dumps(cluster_cfg))
    cluster_cfg_used["eps_mode"] = "fixed"
    cluster_cfg_used["selected_eps"] = float(model_info["selected_eps"])
    cluster_cfg_used["eps"] = float(model_info["selected_eps"])

    clusters, cluster_runtime_meta = cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cluster_cfg_used,
        output_path=None,
        show_progress=bool(request.progress),
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
        "specter": text_runtime_meta,
        "pair_building": pair_meta,
        "pair_scoring": pair_score_runtime_meta,
        "clustering": cluster_runtime_meta,
    }
    write_json(preflight, preflight_path)
    _ui_done(
        f"clusters={_format_count(int(clusters['author_uid'].nunique()))} | "
        f"mentions={_format_count(len(clusters))} | "
        f"backend={cluster_runtime_meta.get('cluster_backend_effective') or cluster_runtime_meta.get('backend_effective')} | "
        f"workers={_format_worker_request(cluster_runtime_meta.get('cpu_workers_effective'))} | "
        f"sharding={_yes_no(bool(cluster_runtime_meta.get('cpu_sharding_enabled')))} "
        f"in {_format_elapsed(perf_counter() - clustering_started_at)}"
    )

    export_started_at = perf_counter()
    _ui_start("Export and reports")
    _ui_info(f"writing {result_paths['publications_disambiguated'].name}")
    if result_paths["references_disambiguated"] is not None:
        _ui_info(f"writing {result_paths['references_disambiguated'].name}")
    assignments = build_source_author_assignments(
        publications=publications,
        references=references,
        canonical_records=canonical_records,
        clusters=clusters,
        uid_scope=uid_scope,
        uid_namespace=uid_namespace,
        output_path=result_paths["source_author_assignments"],
    )
    author_entities = build_author_entities(
        assignments,
        output_path=result_paths["author_entities"],
    )
    clusters = clusters.merge(
        author_entities[["author_uid", "author_display_name"]],
        on="author_uid",
        how="left",
    )
    save_parquet(clusters, result_paths["mention_clusters"], index=False)

    source_export_qc = export_source_mirrored_outputs(
        assignments=assignments,
        publications_path=request.publications_path,
        references_path=request.references_path,
        publications_output_path=result_paths["publications_disambiguated"],
        references_output_path=result_paths["references_disambiguated"],
    )
    write_json(source_export_qc, source_export_qc_path)

    cluster_qc = build_cluster_qc(
        pair_scores=pair_scores,
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
            "specter": text_runtime_meta,
            "pair_building": pair_meta,
            "pair_scoring": pair_score_runtime_meta,
            "clustering": cluster_runtime_meta,
        },
        "precomputed_embeddings": precomputed_embeddings,
        "uid_resolution": uid_mode_meta,
    }
    write_json(cluster_used_payload, cluster_cfg_used_path)
    context_payload["runtime"] = {
        "specter": text_runtime_meta,
        "pair_building": pair_meta,
        "pair_scoring": pair_score_runtime_meta,
        "clustering": cluster_runtime_meta,
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
            "specter": text_runtime_meta,
            "pair_building": pair_meta,
            "pair_scoring": pair_score_runtime_meta,
            "clustering": cluster_runtime_meta,
        },
        precomputed_embeddings=precomputed_embeddings,
    )
    _ui_info(f"writing {result_paths['stage_metrics'].name}")
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
    _ui_done(
        f"GO={go.get('go')} | blockers={len(go.get('blockers', []))} "
        f"in {_format_elapsed(perf_counter() - export_started_at)}"
    )
    _ui_info(f"Run complete | run_id={run_id}")

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
    )
