from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.approaches.nand.build_pairs import assign_lspo_splits, build_pairs_within_blocks, write_pairs
from src.approaches.nand.cluster import cluster_blockwise_dbscan, resolve_dbscan_eps
from src.approaches.nand.export import build_publication_author_mapping
from src.approaches.nand.infer_pairs import score_pairs_with_checkpoint
from src.approaches.nand.train import train_nand_across_seeds
from src.common.cli_ui import CliUI
from src.common.cache_ops import (
    hash_checkpoint_model_state,
    hash_file,
    link_or_copy,
    resolve_shared_cache_root,
    stable_hash,
)
from src.common.config import (
    build_run_dirs,
    find_project_root,
    load_yaml,
    resolve_existing_path,
    resolve_paths_config,
    write_latest_run_context,
    write_run_consistency,
)
from src.common.io_schema import MENTION_REQUIRED_COLUMNS, PAIR_REQUIRED_COLUMNS, PAIR_SCORE_REQUIRED_COLUMNS, read_parquet, save_parquet
from src.common.pipeline_reports import (
    build_cluster_qc,
    build_pairs_qc,
    build_stage_metrics,
    build_subset_summary,
    write_compare_to_baseline,
    write_json,
)
from src.common.run_report import evaluate_go_no_go, load_gate_config, write_go_no_go_report
from src.common.subset_artifacts import (
    atomic_save_parquet,
    compute_source_fp,
    compute_subset_identity,
    resolve_manifest_paths,
    resolve_shared_subset_paths,
)
from src.common.subset_builder import build_stage_subset, write_subset_manifest
from src.data.prepare_ads import prepare_ads_mentions
from src.data.prepare_lspo import prepare_lspo_mentions
from src.features.embed_chars2vec import get_or_create_chars2vec_embeddings
from src.features.embed_specter import get_or_create_specter_embeddings

SUBSET_CACHE_VERSION = "v3"


def _load_run_cfg(path: str | Path) -> dict:
    project_root = find_project_root(Path.cwd())
    cfg_path = resolve_existing_path(path, project_root=project_root) or Path(path)
    cfg = load_yaml(cfg_path)
    return cfg


def _load_model_cfg(path: str | Path) -> dict:
    project_root = find_project_root(Path.cwd())
    cfg_path = resolve_existing_path(path, project_root=project_root) or Path(path)
    return load_yaml(cfg_path)


def _load_paths_cfg(path: str | Path) -> dict:
    project_root = find_project_root(Path.cwd())
    cfg_path = resolve_existing_path(path, project_root=project_root) or Path(path)
    raw = load_yaml(cfg_path)
    return resolve_paths_config(raw, project_root=project_root)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_run_id(stage: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stage}_{ts}_cli{uuid.uuid4().hex[:8]}"


def _configure_library_noise(quiet_libraries: bool) -> None:
    if not quiet_libraries:
        return

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_LOG_LEVEL", "3")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    warnings.filterwarnings(
        "ignore",
        message=r".*`resume_download` is deprecated.*",
        category=FutureWarning,
    )

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    try:  # pragma: no cover
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass

    try:  # pragma: no cover
        from huggingface_hub.utils import disable_progress_bars, logging as hf_logging

        disable_progress_bars()
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    try:  # pragma: no cover
        import absl.logging as absl_logging

        absl_logging.set_verbosity("error")
    except Exception:
        pass


def _resolve_train_seeds(args, run_cfg: dict[str, Any], training_cfg: dict[str, Any]) -> list[int]:
    if getattr(args, "seeds", None):
        return [int(s) for s in args.seeds]
    if run_cfg.get("train_seeds"):
        return [int(s) for s in run_cfg["train_seeds"]]
    return [int(s) for s in training_cfg.get("seeds", [1, 2, 3, 4, 5])]


def _resolve_split_assignment_cfg(run_cfg: dict[str, Any]) -> dict[str, float]:
    cfg = dict(run_cfg.get("split_assignment", {}) or {})
    train_ratio = float(cfg.get("train_ratio", 0.6))
    val_ratio = float(cfg.get("val_ratio", 0.2))
    if train_ratio <= 0.0 or val_ratio < 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"Invalid split_assignment config: train_ratio={train_ratio}, val_ratio={val_ratio}. "
            "Require train_ratio>0, val_ratio>=0, and train_ratio+val_ratio<1."
        )
    return {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": float(1.0 - train_ratio - val_ratio),
    }


def _resolve_pair_build_cfg(run_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(run_cfg.get("pair_building", {}) or {})
    return {
        "exclude_same_bibcode": bool(cfg.get("exclude_same_bibcode", True)),
    }


def _block_size_p95(mentions) -> float:
    if len(mentions) == 0 or "block_key" not in mentions.columns:
        return 0.0
    block_sizes = mentions.groupby("block_key").size()
    if len(block_sizes) == 0:
        return 0.0
    return float(block_sizes.quantile(0.95))


def _singleton_ratio_blocks(mentions) -> float:
    if len(mentions) == 0 or "block_key" not in mentions.columns:
        return 0.0
    block_sizes = mentions.groupby("block_key").size()
    if len(block_sizes) == 0:
        return 0.0
    return float((block_sizes == 1).mean())


def _subset_meta_path(shared_subset_dir: Path, subset_tag: str) -> Path:
    return Path(shared_subset_dir) / f"subset_{subset_tag}.meta.json"


def _validate_cached_subset(
    *,
    lspo_subset,
    ads_subset,
    run_cfg: dict[str, Any],
    stage: str,
    split_assignment_cfg: dict[str, float],
    split_balance_cfg: dict[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    checks: dict[str, Any] = {}

    target_mentions = run_cfg.get("subset_target_mentions")
    if stage != "full" and target_mentions is not None:
        target_n = int(target_mentions)
        if int(len(lspo_subset)) != target_n:
            reasons.append(f"lspo_rows={len(lspo_subset)} expected={target_n}")
        if int(len(ads_subset)) != target_n:
            reasons.append(f"ads_rows={len(ads_subset)} expected={target_n}")
    checks["rows_lspo"] = int(len(lspo_subset))
    checks["rows_ads"] = int(len(ads_subset))

    block_p95 = _block_size_p95(lspo_subset)
    singleton_ratio = _singleton_ratio_blocks(lspo_subset)
    checks["lspo_block_p95"] = float(block_p95)
    checks["lspo_singleton_ratio_blocks"] = float(singleton_ratio)
    if stage in {"mid", "full"}:
        if block_p95 < 2.0:
            reasons.append(f"lspo_block_p95={block_p95:.3f} < 2.0")
        if singleton_ratio > 0.90:
            reasons.append(f"lspo_singleton_ratio_blocks={singleton_ratio:.3f} > 0.90")

    _, feasibility_meta = assign_lspo_splits(
        lspo_subset,
        seed=int(run_cfg.get("seed", 11)),
        train_ratio=float(split_assignment_cfg["train_ratio"]),
        val_ratio=float(split_assignment_cfg["val_ratio"]),
        min_neg_val=int(split_balance_cfg.get("min_neg_val", 0)),
        min_neg_test=int(split_balance_cfg.get("min_neg_test", 0)),
        max_attempts=1,
        return_meta=True,
    )
    max_possible = feasibility_meta.get("max_possible_neg_total")
    required = feasibility_meta.get("required_neg_total")
    checks["max_possible_neg_total"] = max_possible
    checks["required_neg_total"] = required
    if max_possible is not None and required is not None and int(max_possible) < int(required):
        reasons.append(f"max_possible_neg_total={int(max_possible)} < required_neg_total={int(required)}")

    return (len(reasons) == 0), reasons, checks


def _record_cache_ref(
    refs: list[dict[str, Any]],
    *,
    artifact_type: str,
    artifact_id: str,
    shared_path: Path,
    run_path: Path,
    mode: str,
    cache_schema_version: str | None = None,
) -> None:
    row = {
        "artifact_type": str(artifact_type),
        "artifact_id": str(artifact_id),
        "shared_path": str(shared_path),
        "run_path": str(run_path),
        "materialization_mode": str(mode),
    }
    if cache_schema_version is not None:
        row["cache_schema_version"] = str(cache_schema_version)
    refs.append(row)


def _normalize_dataset_id(dataset_id: str) -> str:
    cleaned = str(dataset_id or "").strip()
    if not cleaned:
        raise ValueError("dataset_id must be a non-empty string.")
    if cleaned.startswith("/") or cleaned.startswith("~"):
        raise ValueError("dataset_id must be a folder name under data/raw/ads, not an absolute path.")
    parts = Path(cleaned).parts
    if any(p in {"..", "."} for p in parts):
        raise ValueError("dataset_id must not contain path traversal components ('..' or '.').")
    return cleaned


def _dataset_tag(dataset_id: str) -> str:
    text = _normalize_dataset_id(dataset_id)
    safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in text)
    safe = safe.strip("_")
    return safe or "dataset"


def _file_stamp(path: Path) -> str:
    st = Path(path).stat()
    return f"{st.st_size}-{st.st_mtime_ns}"


def _resolve_ads_dataset_files(data_cfg: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    dataset_id = _normalize_dataset_id(dataset_id)
    raw_ads_base = data_cfg.get("raw_ads_dir")
    if raw_ads_base:
        base_dir = Path(str(raw_ads_base)).resolve()
    else:
        base_dir = Path(str(data_cfg["raw_ads_publications"])).parent.resolve()

    dataset_dir = (base_dir / dataset_id).resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}. "
            f"Expected data/raw/ads/<dataset-id>/ with publications.jsonl or publications.json."
        )

    try:
        dataset_dir.relative_to(base_dir)
    except Exception as exc:
        raise ValueError(f"Dataset path escapes ADS raw base directory: {dataset_dir}") from exc

    pub_candidates = [dataset_dir / "publications.jsonl", dataset_dir / "publications.json"]
    ref_candidates = [dataset_dir / "references.jsonl", dataset_dir / "references.json"]
    publications_path = next((p for p in pub_candidates if p.exists()), None)
    references_path = next((p for p in ref_candidates if p.exists()), None)

    if publications_path is None:
        raise FileNotFoundError(
            "Missing publications file. Expected one of: "
            f"{pub_candidates[0]} or {pub_candidates[1]}"
        )

    dataset_source_fp = stable_hash(
        {
            "publications": _file_stamp(publications_path),
            "references": _file_stamp(references_path) if references_path is not None else "none",
        }
    )
    return {
        "dataset_id": dataset_id,
        "dataset_tag": _dataset_tag(dataset_id),
        "dataset_dir": dataset_dir,
        "publications_path": publications_path,
        "references_path": references_path,
        "references_present": references_path is not None,
        "dataset_source_fp": dataset_source_fp,
    }


def _ensure_columns(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in required_columns:
        if col not in out.columns:
            out[col] = pd.Series(dtype="object")
    return out


def _resolve_model_run_for_inference(
    *,
    paths_cfg: dict[str, Any],
    model_run_id: str,
) -> dict[str, Any]:
    metrics_dir = Path(paths_cfg["artifacts"]["metrics_dir"]) / str(model_run_id)
    manifest_path = metrics_dir / "03_train_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing train manifest for model_run_id={model_run_id}: {manifest_path}"
        )
    train_manifest = _load_json(manifest_path)
    best_checkpoint = Path(str(train_manifest.get("best_checkpoint", ""))).expanduser()
    if not best_checkpoint.exists():
        raise FileNotFoundError(
            "Model run manifest does not reference an existing best checkpoint: "
            f"{best_checkpoint}"
        )

    best_threshold = train_manifest.get("best_threshold")
    if best_threshold is None:
        raise ValueError(
            f"best_threshold missing in {manifest_path}. "
            "A valid model run must contain threshold metadata."
        )
    best_threshold = float(best_threshold)

    cluster_used_path = metrics_dir / "04_clustering_config_used.json"
    if not cluster_used_path.exists():
        raise FileNotFoundError(
            f"Missing clustering metadata for model_run_id={model_run_id}: {cluster_used_path}"
        )
    cluster_used_payload = _load_json(cluster_used_path)
    eps_resolution = dict(cluster_used_payload.get("eps_resolution", {}) or {})
    selected_eps = eps_resolution.get("selected_eps")
    if selected_eps is None:
        raise ValueError(
            "selected_eps missing in model run clustering metadata. "
            f"Expected eps_resolution.selected_eps in {cluster_used_path}."
        )
    selected_eps = float(selected_eps)

    context_path = metrics_dir / "00_context.json"
    context_payload = _load_json(context_path) if context_path.exists() else {}

    run_cfg_path = context_payload.get("run_config")
    run_cfg: dict[str, Any] = {}
    if run_cfg_path:
        try:
            run_cfg = _load_run_cfg(run_cfg_path)
        except Exception:
            run_cfg = {}

    model_cfg_path = context_payload.get("model_config")
    model_cfg: dict[str, Any]
    model_cfg_resolved_path: str | None = None
    if model_cfg_path:
        try:
            model_cfg = _load_model_cfg(model_cfg_path)
            project_root = find_project_root(Path.cwd())
            resolved = resolve_existing_path(model_cfg_path, project_root=project_root) or Path(model_cfg_path)
            model_cfg_resolved_path = str(resolved)
        except Exception:
            model_cfg = _load_model_cfg("configs/model/nand_best.yaml")
            model_cfg_resolved_path = "configs/model/nand_best.yaml"
    else:
        model_cfg = _load_model_cfg("configs/model/nand_best.yaml")
        model_cfg_resolved_path = "configs/model/nand_best.yaml"

    return {
        "model_run_id": str(model_run_id),
        "metrics_dir": metrics_dir,
        "train_manifest_path": manifest_path,
        "train_manifest": train_manifest,
        "best_checkpoint": best_checkpoint,
        "best_threshold": best_threshold,
        "cluster_used_path": cluster_used_path,
        "eps_resolution": eps_resolution,
        "selected_eps": selected_eps,
        "context_path": context_path if context_path.exists() else None,
        "context_payload": context_payload,
        "run_cfg": run_cfg,
        "model_cfg": model_cfg,
        "model_cfg_path": model_cfg_resolved_path,
    }


def _build_eps_values(eps_min: float, eps_max: float, eps_step: float) -> list[float]:
    if eps_step <= 0:
        raise ValueError(f"eps_sweep_step must be > 0, got {eps_step}")
    if eps_min > eps_max:
        raise ValueError(f"eps_sweep_min must be <= eps_sweep_max, got {eps_min} > {eps_max}")
    values = np.arange(eps_min, eps_max + (eps_step * 0.5), eps_step)
    return [float(np.round(v, 6)) for v in values.tolist()]


def _build_eps_sweep_values(cluster_cfg: dict[str, Any]) -> list[float]:
    eps_min = float(cluster_cfg.get("eps_sweep_min", 0.2))
    eps_max = float(cluster_cfg.get("eps_sweep_max", 0.5))
    eps_step = float(cluster_cfg.get("eps_sweep_step", 0.05))
    return _build_eps_values(eps_min=eps_min, eps_max=eps_max, eps_step=eps_step)


def _resolve_precision_mode(run_cfg: dict[str, Any], training_cfg: dict[str, Any]) -> str:
    raw = run_cfg.get("precision_mode", training_cfg.get("precision_mode", "fp32"))
    mode = str(raw or "fp32").strip().lower()
    if mode not in {"fp32", "amp_bf16"}:
        warnings.warn(f"Unknown precision_mode={raw!r}; falling back to fp32.", RuntimeWarning)
        return "fp32"
    return mode


def _cluster_pairwise_metrics(pairs, clusters) -> dict[str, Any]:
    eval_pairs = pairs[pairs["label"].notna()].copy()
    if len(eval_pairs) == 0 or len(clusters) == 0:
        return {
            "f1": None,
            "precision": None,
            "recall": None,
            "accuracy": None,
            "n_pairs": int(len(eval_pairs)),
        }
    diag = eval_pairs.merge(
        clusters[["mention_id", "author_uid"]].rename(columns={"mention_id": "mention_id_1", "author_uid": "author_uid_1"}),
        on="mention_id_1",
        how="left",
    ).merge(
        clusters[["mention_id", "author_uid"]].rename(columns={"mention_id": "mention_id_2", "author_uid": "author_uid_2"}),
        on="mention_id_2",
        how="left",
    )
    pred = (diag["author_uid_1"] == diag["author_uid_2"]).astype(int).to_numpy()
    y = diag["label"].astype(int).to_numpy()
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "n_pairs": int(len(diag)),
    }


def _resolve_stage_eps(
    *,
    cluster_cfg: dict[str, Any],
    best_threshold: float,
    lspo_mentions_split,
    lspo_pairs,
    lspo_chars: np.ndarray,
    lspo_text: np.ndarray,
    checkpoint_path: str,
    score_batch_size: int,
    device: str,
    show_progress: bool,
    precision_mode: str = "fp32",
) -> tuple[float, dict[str, Any]]:
    eps_mode = str(cluster_cfg.get("eps_mode", "fixed")).lower()
    if eps_mode != "val_sweep":
        return resolve_dbscan_eps(cluster_cfg, cosine_threshold=best_threshold)

    sweep_rows: list[dict[str, Any]] = []
    fallback_cfg = dict(cluster_cfg)
    fallback_cfg["selected_eps"] = None
    diag_cfg = dict(cluster_cfg.get("boundary_diagnostics", {}) or {})
    diag_sweep_min = float(diag_cfg.get("diag_min", 0.55))
    diag_sweep_max = float(diag_cfg.get("diag_max", 0.70))
    diag_sweep_step = float(diag_cfg.get("diag_step", 0.05))
    range_limited_delta_f1 = float(diag_cfg.get("delta_f1_threshold", 0.005))

    def _fallback_meta(status: str, base_meta: dict[str, Any]) -> dict[str, Any]:
        payload = dict(base_meta)
        payload.update(
            {
                "sweep_status": status,
                "sweep_results": sweep_rows,
                "n_valid_candidates": 0,
                "boundary_hit": False,
                "boundary_side": None,
                "f1_gap_best_second": None,
                "boundary_diagnostic_run": False,
                "diag_sweep_min": diag_sweep_min,
                "diag_sweep_max": diag_sweep_max,
                "diag_sweep_step": diag_sweep_step,
                "diag_n_valid_candidates": 0,
                "diag_best_eps": None,
                "diag_best_metrics": None,
                "diag_best_minus_canonical_f1": None,
                "range_limited_delta_f1_threshold": range_limited_delta_f1,
                "range_limited": False,
            }
        )
        return payload

    def _eval_rows(
        *,
        eps_values: list[float],
        mentions,
        pairs,
        pair_scores,
        eval_cluster_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for eps in eps_values:
            row: dict[str, Any] = {"eps": float(eps)}
            try:
                cfg = json.loads(json.dumps(eval_cluster_cfg))
                cfg["eps"] = float(eps)
                clusters = cluster_blockwise_dbscan(
                    mentions=mentions,
                    pair_scores=pair_scores,
                    cluster_config=cfg,
                    output_path=None,
                    show_progress=False,
                )
                row.update(_cluster_pairwise_metrics(pairs, clusters))
            except Exception as exc:
                row["error"] = repr(exc)
            rows.append(row)
        return rows

    val_mask = lspo_mentions_split["split"].astype(str) == "val"
    val_mentions = lspo_mentions_split[val_mask].reset_index(drop=True)
    val_pairs = lspo_pairs[(lspo_pairs["split"] == "val") & lspo_pairs["label"].notna()].copy()
    if len(val_mentions) < 2 or len(val_pairs) == 0:
        resolved, base_meta = resolve_dbscan_eps(fallback_cfg, cosine_threshold=best_threshold)
        base_meta = _fallback_meta("fallback_no_val_pairs", base_meta)
        return resolved, base_meta

    val_chars = lspo_chars[val_mask.to_numpy()]
    val_text = lspo_text[val_mask.to_numpy()]
    val_pair_scores = score_pairs_with_checkpoint(
        mentions=val_mentions,
        pairs=val_pairs,
        chars2vec=val_chars,
        text_emb=val_text,
        checkpoint_path=checkpoint_path,
        output_path=None,
        batch_size=int(score_batch_size),
        device=device,
        precision_mode=precision_mode,
        show_progress=show_progress,
    )

    eps_values = _build_eps_sweep_values(cluster_cfg)
    sweep_rows = _eval_rows(
        eps_values=eps_values,
        mentions=val_mentions,
        pairs=val_pairs,
        pair_scores=val_pair_scores,
        eval_cluster_cfg=cluster_cfg,
    )

    valid_rows = [r for r in sweep_rows if r.get("f1") is not None]
    if not valid_rows:
        resolved, base_meta = resolve_dbscan_eps(fallback_cfg, cosine_threshold=best_threshold)
        base_meta = _fallback_meta("fallback_no_valid_candidates", base_meta)
        return resolved, base_meta

    sweep_center = (float(cluster_cfg.get("eps_sweep_min", 0.2)) + float(cluster_cfg.get("eps_sweep_max", 0.5))) / 2.0
    ranked_rows = sorted(valid_rows, key=lambda r: (float(r["f1"]), -abs(float(r["eps"]) - sweep_center)), reverse=True)
    best_row = ranked_rows[0]
    second_row = ranked_rows[1] if len(ranked_rows) > 1 else None
    selected_eps = float(best_row["eps"])
    f1_gap_best_second = None
    if second_row is not None:
        f1_gap_best_second = float(best_row["f1"]) - float(second_row["f1"])
    sweep_min = float(cluster_cfg.get("eps_sweep_min", 0.2))
    sweep_max = float(cluster_cfg.get("eps_sweep_max", 0.5))
    eps_tol = 1e-9
    boundary_side = None
    if abs(selected_eps - sweep_min) <= eps_tol:
        boundary_side = "min"
    elif abs(selected_eps - sweep_max) <= eps_tol:
        boundary_side = "max"
    boundary_hit = boundary_side is not None

    selected_cfg = dict(cluster_cfg)
    selected_cfg["selected_eps"] = selected_eps
    resolved, base_meta = resolve_dbscan_eps(selected_cfg, cosine_threshold=best_threshold)
    base_meta.update(
        {
            "sweep_status": "ok",
            "selected_eps": selected_eps,
            "n_valid_candidates": int(len(valid_rows)),
            "boundary_hit": bool(boundary_hit),
            "boundary_side": boundary_side,
            "f1_gap_best_second": f1_gap_best_second,
            "selected_metrics": {
                "f1": best_row.get("f1"),
                "precision": best_row.get("precision"),
                "recall": best_row.get("recall"),
                "accuracy": best_row.get("accuracy"),
                "n_pairs": best_row.get("n_pairs"),
            },
            "sweep_results": sweep_rows,
            "boundary_diagnostic_run": False,
            "diag_sweep_min": diag_sweep_min,
            "diag_sweep_max": diag_sweep_max,
            "diag_sweep_step": diag_sweep_step,
            "diag_n_valid_candidates": 0,
            "diag_best_eps": None,
            "diag_best_metrics": None,
            "diag_best_minus_canonical_f1": None,
            "range_limited_delta_f1_threshold": range_limited_delta_f1,
            "range_limited": False,
        }
    )

    if boundary_hit:
        base_meta["boundary_diagnostic_run"] = True
        diag_rows: list[dict[str, Any]] = []
        try:
            diag_values = _build_eps_values(
                eps_min=diag_sweep_min,
                eps_max=diag_sweep_max,
                eps_step=diag_sweep_step,
            )
            diag_rows = _eval_rows(
                eps_values=diag_values,
                mentions=val_mentions,
                pairs=val_pairs,
                pair_scores=val_pair_scores,
                eval_cluster_cfg=cluster_cfg,
            )
        except Exception as exc:
            diag_rows = [{"error": repr(exc)}]

        valid_diag = [r for r in diag_rows if r.get("f1") is not None]
        base_meta["diag_sweep_results"] = diag_rows
        base_meta["diag_n_valid_candidates"] = int(len(valid_diag))
        if valid_diag:
            diag_center = (diag_sweep_min + diag_sweep_max) / 2.0
            diag_ranked = sorted(valid_diag, key=lambda r: (float(r["f1"]), -abs(float(r["eps"]) - diag_center)), reverse=True)
            diag_best = diag_ranked[0]
            canonical_f1 = float(best_row["f1"])
            diag_best_f1 = float(diag_best["f1"])
            diag_delta = float(diag_best_f1 - canonical_f1)
            base_meta.update(
                {
                    "diag_best_eps": float(diag_best["eps"]),
                    "diag_best_metrics": {
                        "f1": diag_best.get("f1"),
                        "precision": diag_best.get("precision"),
                        "recall": diag_best.get("recall"),
                        "accuracy": diag_best.get("accuracy"),
                        "n_pairs": diag_best.get("n_pairs"),
                    },
                    "diag_best_minus_canonical_f1": diag_delta,
                    "range_limited": bool(diag_delta >= range_limited_delta_f1),
                }
            )
    return resolved, base_meta


def cmd_prepare_lspo(args):
    paths = _load_paths_cfg(args.paths_config)
    out = args.output or str(Path(paths["data"]["interim_dir"]) / "lspo_mentions.parquet")
    df = prepare_lspo_mentions(
        parquet_path=paths["data"]["raw_lspo_parquet"],
        h5_path=paths["data"].get("raw_lspo_h5"),
        output_path=out,
    )
    print(f"Prepared LSPO mentions: {len(df)} -> {out}")


def cmd_prepare_ads(args):
    paths = _load_paths_cfg(args.paths_config)
    out = args.output or str(Path(paths["data"]["interim_dir"]) / "ads_mentions.parquet")
    df = prepare_ads_mentions(
        publications_path=paths["data"]["raw_ads_publications"],
        references_path=paths["data"]["raw_ads_references"],
        output_path=out,
    )
    print(f"Prepared ADS mentions: {len(df)} -> {out}")


def cmd_subset(args):
    mentions = read_parquet(args.input)
    run_cfg = _load_run_cfg(args.run_config)
    stage = run_cfg["stage"]
    seed = int(run_cfg.get("seed", 11))
    target = run_cfg.get("subset_target_mentions")
    subset_sampling = run_cfg.get("subset_sampling", {})

    subset = build_stage_subset(
        mentions,
        stage=stage,
        seed=seed,
        target_mentions=target,
        subset_sampling=subset_sampling,
    )
    save_parquet(subset, args.output, index=False)
    write_subset_manifest(subset, args.manifest)
    print(f"Subset {stage}: {len(subset)} mentions -> {args.output}")


def cmd_embeddings(args):
    _configure_library_noise(getattr(args, "quiet_libs", True))
    mentions = read_parquet(args.mentions)
    model_cfg = _load_model_cfg(args.model_config)
    rep_cfg = model_cfg.get("representation", {})

    chars = get_or_create_chars2vec_embeddings(
        mentions=mentions,
        output_path=args.chars_out,
        force_recompute=args.force,
        use_stub_if_missing=args.use_stub,
        quiet_libraries=getattr(args, "quiet_libs", True),
    )
    text = get_or_create_specter_embeddings(
        mentions=mentions,
        output_path=args.text_out,
        force_recompute=args.force,
        model_name=rep_cfg.get("text_model_name", "allenai/specter"),
        max_length=int(rep_cfg.get("max_length", 256)),
        batch_size=args.batch_size,
        device=args.device,
        prefer_precomputed=args.prefer_precomputed,
        use_stub_if_missing=args.use_stub,
        show_progress=args.progress,
        quiet_libraries=getattr(args, "quiet_libs", True),
        reuse_model=True,
    )
    print(f"Chars2Vec embeddings: {chars.shape} -> {args.chars_out}")
    print(f"Text embeddings: {text.shape} -> {args.text_out}")


def cmd_pairs(args):
    mentions = read_parquet(args.mentions)
    run_cfg = _load_run_cfg(args.run_config) if args.run_config else {}
    pair_build_cfg = _resolve_pair_build_cfg(run_cfg)

    split_meta = None
    if args.assign_lspo_splits:
        split_cfg = run_cfg.get("split_balance", {})
        split_assignment_cfg = _resolve_split_assignment_cfg(run_cfg)

        min_neg_val = int(args.min_neg_val) if args.min_neg_val is not None else int(split_cfg.get("min_neg_val", 0))
        min_neg_test = int(args.min_neg_test) if args.min_neg_test is not None else int(split_cfg.get("min_neg_test", 0))
        max_attempts = int(args.max_attempts) if args.max_attempts is not None else int(split_cfg.get("max_attempts", 1))

        mentions, split_meta = assign_lspo_splits(
            mentions,
            seed=args.seed,
            train_ratio=float(split_assignment_cfg["train_ratio"]),
            val_ratio=float(split_assignment_cfg["val_ratio"]),
            min_neg_val=min_neg_val,
            min_neg_test=min_neg_test,
            max_attempts=max_attempts,
            return_meta=True,
        )
        save_parquet(mentions, args.mentions, index=False)

    pairs, pair_meta = build_pairs_within_blocks(
        mentions=mentions,
        max_pairs_per_block=args.max_pairs_per_block,
        seed=args.seed,
        require_same_split=not args.allow_cross_split,
        labeled_only=args.labeled_only,
        balance_train=args.balance_train,
        exclude_same_bibcode=bool(pair_build_cfg["exclude_same_bibcode"]),
        show_progress=args.progress,
        return_meta=True,
    )
    write_pairs(pairs, args.output)
    if split_meta is not None:
        print(f"Split balancing: {split_meta}")
    print(f"Pair build meta: {pair_meta}")
    print(f"Built pairs: {len(pairs)} -> {args.output}")


def cmd_train(args):
    mentions = read_parquet(args.mentions)
    pairs = read_parquet(args.pairs)
    chars = np.load(args.chars)
    text = np.load(args.text)

    model_cfg = _load_model_cfg(args.model_config)
    training_cfg = dict(model_cfg.get("training", {}) or {})
    precision_mode = str(args.precision_mode or training_cfg.get("precision_mode", "fp32")).strip().lower()
    if precision_mode not in {"fp32", "amp_bf16"}:
        raise ValueError(f"Unsupported precision_mode={precision_mode!r}. Use fp32 or amp_bf16.")
    training_cfg["precision_mode"] = precision_mode
    seeds = args.seeds or training_cfg.get("seeds", [1, 2, 3, 4, 5])

    manifest = train_nand_across_seeds(
        mentions=mentions,
        pairs=pairs,
        chars2vec=chars,
        text_emb=text,
        model_config=training_cfg,
        seeds=[int(s) for s in seeds],
        run_id=args.run_id,
        output_dir=args.output_dir,
        metrics_output=args.metrics_output,
        device=args.device,
        precision_mode=precision_mode,
        show_progress=args.progress,
    )
    print(f"Training done. Best checkpoint: {manifest['best_checkpoint']}")


def cmd_score(args):
    mentions = read_parquet(args.mentions)
    pairs = read_parquet(args.pairs)
    chars = np.load(args.chars)
    text = np.load(args.text)

    out = score_pairs_with_checkpoint(
        mentions=mentions,
        pairs=pairs,
        chars2vec=chars,
        text_emb=text,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        precision_mode=args.precision_mode,
        show_progress=args.progress,
    )
    print(f"Scored pairs: {len(out)} -> {args.output}")


def cmd_cluster(args):
    mentions = read_parquet(args.mentions)
    pair_scores = read_parquet(args.pair_scores)
    project_root = find_project_root(Path.cwd())
    cluster_cfg_path = resolve_existing_path(args.cluster_config, project_root=project_root) or Path(args.cluster_config)
    cluster_cfg = load_yaml(cluster_cfg_path)
    resolved_eps, _eps_meta = resolve_dbscan_eps(cluster_cfg, cosine_threshold=None)
    cluster_cfg["eps"] = resolved_eps

    clusters = cluster_blockwise_dbscan(
        mentions=mentions,
        pair_scores=pair_scores,
        cluster_config=cluster_cfg,
        output_path=args.output,
        show_progress=args.progress,
    )
    print(f"Cluster assignments: {len(clusters)} -> {args.output}")


def cmd_export(args):
    mentions = read_parquet(args.mentions)
    clusters = read_parquet(args.clusters)
    out = build_publication_author_mapping(mentions=mentions, clusters=clusters, output_path=args.output)
    print(f"Publication-author mapping rows: {len(out)} -> {args.output}")


def cmd_report(args):
    metrics = load_yaml(args.metrics) if str(args.metrics).endswith((".yaml", ".yml")) else None
    if metrics is None:
        metrics = _load_json(args.metrics)

    gate_cfg = load_gate_config(args.gates_config) if args.gates_config else None
    go = evaluate_go_no_go(metrics, gate_config=gate_cfg)
    write_go_no_go_report(go, args.output)
    print(f"Go/No-Go: {'GO' if go['go'] else 'NO-GO'} -> {args.output}")


def _cache_version_num(raw: Any) -> int:
    text = str(raw or "").strip().lower()
    if text.startswith("v"):
        text = text[1:]
    try:
        return int(text)
    except Exception:
        return 0


def _collect_stale_subset_records(data_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    dirs = [
        resolve_shared_cache_root(data_cfg) / "subsets",
        Path(data_cfg["subset_cache_dir"]) / "_shared",
    ]
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for base in dirs:
        if not base.exists():
            continue
        for lspo_path in sorted(base.glob("lspo_mentions_*.parquet")):
            subset_tag = lspo_path.name[len("lspo_mentions_") : -len(".parquet")]
            key = (str(base), subset_tag)
            if key in seen:
                continue
            seen.add(key)
            ads_path = base / f"ads_mentions_{subset_tag}.parquet"
            meta_path = _subset_meta_path(base, subset_tag)
            reason = None
            cache_version = None
            if not ads_path.exists():
                reason = "missing_ads_pair"
            elif not meta_path.exists():
                reason = "missing_meta"
            else:
                try:
                    meta = _load_json(meta_path)
                except Exception:
                    meta = {}
                    reason = "invalid_meta_json"
                cache_version = (meta or {}).get("cache_version")
                if reason is None and _cache_version_num(cache_version) < _cache_version_num(SUBSET_CACHE_VERSION):
                    reason = f"cache_version_lt_{SUBSET_CACHE_VERSION}"
                health = (meta or {}).get("health") or {}
                max_possible = health.get("max_possible_neg_total")
                required = health.get("required_neg_total")
                if reason is None and max_possible is not None and required is not None and int(max_possible) < int(required):
                    reason = "split_feasibility_failed"
            if reason is not None:
                out.append(
                    {
                        "type": "stale_subset",
                        "reason": str(reason),
                        "subset_tag": subset_tag,
                        "cache_version": cache_version,
                        "lspo_path": str(lspo_path),
                        "ads_path": str(ads_path),
                        "meta_path": str(meta_path),
                    }
                )
    return out


def _collect_redundant_run_copies(paths_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    metrics_root = Path(paths_cfg["artifacts"]["metrics_dir"])
    out: list[dict[str, Any]] = []
    if not metrics_root.exists():
        return out
    for ref_file in sorted(metrics_root.glob("*/00_cache_refs.json")):
        payload = _load_json(ref_file)
        run_id = str(payload.get("run_id", ref_file.parent.name))
        for ref in list(payload.get("cache_refs") or []):
            run_path = Path(str(ref.get("run_path", "")))
            shared_path = Path(str(ref.get("shared_path", "")))
            if not run_path.exists() or not shared_path.exists():
                continue
            try:
                if os.path.samefile(run_path, shared_path):
                    continue
            except Exception:
                pass
            if run_path.stat().st_size != shared_path.stat().st_size:
                continue
            try:
                if hash_file(run_path) != hash_file(shared_path):
                    continue
            except Exception:
                continue
            out.append(
                {
                    "type": "redundant_run_copy",
                    "run_id": run_id,
                    "artifact_type": ref.get("artifact_type"),
                    "artifact_id": ref.get("artifact_id"),
                    "run_path": str(run_path),
                    "shared_path": str(shared_path),
                }
            )
    return out


def _collect_shared_path_refs(paths_cfg: dict[str, Any]) -> dict[str, list[str]]:
    metrics_root = Path(paths_cfg["artifacts"]["metrics_dir"])
    refs: dict[str, list[str]] = {}
    if not metrics_root.exists():
        return refs
    for ref_file in sorted(metrics_root.glob("*/00_cache_refs.json")):
        payload = _load_json(ref_file)
        run_id = str(payload.get("run_id", ref_file.parent.name))
        for ref in list(payload.get("cache_refs") or []):
            shared_path = str(ref.get("shared_path", "")).strip()
            if not shared_path:
                continue
            runs = refs.setdefault(shared_path, [])
            if run_id not in runs:
                runs.append(run_id)
    return refs


def _collect_legacy_pair_score_records(paths_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    shared_pair_scores_dir = resolve_shared_cache_root(paths_cfg["data"]) / "pair_scores"
    shared_refs = _collect_shared_path_refs(paths_cfg)
    out: list[dict[str, Any]] = []
    if not shared_pair_scores_dir.exists():
        return out
    for p in sorted(shared_pair_scores_dir.glob("ads_pair_scores_*.parquet")):
        if p.name.startswith("ads_pair_scores_v2_"):
            continue
        referenced_by = list(shared_refs.get(str(p), []))
        out.append(
            {
                "type": "legacy_pair_score",
                "path": str(p),
                "size_bytes": int(p.stat().st_size),
                "referenced_by_runs": referenced_by,
                "referenced": bool(referenced_by),
            }
        )
    return out


def _collect_unused_legacy_pair_score_records(paths_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _collect_legacy_pair_score_records(paths_cfg)
    return [r for r in rows if not bool(r.get("referenced"))]


def cmd_cache_doctor(args):
    paths = _load_paths_cfg(args.paths_config)
    stale_subsets = _collect_stale_subset_records(paths["data"])
    redundant_run_copies = _collect_redundant_run_copies(paths)
    legacy_pair_scores = _collect_legacy_pair_score_records(paths)
    promotable_legacy_hits = [row for row in legacy_pair_scores if bool(row.get("referenced"))]
    payload = {
        "stale_subsets": stale_subsets,
        "redundant_run_copies": redundant_run_copies,
        "legacy_pair_scores_detected": legacy_pair_scores,
        "promotable_legacy_hits": promotable_legacy_hits,
        "counts": {
            "stale_subsets": int(len(stale_subsets)),
            "redundant_run_copies": int(len(redundant_run_copies)),
            "legacy_pair_scores_detected": int(len(legacy_pair_scores)),
            "promotable_legacy_hits": int(len(promotable_legacy_hits)),
        },
    }
    print(json.dumps(payload, indent=2))


def cmd_cache_purge(args):
    paths = _load_paths_cfg(args.paths_config)
    targets = {
        "stale-subsets": _collect_stale_subset_records(paths["data"]),
        "redundant-run-copies": _collect_redundant_run_copies(paths),
        "legacy-pair-scores-unused": _collect_unused_legacy_pair_score_records(paths),
    }
    rows = list(targets[args.target])
    purged = 0
    relinked = 0

    if args.target == "stale-subsets":
        for row in rows:
            for key in ["lspo_path", "ads_path", "meta_path"]:
                p = Path(str(row.get(key, "")))
                if not p.exists():
                    continue
                if args.yes:
                    p.unlink()
                    purged += 1
    elif args.target == "redundant-run-copies":
        for row in rows:
            run_path = Path(str(row["run_path"]))
            shared_path = Path(str(row["shared_path"]))
            if args.yes:
                mode = link_or_copy(shared_path, run_path)
                relinked += int(mode in {"hardlink", "symlink", "existing"})
                purged += 1
    elif args.target == "legacy-pair-scores-unused":
        for row in rows:
            p = Path(str(row.get("path", "")))
            if not p.exists():
                continue
            if args.yes:
                p.unlink()
                purged += 1

    payload = {
        "target": args.target,
        "dry_run": not bool(args.yes),
        "candidates": rows,
        "candidate_count": int(len(rows)),
        "purged_count": int(purged),
        "relinked_count": int(relinked),
    }
    print(json.dumps(payload, indent=2))


def cmd_run_stage(args):
    ui = CliUI(total_steps=11, progress=args.progress)

    try:
        ui.start("Initialize run context")
        _configure_library_noise(args.quiet_libs)
        paths = _load_paths_cfg(args.paths_config)
        data_cfg = paths["data"]
        art_cfg = paths["artifacts"]

        run_cfg_path = args.run_config or f"configs/runs/{args.run_stage}.yaml"
        run_cfg = _load_run_cfg(run_cfg_path)
        run_cfg["stage"] = args.run_stage
        split_assignment_cfg = _resolve_split_assignment_cfg(run_cfg)
        pair_build_cfg = _resolve_pair_build_cfg(run_cfg)

        model_cfg = _load_model_cfg(args.model_config)
        rep_cfg = model_cfg.get("representation", {})
        training_cfg = dict(model_cfg.get("training", {}) or {})
        precision_mode = _resolve_precision_mode(run_cfg=run_cfg, training_cfg=training_cfg)
        if getattr(args, "precision_mode", None):
            precision_mode = str(args.precision_mode).strip().lower()
        training_cfg["precision_mode"] = precision_mode

        project_root = find_project_root(Path.cwd())
        cluster_cfg_path = resolve_existing_path(args.cluster_config, project_root=project_root) or Path(args.cluster_config)
        cluster_cfg = load_yaml(cluster_cfg_path)
        gate_cfg = load_gate_config(args.gates_config) if args.gates_config else None

        run_id = args.run_id or _default_run_id(args.run_stage)
        run_dirs = build_run_dirs(data_cfg, art_cfg, run_id)
        for p in run_dirs.values():
            p.mkdir(parents=True, exist_ok=True)

        latest_context_path = Path(art_cfg["metrics_dir"]) / "latest_run.json"
        write_latest_run_context(
            run_id=run_id,
            run_dirs=run_dirs,
            output_path=latest_context_path,
            stage=args.run_stage,
            extras={"created_utc": datetime.now(timezone.utc).isoformat(), "source": "cli.run-stage"},
        )
        train_seeds = _resolve_train_seeds(args, run_cfg=run_cfg, training_cfg=training_cfg)
        write_json(
            {
                "run_id": run_id,
                "run_stage": args.run_stage,
                "device": args.device,
                "use_stub_embeddings": bool(args.use_stub_embeddings),
                "prefer_precomputed_ads": bool(args.prefer_precomputed_ads),
                "quiet_libs": bool(args.quiet_libs),
                "train_seeds": train_seeds,
                "precision_mode": precision_mode,
                "run_config": str(run_cfg_path),
                "model_config": str(args.model_config),
                "cluster_config": str(cluster_cfg_path),
            },
            Path(run_dirs["metrics"]) / "00_context.json",
        )
        write_run_consistency(
            run_id=run_id,
            run_stage=args.run_stage,
            run_dirs=run_dirs,
            output_path=Path(run_dirs["metrics"]) / "00_run_consistency.json",
            extras={"command": "run-stage", "latest_context_path": str(latest_context_path)},
        )

        stage = args.run_stage
        subset_dir = Path(run_dirs["subset_cache"])
        emb_dir = Path(run_dirs["embeddings"])
        metrics_dir = Path(run_dirs["metrics"])
        checkpoint_dir = Path(run_dirs["checkpoints"])
        pair_score_dir = Path(run_dirs["pair_scores"])
        cluster_dir = Path(run_dirs["clusters"])
        shared_cache_root = resolve_shared_cache_root(data_cfg)
        shared_subsets_dir = Path(run_dirs["shared_subsets"])
        shared_embeddings_dir = Path(run_dirs["shared_embeddings"])
        shared_pairs_dir = Path(run_dirs["shared_pairs"])
        shared_pair_scores_dir = Path(run_dirs["shared_pair_scores"])
        shared_eps_sweeps_dir = Path(run_dirs["shared_eps_sweeps"])

        for p in [shared_cache_root, shared_subsets_dir, shared_embeddings_dir, shared_pairs_dir, shared_pair_scores_dir, shared_eps_sweeps_dir]:
            p.mkdir(parents=True, exist_ok=True)

        lspo_mentions_path = Path(run_dirs["interim"]) / "lspo_mentions.parquet"
        ads_mentions_path = Path(run_dirs["interim"]) / "ads_mentions.parquet"

        lspo_subset_run_path = subset_dir / f"lspo_mentions_{stage}.parquet"
        ads_subset_run_path = subset_dir / f"ads_mentions_{stage}.parquet"
        lspo_pairs_path = subset_dir / f"lspo_pairs_{stage}.parquet"
        ads_pairs_path = subset_dir / f"ads_pairs_{stage}.parquet"

        lspo_chars_path = emb_dir / f"lspo_chars2vec_{stage}.npy"
        lspo_text_path = emb_dir / f"lspo_specter_{stage}.npy"
        ads_chars_path = emb_dir / f"ads_chars2vec_{stage}.npy"
        ads_text_path = emb_dir / f"ads_specter_{stage}.npy"

        train_manifest_path = metrics_dir / "03_train_manifest.json"
        split_meta_path = metrics_dir / "02_split_balance.json"
        pairs_qc_path = metrics_dir / "02_pairs_qc.json"
        pair_scores_path = pair_score_dir / f"ads_pair_scores_{stage}.parquet"

        clusters_path = cluster_dir / f"ads_clusters_{stage}.parquet"
        mention_export_path = cluster_dir / f"mention_author_uid_{stage}.parquet"
        publication_export_path = cluster_dir / f"publication_authors_{stage}.parquet"
        cluster_qc_path = metrics_dir / "04_cluster_qc.json"
        cluster_cfg_used_path = metrics_dir / "04_clustering_config_used.json"

        stage_metrics_path = metrics_dir / f"05_stage_metrics_{stage}.json"
        go_no_go_path = metrics_dir / f"05_go_no_go_{stage}.json"
        compare_path = metrics_dir / "99_compare_to_baseline.json"
        cache_refs_path = metrics_dir / "00_cache_refs.json"
        cache_refs: list[dict[str, Any]] = []

        ui.done(f"Run ID: {run_id}")

        ui.start("Prepare LSPO mentions")
        if lspo_mentions_path.exists() and not args.force:
            lspo_mentions = read_parquet(lspo_mentions_path)
            ui.skip(f"Loaded {len(lspo_mentions)} mentions from cache.")
        else:
            lspo_mentions = prepare_lspo_mentions(
                parquet_path=data_cfg["raw_lspo_parquet"],
                h5_path=data_cfg.get("raw_lspo_h5"),
                output_path=lspo_mentions_path,
            )
            ui.done(f"Prepared {len(lspo_mentions)} mentions.")

        ui.start("Prepare ADS mentions")
        if ads_mentions_path.exists() and not args.force:
            ads_mentions = read_parquet(ads_mentions_path)
            ui.skip(f"Loaded {len(ads_mentions)} mentions from cache.")
        else:
            ads_mentions = prepare_ads_mentions(
                publications_path=data_cfg["raw_ads_publications"],
                references_path=data_cfg["raw_ads_references"],
                output_path=ads_mentions_path,
            )
            ui.done(f"Prepared {len(ads_mentions)} mentions.")

        ui.start("Build or load stage subsets")
        t_all = perf_counter()
        timings: dict[str, float] = {}

        source_fp = compute_source_fp(lspo_mentions_path, ads_mentions_path)
        subset_identity = compute_subset_identity(run_cfg=run_cfg, source_fp=source_fp, sampler_version=SUBSET_CACHE_VERSION)
        subset_paths = resolve_shared_subset_paths(data_cfg=data_cfg, identity=subset_identity)
        manifest_paths = resolve_manifest_paths(
            run_id=run_id,
            manifest_dir=Path(run_dirs["subset_manifests"]),
            identity=subset_identity,
            run_stage=stage,
        )
        subset_paths.shared_dir.mkdir(parents=True, exist_ok=True)
        split_balance_cfg = run_cfg.get("split_balance", {})
        subset_meta_path = _subset_meta_path(subset_paths.shared_dir, subset_identity.subset_tag)

        cache_hit = False
        cache_valid: bool | None = None
        cache_invalid_reason: str | None = None
        cache_rebuilt = False
        cache_source = "none"
        cache_health: dict[str, Any] = {}

        if not args.force and subset_paths.lspo_shared.exists() and subset_paths.ads_shared.exists():
            cache_source = "shared"
        elif (
            not args.force
            and subset_paths.lspo_shared_legacy is not None
            and subset_paths.ads_shared_legacy is not None
            and subset_paths.lspo_shared_legacy.exists()
            and subset_paths.ads_shared_legacy.exists()
        ):
            cache_source = "legacy_shared"

        if cache_source != "none":
            cache_hit = True
            t0 = perf_counter()
            lspo_cache_path = subset_paths.lspo_shared if cache_source == "shared" else subset_paths.lspo_shared_legacy
            ads_cache_path = subset_paths.ads_shared if cache_source == "shared" else subset_paths.ads_shared_legacy
            assert lspo_cache_path is not None
            assert ads_cache_path is not None
            lspo_subset = read_parquet(lspo_cache_path)
            t1 = perf_counter()
            ads_subset = read_parquet(ads_cache_path)
            t2 = perf_counter()
            timings["read_lspo_s"] = t1 - t0
            timings["read_ads_s"] = t2 - t1
            cache_valid, reasons, cache_health = _validate_cached_subset(
                lspo_subset=lspo_subset,
                ads_subset=ads_subset,
                run_cfg=run_cfg,
                stage=stage,
                split_assignment_cfg=split_assignment_cfg,
                split_balance_cfg=split_balance_cfg,
            )
            if not cache_valid:
                cache_invalid_reason = "; ".join(reasons)
                cache_rebuilt = True
                cache_hit = False
            elif cache_source == "legacy_shared":
                atomic_save_parquet(lspo_subset, subset_paths.lspo_shared, index=False)
                atomic_save_parquet(ads_subset, subset_paths.ads_shared, index=False)
                cache_source = "shared_migrated"

        if cache_source == "none" or cache_rebuilt:
            t0 = perf_counter()
            lspo_subset = build_stage_subset(
                lspo_mentions,
                stage=stage,
                seed=int(run_cfg.get("seed", 11)),
                target_mentions=run_cfg.get("subset_target_mentions"),
                subset_sampling=run_cfg.get("subset_sampling", {}),
            )
            t1 = perf_counter()
            ads_subset = build_stage_subset(
                ads_mentions,
                stage=stage,
                seed=int(run_cfg.get("seed", 11)),
                target_mentions=run_cfg.get("subset_target_mentions"),
                subset_sampling=run_cfg.get("subset_sampling", {}),
            )
            t2 = perf_counter()
            atomic_save_parquet(lspo_subset, subset_paths.lspo_shared, index=False)
            t3 = perf_counter()
            atomic_save_parquet(ads_subset, subset_paths.ads_shared, index=False)
            t4 = perf_counter()
            timings["build_lspo_s"] = t1 - t0
            timings["build_ads_s"] = t2 - t1
            timings["save_lspo_shared_s"] = t3 - t2
            timings["save_ads_shared_s"] = t4 - t3
            cache_valid = True
            _, _, cache_health = _validate_cached_subset(
                lspo_subset=lspo_subset,
                ads_subset=ads_subset,
                run_cfg=run_cfg,
                stage=stage,
                split_assignment_cfg=split_assignment_cfg,
                split_balance_cfg=split_balance_cfg,
            )

        write_json(
            {
                "cache_version": SUBSET_CACHE_VERSION,
                "cache_key": subset_identity.subset_tag,
                "sampler_version": subset_identity.sampler_version,
                "run_stage": stage,
                "source_fp": subset_identity.source_fp,
                "subset_target_mentions": run_cfg.get("subset_target_mentions"),
                "health": cache_health,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
            subset_meta_path,
        )

        t5 = perf_counter()
        lspo_subset_link_mode = link_or_copy(subset_paths.lspo_shared, lspo_subset_run_path)
        t6 = perf_counter()
        ads_subset_link_mode = link_or_copy(subset_paths.ads_shared, ads_subset_run_path)
        t7 = perf_counter()
        timings["save_lspo_run_s"] = t6 - t5
        timings["save_ads_run_s"] = t7 - t6
        _record_cache_ref(
            cache_refs,
            artifact_type="subset_lspo",
            artifact_id=subset_identity.subset_tag,
            shared_path=subset_paths.lspo_shared,
            run_path=lspo_subset_run_path,
            mode=lspo_subset_link_mode,
        )
        _record_cache_ref(
            cache_refs,
            artifact_type="subset_ads",
            artifact_id=subset_identity.subset_tag,
            shared_path=subset_paths.ads_shared,
            run_path=ads_subset_run_path,
            mode=ads_subset_link_mode,
        )

        if args.force or not manifest_paths.lspo_primary.exists():
            write_subset_manifest(lspo_subset, manifest_paths.lspo_primary)
        if args.force or not manifest_paths.ads_primary.exists():
            write_subset_manifest(ads_subset, manifest_paths.ads_primary)

        timings["total_s"] = perf_counter() - t_all

        subset_summary = build_subset_summary(
            run_id=run_id,
            stage=stage,
            source_fp=subset_identity.source_fp,
            subset_tag=subset_identity.subset_tag,
            cache_key=subset_identity.subset_tag,
            cache_hit=cache_hit,
            cache_valid=cache_valid,
            cache_invalid_reason=cache_invalid_reason,
            cache_rebuilt=cache_rebuilt,
            cache_version=SUBSET_CACHE_VERSION,
            lspo_subset=lspo_subset,
            ads_subset=ads_subset,
            timings=timings,
        )
        write_json(subset_summary, metrics_dir / "01_subset_summary.json")
        write_run_consistency(
            run_id=run_id,
            run_stage=stage,
            run_dirs=run_dirs,
            output_path=metrics_dir / "01_run_consistency.json",
            extras={"subset_tag": subset_identity.subset_tag, "cache_hit": cache_hit},
        )
        if cache_hit and not cache_rebuilt:
            ui.skip(f"Shared cache hit: {subset_identity.subset_tag}")
        elif cache_rebuilt:
            ui.done(f"Invalid cache rebuilt: {cache_invalid_reason}")
        else:
            ui.done(f"Built subsets ({len(lspo_subset)} LSPO / {len(ads_subset)} ADS).")

        ui.start("Build or load embeddings")
        representation_cfg_hash = stable_hash(rep_cfg)
        model_version = str(model_cfg.get("name", "nand"))
        embedding_id = stable_hash(
            {
                "subset_id": subset_identity.subset_tag,
                "representation_cfg_hash": representation_cfg_hash,
                "model_version": model_version,
            }
        )
        lspo_chars_shared_path = shared_embeddings_dir / f"lspo_chars2vec_{embedding_id}.npy"
        lspo_text_shared_path = shared_embeddings_dir / f"lspo_specter_{embedding_id}.npy"
        ads_chars_shared_path = shared_embeddings_dir / f"ads_chars2vec_{embedding_id}.npy"
        ads_text_shared_path = shared_embeddings_dir / f"ads_specter_{embedding_id}.npy"

        emb_cache_hit = (
            lspo_chars_shared_path.exists()
            and lspo_text_shared_path.exists()
            and ads_chars_shared_path.exists()
            and ads_text_shared_path.exists()
            and not args.force
        )

        lspo_chars = get_or_create_chars2vec_embeddings(
            mentions=lspo_subset,
            output_path=lspo_chars_shared_path,
            force_recompute=args.force,
            use_stub_if_missing=args.use_stub_embeddings,
            quiet_libraries=args.quiet_libs,
        )
        lspo_text = get_or_create_specter_embeddings(
            mentions=lspo_subset,
            output_path=lspo_text_shared_path,
            force_recompute=args.force,
            model_name=rep_cfg.get("text_model_name", "allenai/specter"),
            max_length=int(rep_cfg.get("max_length", 256)),
            batch_size=16,
            device=args.device,
            prefer_precomputed=False,
            use_stub_if_missing=args.use_stub_embeddings,
            show_progress=args.progress,
            quiet_libraries=args.quiet_libs,
            reuse_model=True,
        )
        ads_chars = get_or_create_chars2vec_embeddings(
            mentions=ads_subset,
            output_path=ads_chars_shared_path,
            force_recompute=args.force,
            use_stub_if_missing=args.use_stub_embeddings,
            quiet_libraries=args.quiet_libs,
        )
        ads_text = get_or_create_specter_embeddings(
            mentions=ads_subset,
            output_path=ads_text_shared_path,
            force_recompute=args.force,
            model_name=rep_cfg.get("text_model_name", "allenai/specter"),
            max_length=int(rep_cfg.get("max_length", 256)),
            batch_size=32,
            device=args.device,
            prefer_precomputed=args.prefer_precomputed_ads,
            use_stub_if_missing=args.use_stub_embeddings,
            show_progress=args.progress,
            quiet_libraries=args.quiet_libs,
            reuse_model=True,
        )
        _record_cache_ref(
            cache_refs,
            artifact_type="embedding_lspo_chars",
            artifact_id=embedding_id,
            shared_path=lspo_chars_shared_path,
            run_path=lspo_chars_path,
            mode=link_or_copy(lspo_chars_shared_path, lspo_chars_path),
        )
        _record_cache_ref(
            cache_refs,
            artifact_type="embedding_lspo_text",
            artifact_id=embedding_id,
            shared_path=lspo_text_shared_path,
            run_path=lspo_text_path,
            mode=link_or_copy(lspo_text_shared_path, lspo_text_path),
        )
        _record_cache_ref(
            cache_refs,
            artifact_type="embedding_ads_chars",
            artifact_id=embedding_id,
            shared_path=ads_chars_shared_path,
            run_path=ads_chars_path,
            mode=link_or_copy(ads_chars_shared_path, ads_chars_path),
        )
        _record_cache_ref(
            cache_refs,
            artifact_type="embedding_ads_text",
            artifact_id=embedding_id,
            shared_path=ads_text_shared_path,
            run_path=ads_text_path,
            mode=link_or_copy(ads_text_shared_path, ads_text_path),
        )

        if emb_cache_hit:
            ui.skip("Reused cached embeddings.")
        else:
            ui.done(
                f"Embeddings ready (LSPO {tuple(lspo_chars.shape)}/{tuple(lspo_text.shape)}, "
                f"ADS {tuple(ads_chars.shape)}/{tuple(ads_text.shape)})."
            )

        ui.start("Assign LSPO splits and build LSPO pairs")
        pair_cfg_hash = stable_hash(
            {
                "max_pairs_per_block": run_cfg.get("max_pairs_per_block"),
                "exclude_same_bibcode": bool(pair_build_cfg["exclude_same_bibcode"]),
            }
        )
        split_cfg_hash = stable_hash(
            {
                "split_assignment": split_assignment_cfg,
                "split_balance": split_balance_cfg,
                "seed": int(run_cfg.get("seed", 11)),
            }
        )
        lspo_pairs_id = stable_hash(
            {
                "subset_id": subset_identity.subset_tag,
                "split_cfg_hash": split_cfg_hash,
                "pair_cfg_hash": pair_cfg_hash,
            }
        )
        ads_pairs_id = stable_hash({"subset_id": subset_identity.subset_tag, "pair_cfg_hash": pair_cfg_hash})
        lspo_split_shared_path = shared_pairs_dir / f"lspo_mentions_split_{lspo_pairs_id}.parquet"
        lspo_pairs_shared_path = shared_pairs_dir / f"lspo_pairs_{lspo_pairs_id}.parquet"
        split_meta_shared_path = shared_pairs_dir / f"split_balance_{lspo_pairs_id}.json"
        ads_pairs_shared_path = shared_pairs_dir / f"ads_pairs_{ads_pairs_id}.parquet"
        pairs_qc_shared_path = shared_pairs_dir / f"pairs_qc_{lspo_pairs_id}_{ads_pairs_id}.json"

        if (
            lspo_pairs_shared_path.exists()
            and split_meta_shared_path.exists()
            and lspo_split_shared_path.exists()
            and not args.force
        ):
            lspo_mentions_split = read_parquet(lspo_split_shared_path)
            lspo_pairs = read_parquet(lspo_pairs_shared_path)
            split_meta = _load_json(split_meta_shared_path)
            lspo_pair_meta: dict[str, Any] = {}
            ui.skip(f"Reused LSPO split+pairs ({len(lspo_pairs)} pairs).")
        else:
            lspo_mentions_split, split_meta = assign_lspo_splits(
                lspo_subset,
                seed=int(run_cfg.get("seed", 11)),
                train_ratio=float(split_assignment_cfg["train_ratio"]),
                val_ratio=float(split_assignment_cfg["val_ratio"]),
                min_neg_val=int(split_balance_cfg.get("min_neg_val", 0)),
                min_neg_test=int(split_balance_cfg.get("min_neg_test", 0)),
                max_attempts=int(split_balance_cfg.get("max_attempts", 1)),
                return_meta=True,
            )
            atomic_save_parquet(lspo_mentions_split, lspo_split_shared_path, index=False)
            write_json(split_meta, split_meta_shared_path)
            if str(split_meta.get("status", "")).strip().lower() == "split_balance_infeasible":
                link_or_copy(lspo_split_shared_path, lspo_subset_run_path)
                link_or_copy(split_meta_shared_path, split_meta_path)
                raise RuntimeError(
                    "split_balance_infeasible: "
                    f"max_possible_neg_total={split_meta.get('max_possible_neg_total')} "
                    f"< required_neg_total={split_meta.get('required_neg_total')}"
                )
            lspo_pairs, lspo_pair_meta = build_pairs_within_blocks(
                mentions=lspo_mentions_split,
                max_pairs_per_block=run_cfg.get("max_pairs_per_block"),
                seed=int(run_cfg.get("seed", 11)),
                require_same_split=True,
                labeled_only=False,
                balance_train=True,
                exclude_same_bibcode=bool(pair_build_cfg["exclude_same_bibcode"]),
                show_progress=args.progress,
                return_meta=True,
            )
            write_pairs(lspo_pairs, lspo_pairs_shared_path)
            ui.done(f"Built LSPO pairs ({len(lspo_pairs)} rows).")
        lspo_split_link_mode = link_or_copy(lspo_split_shared_path, lspo_subset_run_path)
        lspo_pairs_link_mode = link_or_copy(lspo_pairs_shared_path, lspo_pairs_path)
        split_meta_link_mode = link_or_copy(split_meta_shared_path, split_meta_path)
        _record_cache_ref(
            cache_refs,
            artifact_type="lspo_split",
            artifact_id=lspo_pairs_id,
            shared_path=lspo_split_shared_path,
            run_path=lspo_subset_run_path,
            mode=lspo_split_link_mode,
        )
        _record_cache_ref(
            cache_refs,
            artifact_type="lspo_pairs",
            artifact_id=lspo_pairs_id,
            shared_path=lspo_pairs_shared_path,
            run_path=lspo_pairs_path,
            mode=lspo_pairs_link_mode,
        )
        _record_cache_ref(
            cache_refs,
            artifact_type="split_meta",
            artifact_id=lspo_pairs_id,
            shared_path=split_meta_shared_path,
            run_path=split_meta_path,
            mode=split_meta_link_mode,
        )

        if str(split_meta.get("status", "")).strip().lower() == "split_balance_infeasible":
            raise RuntimeError(
                "split_balance_infeasible: "
                f"max_possible_neg_total={split_meta.get('max_possible_neg_total')} "
                f"< required_neg_total={split_meta.get('required_neg_total')}"
            )

        ui.start("Build ADS pairs and pair QC")
        if ads_pairs_shared_path.exists() and pairs_qc_shared_path.exists() and not args.force:
            ads_pairs = read_parquet(ads_pairs_shared_path)
            pairs_qc = _load_json(pairs_qc_shared_path)
            lspo_pair_meta = dict(pairs_qc.get("lspo_pair_build", lspo_pair_meta))
            ads_pair_meta = dict(pairs_qc.get("ads_pair_build", {}))
            ui.skip(f"Reused ADS pairs ({len(ads_pairs)} rows).")
        else:
            ads_pairs, ads_pair_meta = build_pairs_within_blocks(
                mentions=ads_subset,
                max_pairs_per_block=run_cfg.get("max_pairs_per_block"),
                seed=int(run_cfg.get("seed", 11)),
                require_same_split=False,
                labeled_only=False,
                balance_train=False,
                exclude_same_bibcode=bool(pair_build_cfg["exclude_same_bibcode"]),
                show_progress=args.progress,
                return_meta=True,
            )
            write_pairs(ads_pairs, ads_pairs_shared_path)
            pairs_qc = build_pairs_qc(
                lspo_mentions=lspo_mentions_split,
                lspo_pairs=lspo_pairs,
                ads_pairs=ads_pairs,
                split_meta=split_meta,
                lspo_pair_build_meta=lspo_pair_meta,
                ads_pair_build_meta=ads_pair_meta,
            )
            write_json(pairs_qc, pairs_qc_shared_path)
            ui.done(f"Built ADS pairs ({len(ads_pairs)} rows).")
        ads_pairs_link_mode = link_or_copy(ads_pairs_shared_path, ads_pairs_path)
        pairs_qc_link_mode = link_or_copy(pairs_qc_shared_path, pairs_qc_path)
        _record_cache_ref(
            cache_refs,
            artifact_type="ads_pairs",
            artifact_id=ads_pairs_id,
            shared_path=ads_pairs_shared_path,
            run_path=ads_pairs_path,
            mode=ads_pairs_link_mode,
        )
        _record_cache_ref(
            cache_refs,
            artifact_type="pairs_qc",
            artifact_id=f"{lspo_pairs_id}_{ads_pairs_id}",
            shared_path=pairs_qc_shared_path,
            run_path=pairs_qc_path,
            mode=pairs_qc_link_mode,
        )

        write_run_consistency(
            run_id=run_id,
            run_stage=stage,
            run_dirs=run_dirs,
            output_path=metrics_dir / "02_run_consistency.json",
            extras={"split_status": split_meta.get("status")},
        )

        ui.start("Train NAND model")
        train_cache_hit = False
        if train_manifest_path.exists() and not args.force:
            train_manifest = _load_json(train_manifest_path)
            best_ckpt = Path(str(train_manifest.get("best_checkpoint", "")))
            if best_ckpt.exists():
                train_cache_hit = True
        if train_cache_hit:
            ui.skip(f"Reused train manifest: {train_manifest.get('best_checkpoint')}")
        else:
            train_manifest = train_nand_across_seeds(
                mentions=lspo_mentions_split,
                pairs=lspo_pairs,
                chars2vec=lspo_chars,
                text_emb=lspo_text,
                model_config=training_cfg,
                seeds=train_seeds,
                run_id=run_id,
                output_dir=checkpoint_dir,
                metrics_output=train_manifest_path,
                device=args.device,
                precision_mode=precision_mode,
                show_progress=args.progress,
            )
            ui.done(f"Best checkpoint: {train_manifest['best_checkpoint']}")

        write_run_consistency(
            run_id=run_id,
            run_stage=stage,
            run_dirs=run_dirs,
            output_path=metrics_dir / "03_run_consistency.json",
            extras={"best_checkpoint": str(train_manifest.get("best_checkpoint"))},
        )

        ui.start("Score ADS pairs")
        score_pipeline_version = "v2"
        try:
            model_state_hash = hash_checkpoint_model_state(
                train_manifest["best_checkpoint"],
                score_pipeline_version=score_pipeline_version,
            )
        except Exception as exc:
            warnings.warn(
                (
                    "Model-state hash failed "
                    f"({exc.__class__.__name__}); falling back to file hash for pair-score cache key."
                ),
                RuntimeWarning,
            )
            model_state_hash = hash_file(train_manifest["best_checkpoint"])

        checkpoint_hash = hash_file(train_manifest["best_checkpoint"])
        score_cfg_hash = stable_hash({"score_batch_size": int(args.score_batch_size)})
        pair_scores_id_v2 = stable_hash(
            {
                "ads_pairs_id": ads_pairs_id,
                "model_state_hash": model_state_hash,
                "score_cfg_hash": score_cfg_hash,
                "score_pipeline_version": score_pipeline_version,
            }
        )
        pair_scores_id_legacy = stable_hash(
            {
                "ads_pairs_id": ads_pairs_id,
                "checkpoint_hash": checkpoint_hash,
                "score_cfg_hash": score_cfg_hash,
            }
        )
        pair_scores_shared_path_v2 = shared_pair_scores_dir / f"ads_pair_scores_v2_{pair_scores_id_v2}.parquet"
        pair_scores_shared_path_legacy = shared_pair_scores_dir / f"ads_pair_scores_{pair_scores_id_legacy}.parquet"
        pair_scores_shared_path_used = pair_scores_shared_path_v2
        pair_scores_cache_schema_version = "v2"
        pair_scores_artifact_id = pair_scores_id_v2

        if pair_scores_shared_path_v2.exists() and not args.force:
            pair_scores = read_parquet(pair_scores_shared_path_v2)
            ui.skip(f"Reused pair scores ({len(pair_scores)} rows).")
        elif pair_scores_shared_path_legacy.exists() and not args.force:
            pair_scores = read_parquet(pair_scores_shared_path_legacy)
            pair_scores_cache_schema_version = "v1"
            pair_scores_artifact_id = pair_scores_id_legacy
            pair_scores_shared_path_used = pair_scores_shared_path_legacy
            promote_mode = link_or_copy(pair_scores_shared_path_legacy, pair_scores_shared_path_v2)
            if pair_scores_shared_path_v2.exists():
                pair_scores_shared_path_used = pair_scores_shared_path_v2
                pair_scores_cache_schema_version = "v2"
                pair_scores_artifact_id = pair_scores_id_v2
            ui.skip(f"Reused legacy pair scores ({len(pair_scores)} rows, promote={promote_mode}).")
        else:
            pair_scores = score_pairs_with_checkpoint(
                mentions=ads_subset,
                pairs=ads_pairs,
                chars2vec=ads_chars,
                text_emb=ads_text,
                checkpoint_path=train_manifest["best_checkpoint"],
                output_path=pair_scores_shared_path_v2,
                batch_size=int(args.score_batch_size),
                device=args.device,
                precision_mode=precision_mode,
                show_progress=args.progress,
            )
            pair_scores_shared_path_used = pair_scores_shared_path_v2
            pair_scores_cache_schema_version = "v2"
            pair_scores_artifact_id = pair_scores_id_v2
            ui.done(f"Scored {len(pair_scores)} ADS pairs.")
        _record_cache_ref(
            cache_refs,
            artifact_type="pair_scores",
            artifact_id=pair_scores_artifact_id,
            shared_path=pair_scores_shared_path_used,
            run_path=pair_scores_path,
            mode=link_or_copy(pair_scores_shared_path_used, pair_scores_path),
            cache_schema_version=pair_scores_cache_schema_version,
        )
        write_json({"run_id": run_id, "cache_refs": cache_refs}, cache_refs_path)

        ui.start("Cluster ADS mentions and export mappings")
        cluster_cache_hit = (
            clusters_path.exists()
            and mention_export_path.exists()
            and publication_export_path.exists()
            and cluster_qc_path.exists()
            and cluster_cfg_used_path.exists()
            and not args.force
        )

        eps_meta: dict[str, Any] = {}
        if cluster_cache_hit:
            clusters = read_parquet(clusters_path)
            cluster_qc = _load_json(cluster_qc_path)
            cfg_payload = _load_json(cluster_cfg_used_path) or {}
            eps_meta = dict(cfg_payload.get("eps_resolution", {}) or {})
            ui.skip(f"Reused clustering outputs ({len(clusters)} mentions).")
        else:
            best_threshold = float(train_manifest["best_threshold"])
            cluster_cfg_used = json.loads(json.dumps(cluster_cfg))
            sweep_cfg_hash = stable_hash(
                {
                    "eps_mode": cluster_cfg_used.get("eps_mode"),
                    "eps_sweep_min": cluster_cfg_used.get("eps_sweep_min"),
                    "eps_sweep_max": cluster_cfg_used.get("eps_sweep_max"),
                    "eps_sweep_step": cluster_cfg_used.get("eps_sweep_step"),
                    "boundary_diagnostics": cluster_cfg_used.get("boundary_diagnostics"),
                    "min_samples": cluster_cfg_used.get("min_samples"),
                    "constraint_mode": cluster_cfg_used.get("constraint_mode"),
                    "constraints": cluster_cfg_used.get("constraints"),
                }
            )
            eps_sweep_id = stable_hash(
                {
                    "lspo_pairs_id": lspo_pairs_id,
                    "model_state_hash": model_state_hash,
                    "sweep_cfg_hash": sweep_cfg_hash,
                }
            )
            resolved_eps, eps_meta = _resolve_stage_eps(
                cluster_cfg=cluster_cfg_used,
                best_threshold=best_threshold,
                lspo_mentions_split=lspo_mentions_split,
                lspo_pairs=lspo_pairs,
                lspo_chars=lspo_chars,
                lspo_text=lspo_text,
                checkpoint_path=str(train_manifest["best_checkpoint"]),
                score_batch_size=int(args.score_batch_size),
                device=args.device,
                precision_mode=precision_mode,
                show_progress=args.progress,
            )
            eps_meta["eps_sweep_id"] = eps_sweep_id
            eps_sweep_shared_path = shared_eps_sweeps_dir / f"eps_sweep_{eps_sweep_id}.json"
            write_json(
                {
                    "eps_sweep_id": eps_sweep_id,
                    "run_stage": stage,
                    "run_id": run_id,
                    "model_state_hash": model_state_hash,
                    "sweep_cfg_hash": sweep_cfg_hash,
                    "eps_resolution": eps_meta,
                },
                eps_sweep_shared_path,
            )
            cluster_cfg_used["eps"] = resolved_eps
            if eps_meta.get("selected_eps") is not None:
                cluster_cfg_used["selected_eps"] = float(eps_meta["selected_eps"])
            write_json(
                {
                    "run_id": run_id,
                    "run_stage": stage,
                    "best_threshold": best_threshold,
                    "eps_resolution": eps_meta,
                    "cluster_config_used": cluster_cfg_used,
                },
                cluster_cfg_used_path,
            )

            clusters = cluster_blockwise_dbscan(
                mentions=ads_subset,
                pair_scores=pair_scores,
                cluster_config=cluster_cfg_used,
                output_path=clusters_path,
                show_progress=args.progress,
            )
            clusters.to_parquet(mention_export_path, index=False)
            _ = build_publication_author_mapping(
                mentions=ads_subset,
                clusters=clusters,
                output_path=publication_export_path,
            )
            cluster_qc = build_cluster_qc(
                pair_scores=pair_scores,
                clusters=clusters,
                threshold=best_threshold,
            )
            write_json(cluster_qc, cluster_qc_path)
            ui.done(f"Clustered {len(clusters)} ADS mentions.")

        write_run_consistency(
            run_id=run_id,
            run_stage=stage,
            run_dirs=run_dirs,
            output_path=metrics_dir / "04_run_consistency.json",
            extras={"cluster_count": int(cluster_qc.get("cluster_count", 0))},
        )

        ui.start("Build stage metrics and go/no-go")
        write_run_consistency(
            run_id=run_id,
            run_stage=stage,
            run_dirs=run_dirs,
            output_path=metrics_dir / "05_run_consistency.json",
            extras={"command": "run-stage"},
        )

        if stage_metrics_path.exists() and go_no_go_path.exists() and not args.force:
            stage_metrics = _load_json(stage_metrics_path)
            go = _load_json(go_no_go_path)
            ui.skip(f"Reused stage reports (GO={go.get('go')}).")
        else:
            if manifest_paths.lspo_primary.exists() and manifest_paths.ads_primary.exists():
                determinism_paths = [manifest_paths.lspo_primary, manifest_paths.ads_primary]
            else:
                determinism_paths = [manifest_paths.lspo_legacy, manifest_paths.ads_legacy]
            consistency_files = [metrics_dir / f"{i:02d}_run_consistency.json" for i in range(0, 6)]

            stage_metrics = build_stage_metrics(
                run_id=run_id,
                run_stage=stage,
                lspo_mentions=lspo_subset,
                lspo_pairs_count=int(len(lspo_pairs)),
                ads_mentions=ads_subset,
                clusters=clusters,
                train_manifest=train_manifest,
                consistency_files=consistency_files,
                determinism_paths=determinism_paths,
                cluster_qc=cluster_qc,
                split_meta=split_meta,
                eps_meta=eps_meta,
                subset_cache_key=subset_identity.subset_tag,
            )
            write_json(stage_metrics, stage_metrics_path)

            go = evaluate_go_no_go(stage_metrics, gate_config=gate_cfg)
            write_go_no_go_report(go, go_no_go_path)
            ui.done(f"GO={go['go']} with blockers={len(go.get('blockers', []))}.")

        if args.baseline_run_id:
            if compare_path.exists() and not args.force:
                ui.info(f"Reused baseline comparison: {compare_path}")
            else:
                write_compare_to_baseline(
                    baseline_run_id=args.baseline_run_id,
                    current_run_id=run_id,
                    run_stage=stage,
                    metrics_root=art_cfg["metrics_dir"],
                    output_path=compare_path,
                )
                ui.info(f"Wrote baseline comparison: {compare_path}")

        ui.info(f"Stage metrics: {stage_metrics_path}")
        ui.info(f"Go/No-Go report: {go_no_go_path}")
        ui.info(f"Run complete: {run_id}")

    except Exception as exc:
        ui.fail(str(exc))
        raise
    finally:
        ui.close()


def cmd_run_infer_ads(args):
    ui = CliUI(total_steps=7, progress=args.progress)

    try:
        ui.start("Initialize run context")
        _configure_library_noise(args.quiet_libs)
        paths = _load_paths_cfg(args.paths_config)
        data_cfg = paths["data"]
        art_cfg = paths["artifacts"]

        project_root = find_project_root(Path.cwd())
        cluster_cfg_path = resolve_existing_path(args.cluster_config, project_root=project_root) or Path(args.cluster_config)
        cluster_cfg = load_yaml(cluster_cfg_path)
        gate_cfg = load_gate_config("configs/gates.yaml")

        dataset_info = _resolve_ads_dataset_files(data_cfg, args.dataset_id)
        model_info = _resolve_model_run_for_inference(paths_cfg=paths, model_run_id=args.model_run_id)
        run_id = args.run_id or _default_run_id("infer_ads")
        stage = "infer_ads"
        run_dirs = build_run_dirs(data_cfg, art_cfg, run_id)
        for p in run_dirs.values():
            p.mkdir(parents=True, exist_ok=True)

        latest_context_path = Path(art_cfg["metrics_dir"]) / "latest_run.json"
        write_latest_run_context(
            run_id=run_id,
            run_dirs=run_dirs,
            output_path=latest_context_path,
            stage=stage,
            extras={"created_utc": datetime.now(timezone.utc).isoformat(), "source": "cli.run-infer-ads"},
        )

        metrics_dir = Path(run_dirs["metrics"])
        emb_dir = Path(run_dirs["embeddings"])
        subset_dir = Path(run_dirs["subset_cache"])
        pair_score_dir = Path(run_dirs["pair_scores"])
        cluster_dir = Path(run_dirs["clusters"])
        interim_dir = Path(run_dirs["interim"])

        shared_embeddings_dir = Path(run_dirs["shared_embeddings"])
        shared_pairs_dir = Path(run_dirs["shared_pairs"])
        shared_pair_scores_dir = Path(run_dirs["shared_pair_scores"])
        for p in [shared_embeddings_dir, shared_pairs_dir, shared_pair_scores_dir]:
            p.mkdir(parents=True, exist_ok=True)

        dataset_tag = str(dataset_info["dataset_tag"])
        dataset_source_fp = str(dataset_info["dataset_source_fp"])
        ads_mentions_path = interim_dir / f"ads_mentions_{dataset_tag}.parquet"
        ads_pairs_path = subset_dir / "ads_pairs_infer_ads.parquet"
        ads_chars_path = emb_dir / "ads_chars2vec_infer_ads.npy"
        ads_text_path = emb_dir / "ads_specter_infer_ads.npy"
        pair_scores_path = pair_score_dir / "ads_pair_scores_infer_ads.parquet"
        clusters_path = cluster_dir / "ads_clusters_infer_ads.parquet"
        publication_export_path = cluster_dir / "publication_authors_infer_ads.parquet"

        input_summary_path = metrics_dir / "01_input_summary.json"
        pairs_qc_path = metrics_dir / "03_pairs_qc.json"
        cluster_qc_path = metrics_dir / "04_cluster_qc.json"
        cluster_cfg_used_path = metrics_dir / "04_clustering_config_used.json"
        stage_metrics_path = metrics_dir / "05_stage_metrics_infer_ads.json"
        go_no_go_path = metrics_dir / "05_go_no_go_infer_ads.json"
        cache_refs_path = metrics_dir / "00_cache_refs.json"
        cache_refs: list[dict[str, Any]] = []

        write_json(
            {
                "run_id": run_id,
                "run_stage": stage,
                "dataset_id": dataset_info["dataset_id"],
                "dataset_dir": str(dataset_info["dataset_dir"]),
                "publications_path": str(dataset_info["publications_path"]),
                "references_path": str(dataset_info["references_path"]) if dataset_info["references_path"] is not None else None,
                "references_present": bool(dataset_info["references_present"]),
                "model_run_id": model_info["model_run_id"],
                "model_train_manifest": str(model_info["train_manifest_path"]),
                "checkpoint": str(model_info["best_checkpoint"]),
                "selected_eps": float(model_info["selected_eps"]),
                "best_threshold": float(model_info["best_threshold"]),
                "model_config": model_info["model_cfg_path"],
                "cluster_config": str(cluster_cfg_path),
                "score_batch_size": int(args.score_batch_size),
                "device": args.device,
                "precision_mode": str(args.precision_mode),
                "quiet_libs": bool(args.quiet_libs),
            },
            metrics_dir / "00_context.json",
        )
        write_run_consistency(
            run_id=run_id,
            run_stage=stage,
            run_dirs=run_dirs,
            output_path=metrics_dir / "00_run_consistency.json",
            extras={"command": "run-infer-ads", "latest_context_path": str(latest_context_path)},
        )
        ui.done(f"Run ID: {run_id}")

        ui.start("Prepare ADS mentions")
        if ads_mentions_path.exists() and not args.force:
            ads_mentions = read_parquet(ads_mentions_path)
            ui.skip(f"Loaded {len(ads_mentions)} mentions from cache.")
        else:
            ads_mentions = prepare_ads_mentions(
                publications_path=dataset_info["publications_path"],
                references_path=dataset_info["references_path"],
                output_path=ads_mentions_path,
            )
            ui.done(f"Prepared {len(ads_mentions)} ADS mentions.")
        if not bool(dataset_info["references_present"]):
            ui.warn("No references file found; continuing with publications-only input.")

        write_json(
            {
                "run_id": run_id,
                "run_stage": stage,
                "dataset_id": dataset_info["dataset_id"],
                "dataset_dir": str(dataset_info["dataset_dir"]),
                "publications_path": str(dataset_info["publications_path"]),
                "references_path": str(dataset_info["references_path"]) if dataset_info["references_path"] is not None else None,
                "references_present": bool(dataset_info["references_present"]),
                "dataset_source_fp": dataset_source_fp,
                "ads_mentions_path": str(ads_mentions_path),
                "ads_mentions": int(len(ads_mentions)),
                "ads_blocks": int(ads_mentions["block_key"].nunique()) if "block_key" in ads_mentions.columns else 0,
                "ads_block_size_p95": float(_block_size_p95(ads_mentions)),
            },
            input_summary_path,
        )
        write_run_consistency(
            run_id=run_id,
            run_stage=stage,
            run_dirs=run_dirs,
            output_path=metrics_dir / "01_run_consistency.json",
            extras={"dataset_id": dataset_info["dataset_id"], "references_present": bool(dataset_info["references_present"])},
        )

        ui.start("Build or load embeddings")
        model_cfg = dict(model_info["model_cfg"] or {})
        rep_cfg = dict(model_cfg.get("representation", {}) or {})
        representation_cfg_hash = stable_hash(rep_cfg)
        model_version = str(model_cfg.get("name", "nand"))
        ads_mentions_id = stable_hash({"dataset_id": dataset_info["dataset_id"], "dataset_source_fp": dataset_source_fp})
        embedding_id = stable_hash(
            {
                "ads_mentions_id": ads_mentions_id,
                "representation_cfg_hash": representation_cfg_hash,
                "model_version": model_version,
            }
        )
        ads_chars_shared_path = shared_embeddings_dir / f"ads_chars2vec_{embedding_id}.npy"
        ads_text_shared_path = shared_embeddings_dir / f"ads_specter_{embedding_id}.npy"
        emb_cache_hit = ads_chars_shared_path.exists() and ads_text_shared_path.exists() and not args.force

        ads_chars = get_or_create_chars2vec_embeddings(
            mentions=ads_mentions,
            output_path=ads_chars_shared_path,
            force_recompute=args.force,
            use_stub_if_missing=False,
            quiet_libraries=args.quiet_libs,
        )
        ads_text = get_or_create_specter_embeddings(
            mentions=ads_mentions,
            output_path=ads_text_shared_path,
            force_recompute=args.force,
            model_name=rep_cfg.get("text_model_name", "allenai/specter"),
            max_length=int(rep_cfg.get("max_length", 256)),
            batch_size=32,
            device=args.device,
            prefer_precomputed=True,
            use_stub_if_missing=False,
            show_progress=args.progress,
            quiet_libraries=args.quiet_libs,
            reuse_model=True,
        )
        _record_cache_ref(
            cache_refs,
            artifact_type="embedding_ads_chars",
            artifact_id=embedding_id,
            shared_path=ads_chars_shared_path,
            run_path=ads_chars_path,
            mode=link_or_copy(ads_chars_shared_path, ads_chars_path),
        )
        _record_cache_ref(
            cache_refs,
            artifact_type="embedding_ads_text",
            artifact_id=embedding_id,
            shared_path=ads_text_shared_path,
            run_path=ads_text_path,
            mode=link_or_copy(ads_text_shared_path, ads_text_path),
        )
        if emb_cache_hit:
            ui.skip("Reused cached embeddings.")
        else:
            ui.done(f"Embeddings ready (ADS {tuple(ads_chars.shape)}/{tuple(ads_text.shape)}).")

        ui.start("Build ADS pairs and pair QC")
        run_cfg_from_model = dict(model_info.get("run_cfg", {}) or {})
        pair_build_cfg = _resolve_pair_build_cfg(run_cfg_from_model)
        max_pairs_per_block = run_cfg_from_model.get("max_pairs_per_block")
        pair_cfg_hash = stable_hash(
            {
                "max_pairs_per_block": max_pairs_per_block,
                "exclude_same_bibcode": bool(pair_build_cfg["exclude_same_bibcode"]),
            }
        )
        ads_pairs_id = stable_hash({"ads_mentions_id": ads_mentions_id, "pair_cfg_hash": pair_cfg_hash})
        ads_pairs_shared_path = shared_pairs_dir / f"ads_pairs_{ads_pairs_id}.parquet"
        pairs_qc_shared_path = shared_pairs_dir / f"pairs_qc_infer_ads_{ads_pairs_id}.json"

        if ads_pairs_shared_path.exists() and pairs_qc_shared_path.exists() and not args.force:
            ads_pairs = read_parquet(ads_pairs_shared_path)
            pairs_qc = _load_json(pairs_qc_shared_path)
            ui.skip(f"Reused ADS pairs ({len(ads_pairs)} rows).")
        else:
            ads_pairs, ads_pair_meta = build_pairs_within_blocks(
                mentions=ads_mentions,
                max_pairs_per_block=max_pairs_per_block,
                seed=11,
                require_same_split=False,
                labeled_only=False,
                balance_train=False,
                exclude_same_bibcode=bool(pair_build_cfg["exclude_same_bibcode"]),
                show_progress=args.progress,
                return_meta=True,
            )
            ads_pairs = _ensure_columns(ads_pairs, PAIR_REQUIRED_COLUMNS + ["label"])
            if len(ads_pairs) == 0:
                save_parquet(ads_pairs, ads_pairs_shared_path, index=False)
            else:
                write_pairs(ads_pairs, ads_pairs_shared_path)
            empty_mentions = pd.DataFrame(columns=MENTION_REQUIRED_COLUMNS + ["orcid", "split"])
            empty_pairs = pd.DataFrame(columns=PAIR_REQUIRED_COLUMNS + ["label"])
            pairs_qc = build_pairs_qc(
                lspo_mentions=empty_mentions,
                lspo_pairs=empty_pairs,
                ads_pairs=ads_pairs,
                split_meta={"status": "not_applicable"},
                lspo_pair_build_meta={},
                ads_pair_build_meta=ads_pair_meta,
            )
            pairs_qc["dataset_id"] = dataset_info["dataset_id"]
            pairs_qc["model_run_id"] = model_info["model_run_id"]
            write_json(pairs_qc, pairs_qc_shared_path)
            ui.done(f"Built ADS pairs ({len(ads_pairs)} rows).")

        _record_cache_ref(
            cache_refs,
            artifact_type="ads_pairs",
            artifact_id=ads_pairs_id,
            shared_path=ads_pairs_shared_path,
            run_path=ads_pairs_path,
            mode=link_or_copy(ads_pairs_shared_path, ads_pairs_path),
        )
        _record_cache_ref(
            cache_refs,
            artifact_type="pairs_qc",
            artifact_id=f"infer_ads_{ads_pairs_id}",
            shared_path=pairs_qc_shared_path,
            run_path=pairs_qc_path,
            mode=link_or_copy(pairs_qc_shared_path, pairs_qc_path),
        )

        ui.start("Score ADS pairs")
        score_pipeline_version = "v2"
        try:
            model_state_hash = hash_checkpoint_model_state(
                model_info["best_checkpoint"],
                score_pipeline_version=score_pipeline_version,
            )
        except Exception as exc:
            warnings.warn(
                (
                    "Model-state hash failed "
                    f"({exc.__class__.__name__}); falling back to file hash for pair-score cache key."
                ),
                RuntimeWarning,
            )
            model_state_hash = hash_file(model_info["best_checkpoint"])

        checkpoint_hash = hash_file(model_info["best_checkpoint"])
        score_cfg_hash = stable_hash(
            {
                "score_batch_size": int(args.score_batch_size),
                "precision_mode": str(args.precision_mode),
            }
        )
        pair_scores_id_v2 = stable_hash(
            {
                "ads_pairs_id": ads_pairs_id,
                "model_state_hash": model_state_hash,
                "score_cfg_hash": score_cfg_hash,
                "score_pipeline_version": score_pipeline_version,
            }
        )
        pair_scores_id_legacy = stable_hash(
            {
                "ads_pairs_id": ads_pairs_id,
                "checkpoint_hash": checkpoint_hash,
                "score_cfg_hash": score_cfg_hash,
            }
        )
        pair_scores_shared_path_v2 = shared_pair_scores_dir / f"ads_pair_scores_v2_{pair_scores_id_v2}.parquet"
        pair_scores_shared_path_legacy = shared_pair_scores_dir / f"ads_pair_scores_{pair_scores_id_legacy}.parquet"
        pair_scores_shared_path_used = pair_scores_shared_path_v2
        pair_scores_cache_schema_version = "v2"
        pair_scores_artifact_id = pair_scores_id_v2

        if pair_scores_shared_path_v2.exists() and not args.force:
            pair_scores = read_parquet(pair_scores_shared_path_v2)
            ui.skip(f"Reused pair scores ({len(pair_scores)} rows).")
        elif pair_scores_shared_path_legacy.exists() and not args.force:
            pair_scores = read_parquet(pair_scores_shared_path_legacy)
            pair_scores_cache_schema_version = "v1"
            pair_scores_artifact_id = pair_scores_id_legacy
            pair_scores_shared_path_used = pair_scores_shared_path_legacy
            promote_mode = link_or_copy(pair_scores_shared_path_legacy, pair_scores_shared_path_v2)
            if pair_scores_shared_path_v2.exists():
                pair_scores_shared_path_used = pair_scores_shared_path_v2
                pair_scores_cache_schema_version = "v2"
                pair_scores_artifact_id = pair_scores_id_v2
            ui.skip(f"Reused legacy pair scores ({len(pair_scores)} rows, promote={promote_mode}).")
        else:
            if len(ads_pairs) == 0:
                pair_scores = pd.DataFrame(columns=PAIR_SCORE_REQUIRED_COLUMNS)
                save_parquet(pair_scores, pair_scores_shared_path_v2, index=False)
                ui.done("No ADS pairs to score; wrote empty pair_scores artifact.")
            else:
                pair_scores = score_pairs_with_checkpoint(
                    mentions=ads_mentions,
                    pairs=ads_pairs,
                    chars2vec=ads_chars,
                    text_emb=ads_text,
                    checkpoint_path=model_info["best_checkpoint"],
                    output_path=pair_scores_shared_path_v2,
                    batch_size=int(args.score_batch_size),
                    device=args.device,
                    precision_mode=str(args.precision_mode),
                    show_progress=args.progress,
                )
                ui.done(f"Scored {len(pair_scores)} ADS pairs.")
            pair_scores_shared_path_used = pair_scores_shared_path_v2
            pair_scores_cache_schema_version = "v2"
            pair_scores_artifact_id = pair_scores_id_v2

        _record_cache_ref(
            cache_refs,
            artifact_type="pair_scores",
            artifact_id=pair_scores_artifact_id,
            shared_path=pair_scores_shared_path_used,
            run_path=pair_scores_path,
            mode=link_or_copy(pair_scores_shared_path_used, pair_scores_path),
            cache_schema_version=pair_scores_cache_schema_version,
        )
        write_json({"run_id": run_id, "cache_refs": cache_refs}, cache_refs_path)

        ui.start("Cluster ADS mentions and export mappings")
        cluster_cache_hit = (
            clusters_path.exists()
            and publication_export_path.exists()
            and cluster_qc_path.exists()
            and cluster_cfg_used_path.exists()
            and not args.force
        )
        eps_meta: dict[str, Any] = {}
        if cluster_cache_hit:
            clusters = read_parquet(clusters_path)
            cluster_qc = _load_json(cluster_qc_path)
            cfg_payload = _load_json(cluster_cfg_used_path) or {}
            eps_meta = dict(cfg_payload.get("eps_resolution", {}) or {})
            ui.skip(f"Reused clustering outputs ({len(clusters)} mentions).")
        else:
            selected_eps = float(model_info["selected_eps"])
            eps_min = float(cluster_cfg.get("eps_min", 0.0))
            eps_max = float(cluster_cfg.get("eps_max", 1.0))
            resolved_eps = float(np.clip(selected_eps, eps_min, eps_max))
            if abs(resolved_eps - selected_eps) > 1e-9:
                ui.warn(
                    f"Model selected_eps={selected_eps:.4f} clipped to [{eps_min:.4f}, {eps_max:.4f}] -> {resolved_eps:.4f}"
                )
            cluster_cfg_used = json.loads(json.dumps(cluster_cfg))
            cluster_cfg_used["eps_mode"] = "fixed"
            cluster_cfg_used["selected_eps"] = selected_eps
            cluster_cfg_used["eps"] = resolved_eps

            eps_meta = {
                "eps_mode": "model_run_selected",
                "source": "model_run_selected_eps",
                "model_run_id": model_info["model_run_id"],
                "raw_eps": selected_eps,
                "selected_eps": selected_eps,
                "resolved_eps": resolved_eps,
                "eps_min": eps_min,
                "eps_max": eps_max,
            }
            write_json(
                {
                    "run_id": run_id,
                    "run_stage": stage,
                    "model_run_id": model_info["model_run_id"],
                    "best_threshold": float(model_info["best_threshold"]),
                    "eps_resolution": eps_meta,
                    "cluster_config_used": cluster_cfg_used,
                },
                cluster_cfg_used_path,
            )
            clusters = cluster_blockwise_dbscan(
                mentions=ads_mentions,
                pair_scores=pair_scores,
                cluster_config=cluster_cfg_used,
                output_path=clusters_path,
                show_progress=args.progress,
            )
            _ = build_publication_author_mapping(
                mentions=ads_mentions,
                clusters=clusters,
                output_path=publication_export_path,
            )
            cluster_qc = build_cluster_qc(
                pair_scores=pair_scores,
                clusters=clusters,
                threshold=float(model_info["best_threshold"]),
            )
            write_json(cluster_qc, cluster_qc_path)
            ui.done(f"Clustered {len(clusters)} ADS mentions.")

        write_run_consistency(
            run_id=run_id,
            run_stage=stage,
            run_dirs=run_dirs,
            output_path=metrics_dir / "04_run_consistency.json",
            extras={"cluster_count": int(cluster_qc.get("cluster_count", 0))},
        )

        ui.start("Build stage metrics and go/no-go")
        write_run_consistency(
            run_id=run_id,
            run_stage=stage,
            run_dirs=run_dirs,
            output_path=metrics_dir / "05_run_consistency.json",
            extras={"command": "run-infer-ads"},
        )

        if stage_metrics_path.exists() and go_no_go_path.exists() and not args.force:
            stage_metrics = _load_json(stage_metrics_path)
            go = _load_json(go_no_go_path)
            ui.skip(f"Reused stage reports (GO={go.get('go')}).")
        else:
            empty_lspo = pd.DataFrame(columns=MENTION_REQUIRED_COLUMNS)
            consistency_files = [
                metrics_dir / "00_run_consistency.json",
                metrics_dir / "01_run_consistency.json",
                metrics_dir / "04_run_consistency.json",
                metrics_dir / "05_run_consistency.json",
            ]
            stage_metrics = build_stage_metrics(
                run_id=run_id,
                run_stage=stage,
                lspo_mentions=empty_lspo,
                lspo_pairs_count=None,
                ads_mentions=ads_mentions,
                clusters=clusters,
                train_manifest=model_info["train_manifest"],
                consistency_files=consistency_files,
                determinism_paths=[ads_mentions_path],
                cluster_qc=cluster_qc,
                split_meta={"status": "not_applicable"},
                eps_meta=eps_meta,
                subset_cache_key=dataset_source_fp,
            )
            write_json(stage_metrics, stage_metrics_path)

            go = evaluate_go_no_go(stage_metrics, gate_config=gate_cfg)
            write_go_no_go_report(go, go_no_go_path)
            ui.done(f"GO={go['go']} with blockers={len(go.get('blockers', []))}.")

        ui.info(f"Stage metrics: {stage_metrics_path}")
        ui.info(f"Go/No-Go report: {go_no_go_path}")
        ui.info(f"Run complete: {run_id}")

    except Exception as exc:
        ui.fail(str(exc))
        raise
    finally:
        ui.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NAND research CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("run-stage")
    sp.add_argument("--run-stage", required=True, choices=["smoke", "mini", "mid", "full"])
    sp.add_argument("--paths-config", default="configs/paths.local.yaml")
    sp.add_argument("--run-config", default=None)
    sp.add_argument("--model-config", default="configs/model/nand_best.yaml")
    sp.add_argument("--cluster-config", default="configs/clustering/dbscan_paper.yaml")
    sp.add_argument("--gates-config", default="configs/gates.yaml")
    sp.add_argument("--run-id", default=None)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--precision-mode", choices=["fp32", "amp_bf16"], default=None)
    sp.add_argument("--seeds", nargs="+", type=int, default=None)
    sp.add_argument("--use-stub-embeddings", action="store_true")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--baseline-run-id", default=None)
    sp.add_argument("--score-batch-size", type=int, default=8192)

    sp.add_argument("--prefer-precomputed-ads", dest="prefer_precomputed_ads", action="store_true")
    sp.add_argument("--no-prefer-precomputed-ads", dest="prefer_precomputed_ads", action="store_false")
    sp.set_defaults(prefer_precomputed_ads=True)

    sp.add_argument("--progress", dest="progress", action="store_true")
    sp.add_argument("--no-progress", dest="progress", action="store_false")
    sp.set_defaults(progress=True)

    sp.add_argument("--quiet-libs", dest="quiet_libs", action="store_true")
    sp.add_argument("--verbose-libs", dest="quiet_libs", action="store_false")
    sp.set_defaults(quiet_libs=True)

    sp.set_defaults(func=cmd_run_stage)

    sp = sub.add_parser("run-infer-ads")
    sp.add_argument("--dataset-id", required=True)
    sp.add_argument("--model-run-id", required=True)
    sp.add_argument("--paths-config", default="configs/paths.local.yaml")
    sp.add_argument("--cluster-config", default="configs/clustering/dbscan_paper.yaml")
    sp.add_argument("--run-id", default=None)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--precision-mode", choices=["fp32", "amp_bf16"], default="fp32")
    sp.add_argument("--score-batch-size", type=int, default=8192)
    sp.add_argument("--force", action="store_true")

    sp.add_argument("--progress", dest="progress", action="store_true")
    sp.add_argument("--no-progress", dest="progress", action="store_false")
    sp.set_defaults(progress=True)

    sp.add_argument("--quiet-libs", dest="quiet_libs", action="store_true")
    sp.add_argument("--verbose-libs", dest="quiet_libs", action="store_false")
    sp.set_defaults(quiet_libs=True)

    sp.set_defaults(func=cmd_run_infer_ads)

    sp = sub.add_parser("prepare-lspo")
    sp.add_argument("--paths-config", default="configs/paths.local.yaml")
    sp.add_argument("--output", default=None)
    sp.set_defaults(func=cmd_prepare_lspo)

    sp = sub.add_parser("prepare-ads")
    sp.add_argument("--paths-config", default="configs/paths.local.yaml")
    sp.add_argument("--output", default=None)
    sp.set_defaults(func=cmd_prepare_ads)

    sp = sub.add_parser("subset")
    sp.add_argument("--input", required=True)
    sp.add_argument("--run-config", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--manifest", required=True)
    sp.set_defaults(func=cmd_subset)

    sp = sub.add_parser("embeddings")
    sp.add_argument("--mentions", required=True)
    sp.add_argument("--model-config", default="configs/model/nand_best.yaml")
    sp.add_argument("--chars-out", required=True)
    sp.add_argument("--text-out", required=True)
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--prefer-precomputed", action="store_true")
    sp.add_argument("--use-stub", action="store_true")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--progress", action="store_true")
    sp.add_argument("--quiet-libs", dest="quiet_libs", action="store_true")
    sp.add_argument("--verbose-libs", dest="quiet_libs", action="store_false")
    sp.set_defaults(quiet_libs=True)
    sp.set_defaults(func=cmd_embeddings)

    sp = sub.add_parser("pairs")
    sp.add_argument("--mentions", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--seed", type=int, default=11)
    sp.add_argument("--max-pairs-per-block", type=int, default=None)
    sp.add_argument("--allow-cross-split", action="store_true")
    sp.add_argument("--labeled-only", action="store_true")
    sp.add_argument("--balance-train", action="store_true")
    sp.add_argument("--assign-lspo-splits", action="store_true")
    sp.add_argument("--run-config", default=None)
    sp.add_argument("--min-neg-val", type=int, default=None)
    sp.add_argument("--min-neg-test", type=int, default=None)
    sp.add_argument("--max-attempts", type=int, default=None)
    sp.add_argument("--progress", action="store_true")
    sp.set_defaults(func=cmd_pairs)

    sp = sub.add_parser("train")
    sp.add_argument("--mentions", required=True)
    sp.add_argument("--pairs", required=True)
    sp.add_argument("--chars", required=True)
    sp.add_argument("--text", required=True)
    sp.add_argument("--model-config", default="configs/model/nand_best.yaml")
    sp.add_argument("--seeds", nargs="*", type=int)
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--metrics-output", required=True)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--precision-mode", choices=["fp32", "amp_bf16"], default=None)
    sp.add_argument("--progress", action="store_true")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("score")
    sp.add_argument("--mentions", required=True)
    sp.add_argument("--pairs", required=True)
    sp.add_argument("--chars", required=True)
    sp.add_argument("--text", required=True)
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--batch-size", type=int, default=8192)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--precision-mode", choices=["fp32", "amp_bf16"], default="fp32")
    sp.add_argument("--progress", action="store_true")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("cluster")
    sp.add_argument("--mentions", required=True)
    sp.add_argument("--pair-scores", required=True)
    sp.add_argument("--cluster-config", default="configs/clustering/dbscan_paper.yaml")
    sp.add_argument("--output", required=True)
    sp.add_argument("--progress", action="store_true")
    sp.set_defaults(func=cmd_cluster)

    sp = sub.add_parser("export")
    sp.add_argument("--mentions", required=True)
    sp.add_argument("--clusters", required=True)
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("report")
    sp.add_argument("--metrics", required=True)
    sp.add_argument("--gates-config", default="configs/gates.yaml")
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("cache")
    cache_sub = sp.add_subparsers(dest="cache_command", required=True)

    cdoc = cache_sub.add_parser("doctor")
    cdoc.add_argument("--paths-config", default="configs/paths.local.yaml")
    cdoc.set_defaults(func=cmd_cache_doctor)

    cpurge = cache_sub.add_parser("purge")
    cpurge.add_argument("--paths-config", default="configs/paths.local.yaml")
    cpurge.add_argument(
        "--target",
        required=True,
        choices=["stale-subsets", "redundant-run-copies", "legacy-pair-scores-unused"],
    )
    cpurge.add_argument("--yes", action="store_true")
    cpurge.set_defaults(func=cmd_cache_purge)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
