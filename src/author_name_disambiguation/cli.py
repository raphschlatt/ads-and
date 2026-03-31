from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from author_name_disambiguation.approaches.nand.build_pairs import assign_lspo_splits, build_pairs_within_blocks, write_pairs
from author_name_disambiguation.approaches.nand.cluster import cluster_blockwise_dbscan, resolve_dbscan_eps
from author_name_disambiguation.approaches.nand.infer_pairs import score_pairs_with_checkpoint
from author_name_disambiguation.approaches.nand.train import train_nand_across_seeds
from author_name_disambiguation.common.cli_ui import CliUI
from author_name_disambiguation.common.cache_ops import (
    hash_checkpoint_model_state,
    hash_file,
    link_or_copy,
    stable_hash,
)
from author_name_disambiguation.common.config import (
    build_run_dirs,
    build_workspace_paths,
    load_yaml,
    write_latest_run_context,
    write_run_consistency,
)
from author_name_disambiguation.common.io_schema import PAIR_REQUIRED_COLUMNS, read_parquet
from author_name_disambiguation.common.package_resources import load_yaml_like
from author_name_disambiguation.common.pipeline_reports import (
    build_pairs_qc,
    build_train_stage_metrics,
    build_subset_summary,
    default_run_id,
    load_json,
    write_compare_infer_to_baseline,
    write_compare_train_to_baseline,
    write_json,
)
from author_name_disambiguation.common.run_report import evaluate_go_no_go, load_gate_config, write_go_no_go_report
from author_name_disambiguation.common.subset_artifacts import (
    LSPO_SOURCE_FP_SCHEME,
    LSPO_SOURCE_FP_SCHEME_LEGACY,
    atomic_save_parquet,
    compute_lspo_source_fp,
    compute_lspo_source_fp_legacy,
    compute_subset_identity,
    resolve_manifest_paths,
    resolve_shared_subset_paths,
)
from author_name_disambiguation.common.subset_builder import build_stage_subset, write_subset_manifest
from author_name_disambiguation.data.prepare_lspo import prepare_lspo_mentions
from author_name_disambiguation.embedding_contract import build_bundle_embedding_contract
from author_name_disambiguation.features.embed_chars2vec import get_or_create_chars2vec_embeddings
from author_name_disambiguation.features.embed_specter import get_or_create_specter_embeddings

SUBSET_CACHE_VERSION = "v3"
MODEL_BUNDLE_SCHEMA_VERSION = "v1"
_CLUSTER_OVERRIDE_EPS_FIELDS = ("eps", "selected_eps", "eps_mode")
_REPORT_TAG_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _resolved_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _load_train_run_cfg(run_stage: str, override_path: str | Path | None) -> tuple[dict[str, Any], str | None]:
    cfg = load_yaml_like(
        override_path,
        default_resource=f"resources/train_runs/{run_stage}.yaml",
        param_name="run_config",
    )
    cfg["stage"] = run_stage
    resolved = None if override_path is None else str(_resolved_path(override_path))
    return cfg, resolved


def _load_model_cfg(path: str | Path | None) -> tuple[dict[str, Any], str | None]:
    resolved = None if path is None else str(_resolved_path(path))
    cfg = load_yaml_like(path, default_resource="resources/models/nand_best.yaml", param_name="model_config")
    return cfg, resolved


def _load_cluster_cfg(path: str | Path | None) -> tuple[dict[str, Any], str | None]:
    resolved = None if path is None else str(_resolved_path(path))
    cfg = load_yaml_like(path, default_resource="resources/clustering/default.yaml", param_name="cluster_config")
    return cfg, resolved


def _build_public_workspace_paths(args) -> dict[str, Any]:
    return build_workspace_paths(
        data_root=args.data_root,
        artifacts_root=args.artifacts_root,
        raw_lspo_parquet=getattr(args, "raw_lspo_parquet", None),
        raw_lspo_h5=getattr(args, "raw_lspo_h5", None),
    )


def _cli_run_id(stage: str) -> str:
    return default_run_id(stage, tag="cli")


def _configure_library_noise(quiet_libraries: bool) -> None:
    if not quiet_libraries:
        return

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["ABSL_LOG_LEVEL"] = "3"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

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


def _ensure_run_dirs(run_dirs: dict[str, Path], keys: list[str]) -> None:
    for key in keys:
        Path(run_dirs[key]).mkdir(parents=True, exist_ok=True)


def _resolve_model_run_for_inference(
    *,
    artifacts_root: str | Path,
    model_run_id: str,
) -> dict[str, Any]:
    metrics_dir = Path(artifacts_root).expanduser().resolve() / "metrics" / str(model_run_id)
    manifest_path = metrics_dir / "03_train_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing train manifest for model_run_id={model_run_id}: {manifest_path}"
        )
    train_manifest = load_json(manifest_path)
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
    cluster_used_payload = load_json(cluster_used_path)
    eps_resolution = dict(cluster_used_payload.get("eps_resolution", {}) or {})
    selected_eps = eps_resolution.get("selected_eps")
    if selected_eps is None:
        selected_eps = eps_resolution.get("resolved_eps")
    if selected_eps is None:
        selected_eps = (cluster_used_payload.get("cluster_config_used", {}) or {}).get("eps")
    if selected_eps is None:
        raise ValueError(
            "selected_eps missing in model run clustering metadata. "
            f"Expected eps_resolution.selected_eps or resolved_eps/cluster_config_used.eps in {cluster_used_path}."
        )
    selected_eps = float(selected_eps)

    context_path = metrics_dir / "00_context.json"
    context_payload = load_json(context_path) if context_path.exists() else {}

    run_cfg = dict(context_payload.get("run_config_payload") or {})
    run_cfg_path = context_payload.get("run_config")
    if not run_cfg and run_cfg_path:
        run_cfg = load_yaml(str(run_cfg_path))

    model_cfg = dict(context_payload.get("model_config_payload") or {})
    model_cfg_path = context_payload.get("model_config")
    model_cfg_resolved_path: str | None = None if model_cfg_path is None else str(_resolved_path(str(model_cfg_path)))
    if not model_cfg:
        model_cfg, model_cfg_resolved_path = _load_model_cfg(model_cfg_path)

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


def _write_model_bundle(
    *,
    artifacts_root: str | Path,
    model_run_id: str,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    model_info = _resolve_model_run_for_inference(artifacts_root=artifacts_root, model_run_id=model_run_id)
    if output_dir is None:
        models_root = Path(artifacts_root).expanduser().resolve() / "models"
        bundle_dir = models_root / str(model_run_id) / "bundle_v1"
    else:
        bundle_dir = Path(output_dir).expanduser().resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dst = bundle_dir / "checkpoint.pt"
    model_cfg_dst = bundle_dir / "model_config.yaml"
    clustering_dst = bundle_dir / "clustering_resolved.json"
    manifest_dst = bundle_dir / "bundle_manifest.json"

    shutil.copy2(model_info["best_checkpoint"], checkpoint_dst)
    with model_cfg_dst.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(model_info["model_cfg"]), f, sort_keys=False)

    cluster_used_payload = load_json(model_info["cluster_used_path"])
    eps_resolution = dict(cluster_used_payload.get("eps_resolution", {}) or {})
    if eps_resolution.get("selected_eps") is None:
        eps_resolution["selected_eps"] = float(model_info["selected_eps"])
    cluster_config_used = dict(cluster_used_payload.get("cluster_config_used", {}) or {})
    clustering_payload = {
        "source_model_run_id": str(model_run_id),
        "eps_resolution": eps_resolution,
        "cluster_config_used": cluster_config_used,
    }
    write_json(clustering_payload, clustering_dst)

    run_cfg = dict(model_info.get("run_cfg", {}) or {})
    pair_building = dict(run_cfg.get("pair_building", {}) or {})
    manifest_payload = {
        "bundle_schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
        "source_model_run_id": str(model_run_id),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_hash": hash_file(checkpoint_dst),
        "selected_eps": float(model_info["selected_eps"]),
        "best_threshold": float(model_info["best_threshold"]),
        "precision_mode": str(model_info["train_manifest"].get("precision_mode", "fp32")),
        "best_test_f1": model_info["train_manifest"].get("best_test_f1"),
        "best_val_f1": model_info["train_manifest"].get("best_val_f1"),
        "best_val_class_counts": model_info["train_manifest"].get("best_val_class_counts", {}),
        "best_test_class_counts": model_info["train_manifest"].get("best_test_class_counts", {}),
        "max_pairs_per_block": run_cfg.get("max_pairs_per_block"),
        "pair_building": pair_building,
        "embedding_contract": build_bundle_embedding_contract(model_info["model_cfg"]),
        "paths": {
            "checkpoint": str(checkpoint_dst.name),
            "model_config": str(model_cfg_dst.name),
            "clustering_resolved": str(clustering_dst.name),
        },
    }
    write_json(manifest_payload, manifest_dst)

    return {
        "bundle_dir": bundle_dir,
        "bundle_manifest_path": manifest_dst,
        "checkpoint_path": checkpoint_dst,
        "model_config_path": model_cfg_dst,
        "clustering_resolved_path": clustering_dst,
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


def _compute_mean_sem(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) == 0:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, None
    sem = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    return mean, sem


def _summarize_cluster_test_rows(per_seed_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if len(per_seed_rows) == 0:
        return {}

    metric_keys = ["accuracy", "precision", "recall", "f1"]
    summary: dict[str, dict[str, Any]] = {}
    variants = sorted({str(row["variant"]) for row in per_seed_rows})
    for variant in variants:
        rows = [row for row in per_seed_rows if str(row["variant"]) == variant]
        payload: dict[str, Any] = {
            "seed_count": int(len(rows)),
        }
        for key in metric_keys:
            values = [float(row[key]) for row in rows]
            mean, sem = _compute_mean_sem(values)
            payload[f"{key}_mean"] = mean
            payload[f"{key}_sem"] = sem
        n_pairs_vals = [int(row["n_pairs"]) for row in rows]
        n_pairs_mean, _ = _compute_mean_sem([float(v) for v in n_pairs_vals])
        payload["n_pairs_mean"] = n_pairs_mean
        payload["n_pairs_total"] = int(sum(n_pairs_vals))
        summary[variant] = payload
    return summary


def _build_cluster_test_report_markdown(report: dict[str, Any]) -> str:
    def _fmt(value: Any, digits: int = 6) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value)

    lines: list[str] = []
    lines.append("# Final Clustering Test Report")
    lines.append("")
    lines.append(f"- model_run_id: `{report.get('model_run_id')}`")
    lines.append(f"- run_stage: `{report.get('run_stage')}`")
    lines.append(f"- generated_utc: `{report.get('generated_utc')}`")
    lines.append(f"- selected_eps: `{_fmt(report.get('selected_eps'))}`")
    lines.append(f"- min_samples: `{_fmt(report.get('min_samples'))}`")
    lines.append(f"- metric: `{report.get('metric')}`")
    lines.append(f"- seeds_expected: `{report.get('seeds_expected')}`")
    lines.append(f"- seeds_evaluated: `{report.get('seeds_evaluated')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| variant | accuracy_mean | accuracy_sem | precision_mean | precision_sem | recall_mean | recall_sem | f1_mean | f1_sem | n_pairs_mean | n_pairs_total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    summary = dict(report.get("variants", {}) or {})
    for variant in sorted(summary.keys()):
        row = dict(summary.get(variant, {}) or {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(variant),
                    _fmt(row.get("accuracy_mean")),
                    _fmt(row.get("accuracy_sem")),
                    _fmt(row.get("precision_mean")),
                    _fmt(row.get("precision_sem")),
                    _fmt(row.get("recall_mean")),
                    _fmt(row.get("recall_sem")),
                    _fmt(row.get("f1_mean")),
                    _fmt(row.get("f1_sem")),
                    _fmt(row.get("n_pairs_mean")),
                    _fmt(row.get("n_pairs_total")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Delta (with_constraints - no_constraints)")
    lines.append("")
    delta = dict(report.get("delta_with_constraints_minus_no_constraints", {}) or {})
    lines.append(
        "| accuracy | precision | recall | f1 |\n"
        "|---:|---:|---:|---:|\n"
        f"| {_fmt(delta.get('accuracy'))} | {_fmt(delta.get('precision'))} | {_fmt(delta.get('recall'))} | {_fmt(delta.get('f1'))} |"
    )
    lines.append("")
    lines.append("## Per Seed")
    lines.append("")
    lines.append("| seed | variant | threshold | accuracy | precision | recall | f1 | n_pairs | checkpoint |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
    per_seed_rows = sorted(
        list(report.get("per_seed_rows", []) or []),
        key=lambda r: (str(r.get("variant", "")), int(r.get("seed", 0))),
    )
    for row in per_seed_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("seed"), digits=0),
                    str(row.get("variant")),
                    _fmt(row.get("threshold")),
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("precision")),
                    _fmt(row.get("recall")),
                    _fmt(row.get("f1")),
                    _fmt(row.get("n_pairs"), digits=0),
                    str(row.get("checkpoint")),
                ]
            )
            + " |"
        )

    lines.append("")
    return "\n".join(lines)


def _resolve_selected_eps(cluster_used_payload: dict[str, Any], *, source_path: Path) -> float:
    eps_resolution = dict(cluster_used_payload.get("eps_resolution", {}) or {})
    selected_eps = eps_resolution.get("selected_eps")
    if selected_eps is None:
        selected_eps = eps_resolution.get("resolved_eps")
    if selected_eps is None:
        selected_eps = (cluster_used_payload.get("cluster_config_used", {}) or {}).get("eps")
    if selected_eps is None:
        raise ValueError(
            "selected_eps missing in clustering metadata. "
            f"Expected eps_resolution.selected_eps/resolved_eps or cluster_config_used.eps in {source_path}."
        )
    return float(selected_eps)


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(dict(merged[key]), value)
        else:
            merged[key] = json.loads(json.dumps(value))
    return merged


def _sanitize_report_tag(tag: str | None) -> str | None:
    if tag is None:
        return None
    value = str(tag).strip()
    if value == "":
        raise ValueError("report_tag must be non-empty when provided.")
    if _REPORT_TAG_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "Invalid report_tag. Allowed characters are: a-z, A-Z, 0-9, '.', '_' and '-'."
        )
    return value


def _resolve_report_paths(metrics_dir: Path, report_tag: str | None) -> dict[str, Path]:
    suffix = "" if report_tag is None else f"__{report_tag}"
    return {
        "json": metrics_dir / f"06_clustering_test_report{suffix}.json",
        "summary_csv": metrics_dir / f"06_clustering_test_summary{suffix}.csv",
        "per_seed_csv": metrics_dir / f"06_clustering_test_per_seed{suffix}.csv",
        "markdown": metrics_dir / f"06_clustering_test_report{suffix}.md",
    }


def _apply_cluster_config_override(
    *,
    base_cluster_config: dict[str, Any],
    override_path: str | Path | None,
) -> tuple[dict[str, Any], str, str | None, list[str]]:
    base = json.loads(json.dumps(base_cluster_config))
    if override_path is None:
        return base, "train_only", None, []

    resolved_override_path = _resolved_path(override_path)
    override_cfg = load_yaml(resolved_override_path)
    if not isinstance(override_cfg, dict):
        raise ValueError(
            f"Cluster config override must be a YAML object/map: {resolved_override_path}"
        )

    ignored = [field for field in _CLUSTER_OVERRIDE_EPS_FIELDS if field in override_cfg]
    merged = _deep_merge_dict(base, override_cfg)
    return merged, "train_plus_override", str(Path(resolved_override_path).resolve()), ignored


def _resolve_train_seed_runs(train_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    runs = train_manifest.get("runs")
    default_threshold = train_manifest.get("best_threshold")
    if default_threshold is None:
        raise ValueError("best_threshold missing in train manifest.")
    default_threshold = float(default_threshold)

    if not isinstance(runs, list) or len(runs) == 0:
        best_checkpoint = train_manifest.get("best_checkpoint")
        if best_checkpoint is None:
            raise ValueError(
                "Train manifest does not contain per-seed runs and is missing best_checkpoint. "
                "Expected non-empty `runs` or `best_checkpoint` in 03_train_manifest.json."
            )
        warnings.warn(
            "Train manifest has no per-seed `runs`; falling back to a single-seed clustering report from best_checkpoint.",
            RuntimeWarning,
        )
        best_seed = int(train_manifest.get("best_seed", 1))
        return [
            {
                "seed": best_seed,
                "checkpoint": Path(str(best_checkpoint)).expanduser(),
                "threshold": default_threshold,
            }
        ]

    seed_runs: list[dict[str, Any]] = []
    for row in runs:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid run entry in manifest: {row!r}")
        if row.get("seed") is None:
            raise ValueError(f"Manifest run entry missing seed: {row!r}")
        if row.get("checkpoint") is None:
            raise ValueError(f"Manifest run entry missing checkpoint: {row!r}")
        seed = int(row["seed"])
        checkpoint = Path(str(row["checkpoint"])).expanduser()
        threshold = float(row.get("threshold", default_threshold))
        seed_runs.append(
            {
                "seed": seed,
                "checkpoint": checkpoint,
                "threshold": threshold,
            }
        )

    seed_runs = sorted(seed_runs, key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in seed_runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds found in manifest runs: {seeds}")
    return seed_runs


def _normalize_split_label_counts(value: Any) -> dict[str, dict[str, int]] | None:
    if isinstance(value, dict):
        out: dict[str, dict[str, int]] = {}
        for split, row in value.items():
            if not isinstance(row, dict):
                continue
            out[str(split)] = {
                "pos": int(row.get("pos", 0)),
                "neg": int(row.get("neg", 0)),
                "labeled_pairs": int(row.get("labeled_pairs", 0)),
            }
        return out
    if isinstance(value, list):
        out = {}
        for row in value:
            if not isinstance(row, dict):
                continue
            split = str(row.get("split", "")).strip()
            if split == "":
                continue
            out[split] = {
                "pairs": int(row.get("pairs", 0)),
                "labeled_pairs": int(row.get("labeled_pairs", 0)),
                "pos": int(row.get("pos", 0)),
                "neg": int(row.get("neg", 0)),
            }
        return out
    return None


def _snapshot_legacy_subset_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "lspo_mentions": int(payload.get("lspo_mentions", 0)),
        "lspo_blocks": int(payload.get("lspo_blocks", 0)),
        "lspo_block_size_p95": float(payload.get("lspo_block_size_p95", 0.0)),
    }


def _snapshot_legacy_split_balance(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "attempts": int(payload.get("attempts", 0)),
        "min_neg_val": int(payload.get("min_neg_val", 0)),
        "min_neg_test": int(payload.get("min_neg_test", 0)),
        "required_neg_total": int(payload.get("required_neg_total", 0)),
        "max_possible_neg_total": int(payload.get("max_possible_neg_total", 0)),
        "train_ratio": float(payload.get("train_ratio", 0.0)),
        "val_ratio": float(payload.get("val_ratio", 0.0)),
        "test_ratio": float(payload.get("test_ratio", 0.0)),
        "split_label_counts": _normalize_split_label_counts(payload.get("split_label_counts")),
    }


def _snapshot_legacy_pairs_qc(payload: dict[str, Any]) -> dict[str, Any]:
    pair_build = dict(payload.get("lspo_pair_build", {}) or {})
    return {
        "orcid_leakage_groups": int(payload.get("orcid_leakage_groups", 0)),
        "lspo_pairs": int(payload.get("lspo_pairs", 0)),
        "split_label_counts": _normalize_split_label_counts(payload.get("split_label_counts")),
        "lspo_pair_build": {
            "exclude_same_bibcode": bool(pair_build.get("exclude_same_bibcode", False)),
            "same_publication_pairs_skipped": int(pair_build.get("same_publication_pairs_skipped", 0)),
            "balance_train": bool(pair_build.get("balance_train", False)),
            "pairs_written": int(pair_build.get("pairs_written", 0)),
            "train_balance_before": dict(pair_build.get("train_balance_before", {}) or {}),
            "train_balance_after": dict(pair_build.get("train_balance_after", {}) or {}),
        },
    }


def _collect_snapshot_mismatches(section: str, expected: dict[str, Any], current: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for key in sorted(set(expected) | set(current)):
        if expected.get(key) != current.get(key):
            issues.append(
                f"{section}.{key} mismatch: expected={expected.get(key)!r}, current={current.get(key)!r}"
            )
    return issues


def _build_cluster_variant_config(base_cluster_cfg: dict[str, Any], *, enable_constraints: bool) -> dict[str, Any]:
    out = json.loads(json.dumps(base_cluster_cfg))
    constraints_cfg = dict(out.get("constraints", {}) or {})
    constraints_cfg["enabled"] = bool(enable_constraints)
    out["constraints"] = constraints_cfg
    return out


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


def _validate_cached_lspo_subset(
    *,
    lspo_subset,
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
    checks["rows_lspo"] = int(len(lspo_subset))

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


def cmd_run_train_stage(args):
    ui = CliUI(total_steps=9, progress=args.progress)

    try:
        ui.start("Initialize run context")
        _configure_library_noise(args.quiet_libs)
        paths = _build_public_workspace_paths(args)
        data_cfg = paths["data"]
        art_cfg = paths["artifacts"]

        run_cfg, run_cfg_path = _load_train_run_cfg(args.run_stage, args.run_config)
        split_assignment_cfg = _resolve_split_assignment_cfg(run_cfg)
        pair_build_cfg = _resolve_pair_build_cfg(run_cfg)
        split_balance_cfg = run_cfg.get("split_balance", {})

        model_cfg, model_cfg_path = _load_model_cfg(args.model_config)
        rep_cfg = model_cfg.get("representation", {})
        training_cfg = dict(model_cfg.get("training", {}) or {})
        precision_mode = _resolve_precision_mode(run_cfg=run_cfg, training_cfg=training_cfg)
        if getattr(args, "precision_mode", None):
            precision_mode = str(args.precision_mode).strip().lower()
        training_cfg["precision_mode"] = precision_mode

        cluster_cfg, cluster_cfg_path = _load_cluster_cfg(args.cluster_config)
        gate_cfg = load_gate_config(args.gates_config)

        stage = args.run_stage
        run_id = args.run_id or _cli_run_id(stage)
        run_dirs = build_run_dirs(data_cfg, art_cfg, run_id)
        _ensure_run_dirs(
            run_dirs,
            [
                "metrics",
                "checkpoints",
                "embeddings",
                "subset_cache",
                "subset_manifests",
                "interim",
                "shared_cache_root",
                "shared_subsets",
                "shared_embeddings",
                "shared_pairs",
                "shared_eps_sweeps",
                "models",
            ],
        )

        command_name = "run-train-stage"
        latest_context_path = Path(art_cfg["metrics_dir"]) / "latest_run.json"
        write_latest_run_context(
            run_id=run_id,
            run_dirs=run_dirs,
            output_path=latest_context_path,
            stage=stage,
            extras={"created_utc": datetime.now(timezone.utc).isoformat(), "source": f"cli.{command_name}"},
        )
        train_seeds = _resolve_train_seeds(args, run_cfg=run_cfg, training_cfg=training_cfg)
        metrics_dir = Path(run_dirs["metrics"])
        write_json(
            {
                "run_id": run_id,
                "run_stage": stage,
                "pipeline_scope": "train",
                "device": args.device,
                "quiet_libs": bool(args.quiet_libs),
                "train_seeds": train_seeds,
                "precision_mode": precision_mode,
                "run_config": run_cfg_path,
                "run_config_payload": run_cfg,
                "model_config": model_cfg_path,
                "model_config_payload": model_cfg,
                "cluster_config": cluster_cfg_path,
                "cluster_config_payload": cluster_cfg,
            },
            metrics_dir / "00_context.json",
        )
        write_run_consistency(
            run_id=run_id,
            run_stage=stage,
            run_dirs=run_dirs,
            output_path=metrics_dir / "00_run_consistency.json",
            extras={"command": command_name, "latest_context_path": str(latest_context_path)},
        )
        ui.done(f"Run ID: {run_id}")

        subset_dir = Path(run_dirs["subset_cache"])
        emb_dir = Path(run_dirs["embeddings"])
        checkpoint_dir = Path(run_dirs["checkpoints"])
        shared_subsets_dir = Path(run_dirs["shared_subsets"])
        shared_embeddings_dir = Path(run_dirs["shared_embeddings"])
        shared_pairs_dir = Path(run_dirs["shared_pairs"])
        shared_eps_sweeps_dir = Path(run_dirs["shared_eps_sweeps"])

        lspo_mentions_path = Path(run_dirs["interim"]) / "lspo_mentions.parquet"
        lspo_subset_run_path = subset_dir / f"lspo_mentions_{stage}.parquet"
        lspo_split_run_path = subset_dir / f"lspo_mentions_split_{stage}.parquet"
        lspo_pairs_path = subset_dir / f"lspo_pairs_{stage}.parquet"
        lspo_chars_path = emb_dir / f"lspo_chars2vec_{stage}.npy"
        lspo_text_path = emb_dir / f"lspo_specter_{stage}.npy"

        train_manifest_path = metrics_dir / "03_train_manifest.json"
        split_meta_path = metrics_dir / "02_split_balance.json"
        pairs_qc_path = metrics_dir / "02_pairs_qc.json"
        cluster_cfg_used_path = metrics_dir / "04_clustering_config_used.json"
        stage_metrics_path = metrics_dir / f"05_stage_metrics_{stage}.json"
        go_no_go_path = metrics_dir / f"05_go_no_go_{stage}.json"
        compare_path = metrics_dir / "99_compare_train_to_baseline.json"
        cache_refs_path = metrics_dir / "00_cache_refs.json"
        cache_refs: list[dict[str, Any]] = []

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

        ui.start("Build or load LSPO subset")
        t_all = perf_counter()
        timings: dict[str, float] = {}
        source_fp = compute_lspo_source_fp(lspo_mentions_path)
        subset_identity = compute_subset_identity(run_cfg=run_cfg, source_fp=source_fp, sampler_version=SUBSET_CACHE_VERSION)
        subset_paths = resolve_shared_subset_paths(data_cfg=data_cfg, identity=subset_identity)
        manifest_paths = resolve_manifest_paths(
            run_id=run_id,
            manifest_dir=Path(run_dirs["subset_manifests"]),
            identity=subset_identity,
            run_stage=stage,
        )
        shared_subsets_dir.mkdir(parents=True, exist_ok=True)
        subset_meta_path = _subset_meta_path(subset_paths.shared_dir, subset_identity.subset_tag)

        cache_hit = False
        cache_valid: bool | None = None
        cache_invalid_reason: str | None = None
        cache_rebuilt = False
        cache_source = "none"
        cache_health: dict[str, Any] = {}

        if not args.force and subset_paths.lspo_shared.exists():
            cache_source = "shared"
        elif not args.force and subset_paths.lspo_shared_legacy is not None and subset_paths.lspo_shared_legacy.exists():
            cache_source = "legacy_shared"

        if cache_source != "none":
            cache_hit = True
            t0 = perf_counter()
            lspo_cache_path = subset_paths.lspo_shared if cache_source == "shared" else subset_paths.lspo_shared_legacy
            assert lspo_cache_path is not None
            lspo_subset = read_parquet(lspo_cache_path)
            t1 = perf_counter()
            timings["read_lspo_s"] = t1 - t0
            cache_valid, reasons, cache_health = _validate_cached_lspo_subset(
                lspo_subset=lspo_subset,
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
            atomic_save_parquet(lspo_subset, subset_paths.lspo_shared, index=False)
            t2 = perf_counter()
            timings["build_lspo_s"] = t1 - t0
            timings["save_lspo_shared_s"] = t2 - t1
            cache_valid = True
            _, _, cache_health = _validate_cached_lspo_subset(
                lspo_subset=lspo_subset,
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
                "pipeline_scope": "train",
                "source_fp": subset_identity.source_fp,
                "source_fingerprint_scheme": LSPO_SOURCE_FP_SCHEME,
                "subset_target_mentions": run_cfg.get("subset_target_mentions"),
                "health": cache_health,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
            subset_meta_path,
        )

        t3 = perf_counter()
        lspo_subset_link_mode = link_or_copy(subset_paths.lspo_shared, lspo_subset_run_path)
        t4 = perf_counter()
        timings["save_lspo_run_s"] = t4 - t3
        timings["total_s"] = perf_counter() - t_all
        _record_cache_ref(
            cache_refs,
            artifact_type="subset_lspo",
            artifact_id=subset_identity.subset_tag,
            shared_path=subset_paths.lspo_shared,
            run_path=lspo_subset_run_path,
            mode=lspo_subset_link_mode,
        )

        if args.force or not manifest_paths.lspo_primary.exists():
            write_subset_manifest(lspo_subset, manifest_paths.lspo_primary)

        empty_ads = pd.DataFrame(columns=lspo_subset.columns)
        subset_summary = build_subset_summary(
            run_id=run_id,
            stage=stage,
            source_fp=subset_identity.source_fp,
            source_fingerprint_scheme=LSPO_SOURCE_FP_SCHEME,
            subset_tag=subset_identity.subset_tag,
            cache_key=subset_identity.subset_tag,
            cache_hit=cache_hit,
            cache_valid=cache_valid,
            cache_invalid_reason=cache_invalid_reason,
            cache_rebuilt=cache_rebuilt,
            cache_version=SUBSET_CACHE_VERSION,
            lspo_subset=lspo_subset,
            ads_subset=empty_ads,
            timings=timings,
        )
        subset_summary["pipeline_scope"] = "train"
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
            ui.done(f"Built LSPO subset ({len(lspo_subset)} mentions).")

        ui.start("Build or load LSPO embeddings")
        representation_cfg_hash = stable_hash(rep_cfg)
        model_version = str(model_cfg.get("name", "nand"))
        embedding_id = stable_hash(
            {
                "subset_id": subset_identity.subset_tag,
                "representation_cfg_hash": representation_cfg_hash,
                "model_version": model_version,
                "pipeline_scope": "train",
            }
        )
        lspo_chars_shared_path = shared_embeddings_dir / f"lspo_chars2vec_{embedding_id}.npy"
        lspo_text_shared_path = shared_embeddings_dir / f"lspo_specter_{embedding_id}.npy"
        emb_cache_hit = lspo_chars_shared_path.exists() and lspo_text_shared_path.exists() and not args.force

        lspo_chars = get_or_create_chars2vec_embeddings(
            mentions=lspo_subset,
            output_path=lspo_chars_shared_path,
            force_recompute=args.force,
            batch_size=32,
            execution_mode="predict",
            use_stub_if_missing=args.use_stub_embeddings,
            quiet_libraries=args.quiet_libs,
            show_progress=bool(args.progress),
        )
        lspo_text = get_or_create_specter_embeddings(
            mentions=lspo_subset,
            output_path=lspo_text_shared_path,
            force_recompute=args.force,
            model_name=rep_cfg.get("text_model_name", "allenai/specter"),
            text_backend=rep_cfg.get("text_backend", "transformers"),
            text_adapter_name=rep_cfg.get("text_adapter_name"),
            text_adapter_alias=rep_cfg.get("text_adapter_alias", "specter2"),
            max_length=int(rep_cfg.get("max_length", 256)),
            batch_size=16,
            device=args.device,
            prefer_precomputed=False,
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
        if emb_cache_hit:
            ui.skip("Reused cached LSPO embeddings.")
        else:
            ui.done(f"Embeddings ready (LSPO {tuple(lspo_chars.shape)}/{tuple(lspo_text.shape)}).")

        ui.start("Assign splits and build LSPO pairs")
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
                "pipeline_scope": "train",
            }
        )
        lspo_split_shared_path = shared_pairs_dir / f"lspo_mentions_split_{lspo_pairs_id}.parquet"
        lspo_pairs_shared_path = shared_pairs_dir / f"lspo_pairs_{lspo_pairs_id}.parquet"
        split_meta_shared_path = shared_pairs_dir / f"split_balance_{lspo_pairs_id}.json"
        pairs_qc_shared_path = shared_pairs_dir / f"pairs_qc_train_{lspo_pairs_id}.json"

        if (
            lspo_pairs_shared_path.exists()
            and split_meta_shared_path.exists()
            and lspo_split_shared_path.exists()
            and pairs_qc_shared_path.exists()
            and not args.force
        ):
            lspo_mentions_split = read_parquet(lspo_split_shared_path)
            lspo_pairs = read_parquet(lspo_pairs_shared_path)
            split_meta = load_json(split_meta_shared_path)
            pairs_qc = load_json(pairs_qc_shared_path)
            ui.skip(f"Reused LSPO split+pairs ({len(lspo_pairs)} rows).")
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
                link_or_copy(lspo_split_shared_path, lspo_split_run_path)
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
            empty_ads_pairs = pd.DataFrame(columns=PAIR_REQUIRED_COLUMNS + ["label"])
            pairs_qc = build_pairs_qc(
                lspo_mentions=lspo_mentions_split,
                lspo_pairs=lspo_pairs,
                ads_pairs=empty_ads_pairs,
                split_meta=split_meta,
                lspo_pair_build_meta=lspo_pair_meta,
                ads_pair_build_meta={},
            )
            pairs_qc["pipeline_scope"] = "train"
            write_json(pairs_qc, pairs_qc_shared_path)
            ui.done(f"Built LSPO pairs ({len(lspo_pairs)} rows).")

        lspo_split_link_mode = link_or_copy(lspo_split_shared_path, lspo_split_run_path)
        lspo_pairs_link_mode = link_or_copy(lspo_pairs_shared_path, lspo_pairs_path)
        split_meta_link_mode = link_or_copy(split_meta_shared_path, split_meta_path)
        pairs_qc_link_mode = link_or_copy(pairs_qc_shared_path, pairs_qc_path)
        _record_cache_ref(
            cache_refs,
            artifact_type="lspo_split",
            artifact_id=lspo_pairs_id,
            shared_path=lspo_split_shared_path,
            run_path=lspo_split_run_path,
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
        _record_cache_ref(
            cache_refs,
            artifact_type="pairs_qc",
            artifact_id=f"train_{lspo_pairs_id}",
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
        if str(split_meta.get("status", "")).strip().lower() == "split_balance_infeasible":
            raise RuntimeError(
                "split_balance_infeasible: "
                f"max_possible_neg_total={split_meta.get('max_possible_neg_total')} "
                f"< required_neg_total={split_meta.get('required_neg_total')}"
            )

        ui.start("Train NAND model")
        train_cache_hit = False
        if train_manifest_path.exists() and not args.force:
            train_manifest = load_json(train_manifest_path)
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

        ui.start("Resolve clustering eps from LSPO val sweep")
        best_threshold = float(train_manifest["best_threshold"])
        try:
            model_state_hash = hash_checkpoint_model_state(
                train_manifest["best_checkpoint"],
                score_pipeline_version="v3",
            )
        except Exception:
            model_state_hash = hash_file(train_manifest["best_checkpoint"])
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
        if eps_meta.get("selected_eps") is None:
            eps_meta["selected_eps"] = float(resolved_eps)
        eps_meta["eps_sweep_id"] = eps_sweep_id
        eps_sweep_shared_path = shared_eps_sweeps_dir / f"eps_sweep_{eps_sweep_id}.json"
        write_json(
            {
                "eps_sweep_id": eps_sweep_id,
                "run_stage": stage,
                "run_id": run_id,
                "pipeline_scope": "train",
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
                "pipeline_scope": "train",
                "best_threshold": best_threshold,
                "eps_resolution": eps_meta,
                "cluster_config_used": cluster_cfg_used,
            },
            cluster_cfg_used_path,
        )
        _record_cache_ref(
            cache_refs,
            artifact_type="eps_sweep",
            artifact_id=eps_sweep_id,
            shared_path=eps_sweep_shared_path,
            run_path=cluster_cfg_used_path,
            mode=link_or_copy(eps_sweep_shared_path, metrics_dir / "04_eps_sweep.json"),
        )
        ui.done(f"Resolved eps={resolved_eps:.4f}")
        write_run_consistency(
            run_id=run_id,
            run_stage=stage,
            run_dirs=run_dirs,
            output_path=metrics_dir / "04_run_consistency.json",
            extras={"selected_eps": resolved_eps},
        )

        ui.start("Build stage metrics and go/no-go")
        write_run_consistency(
            run_id=run_id,
            run_stage=stage,
            run_dirs=run_dirs,
            output_path=metrics_dir / "05_run_consistency.json",
            extras={"command": command_name},
        )
        if stage_metrics_path.exists() and go_no_go_path.exists() and not args.force:
            stage_metrics = load_json(stage_metrics_path)
            go = load_json(go_no_go_path)
            ui.skip(f"Reused stage reports (GO={go.get('go')}).")
        else:
            determinism_paths = [manifest_paths.lspo_primary] if manifest_paths.lspo_primary.exists() else [manifest_paths.lspo_legacy]
            consistency_files = [metrics_dir / f"{i:02d}_run_consistency.json" for i in range(0, 6)]
            stage_metrics = build_train_stage_metrics(
                run_id=run_id,
                run_stage=stage,
                lspo_mentions=lspo_subset,
                train_manifest=train_manifest,
                consistency_files=consistency_files,
                determinism_paths=determinism_paths,
                split_meta=split_meta,
                eps_meta=eps_meta,
                subset_cache_key=subset_identity.subset_tag,
                lspo_source_fingerprint=subset_identity.source_fp,
                lspo_source_fingerprint_scheme=LSPO_SOURCE_FP_SCHEME,
                lspo_pairs_count=int(len(lspo_pairs)),
            )
            write_json(stage_metrics, stage_metrics_path)
            go = evaluate_go_no_go(stage_metrics, gate_config=gate_cfg)
            write_go_no_go_report(go, go_no_go_path)
            ui.done(f"GO={go['go']} with blockers={len(go.get('blockers', []))}.")
        write_json({"run_id": run_id, "cache_refs": cache_refs}, cache_refs_path)

        if args.baseline_run_id:
            if compare_path.exists() and not args.force:
                ui.info(f"Reused baseline comparison: {compare_path}")
            else:
                write_compare_train_to_baseline(
                    baseline_run_id=args.baseline_run_id,
                    current_run_id=run_id,
                    run_stage=stage,
                    metrics_root=art_cfg["metrics_dir"],
                    output_path=compare_path,
                )
                ui.info(f"Wrote baseline comparison: {compare_path}")
        ui.info(f"Stage metrics: {stage_metrics_path}")
        ui.info(f"Go/No-Go report: {go_no_go_path}")

        ui.start("Finalize train run")
        ui.done("Training artifacts ready.")

        ui.info(f"Run complete: {run_id}")

    except Exception as exc:
        ui.fail(str(exc))
        raise
    finally:
        ui.close()


def cmd_run_infer_sources(args):
    ui = CliUI(total_steps=8, progress=args.progress)
    try:
        from author_name_disambiguation.infer_sources import InferSourcesRequest, run_infer_sources

        result = run_infer_sources(
            InferSourcesRequest(
                publications_path=args.publications_path,
                references_path=args.references_path,
                output_root=args.output_root,
                dataset_id=args.dataset_id,
                model_bundle=args.model_bundle,
                uid_scope=args.uid_scope,
                uid_namespace=args.uid_namespace,
                infer_stage=args.infer_stage,
                cluster_config=args.cluster_config,
                gates_config=args.gates_config,
                device=args.device,
                precision_mode=args.precision_mode,
                specter_runtime_backend=args.specter_runtime_backend,
                cluster_backend=args.cluster_backend,
                force=bool(args.force),
                progress=bool(args.progress),
            )
        )
        payload = {
            "run_id": result.run_id,
            "go": result.go,
            "output_root": str(result.output_root),
            "publications_disambiguated_path": str(result.publications_disambiguated_path),
            "references_disambiguated_path": (
                None if result.references_disambiguated_path is None else str(result.references_disambiguated_path)
            ),
            "source_author_assignments_path": str(result.source_author_assignments_path),
            "author_entities_path": str(result.author_entities_path),
            "mention_clusters_path": str(result.mention_clusters_path),
            "stage_metrics_path": str(result.stage_metrics_path),
            "go_no_go_path": str(result.go_no_go_path),
        }
        print(json.dumps(payload, indent=2))
        return payload
    except Exception as exc:
        ui.fail(str(exc))
        raise
    finally:
        ui.close()


def cmd_precompute_source_embeddings(args):
    ui = CliUI(total_steps=1, progress=args.progress)
    try:
        from author_name_disambiguation.precompute_source_embeddings import (
            PrecomputeSourceEmbeddingsRequest,
            precompute_source_embeddings,
        )

        ui.start("Precompute remote source embeddings")
        ui.info(
            f"provider={args.provider} | model={args.model_name} | batch_size={int(args.batch_size)} | "
            f"output_root={Path(args.output_root).expanduser().resolve()}"
        )
        result = precompute_source_embeddings(
            PrecomputeSourceEmbeddingsRequest(
                publications_path=args.publications_path,
                references_path=args.references_path,
                output_root=args.output_root,
                dataset_id=args.dataset_id,
                provider=args.provider,
                model_name=args.model_name,
                hf_token_env_var=args.hf_token_env_var,
                batch_size=int(args.batch_size),
                max_retries=int(args.max_retries),
                base_backoff_seconds=float(args.base_backoff_seconds),
                max_backoff_seconds=float(args.max_backoff_seconds),
                force=bool(args.force),
                progress=bool(args.progress),
            )
        )
        payload = {
            "run_id": result.run_id,
            "output_root": str(result.output_root),
            "publications_output_path": str(result.publications_output_path),
            "references_output_path": None if result.references_output_path is None else str(result.references_output_path),
            "report_path": str(result.report_path),
        }
        ui.done(f"Wrote precomputed source artifacts to {result.output_root}")
        print(json.dumps(payload, indent=2))
        return payload
    except Exception as exc:
        ui.fail(str(exc))
        raise
    finally:
        ui.close()


def cmd_compare_infer_baseline(args):
    payload = None
    ui = CliUI(total_steps=1, progress=args.progress)
    try:
        ui.start("Compare infer run to baseline")
        baseline_ref = str(args.baseline_run_id)
        current_ref = str(args.current_run_id)
        metrics_root = Path(args.metrics_root).expanduser().resolve()
        current_candidate = Path(current_ref).expanduser()
        current_dir = current_candidate.resolve() if current_candidate.exists() else (metrics_root / current_ref).resolve()
        output_path = (
            Path(args.output_path).expanduser().resolve()
            if args.output_path is not None
            else (current_dir / "99_compare_infer_to_baseline.json").resolve()
        )
        ui.info(f"baseline={baseline_ref} | current={current_ref} | metrics_root={metrics_root}")
        report_path = write_compare_infer_to_baseline(
            baseline_run_id=baseline_ref,
            current_run_id=current_ref,
            run_stage="infer_sources",
            metrics_root=metrics_root,
            output_path=output_path,
        )
        payload = load_json(report_path)
        payload["output_path"] = str(report_path)
        ui.done(f"Wrote {report_path.name}")
        print(json.dumps(payload, indent=2))
        return payload
    except Exception as exc:
        ui.fail(str(exc))
        raise
    finally:
        ui.close()


def cmd_run_hf_compatibility_report(args):
    ui = CliUI(total_steps=1, progress=args.progress)
    try:
        from author_name_disambiguation.hf_compatibility_report import (
            HfCompatibilityReportRequest,
            run_hf_compatibility_report,
        )

        ui.start("Run HF compatibility report")
        ui.info(
            f"dataset={args.dataset_id} | bundle={Path(args.model_bundle).expanduser().resolve()} | "
            f"sample_size={int(args.sample_size)} | provider={args.provider} | model={args.model_name}"
        )
        result = run_hf_compatibility_report(
            HfCompatibilityReportRequest(
                publications_path=args.publications_path,
                references_path=args.references_path,
                output_root=args.output_root,
                dataset_id=args.dataset_id,
                model_bundle=args.model_bundle,
                sample_size=int(args.sample_size),
                provider=args.provider,
                model_name=args.model_name,
                hf_token_env_var=args.hf_token_env_var,
                batch_size=int(args.batch_size),
                device=args.device,
                force=bool(args.force),
                progress=bool(args.progress),
            )
        )
        payload = {
            "run_id": result.run_id,
            "output_root": str(result.output_root),
            "report_json_path": str(result.report_json_path),
            "report_markdown_path": str(result.report_markdown_path),
            "compatible": bool(result.compatible),
        }
        ui.done(f"Compatibility={result.compatible}")
        print(json.dumps(payload, indent=2))
        return payload
    except Exception as exc:
        ui.fail(str(exc))
        raise
    finally:
        ui.close()


def cmd_run_specter_benchmark(args):
    ui = CliUI(total_steps=1, progress=args.progress)
    try:
        from author_name_disambiguation.specter_benchmark import (
            SpecterBenchmarkRequest,
            run_specter_benchmark,
        )

        ui.start("Run SPECTER benchmark")
        ui.info(
            f"dataset={args.dataset_id} | bundle={Path(args.model_bundle).expanduser().resolve()} | "
            f"parity_sample={int(args.parity_sample_size)} | throughput_sample={int(args.throughput_sample_size)} | "
            f"provider={args.provider} | model={args.model_name}"
        )
        result = run_specter_benchmark(
            SpecterBenchmarkRequest(
                publications_path=args.publications_path,
                references_path=args.references_path,
                output_root=args.output_root,
                dataset_id=args.dataset_id,
                model_bundle=args.model_bundle,
                provider=args.provider,
                model_name=args.model_name,
                hf_token_env_var=args.hf_token_env_var,
                parity_sample_size=int(args.parity_sample_size),
                throughput_sample_size=int(args.throughput_sample_size),
                local_batch_size=args.local_batch_size,
                cpu_device=args.cpu_device,
                gpu_device=args.gpu_device,
                api_concurrency=int(args.api_concurrency),
                force=bool(args.force),
                progress=bool(args.progress),
            )
        )
        payload = {
            "run_id": result.run_id,
            "output_root": str(result.output_root),
            "report_json_path": str(result.report_json_path),
            "report_markdown_path": str(result.report_markdown_path),
            "recommendation": str(result.recommendation),
        }
        ui.done(str(result.recommendation))
        print(json.dumps(payload, indent=2))
        return payload
    except Exception as exc:
        ui.fail(str(exc))
        raise
    finally:
        ui.close()


def cmd_run_specter_hf_lab_benchmark(args):
    ui = CliUI(total_steps=1, progress=args.progress)
    try:
        from author_name_disambiguation.specter_hf_lab_benchmark import (
            SpecterHFLabBenchmarkRequest,
            run_specter_hf_lab_benchmark,
        )

        profiles = tuple(part.strip() for part in str(args.profiles).split(",") if part.strip())
        concurrency_values = tuple(
            int(part.strip()) for part in str(args.concurrency_values).split(",") if part.strip()
        )
        ui.start("Run SPECTER HF lab benchmark")
        ui.info(
            f"dataset={args.dataset_id} | bundle={Path(args.model_bundle).expanduser().resolve()} | "
            f"profiles={','.join(profiles or ('all',))} | concurrency={','.join(str(v) for v in concurrency_values or (4, 16, 64))}"
        )
        result = run_specter_hf_lab_benchmark(
            SpecterHFLabBenchmarkRequest(
                publications_path=args.publications_path,
                references_path=args.references_path,
                output_root=args.output_root,
                dataset_id=args.dataset_id,
                model_bundle=args.model_bundle,
                provider=args.provider,
                model_name=args.model_name,
                hf_token_env_var=args.hf_token_env_var,
                profiles=profiles or ("all",),
                concurrency_values=concurrency_values or (4, 16, 64),
                realistic_sample_size=int(args.realistic_sample_size),
                micro_repeat_count=int(args.micro_repeat_count),
                force=bool(args.force),
                progress=bool(args.progress),
            )
        )
        payload = {
            "run_id": result.run_id,
            "output_root": str(result.output_root),
            "report_json_path": str(result.report_json_path),
            "report_markdown_path": str(result.report_markdown_path),
            "summary": str(result.summary),
        }
        ui.done(str(result.summary))
        print(json.dumps(payload, indent=2))
        return payload
    except Exception as exc:
        ui.fail(str(exc))
        raise
    finally:
        ui.close()


def cmd_run_cluster_test_report(args):
    ui = CliUI(total_steps=6, progress=args.progress)
    try:
        ui.start("Initialize clustering test report context")
        _configure_library_noise(args.quiet_libs)

        paths = _build_public_workspace_paths(args)
        data_cfg = dict(paths["data"])
        art_cfg = dict(paths["artifacts"])
        run_dirs = build_run_dirs(data_cfg, art_cfg, str(args.model_run_id))

        model_run_id = str(args.model_run_id)
        metrics_dir = Path(str(art_cfg["metrics_dir"])) / model_run_id
        if not metrics_dir.exists():
            raise FileNotFoundError(f"Train metrics directory not found for model_run_id={model_run_id}: {metrics_dir}")
        report_tag = _sanitize_report_tag(args.report_tag)
        if args.cluster_config_override and report_tag is None:
            raise ValueError("--cluster-config-override requires --report-tag to avoid overwriting baseline 06_* reports.")

        context_path = metrics_dir / "00_context.json"
        train_manifest_path = metrics_dir / "03_train_manifest.json"
        cluster_used_path = metrics_dir / "04_clustering_config_used.json"
        for p in [context_path, train_manifest_path, cluster_used_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing required train artifact for clustering report: {p}")

        context_payload = load_json(context_path)
        if str(context_payload.get("pipeline_scope", "")).strip().lower() != "train":
            raise ValueError(
                f"Expected pipeline_scope=train in {context_path}, got {context_payload.get('pipeline_scope')!r}."
            )

        run_stage = str(context_payload.get("run_stage", "")).strip()
        if not run_stage:
            raise ValueError(f"run_stage missing in {context_path}.")

        stage_metrics_path = metrics_dir / f"05_stage_metrics_{run_stage}.json"
        if not stage_metrics_path.exists():
            raise FileNotFoundError(
                f"Missing stage metrics for run_stage={run_stage}: {stage_metrics_path}"
            )
        stage_metrics = load_json(stage_metrics_path)
        expected_subset_cache_key = str(stage_metrics.get("subset_cache_key", "")).strip()
        if not expected_subset_cache_key:
            raise ValueError(
                f"subset_cache_key missing in {stage_metrics_path}; cannot verify reproducibility."
            )
        expected_lspo_source_fingerprint_raw = stage_metrics.get("lspo_source_fingerprint")
        expected_lspo_source_fingerprint = (
            str(expected_lspo_source_fingerprint_raw).strip()
            if expected_lspo_source_fingerprint_raw is not None
            else None
        ) or None
        expected_lspo_source_fingerprint_scheme_raw = stage_metrics.get("lspo_source_fingerprint_scheme")
        expected_lspo_source_fingerprint_scheme = (
            str(expected_lspo_source_fingerprint_scheme_raw).strip()
            if expected_lspo_source_fingerprint_scheme_raw is not None
            else None
        ) or None

        train_manifest = load_json(train_manifest_path)
        seed_runs = _resolve_train_seed_runs(train_manifest)
        for row in seed_runs:
            ckpt = Path(row["checkpoint"])
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint for seed={row['seed']} does not exist: {ckpt}")

        cluster_used_payload = load_json(cluster_used_path)
        selected_eps = _resolve_selected_eps(cluster_used_payload, source_path=cluster_used_path)
        cluster_config_used = dict(cluster_used_payload.get("cluster_config_used", {}) or {})
        if len(cluster_config_used) == 0:
            raise ValueError(
                f"cluster_config_used missing or empty in {cluster_used_path}; cannot run clustering benchmark."
            )
        base_cluster_cfg, cluster_config_source_mode, cluster_config_override_path, override_ignored_fields = (
            _apply_cluster_config_override(
                base_cluster_config=cluster_config_used,
                override_path=args.cluster_config_override,
            )
        )
        base_cluster_cfg["eps"] = float(selected_eps)
        base_cluster_cfg["selected_eps"] = float(selected_eps)
        base_cluster_cfg["eps_mode"] = "fixed"
        min_samples = int(base_cluster_cfg.get("min_samples", 1))
        metric = str(base_cluster_cfg.get("metric", "precomputed"))

        raw_lspo_parquet = Path(str(data_cfg["raw_lspo_parquet"]))
        if not raw_lspo_parquet.exists():
            raise FileNotFoundError(
                f"LSPO parquet is required for report generation but was not found: {raw_lspo_parquet}"
            )
        raw_lspo_h5_val = data_cfg.get("raw_lspo_h5")
        raw_lspo_h5 = Path(str(raw_lspo_h5_val)) if raw_lspo_h5_val else None
        if raw_lspo_h5 is not None and not raw_lspo_h5.exists():
            warnings.warn(
                f"Optional LSPO H5 path is configured but missing: {raw_lspo_h5}",
                RuntimeWarning,
            )

        run_cfg = dict(context_payload.get("run_config_payload") or {})
        run_cfg_path = context_payload.get("run_config")
        if not run_cfg:
            if run_cfg_path:
                run_cfg = load_yaml(str(run_cfg_path))
            else:
                run_cfg, _ = _load_train_run_cfg(run_stage, None)
        run_cfg["stage"] = run_stage
        split_assignment_cfg = _resolve_split_assignment_cfg(run_cfg)
        pair_build_cfg = _resolve_pair_build_cfg(run_cfg)
        split_balance_cfg = dict(run_cfg.get("split_balance", {}) or {})

        model_cfg = dict(context_payload.get("model_config_payload") or {})
        model_cfg_path = context_payload.get("model_config")
        if not model_cfg:
            model_cfg, _ = _load_model_cfg(model_cfg_path)
        ui.done(f"Loaded train context for {model_run_id} (stage={run_stage}, seeds={len(seed_runs)}).")

        ui.start("Rebuild LSPO subset and verify subset fingerprint")
        lspo_mentions_path = Path(str(data_cfg["interim_dir"])) / "lspo_mentions.parquet"
        if lspo_mentions_path.exists() and not args.force:
            lspo_mentions = read_parquet(lspo_mentions_path)
        else:
            lspo_mentions = prepare_lspo_mentions(
                parquet_path=raw_lspo_parquet,
                h5_path=raw_lspo_h5,
                output_path=lspo_mentions_path,
            )

        source_fp = compute_lspo_source_fp(lspo_mentions_path)
        legacy_source_fp = compute_lspo_source_fp_legacy(lspo_mentions_path)
        subset_identity = compute_subset_identity(
            run_cfg=run_cfg,
            source_fp=source_fp,
            sampler_version=SUBSET_CACHE_VERSION,
        )
        legacy_subset_identity = compute_subset_identity(
            run_cfg=run_cfg,
            source_fp=legacy_source_fp,
            sampler_version=SUBSET_CACHE_VERSION,
        )

        lspo_subset = build_stage_subset(
            lspo_mentions,
            stage=run_stage,
            seed=int(run_cfg.get("seed", 11)),
            target_mentions=run_cfg.get("subset_target_mentions"),
            subset_sampling=run_cfg.get("subset_sampling", {}),
        )
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
        if str(split_meta.get("status", "")).strip().lower() == "split_balance_infeasible":
            raise RuntimeError(
                "LSPO split assignment is infeasible under current split balance config; "
                f"split_meta={split_meta}"
            )

        pair_result = build_pairs_within_blocks(
            mentions=lspo_mentions_split,
            max_pairs_per_block=run_cfg.get("max_pairs_per_block"),
            seed=int(run_cfg.get("seed", 11)),
            require_same_split=True,
            labeled_only=False,
            balance_train=True,
            exclude_same_bibcode=bool(pair_build_cfg.get("exclude_same_bibcode", True)),
            show_progress=False,
            return_meta=True,
        )
        if isinstance(pair_result, tuple):
            lspo_pairs, lspo_pair_build_meta = pair_result
        else:
            lspo_pairs = pair_result
            lspo_pair_build_meta = {}
        if lspo_pairs is None:
            raise RuntimeError("Pair builder returned None during clustering report generation.")
        empty_ads_pairs = pd.DataFrame(columns=PAIR_REQUIRED_COLUMNS + ["label"])
        current_pairs_qc = build_pairs_qc(
            lspo_mentions=lspo_mentions_split,
            lspo_pairs=lspo_pairs,
            ads_pairs=empty_ads_pairs,
            split_meta=split_meta,
            lspo_pair_build_meta=lspo_pair_build_meta,
            ads_pair_build_meta={},
        )

        subset_verification_mode = "strict"
        if subset_identity.subset_tag != expected_subset_cache_key:
            legacy_mode_allowed = expected_lspo_source_fingerprint_scheme in (None, "", LSPO_SOURCE_FP_SCHEME_LEGACY)
            if not args.allow_legacy_lspo_compat:
                raise ValueError(
                    "Subset reproducibility check failed: computed stable subset_cache_key does not match train stage metrics. "
                    f"stable={subset_identity.subset_tag}, legacy={legacy_subset_identity.subset_tag}, "
                    f"expected={expected_subset_cache_key}."
                )
            if not legacy_mode_allowed:
                raise ValueError(
                    "Legacy LSPO compatibility is only supported for model runs without stable LSPO fingerprint metadata. "
                    f"expected_scheme={expected_lspo_source_fingerprint_scheme!r}."
                )

            subset_summary_path = metrics_dir / "01_subset_summary.json"
            split_meta_expected_path = metrics_dir / "02_split_balance.json"
            pairs_qc_expected_path = metrics_dir / "02_pairs_qc.json"
            for required_path in [subset_summary_path, split_meta_expected_path, pairs_qc_expected_path]:
                if not required_path.exists():
                    raise FileNotFoundError(
                        f"Legacy LSPO compatibility requires historical train artifact: {required_path}"
                    )

            expected_subset_summary = load_json(subset_summary_path)
            expected_split_meta = load_json(split_meta_expected_path)
            expected_pairs_qc = load_json(pairs_qc_expected_path)
            current_subset_summary = {
                "lspo_mentions": len(lspo_subset),
                "lspo_blocks": int(lspo_subset["block_key"].nunique()) if "block_key" in lspo_subset.columns else 0,
                "lspo_block_size_p95": float(lspo_subset.groupby("block_key").size().quantile(0.95))
                if len(lspo_subset) and "block_key" in lspo_subset.columns
                else 0.0,
            }
            compat_issues: list[str] = []
            compat_issues.extend(
                _collect_snapshot_mismatches(
                    "subset_summary",
                    _snapshot_legacy_subset_summary(expected_subset_summary),
                    _snapshot_legacy_subset_summary(current_subset_summary),
                )
            )
            compat_issues.extend(
                _collect_snapshot_mismatches(
                    "split_balance",
                    _snapshot_legacy_split_balance(expected_split_meta),
                    _snapshot_legacy_split_balance(split_meta),
                )
            )
            compat_issues.extend(
                _collect_snapshot_mismatches(
                    "pairs_qc",
                    _snapshot_legacy_pairs_qc(expected_pairs_qc),
                    _snapshot_legacy_pairs_qc(current_pairs_qc),
                )
            )
            if compat_issues:
                preview = "; ".join(compat_issues[:6])
                if len(compat_issues) > 6:
                    preview = f"{preview}; ... ({len(compat_issues)} mismatches total)"
                raise ValueError(
                    "Legacy LSPO compatibility check failed: reconstructed subset artifacts do not match historical train artifacts. "
                    f"{preview}"
                )
            subset_verification_mode = "legacy_compat"

        test_mentions = lspo_mentions_split[lspo_mentions_split["split"].astype(str) == "test"].reset_index(drop=True)
        test_pairs = lspo_pairs[
            (lspo_pairs["split"].astype(str) == "test") & lspo_pairs["label"].notna()
        ].reset_index(drop=True)
        if len(test_mentions) < 2:
            raise RuntimeError(
                f"Test split has too few mentions for clustering benchmark: {len(test_mentions)}"
            )
        if len(test_pairs) == 0:
            raise RuntimeError("No labeled LSPO test pairs available for clustering benchmark.")
        ui.done(
            "Prepared LSPO test split "
            f"({len(test_mentions)} mentions, {len(test_pairs)} labeled pairs) "
            f"| verification={subset_verification_mode}"
        )

        ui.start("Build or load LSPO embeddings")
        rep_cfg = dict(model_cfg.get("representation", {}) or {})
        representation_cfg_hash = stable_hash(rep_cfg)
        model_version = str(model_cfg.get("name", "nand"))
        embedding_id = stable_hash(
            {
                "subset_id": subset_identity.subset_tag,
                "representation_cfg_hash": representation_cfg_hash,
                "model_version": model_version,
                "pipeline_scope": "train",
            }
        )
        shared_embeddings_dir = Path(run_dirs["shared_embeddings"])
        shared_embeddings_dir.mkdir(parents=True, exist_ok=True)
        lspo_chars_path = shared_embeddings_dir / f"lspo_chars2vec_{embedding_id}.npy"
        lspo_text_path = shared_embeddings_dir / f"lspo_specter_{embedding_id}.npy"

        lspo_chars = get_or_create_chars2vec_embeddings(
            mentions=lspo_subset,
            output_path=lspo_chars_path,
            force_recompute=bool(args.force),
            batch_size=32,
            execution_mode="predict",
            use_stub_if_missing=False,
            quiet_libraries=bool(args.quiet_libs),
            show_progress=bool(args.progress),
        )
        lspo_text = get_or_create_specter_embeddings(
            mentions=lspo_subset,
            output_path=lspo_text_path,
            force_recompute=bool(args.force),
            model_name=rep_cfg.get("text_model_name", "allenai/specter"),
            text_backend=rep_cfg.get("text_backend", "transformers"),
            text_adapter_name=rep_cfg.get("text_adapter_name"),
            text_adapter_alias=rep_cfg.get("text_adapter_alias", "specter2"),
            max_length=int(rep_cfg.get("max_length", 256)),
            batch_size=16,
            device=args.device,
            prefer_precomputed=False,
            use_stub_if_missing=False,
            show_progress=bool(args.progress),
            quiet_libraries=bool(args.quiet_libs),
            reuse_model=True,
        )
        subset_index = {
            str(m): int(i)
            for i, m in enumerate(lspo_subset["mention_id"].astype(str).tolist())
        }
        test_idx = test_mentions["mention_id"].astype(str).map(subset_index)
        if test_idx.isna().any():
            raise RuntimeError("Failed to align test mentions with embedding indices from reconstructed subset.")
        test_idx_np = test_idx.astype(int).to_numpy()
        test_chars = lspo_chars[test_idx_np]
        test_text = lspo_text[test_idx_np]
        ui.done(f"Embeddings ready for test split ({tuple(test_chars.shape)}/{tuple(test_text.shape)}).")

        ui.start("Evaluate clustering variants across train seeds")
        variants = [
            ("dbscan_no_constraints", False),
            ("dbscan_with_constraints", True),
        ]
        per_seed_rows: list[dict[str, Any]] = []
        for run_row in seed_runs:
            seed = int(run_row["seed"])
            checkpoint = Path(run_row["checkpoint"])
            threshold = float(run_row["threshold"])

            pair_scores = score_pairs_with_checkpoint(
                mentions=test_mentions,
                pairs=test_pairs,
                chars2vec=test_chars,
                text_emb=test_text,
                checkpoint_path=checkpoint,
                output_path=None,
                batch_size=int(args.score_batch_size),
                device=args.device,
                precision_mode=args.precision_mode,
                show_progress=False,
            )

            for variant_name, enable_constraints in variants:
                eval_cfg = _build_cluster_variant_config(
                    base_cluster_cfg,
                    enable_constraints=enable_constraints,
                )
                clusters = cluster_blockwise_dbscan(
                    mentions=test_mentions,
                    pair_scores=pair_scores,
                    cluster_config=eval_cfg,
                    output_path=None,
                    show_progress=False,
                )
                metrics = _cluster_pairwise_metrics(test_pairs, clusters)
                if metrics.get("f1") is None:
                    raise RuntimeError(
                        "Cluster metrics are empty for test split; cannot build final report."
                    )
                per_seed_rows.append(
                    {
                        "seed": seed,
                        "checkpoint": str(checkpoint),
                        "threshold": threshold,
                        "variant": variant_name,
                        "accuracy": float(metrics["accuracy"]),
                        "precision": float(metrics["precision"]),
                        "recall": float(metrics["recall"]),
                        "f1": float(metrics["f1"]),
                        "n_pairs": int(metrics["n_pairs"]),
                    }
                )
        ui.done(f"Evaluated {len(seed_runs)} seeds across {len(variants)} variants.")

        ui.start("Write final clustering test report artifacts")
        summary_payload = _summarize_cluster_test_rows(per_seed_rows)
        summary_rows: list[dict[str, Any]] = []
        for variant in sorted(summary_payload.keys()):
            summary_rows.append(
                {
                    "variant": variant,
                    **dict(summary_payload[variant]),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["variant", "seed"]).reset_index(drop=True)

        with_constraints = dict(summary_payload.get("dbscan_with_constraints", {}) or {})
        no_constraints = dict(summary_payload.get("dbscan_no_constraints", {}) or {})
        delta = {}
        for key in ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]:
            a = with_constraints.get(key)
            b = no_constraints.get(key)
            delta_key = key.replace("_mean", "")
            delta[delta_key] = None if a is None or b is None else float(a) - float(b)

        report_payload = {
            "model_run_id": model_run_id,
            "run_stage": run_stage,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_scope": "train",
            "source_context_path": str(context_path),
            "train_manifest_path": str(train_manifest_path),
            "cluster_config_used_path": str(cluster_used_path),
            "cluster_config_source_mode": str(cluster_config_source_mode),
            "cluster_config_override_path": cluster_config_override_path,
            "override_ignored_fields": override_ignored_fields,
            "report_tag": report_tag,
            "lspo_source_paths": {
                "raw_lspo_parquet": str(raw_lspo_parquet),
                "raw_lspo_h5": None if raw_lspo_h5 is None else str(raw_lspo_h5),
                "interim_lspo_mentions": str(lspo_mentions_path),
            },
            "lspo_source_fingerprint": subset_identity.source_fp,
            "lspo_source_fingerprint_scheme": LSPO_SOURCE_FP_SCHEME,
            "lspo_source_fingerprint_expected": expected_lspo_source_fingerprint,
            "lspo_source_fingerprint_scheme_expected": expected_lspo_source_fingerprint_scheme,
            "subset_verification_mode": subset_verification_mode,
            "subset_cache_key_expected": expected_subset_cache_key,
            "subset_cache_key_computed": subset_identity.subset_tag,
            "subset_cache_key_stable_computed": subset_identity.subset_tag,
            "subset_cache_key_legacy_computed": legacy_subset_identity.subset_tag,
            "seeds_expected": [int(row["seed"]) for row in seed_runs],
            "seeds_evaluated": sorted({int(row["seed"]) for row in per_seed_rows}),
            "selected_eps": float(selected_eps),
            "min_samples": int(min_samples),
            "metric": str(metric),
            "cluster_config_effective": base_cluster_cfg,
            "variants": summary_payload,
            "per_seed_rows": per_seed_rows,
            "delta_with_constraints_minus_no_constraints": delta,
            "status": "ok",
        }
        report_md = _build_cluster_test_report_markdown(report_payload)

        report_paths = _resolve_report_paths(metrics_dir, report_tag=report_tag)
        report_json_path = report_paths["json"]
        report_summary_csv_path = report_paths["summary_csv"]
        report_per_seed_csv_path = report_paths["per_seed_csv"]
        report_md_path = report_paths["markdown"]

        write_json(report_payload, report_json_path)
        summary_df.to_csv(report_summary_csv_path, index=False)
        per_seed_df.to_csv(report_per_seed_csv_path, index=False)
        report_md_path.write_text(report_md, encoding="utf-8")
        ui.done("Wrote 06_clustering_test_report.{json,csv,md} artifacts.")

        ui.start("Finalize")
        ui.info(f"Report JSON: {report_json_path}")
        ui.info(f"Summary CSV: {report_summary_csv_path}")
        ui.info(f"Per-seed CSV: {report_per_seed_csv_path}")
        ui.info(f"Report MD: {report_md_path}")
        ui.done(f"Run complete: {model_run_id}")
    except Exception as exc:
        ui.fail(str(exc))
        raise
    finally:
        ui.close()


def cmd_export_model_bundle(args):
    meta = _write_model_bundle(
        artifacts_root=args.artifacts_root,
        model_run_id=args.model_run_id,
        output_dir=args.output_dir,
    )
    print(json.dumps({"model_run_id": args.model_run_id, **{k: str(v) for k, v in meta.items()}}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="author_name_disambiguation operator CLI")
    sub = p.add_subparsers(dest="command", required=True)

    def _add_progress_and_logging_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--progress", dest="progress", action="store_true")
        sp.add_argument("--no-progress", dest="progress", action="store_false")
        sp.set_defaults(progress=True)
        sp.add_argument("--quiet-libs", dest="quiet_libs", action="store_true")
        sp.add_argument("--verbose-libs", dest="quiet_libs", action="store_false")
        sp.set_defaults(quiet_libs=True)

    def _add_public_workspace_args(
        sp: argparse.ArgumentParser,
        *,
        include_raw_lspo: bool,
    ) -> None:
        sp.add_argument("--data-root", required=True)
        sp.add_argument("--artifacts-root", required=True)
        if include_raw_lspo:
            sp.add_argument("--raw-lspo-parquet", required=True)
            sp.add_argument("--raw-lspo-h5", default=None)

    def _add_train_stage_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--run-stage", required=True, choices=["smoke", "mini", "mid", "full"])
        _add_public_workspace_args(sp, include_raw_lspo=True)
        sp.add_argument("--run-config", default=None)
        sp.add_argument("--model-config", default=None)
        sp.add_argument("--cluster-config", default=None)
        sp.add_argument("--gates-config", default=None)
        sp.add_argument("--run-id", default=None)
        sp.add_argument("--device", default="auto")
        sp.add_argument("--precision-mode", choices=["fp32", "amp_bf16"], default=None)
        sp.add_argument("--seeds", nargs="+", type=int, default=None)
        sp.add_argument("--use-stub-embeddings", action="store_true")
        sp.add_argument("--force", action="store_true")
        sp.add_argument("--baseline-run-id", default=None)
        sp.add_argument("--score-batch-size", type=int, default=8192)
        _add_progress_and_logging_args(sp)

    sp = sub.add_parser("run-train-stage")
    _add_train_stage_args(sp)
    sp.set_defaults(func=cmd_run_train_stage)

    sp = sub.add_parser("run-infer-sources")
    sp.add_argument("--publications-path", required=True)
    sp.add_argument("--references-path", default=None)
    sp.add_argument("--output-root", required=True)
    sp.add_argument("--dataset-id", required=True)
    sp.add_argument("--model-bundle", required=True)
    sp.add_argument("--infer-stage", choices=["smoke", "mini", "mid", "full"], default="full")
    sp.add_argument("--cluster-config", default=None)
    sp.add_argument("--gates-config", default=None)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--precision-mode", choices=["fp32", "amp_bf16"], default="fp32")
    sp.add_argument("--specter-runtime-backend", choices=["transformers", "onnx_fp32"], default=None)
    sp.add_argument("--cluster-backend", choices=["auto", "sklearn_cpu", "cuml_gpu"], default=None)
    sp.add_argument("--uid-scope", choices=["dataset", "local", "registry"], default="dataset")
    sp.add_argument("--uid-namespace", default=None)
    sp.add_argument("--force", action="store_true")
    _add_progress_and_logging_args(sp)
    sp.set_defaults(func=cmd_run_infer_sources)

    sp = sub.add_parser("precompute-source-embeddings")
    sp.add_argument("--publications-path", required=True)
    sp.add_argument("--references-path", default=None)
    sp.add_argument("--output-root", required=True)
    sp.add_argument("--dataset-id", default=None)
    sp.add_argument("--provider", default="hf-inference")
    sp.add_argument("--model-name", default="allenai/specter")
    sp.add_argument("--hf-token-env-var", default="HF_TOKEN")
    sp.add_argument("--batch-size", type=int, default=32)
    sp.add_argument("--max-retries", type=int, default=5)
    sp.add_argument("--base-backoff-seconds", type=float, default=1.0)
    sp.add_argument("--max-backoff-seconds", type=float, default=30.0)
    sp.add_argument("--force", action="store_true")
    _add_progress_and_logging_args(sp)
    sp.set_defaults(func=cmd_precompute_source_embeddings)

    sp = sub.add_parser("compare-infer-baseline")
    sp.add_argument("--baseline-run-id", required=True)
    sp.add_argument("--current-run-id", required=True)
    sp.add_argument("--metrics-root", required=True)
    sp.add_argument("--output-path", default=None)
    _add_progress_and_logging_args(sp)
    sp.set_defaults(func=cmd_compare_infer_baseline)

    sp = sub.add_parser("run-hf-compatibility-report")
    sp.add_argument("--publications-path", required=True)
    sp.add_argument("--references-path", default=None)
    sp.add_argument("--output-root", required=True)
    sp.add_argument("--dataset-id", required=True)
    sp.add_argument("--model-bundle", required=True)
    sp.add_argument("--sample-size", type=int, default=128)
    sp.add_argument("--provider", default="hf-inference")
    sp.add_argument("--model-name", default="allenai/specter")
    sp.add_argument("--hf-token-env-var", default="HF_TOKEN")
    sp.add_argument("--batch-size", type=int, default=32)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--force", action="store_true")
    _add_progress_and_logging_args(sp)
    sp.set_defaults(func=cmd_run_hf_compatibility_report)

    sp = sub.add_parser("run-specter-benchmark")
    sp.add_argument("--publications-path", required=True)
    sp.add_argument("--references-path", default=None)
    sp.add_argument("--output-root", required=True)
    sp.add_argument("--dataset-id", required=True)
    sp.add_argument("--model-bundle", required=True)
    sp.add_argument("--provider", default="hf-inference")
    sp.add_argument("--model-name", default="allenai/specter")
    sp.add_argument("--hf-token-env-var", default="HF_TOKEN")
    sp.add_argument("--parity-sample-size", type=int, default=128)
    sp.add_argument("--throughput-sample-size", type=int, default=2048)
    sp.add_argument("--local-batch-size", type=int, default=None)
    sp.add_argument("--cpu-device", default="cpu")
    sp.add_argument("--gpu-device", default="cuda")
    sp.add_argument("--api-concurrency", type=int, default=4)
    sp.add_argument("--force", action="store_true")
    _add_progress_and_logging_args(sp)
    sp.set_defaults(func=cmd_run_specter_benchmark)

    sp = sub.add_parser("run-specter-hf-lab-benchmark")
    sp.add_argument("--publications-path", required=True)
    sp.add_argument("--references-path", default=None)
    sp.add_argument("--output-root", required=True)
    sp.add_argument("--dataset-id", required=True)
    sp.add_argument("--model-bundle", required=True)
    sp.add_argument("--provider", default="hf-inference")
    sp.add_argument("--model-name", default="allenai/specter")
    sp.add_argument("--hf-token-env-var", default="HF_TOKEN")
    sp.add_argument("--profiles", default="all")
    sp.add_argument("--concurrency-values", default="4,16,64")
    sp.add_argument("--realistic-sample-size", type=int, default=128)
    sp.add_argument("--micro-repeat-count", type=int, default=1000)
    sp.add_argument("--force", action="store_true")
    _add_progress_and_logging_args(sp)
    sp.set_defaults(func=cmd_run_specter_hf_lab_benchmark)

    sp = sub.add_parser("run-cluster-test-report")
    sp.add_argument("--model-run-id", required=True)
    _add_public_workspace_args(sp, include_raw_lspo=True)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--precision-mode", choices=["fp32", "amp_bf16"], default="fp32")
    sp.add_argument("--score-batch-size", type=int, default=8192)
    sp.add_argument("--cluster-config-override", default=None)
    sp.add_argument("--report-tag", default=None)
    sp.add_argument("--allow-legacy-lspo-compat", action="store_true")
    sp.add_argument("--force", action="store_true")
    _add_progress_and_logging_args(sp)
    sp.set_defaults(func=cmd_run_cluster_test_report)

    sp = sub.add_parser("export-model-bundle")
    sp.add_argument("--model-run-id", required=True)
    sp.add_argument("--artifacts-root", required=True)
    sp.add_argument("--output-dir", default=None)
    sp.set_defaults(func=cmd_export_model_bundle)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    _configure_library_noise(bool(getattr(args, "quiet_libs", False)))
    args.func(args)


if __name__ == "__main__":
    main()
