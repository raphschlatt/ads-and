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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.approaches.nand.build_pairs import assign_lspo_splits, build_pairs_within_blocks, write_pairs
from src.approaches.nand.cluster import cluster_blockwise_dbscan, resolve_dbscan_eps
from src.approaches.nand.export import build_publication_author_mapping
from src.approaches.nand.infer_pairs import score_pairs_with_checkpoint
from src.approaches.nand.train import train_nand_across_seeds
from src.common.cli_ui import CliUI
from src.common.config import (
    build_run_dirs,
    find_project_root,
    load_yaml,
    resolve_existing_path,
    resolve_paths_config,
    write_latest_run_context,
    write_run_consistency,
)
from src.common.io_schema import read_parquet, save_parquet
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


def _build_eps_sweep_values(cluster_cfg: dict[str, Any]) -> list[float]:
    eps_min = float(cluster_cfg.get("eps_sweep_min", 0.2))
    eps_max = float(cluster_cfg.get("eps_sweep_max", 0.5))
    eps_step = float(cluster_cfg.get("eps_sweep_step", 0.05))
    if eps_step <= 0:
        raise ValueError(f"eps_sweep_step must be > 0, got {eps_step}")
    if eps_min > eps_max:
        raise ValueError(f"eps_sweep_min must be <= eps_sweep_max, got {eps_min} > {eps_max}")
    values = np.arange(eps_min, eps_max + (eps_step * 0.5), eps_step)
    return [float(np.round(v, 6)) for v in values.tolist()]


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
) -> tuple[float, dict[str, Any]]:
    eps_mode = str(cluster_cfg.get("eps_mode", "fixed")).lower()
    if eps_mode != "val_sweep":
        return resolve_dbscan_eps(cluster_cfg, cosine_threshold=best_threshold)

    sweep_rows: list[dict[str, Any]] = []
    fallback_cfg = dict(cluster_cfg)
    fallback_cfg["selected_eps"] = None

    val_mask = lspo_mentions_split["split"].astype(str) == "val"
    val_mentions = lspo_mentions_split[val_mask].reset_index(drop=True)
    val_pairs = lspo_pairs[(lspo_pairs["split"] == "val") & lspo_pairs["label"].notna()].copy()
    if len(val_mentions) < 2 or len(val_pairs) == 0:
        resolved, base_meta = resolve_dbscan_eps(fallback_cfg, cosine_threshold=best_threshold)
        base_meta.update(
            {
                "sweep_status": "fallback_no_val_pairs",
                "sweep_results": sweep_rows,
                "n_valid_candidates": 0,
                "boundary_hit": False,
                "boundary_side": None,
                "f1_gap_best_second": None,
            }
        )
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
        show_progress=show_progress,
    )

    eps_values = _build_eps_sweep_values(cluster_cfg)
    for eps in eps_values:
        row: dict[str, Any] = {"eps": float(eps)}
        try:
            eval_cfg = json.loads(json.dumps(cluster_cfg))
            eval_cfg["eps"] = float(eps)
            clusters = cluster_blockwise_dbscan(
                mentions=val_mentions,
                pair_scores=val_pair_scores,
                cluster_config=eval_cfg,
                output_path=None,
                show_progress=False,
            )
            metrics = _cluster_pairwise_metrics(val_pairs, clusters)
            row.update(metrics)
        except Exception as exc:
            row["error"] = repr(exc)
        sweep_rows.append(row)

    valid_rows = [r for r in sweep_rows if r.get("f1") is not None]
    if not valid_rows:
        resolved, base_meta = resolve_dbscan_eps(fallback_cfg, cosine_threshold=best_threshold)
        base_meta.update(
            {
                "sweep_status": "fallback_no_valid_candidates",
                "sweep_results": sweep_rows,
                "n_valid_candidates": 0,
                "boundary_hit": False,
                "boundary_side": None,
                "f1_gap_best_second": None,
            }
        )
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
    training_cfg = model_cfg.get("training", {})
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
        training_cfg = model_cfg.get("training", {})

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
        subset_identity = compute_subset_identity(run_cfg=run_cfg, source_fp=source_fp, sampler_version="v2")
        subset_paths = resolve_shared_subset_paths(data_cfg=data_cfg, identity=subset_identity)
        manifest_paths = resolve_manifest_paths(
            run_id=run_id,
            manifest_dir=Path(run_dirs["subset_manifests"]),
            identity=subset_identity,
            run_stage=stage,
        )
        subset_paths.shared_dir.mkdir(parents=True, exist_ok=True)

        cache_hit = False
        if subset_paths.lspo_shared.exists() and subset_paths.ads_shared.exists() and not args.force:
            cache_hit = True
            t0 = perf_counter()
            lspo_subset = read_parquet(subset_paths.lspo_shared)
            t1 = perf_counter()
            ads_subset = read_parquet(subset_paths.ads_shared)
            t2 = perf_counter()
            timings["read_lspo_s"] = t1 - t0
            timings["read_ads_s"] = t2 - t1
        else:
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

        t5 = perf_counter()
        atomic_save_parquet(lspo_subset, lspo_subset_run_path, index=False)
        t6 = perf_counter()
        atomic_save_parquet(ads_subset, ads_subset_run_path, index=False)
        t7 = perf_counter()
        timings["save_lspo_run_s"] = t6 - t5
        timings["save_ads_run_s"] = t7 - t6

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
            cache_hit=cache_hit,
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
        if cache_hit:
            ui.skip(f"Shared cache hit: {subset_identity.subset_tag}")
        else:
            ui.done(f"Built subsets ({len(lspo_subset)} LSPO / {len(ads_subset)} ADS).")

        ui.start("Build or load embeddings")
        emb_cache_hit = (
            lspo_chars_path.exists()
            and lspo_text_path.exists()
            and ads_chars_path.exists()
            and ads_text_path.exists()
            and not args.force
        )

        lspo_chars = get_or_create_chars2vec_embeddings(
            mentions=lspo_subset,
            output_path=lspo_chars_path,
            force_recompute=args.force,
            use_stub_if_missing=args.use_stub_embeddings,
            quiet_libraries=args.quiet_libs,
        )
        lspo_text = get_or_create_specter_embeddings(
            mentions=lspo_subset,
            output_path=lspo_text_path,
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
            output_path=ads_chars_path,
            force_recompute=args.force,
            use_stub_if_missing=args.use_stub_embeddings,
            quiet_libraries=args.quiet_libs,
        )
        ads_text = get_or_create_specter_embeddings(
            mentions=ads_subset,
            output_path=ads_text_path,
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

        if emb_cache_hit:
            ui.skip("Reused cached embeddings.")
        else:
            ui.done(
                f"Embeddings ready (LSPO {tuple(lspo_chars.shape)}/{tuple(lspo_text.shape)}, "
                f"ADS {tuple(ads_chars.shape)}/{tuple(ads_text.shape)})."
            )

        ui.start("Assign LSPO splits and build LSPO pairs")
        if lspo_pairs_path.exists() and split_meta_path.exists() and lspo_subset_run_path.exists() and not args.force:
            lspo_mentions_split = read_parquet(lspo_subset_run_path)
            lspo_pairs = read_parquet(lspo_pairs_path)
            split_meta = _load_json(split_meta_path)
            lspo_pair_meta: dict[str, Any] = {}
            ui.skip(f"Reused LSPO split+pairs ({len(lspo_pairs)} pairs).")
        else:
            split_balance_cfg = run_cfg.get("split_balance", {})
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
            write_pairs(lspo_pairs, lspo_pairs_path)
            save_parquet(lspo_mentions_split, lspo_subset_run_path, index=False)
            write_json(split_meta, split_meta_path)
            ui.done(f"Built LSPO pairs ({len(lspo_pairs)} rows).")

        ui.start("Build ADS pairs and pair QC")
        if ads_pairs_path.exists() and pairs_qc_path.exists() and not args.force:
            ads_pairs = read_parquet(ads_pairs_path)
            pairs_qc = _load_json(pairs_qc_path)
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
            write_pairs(ads_pairs, ads_pairs_path)
            pairs_qc = build_pairs_qc(
                lspo_mentions=lspo_mentions_split,
                lspo_pairs=lspo_pairs,
                ads_pairs=ads_pairs,
                split_meta=split_meta,
                lspo_pair_build_meta=lspo_pair_meta,
                ads_pair_build_meta=ads_pair_meta,
            )
            write_json(pairs_qc, pairs_qc_path)
            ui.done(f"Built ADS pairs ({len(ads_pairs)} rows).")

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
        if pair_scores_path.exists() and not args.force:
            pair_scores = read_parquet(pair_scores_path)
            ui.skip(f"Reused pair scores ({len(pair_scores)} rows).")
        else:
            pair_scores = score_pairs_with_checkpoint(
                mentions=ads_subset,
                pairs=ads_pairs,
                chars2vec=ads_chars,
                text_emb=ads_text,
                checkpoint_path=train_manifest["best_checkpoint"],
                output_path=pair_scores_path,
                batch_size=int(args.score_batch_size),
                device=args.device,
                show_progress=args.progress,
            )
            ui.done(f"Scored {len(pair_scores)} ADS pairs.")

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
                show_progress=args.progress,
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
                ads_mentions=ads_subset,
                clusters=clusters,
                train_manifest=train_manifest,
                consistency_files=consistency_files,
                determinism_paths=determinism_paths,
                cluster_qc=cluster_qc,
                split_meta=split_meta,
                eps_meta=eps_meta,
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

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
