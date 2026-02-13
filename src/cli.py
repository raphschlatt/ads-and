from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

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
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{stage}_{ts}_cli{uuid.uuid4().hex[:8]}"


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
    mentions = read_parquet(args.mentions)
    model_cfg = _load_model_cfg(args.model_config)
    rep_cfg = model_cfg.get("representation", {})

    chars = get_or_create_chars2vec_embeddings(
        mentions=mentions,
        output_path=args.chars_out,
        force_recompute=args.force,
        use_stub_if_missing=args.use_stub,
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
    )
    print(f"Chars2Vec embeddings: {chars.shape} -> {args.chars_out}")
    print(f"Text embeddings: {text.shape} -> {args.text_out}")


def cmd_pairs(args):
    mentions = read_parquet(args.mentions)

    split_meta = None
    if args.assign_lspo_splits:
        split_cfg = {}
        if args.run_config:
            split_cfg = _load_run_cfg(args.run_config).get("split_balance", {})

        min_neg_val = int(args.min_neg_val) if args.min_neg_val is not None else int(split_cfg.get("min_neg_val", 0))
        min_neg_test = int(args.min_neg_test) if args.min_neg_test is not None else int(split_cfg.get("min_neg_test", 0))
        max_attempts = int(args.max_attempts) if args.max_attempts is not None else int(split_cfg.get("max_attempts", 1))

        mentions, split_meta = assign_lspo_splits(
            mentions,
            seed=args.seed,
            min_neg_val=min_neg_val,
            min_neg_test=min_neg_test,
            max_attempts=max_attempts,
            return_meta=True,
        )
        save_parquet(mentions, args.mentions, index=False)

    pairs = build_pairs_within_blocks(
        mentions=mentions,
        max_pairs_per_block=args.max_pairs_per_block,
        seed=args.seed,
        require_same_split=not args.allow_cross_split,
        labeled_only=args.labeled_only,
        balance_train=args.balance_train,
        show_progress=args.progress,
    )
    write_pairs(pairs, args.output)
    if split_meta is not None:
        print(f"Split balancing: {split_meta}")
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
        paths = _load_paths_cfg(args.paths_config)
        data_cfg = paths["data"]
        art_cfg = paths["artifacts"]

        run_cfg_path = args.run_config or f"configs/runs/{args.run_stage}.yaml"
        run_cfg = _load_run_cfg(run_cfg_path)
        run_cfg["stage"] = args.run_stage

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
            extras={"created_utc": datetime.utcnow().isoformat(), "source": "cli.run-stage"},
        )
        write_json(
            {
                "run_id": run_id,
                "run_stage": args.run_stage,
                "device": args.device,
                "use_stub_embeddings": bool(args.use_stub_embeddings),
                "prefer_precomputed_ads": bool(args.prefer_precomputed_ads),
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
        )
        ads_chars = get_or_create_chars2vec_embeddings(
            mentions=ads_subset,
            output_path=ads_chars_path,
            force_recompute=args.force,
            use_stub_if_missing=args.use_stub_embeddings,
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
            ui.skip(f"Reused LSPO split+pairs ({len(lspo_pairs)} pairs).")
        else:
            split_balance_cfg = run_cfg.get("split_balance", {})
            lspo_mentions_split, split_meta = assign_lspo_splits(
                lspo_subset,
                seed=int(run_cfg.get("seed", 11)),
                min_neg_val=int(split_balance_cfg.get("min_neg_val", 0)),
                min_neg_test=int(split_balance_cfg.get("min_neg_test", 0)),
                max_attempts=int(split_balance_cfg.get("max_attempts", 1)),
                return_meta=True,
            )
            lspo_pairs = build_pairs_within_blocks(
                mentions=lspo_mentions_split,
                max_pairs_per_block=run_cfg.get("max_pairs_per_block"),
                seed=int(run_cfg.get("seed", 11)),
                require_same_split=True,
                labeled_only=False,
                balance_train=True,
                show_progress=args.progress,
            )
            write_pairs(lspo_pairs, lspo_pairs_path)
            save_parquet(lspo_mentions_split, lspo_subset_run_path, index=False)
            write_json(split_meta, split_meta_path)
            ui.done(f"Built LSPO pairs ({len(lspo_pairs)} rows).")

        ui.start("Build ADS pairs and pair QC")
        if ads_pairs_path.exists() and pairs_qc_path.exists() and not args.force:
            ads_pairs = read_parquet(ads_pairs_path)
            pairs_qc = _load_json(pairs_qc_path)
            ui.skip(f"Reused ADS pairs ({len(ads_pairs)} rows).")
        else:
            ads_pairs = build_pairs_within_blocks(
                mentions=ads_subset,
                max_pairs_per_block=run_cfg.get("max_pairs_per_block"),
                seed=int(run_cfg.get("seed", 11)),
                require_same_split=False,
                labeled_only=False,
                balance_train=False,
                show_progress=args.progress,
            )
            write_pairs(ads_pairs, ads_pairs_path)
            pairs_qc = build_pairs_qc(
                lspo_mentions=lspo_mentions_split,
                lspo_pairs=lspo_pairs,
                ads_pairs=ads_pairs,
                split_meta=split_meta,
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
            seeds = [int(s) for s in training_cfg.get("seeds", [1, 2, 3, 4, 5])]
            train_manifest = train_nand_across_seeds(
                mentions=lspo_mentions_split,
                pairs=lspo_pairs,
                chars2vec=lspo_chars,
                text_emb=lspo_text,
                model_config=training_cfg,
                seeds=seeds,
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

        if cluster_cache_hit:
            clusters = read_parquet(clusters_path)
            cluster_qc = _load_json(cluster_qc_path)
            ui.skip(f"Reused clustering outputs ({len(clusters)} mentions).")
        else:
            best_threshold = float(train_manifest["best_threshold"])
            cluster_cfg_used = json.loads(json.dumps(cluster_cfg))
            resolved_eps, eps_meta = resolve_dbscan_eps(cluster_cfg_used, cosine_threshold=best_threshold)
            cluster_cfg_used["eps"] = resolved_eps
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
