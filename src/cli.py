from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_yaml, resolve_paths_config, find_project_root, resolve_existing_path
from src.common.io_schema import read_parquet, save_parquet
from src.common.run_report import evaluate_go_no_go, write_go_no_go_report
from src.common.subset_builder import build_stage_subset, write_subset_manifest
from src.data.prepare_lspo import prepare_lspo_mentions
from src.data.prepare_ads import prepare_ads_mentions
from src.approaches.nand.build_pairs import assign_lspo_splits, build_pairs_within_blocks, write_pairs
from src.approaches.nand.train import train_nand_across_seeds
from src.approaches.nand.infer_pairs import score_pairs_with_checkpoint
from src.approaches.nand.cluster import cluster_blockwise_dbscan
from src.approaches.nand.export import build_publication_author_mapping
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

    subset = build_stage_subset(mentions, stage=stage, seed=seed, target_mentions=target)
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
    )
    print(f"Chars2Vec embeddings: {chars.shape} -> {args.chars_out}")
    print(f"Text embeddings: {text.shape} -> {args.text_out}")


def cmd_pairs(args):
    mentions = read_parquet(args.mentions)

    if args.assign_lspo_splits:
        mentions = assign_lspo_splits(mentions, seed=args.seed)
        save_parquet(mentions, args.mentions, index=False)

    pairs = build_pairs_within_blocks(
        mentions=mentions,
        max_pairs_per_block=args.max_pairs_per_block,
        seed=args.seed,
        require_same_split=not args.allow_cross_split,
        labeled_only=args.labeled_only,
        balance_train=args.balance_train,
    )
    write_pairs(pairs, args.output)
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
        import json

        with open(args.metrics, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    go = evaluate_go_no_go(metrics)
    write_go_no_go_report(go, args.output)
    print(f"Go/No-Go: {'GO' if go['go'] else 'NO-GO'} -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NAND research CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prepare-lspo")
    sp.add_argument("--paths-config", default="configs/paths.colab.yaml")
    sp.add_argument("--output", default=None)
    sp.set_defaults(func=cmd_prepare_lspo)

    sp = sub.add_parser("prepare-ads")
    sp.add_argument("--paths-config", default="configs/paths.colab.yaml")
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
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("cluster")
    sp.add_argument("--mentions", required=True)
    sp.add_argument("--pair-scores", required=True)
    sp.add_argument("--cluster-config", default="configs/clustering/dbscan_paper.yaml")
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=cmd_cluster)

    sp = sub.add_parser("export")
    sp.add_argument("--mentions", required=True)
    sp.add_argument("--clusters", required=True)
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("report")
    sp.add_argument("--metrics", required=True)
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=cmd_report)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
