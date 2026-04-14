from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from author_name_disambiguation import cli
from author_name_disambiguation.common.subset_artifacts import (
    LSPO_SOURCE_FP_SCHEME,
    compute_lspo_source_fp,
    compute_lspo_source_fp_legacy,
    compute_subset_identity,
)


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def _toy_lspo_mentions() -> pd.DataFrame:
    rows = [
        {
            "mention_id": "m0",
            "bibcode": "b0",
            "author_idx": 0,
            "author_raw": "Doe, J",
            "title": "T0",
            "abstract": "A0",
            "year": 2000,
            "source_type": "lspo",
            "block_key": "blk0",
            "orcid": "o0",
        },
        {
            "mention_id": "m1",
            "bibcode": "b1",
            "author_idx": 0,
            "author_raw": "Doe, J.",
            "title": "T1",
            "abstract": "A1",
            "year": 2001,
            "source_type": "lspo",
            "block_key": "blk0",
            "orcid": "o0",
        },
        {
            "mention_id": "m2",
            "bibcode": "b2",
            "author_idx": 0,
            "author_raw": "Smith, A",
            "title": "T2",
            "abstract": "A2",
            "year": 2002,
            "source_type": "lspo",
            "block_key": "blk1",
            "orcid": "o1",
        },
        {
            "mention_id": "m3",
            "bibcode": "b3",
            "author_idx": 0,
            "author_raw": "Smith, A.",
            "title": "T3",
            "abstract": "A3",
            "year": 2003,
            "source_type": "lspo",
            "block_key": "blk1",
            "orcid": "o1",
        },
    ]
    return pd.DataFrame(rows)


def _make_configs(tmp_path: Path) -> dict[str, Path]:
    paths_cfg = {
        "project_root": str(tmp_path),
        "data": {
            "raw_lspo_parquet": str(tmp_path / "data/raw/lspo/LSPO_v1.parquet"),
            "raw_lspo_h5": str(tmp_path / "data/raw/lspo/LSPO_v1.h5"),
            "raw_ads_publications": str(tmp_path / "data/raw/ads/legacy_publications.jsonl"),
            "raw_ads_references": str(tmp_path / "data/raw/ads/legacy_references.json"),
            "interim_dir": str(tmp_path / "data/interim"),
            "processed_dir": str(tmp_path / "data/processed"),
            "subset_cache_dir": str(tmp_path / "data/subsets/cache"),
            "subset_manifest_dir": str(tmp_path / "data/subsets/manifests"),
        },
        "artifacts": {
            "root": str(tmp_path / "artifacts"),
            "embeddings_dir": str(tmp_path / "artifacts/embeddings"),
            "checkpoints_dir": str(tmp_path / "artifacts/checkpoints"),
            "pair_scores_dir": str(tmp_path / "artifacts/pair_scores"),
            "clusters_dir": str(tmp_path / "artifacts/clusters"),
            "metrics_dir": str(tmp_path / "artifacts/metrics"),
            "models_dir": str(tmp_path / "artifacts/models"),
        },
    }
    run_cfg = {
        "stage": "full",
        "subset_target_mentions": None,
        "seed": 11,
        "subset_sampling": {"target_mean_block_size": 2},
        "split_assignment": {"train_ratio": 0.6, "val_ratio": 0.2},
        "pair_building": {"exclude_same_bibcode": True},
        "max_pairs_per_block": 100,
        "split_balance": {"min_neg_val": 0, "min_neg_test": 0, "max_attempts": 3},
    }
    model_cfg = {
        "name": "mock-nand",
        "representation": {
            "text_model_name": "mock-specter",
            "max_length": 64,
        },
        "training": {
            "precision_mode": "fp32",
        },
    }
    cfg_dir = tmp_path / "cfg"
    return {
        "paths": _write_yaml(cfg_dir / "paths.yaml", paths_cfg),
        "run": _write_yaml(cfg_dir / "run.yaml", run_cfg),
        "model": _write_yaml(cfg_dir / "model.yaml", model_cfg),
        "data_root": tmp_path / "data",
        "artifacts_root": tmp_path / "artifacts",
        "raw_lspo_parquet": Path(paths_cfg["data"]["raw_lspo_parquet"]),
        "raw_lspo_h5": Path(paths_cfg["data"]["raw_lspo_h5"]),
        "metrics_dir": Path(paths_cfg["artifacts"]["metrics_dir"]),
        "paths_payload": paths_cfg,
        "run_payload": run_cfg,
    }


def _write_train_artifacts(
    tmp_path: Path,
    cfg: dict[str, Any],
    model_run_id: str,
    subset_cache_key: str,
    *,
    lspo_source_fingerprint: str | None = None,
    lspo_source_fingerprint_scheme: str | None = None,
) -> None:
    metrics_dir = cfg["metrics_dir"] / model_run_id
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = tmp_path / "artifacts" / "checkpoints" / model_run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt1 = ckpt_dir / f"{model_run_id}_seed1.pt"
    ckpt2 = ckpt_dir / f"{model_run_id}_seed2.pt"
    ckpt1.write_text("checkpoint-1", encoding="utf-8")
    ckpt2.write_text("checkpoint-2", encoding="utf-8")

    with (metrics_dir / "00_context.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": model_run_id,
                "run_stage": "full",
                "pipeline_scope": "train",
                "run_config": str(cfg["run"]),
                "model_config": str(cfg["model"]),
            },
            f,
            indent=2,
        )

    with (metrics_dir / "03_train_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": model_run_id,
                "best_threshold": 0.5,
                "runs": [
                    {"seed": 1, "checkpoint": str(ckpt1), "threshold": 0.5},
                    {"seed": 2, "checkpoint": str(ckpt2), "threshold": 0.5},
                ],
            },
            f,
            indent=2,
        )

    with (metrics_dir / "04_clustering_config_used.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "eps_resolution": {"selected_eps": 0.35},
                "cluster_config_used": {
                    "method": "dbscan",
                    "eps": 0.35,
                    "min_samples": 1,
                    "metric": "precomputed",
                    "constraints": {
                        "enabled": True,
                    },
                },
            },
            f,
            indent=2,
        )

    with (metrics_dir / "05_stage_metrics_full.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": model_run_id,
                "stage": "full",
                "metric_scope": "train",
                "subset_cache_key": subset_cache_key,
                "lspo_source_fingerprint": lspo_source_fingerprint,
                "lspo_source_fingerprint_scheme": lspo_source_fingerprint_scheme,
            },
            f,
            indent=2,
        )


def _write_legacy_compat_artifacts(tmp_path: Path, cfg: dict[str, Any], metrics_dir: Path) -> None:
    lspo_mentions = pd.read_parquet(Path(cfg["paths_payload"]["data"]["interim_dir"]) / "lspo_mentions.parquet")
    lspo_subset = cli.build_stage_subset(
        lspo_mentions,
        stage="full",
        seed=int(cfg["run_payload"].get("seed", 11)),
        target_mentions=cfg["run_payload"].get("subset_target_mentions"),
        subset_sampling=cfg["run_payload"].get("subset_sampling", {}),
    )
    lspo_mentions_split, split_meta = cli.assign_lspo_splits(
        lspo_subset,
        seed=int(cfg["run_payload"].get("seed", 11)),
        train_ratio=float(cfg["run_payload"]["split_assignment"]["train_ratio"]),
        val_ratio=float(cfg["run_payload"]["split_assignment"]["val_ratio"]),
        min_neg_val=int(cfg["run_payload"]["split_balance"]["min_neg_val"]),
        min_neg_test=int(cfg["run_payload"]["split_balance"]["min_neg_test"]),
        max_attempts=int(cfg["run_payload"]["split_balance"]["max_attempts"]),
        return_meta=True,
    )
    pair_result = cli.build_pairs_within_blocks(
        mentions=lspo_mentions_split,
        max_pairs_per_block=cfg["run_payload"].get("max_pairs_per_block"),
        seed=int(cfg["run_payload"].get("seed", 11)),
        require_same_split=True,
        labeled_only=False,
        balance_train=True,
        exclude_same_bibcode=True,
        show_progress=False,
        return_meta=True,
    )
    if isinstance(pair_result, tuple):
        lspo_pairs, lspo_pair_build_meta = pair_result
    else:
        lspo_pairs = pair_result
        lspo_pair_build_meta = {}
    pairs_qc = cli.build_pairs_qc(
        lspo_mentions=lspo_mentions_split,
        lspo_pairs=lspo_pairs,
        ads_pairs=pd.DataFrame(columns=cli.PAIR_REQUIRED_COLUMNS + ["label"]),
        split_meta=split_meta,
        lspo_pair_build_meta=lspo_pair_build_meta,
        ads_pair_build_meta={},
    )
    with (metrics_dir / "01_subset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "lspo_mentions": int(len(lspo_subset)),
                "lspo_blocks": int(lspo_subset["block_key"].nunique()),
                "lspo_block_size_p95": float(lspo_subset.groupby("block_key").size().quantile(0.95)),
            },
            f,
            indent=2,
        )
    with (metrics_dir / "02_split_balance.json").open("w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
    with (metrics_dir / "02_pairs_qc.json").open("w", encoding="utf-8") as f:
        json.dump(pairs_qc, f, indent=2, default=int)


def _apply_fast_mocks(monkeypatch) -> None:
    def _assign(mentions, return_meta=False, **_kwargs):
        out = mentions.copy()
        out["split"] = "test"
        meta = {
            "status": "ok",
            "attempts": 1,
            "split_label_counts": {
                "train": {"pos": 0, "neg": 0, "labeled_pairs": 0},
                "val": {"pos": 0, "neg": 0, "labeled_pairs": 0},
                "test": {"pos": 2, "neg": 0, "labeled_pairs": 2},
            },
        }
        return (out, meta) if return_meta else out

    def _pairs(mentions, **_kwargs):
        rows = []
        for block_key, grp in mentions.groupby("block_key", sort=False):
            if len(grp) < 2:
                continue
            a = grp.iloc[0]
            b = grp.iloc[1]
            rows.append(
                {
                    "pair_id": f"{a['mention_id']}__{b['mention_id']}",
                    "mention_id_1": str(a["mention_id"]),
                    "mention_id_2": str(b["mention_id"]),
                    "block_key": str(block_key),
                    "split": str(a.get("split", "test")),
                    "label": int(str(a.get("orcid", "")) == str(b.get("orcid", ""))),
                }
            )
        out = pd.DataFrame(rows)
        if _kwargs.get("return_meta"):
            return out, {
                "exclude_same_bibcode": bool(_kwargs.get("exclude_same_bibcode", True)),
                "same_publication_pairs_skipped": 0,
                "balance_train": bool(_kwargs.get("balance_train", False)),
                "pairs_written": int(len(out)),
                "train_balance_before": {"pos": int((out["label"] == 1).sum()) if len(out) else 0, "neg": int((out["label"] == 0).sum()) if len(out) else 0},
                "train_balance_after": {"pos": int((out["label"] == 1).sum()) if len(out) else 0, "neg": int((out["label"] == 0).sum()) if len(out) else 0},
            }
        return out

    def _chars(mentions, output_path, force_recompute=False, **_kwargs):
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and not force_recompute:
            return np.load(p)
        arr = np.ones((len(mentions), 50), dtype=np.float32)
        np.save(p, arr)
        return arr

    def _text(mentions, output_path, force_recompute=False, **_kwargs):
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and not force_recompute:
            return np.load(p)
        arr = np.ones((len(mentions), 768), dtype=np.float32)
        np.save(p, arr)
        return arr

    def _score(mentions, pairs, output_path=None, **_kwargs):
        out = pairs[["pair_id", "mention_id_1", "mention_id_2", "block_key"]].copy()
        out["cosine_sim"] = 0.9
        out["distance"] = 0.1
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(output_path, index=False)
        return out

    def _cluster(mentions, pair_scores, cluster_config, output_path=None, **_kwargs):
        enabled = bool((cluster_config.get("constraints", {}) or {}).get("enabled", False))
        out = mentions[["mention_id", "block_key"]].copy()
        if enabled:
            out["author_uid"] = out["block_key"].astype(str) + "::0"
        else:
            out["author_uid"] = [
                f"{blk}::{idx}"
                for idx, blk in enumerate(out["block_key"].astype(str).tolist())
            ]
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(output_path, index=False)
        return out

    monkeypatch.setattr(cli, "assign_lspo_splits", _assign)
    monkeypatch.setattr(cli, "build_pairs_within_blocks", _pairs)
    monkeypatch.setattr(cli, "get_or_create_chars2vec_embeddings", _chars)
    monkeypatch.setattr(cli, "get_or_create_specter_embeddings", _text)
    monkeypatch.setattr(cli, "score_pairs_with_checkpoint", _score)
    monkeypatch.setattr(cli, "cluster_blockwise_dbscan", _cluster)


def _run_report(
    parser: argparse.ArgumentParser,
    cfg: dict[str, Any],
    model_run_id: str,
    extra: list[str] | None = None,
) -> None:
    argv = [
        "run-cluster-test-report",
        "--model-run-id",
        model_run_id,
        "--data-root",
        str(cfg["data_root"]),
        "--artifacts-root",
        str(cfg["artifacts_root"]),
        "--raw-lspo-parquet",
        str(cfg["raw_lspo_parquet"]),
        "--raw-lspo-h5",
        str(cfg["raw_lspo_h5"]),
        "--no-progress",
    ]
    if extra:
        argv.extend(extra)
    args = parser.parse_args(argv)
    args.func(args)


def test_cli_run_cluster_test_report_writes_outputs(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    lspo_df = _toy_lspo_mentions()

    raw_lspo_parquet = Path(cfg["paths_payload"]["data"]["raw_lspo_parquet"])
    raw_lspo_parquet.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(raw_lspo_parquet, index=False)
    Path(cfg["paths_payload"]["data"]["raw_lspo_h5"]).write_text("stub", encoding="utf-8")

    interim_lspo = Path(cfg["paths_payload"]["data"]["interim_dir"]) / "lspo_mentions.parquet"
    interim_lspo.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(interim_lspo, index=False)

    source_fp = compute_lspo_source_fp(interim_lspo)
    subset_identity = compute_subset_identity(run_cfg=cfg["run_payload"], source_fp=source_fp, sampler_version="v3")

    model_run_id = "full_20260218T111506Z_cli02681429"
    _write_train_artifacts(
        tmp_path,
        cfg,
        model_run_id,
        subset_cache_key=subset_identity.subset_tag,
        lspo_source_fingerprint=source_fp,
        lspo_source_fingerprint_scheme=LSPO_SOURCE_FP_SCHEME,
    )
    _apply_fast_mocks(monkeypatch)

    parser = cli.build_parser()
    _run_report(parser, cfg, model_run_id=model_run_id)

    metrics_dir = cfg["metrics_dir"] / model_run_id
    assert (metrics_dir / "06_clustering_test_report.json").exists()
    assert (metrics_dir / "06_clustering_test_summary.csv").exists()
    assert (metrics_dir / "06_clustering_test_per_seed.csv").exists()
    assert (metrics_dir / "06_clustering_test_report.md").exists()

    payload = json.loads((metrics_dir / "06_clustering_test_report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["model_run_id"] == model_run_id
    assert payload["wall_seconds"] >= 0.0
    assert payload["pipeline_scope"] == "train"
    assert payload["cluster_config_source_mode"] == "train_only"
    assert payload["cluster_config_override_path"] is None
    assert payload["override_ignored_fields"] == []
    assert payload["report_tag"] is None
    assert payload["lspo_source_fingerprint"] == source_fp
    assert payload["lspo_source_fingerprint_scheme"] == LSPO_SOURCE_FP_SCHEME
    assert payload["subset_verification_mode"] == "strict"
    assert payload["subset_cache_key_stable_computed"] == subset_identity.subset_tag
    assert payload["subset_cache_key_legacy_computed"] is not None
    assert payload["seeds_expected"] == [1, 2]
    assert payload["seeds_evaluated"] == [1, 2]
    assert set(payload["variants"].keys()) == {"dbscan_no_constraints", "dbscan_with_constraints"}


def test_cli_run_cluster_test_report_override_requires_report_tag(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    lspo_df = _toy_lspo_mentions()

    raw_lspo_parquet = Path(cfg["paths_payload"]["data"]["raw_lspo_parquet"])
    raw_lspo_parquet.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(raw_lspo_parquet, index=False)

    interim_lspo = Path(cfg["paths_payload"]["data"]["interim_dir"]) / "lspo_mentions.parquet"
    interim_lspo.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(interim_lspo, index=False)

    source_fp = compute_lspo_source_fp(interim_lspo)
    subset_identity = compute_subset_identity(run_cfg=cfg["run_payload"], source_fp=source_fp, sampler_version="v3")
    model_run_id = "full_20260218T111506Z_cli02681429"
    _write_train_artifacts(
        tmp_path,
        cfg,
        model_run_id,
        subset_cache_key=subset_identity.subset_tag,
        lspo_source_fingerprint=source_fp,
        lspo_source_fingerprint_scheme=LSPO_SOURCE_FP_SCHEME,
    )
    _apply_fast_mocks(monkeypatch)

    override_cfg_path = tmp_path / "cfg" / "cluster_override.yaml"
    _write_yaml(
        override_cfg_path,
        {
            "eps": 0.40,
            "eps_mode": "val_sweep",
            "selected_eps": 0.40,
            "constraints": {"enabled": False},
        },
    )

    parser = cli.build_parser()
    with pytest.raises(ValueError, match="requires --report-tag"):
        _run_report(
            parser,
            cfg,
            model_run_id=model_run_id,
            extra=["--cluster-config-override", str(override_cfg_path)],
        )


def test_cli_run_cluster_test_report_override_writes_tagged_outputs(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    lspo_df = _toy_lspo_mentions()

    raw_lspo_parquet = Path(cfg["paths_payload"]["data"]["raw_lspo_parquet"])
    raw_lspo_parquet.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(raw_lspo_parquet, index=False)
    Path(cfg["paths_payload"]["data"]["raw_lspo_h5"]).write_text("stub", encoding="utf-8")

    interim_lspo = Path(cfg["paths_payload"]["data"]["interim_dir"]) / "lspo_mentions.parquet"
    interim_lspo.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(interim_lspo, index=False)

    source_fp = compute_lspo_source_fp(interim_lspo)
    subset_identity = compute_subset_identity(run_cfg=cfg["run_payload"], source_fp=source_fp, sampler_version="v3")
    model_run_id = "full_20260218T111506Z_cli02681429"
    _write_train_artifacts(
        tmp_path,
        cfg,
        model_run_id,
        subset_cache_key=subset_identity.subset_tag,
        lspo_source_fingerprint=source_fp,
        lspo_source_fingerprint_scheme=LSPO_SOURCE_FP_SCHEME,
    )
    _apply_fast_mocks(monkeypatch)

    parser = cli.build_parser()
    _run_report(parser, cfg, model_run_id=model_run_id)

    override_cfg_path = tmp_path / "cfg" / "cluster_override.yaml"
    _write_yaml(
        override_cfg_path,
        {
            "eps": 0.60,
            "selected_eps": 0.60,
            "eps_mode": "val_sweep",
            "constraints": {"enabled": False},
            "eps_block_policy": {"enabled": True},
        },
    )
    _run_report(
        parser,
        cfg,
        model_run_id=model_run_id,
        extra=[
            "--cluster-config-override",
            str(override_cfg_path),
            "--report-tag",
            "epsbkt_v1",
        ],
    )

    metrics_dir = cfg["metrics_dir"] / model_run_id
    baseline_json = metrics_dir / "06_clustering_test_report.json"
    tagged_json = metrics_dir / "06_clustering_test_report__epsbkt_v1.json"
    tagged_summary = metrics_dir / "06_clustering_test_summary__epsbkt_v1.csv"
    tagged_per_seed = metrics_dir / "06_clustering_test_per_seed__epsbkt_v1.csv"
    tagged_md = metrics_dir / "06_clustering_test_report__epsbkt_v1.md"

    assert baseline_json.exists()
    assert tagged_json.exists()
    assert tagged_summary.exists()
    assert tagged_per_seed.exists()
    assert tagged_md.exists()

    payload = json.loads(tagged_json.read_text(encoding="utf-8"))
    assert payload["report_tag"] == "epsbkt_v1"
    assert payload["cluster_config_source_mode"] == "train_plus_override"
    assert payload["cluster_config_override_path"] == str(override_cfg_path.resolve())
    assert payload["override_ignored_fields"] == ["eps", "selected_eps", "eps_mode"]
    assert payload["selected_eps"] == 0.35
    assert payload["cluster_config_effective"]["eps"] == 0.35
    assert payload["cluster_config_effective"]["eps_mode"] == "fixed"


def test_cli_run_cluster_test_report_fails_when_lspo_raw_missing(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    lspo_df = _toy_lspo_mentions()

    interim_lspo = Path(cfg["paths_payload"]["data"]["interim_dir"]) / "lspo_mentions.parquet"
    interim_lspo.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(interim_lspo, index=False)

    source_fp = compute_lspo_source_fp(interim_lspo)
    subset_identity = compute_subset_identity(run_cfg=cfg["run_payload"], source_fp=source_fp, sampler_version="v3")
    model_run_id = "full_20260218T111506Z_cli02681429"
    _write_train_artifacts(
        tmp_path,
        cfg,
        model_run_id,
        subset_cache_key=subset_identity.subset_tag,
        lspo_source_fingerprint=source_fp,
        lspo_source_fingerprint_scheme=LSPO_SOURCE_FP_SCHEME,
    )
    _apply_fast_mocks(monkeypatch)

    parser = cli.build_parser()
    with pytest.raises(FileNotFoundError, match="LSPO raw source not found for research workflow"):
        _run_report(parser, cfg, model_run_id=model_run_id)


def test_cli_run_cluster_test_report_accepts_h5_when_parquet_missing(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    lspo_df = _toy_lspo_mentions()

    raw_lspo_h5 = Path(cfg["paths_payload"]["data"]["raw_lspo_h5"])
    raw_lspo_h5.parent.mkdir(parents=True, exist_ok=True)
    raw_lspo_h5.write_text("stub", encoding="utf-8")

    interim_lspo = Path(cfg["paths_payload"]["data"]["interim_dir"]) / "lspo_mentions.parquet"
    interim_lspo.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(interim_lspo, index=False)

    source_fp = compute_lspo_source_fp(interim_lspo)
    subset_identity = compute_subset_identity(run_cfg=cfg["run_payload"], source_fp=source_fp, sampler_version="v3")
    model_run_id = "full_20260218T111506Z_cli02681429"
    _write_train_artifacts(
        tmp_path,
        cfg,
        model_run_id,
        subset_cache_key=subset_identity.subset_tag,
        lspo_source_fingerprint=source_fp,
        lspo_source_fingerprint_scheme=LSPO_SOURCE_FP_SCHEME,
    )
    _apply_fast_mocks(monkeypatch)

    parser = cli.build_parser()
    _run_report(parser, cfg, model_run_id=model_run_id)

    assert (cfg["metrics_dir"] / model_run_id / "06_clustering_test_report.json").exists()


def test_cli_run_cluster_test_report_fails_when_metrics_dir_missing(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    parser = cli.build_parser()

    with pytest.raises(FileNotFoundError, match="Missing mandatory train metrics directory"):
        _run_report(parser, cfg, model_run_id="full_20260218T111506Z_cli02681429")


def test_cli_run_cluster_test_report_fails_when_context_missing(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    lspo_df = _toy_lspo_mentions()

    raw_lspo_parquet = Path(cfg["paths_payload"]["data"]["raw_lspo_parquet"])
    raw_lspo_parquet.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(raw_lspo_parquet, index=False)

    interim_lspo = Path(cfg["paths_payload"]["data"]["interim_dir"]) / "lspo_mentions.parquet"
    interim_lspo.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(interim_lspo, index=False)

    source_fp = compute_lspo_source_fp(interim_lspo)
    subset_identity = compute_subset_identity(run_cfg=cfg["run_payload"], source_fp=source_fp, sampler_version="v3")
    model_run_id = "full_20260218T111506Z_cli02681429"
    _write_train_artifacts(
        tmp_path,
        cfg,
        model_run_id,
        subset_cache_key=subset_identity.subset_tag,
        lspo_source_fingerprint=source_fp,
        lspo_source_fingerprint_scheme=LSPO_SOURCE_FP_SCHEME,
    )
    _apply_fast_mocks(monkeypatch)
    (cfg["metrics_dir"] / model_run_id / "00_context.json").unlink()

    parser = cli.build_parser()
    with pytest.raises(FileNotFoundError, match="Missing mandatory train artifact for clustering report"):
        _run_report(parser, cfg, model_run_id=model_run_id)


def test_cli_run_cluster_test_report_fails_when_train_manifest_missing(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    lspo_df = _toy_lspo_mentions()

    raw_lspo_parquet = Path(cfg["paths_payload"]["data"]["raw_lspo_parquet"])
    raw_lspo_parquet.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(raw_lspo_parquet, index=False)

    interim_lspo = Path(cfg["paths_payload"]["data"]["interim_dir"]) / "lspo_mentions.parquet"
    interim_lspo.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(interim_lspo, index=False)

    source_fp = compute_lspo_source_fp(interim_lspo)
    subset_identity = compute_subset_identity(run_cfg=cfg["run_payload"], source_fp=source_fp, sampler_version="v3")
    model_run_id = "full_20260218T111506Z_cli02681429"
    _write_train_artifacts(
        tmp_path,
        cfg,
        model_run_id,
        subset_cache_key=subset_identity.subset_tag,
        lspo_source_fingerprint=source_fp,
        lspo_source_fingerprint_scheme=LSPO_SOURCE_FP_SCHEME,
    )
    _apply_fast_mocks(monkeypatch)
    (cfg["metrics_dir"] / model_run_id / "03_train_manifest.json").unlink()

    parser = cli.build_parser()
    with pytest.raises(FileNotFoundError, match="Missing mandatory train artifact for clustering report"):
        _run_report(parser, cfg, model_run_id=model_run_id)


def test_cli_run_cluster_test_report_fails_when_clustering_config_missing(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    lspo_df = _toy_lspo_mentions()

    raw_lspo_parquet = Path(cfg["paths_payload"]["data"]["raw_lspo_parquet"])
    raw_lspo_parquet.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(raw_lspo_parquet, index=False)

    interim_lspo = Path(cfg["paths_payload"]["data"]["interim_dir"]) / "lspo_mentions.parquet"
    interim_lspo.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(interim_lspo, index=False)

    source_fp = compute_lspo_source_fp(interim_lspo)
    subset_identity = compute_subset_identity(run_cfg=cfg["run_payload"], source_fp=source_fp, sampler_version="v3")
    model_run_id = "full_20260218T111506Z_cli02681429"
    _write_train_artifacts(
        tmp_path,
        cfg,
        model_run_id,
        subset_cache_key=subset_identity.subset_tag,
        lspo_source_fingerprint=source_fp,
        lspo_source_fingerprint_scheme=LSPO_SOURCE_FP_SCHEME,
    )
    _apply_fast_mocks(monkeypatch)
    (cfg["metrics_dir"] / model_run_id / "04_clustering_config_used.json").unlink()

    parser = cli.build_parser()
    with pytest.raises(FileNotFoundError, match="Missing mandatory train artifact for clustering report"):
        _run_report(parser, cfg, model_run_id=model_run_id)


def test_cli_run_cluster_test_report_fails_on_missing_checkpoint(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    lspo_df = _toy_lspo_mentions()

    raw_lspo_parquet = Path(cfg["paths_payload"]["data"]["raw_lspo_parquet"])
    raw_lspo_parquet.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(raw_lspo_parquet, index=False)

    interim_lspo = Path(cfg["paths_payload"]["data"]["interim_dir"]) / "lspo_mentions.parquet"
    interim_lspo.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(interim_lspo, index=False)

    source_fp = compute_lspo_source_fp(interim_lspo)
    subset_identity = compute_subset_identity(run_cfg=cfg["run_payload"], source_fp=source_fp, sampler_version="v3")
    model_run_id = "full_20260218T111506Z_cli02681429"
    _write_train_artifacts(
        tmp_path,
        cfg,
        model_run_id,
        subset_cache_key=subset_identity.subset_tag,
        lspo_source_fingerprint=source_fp,
        lspo_source_fingerprint_scheme=LSPO_SOURCE_FP_SCHEME,
    )
    _apply_fast_mocks(monkeypatch)

    ckpt = tmp_path / "artifacts" / "checkpoints" / model_run_id / f"{model_run_id}_seed2.pt"
    ckpt.unlink()

    parser = cli.build_parser()
    with pytest.raises(FileNotFoundError, match="Missing mandatory checkpoint for seed=2"):
        _run_report(parser, cfg, model_run_id=model_run_id)


def test_cli_run_cluster_test_report_fails_on_subset_key_mismatch(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    lspo_df = _toy_lspo_mentions()

    raw_lspo_parquet = Path(cfg["paths_payload"]["data"]["raw_lspo_parquet"])
    raw_lspo_parquet.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(raw_lspo_parquet, index=False)

    interim_lspo = Path(cfg["paths_payload"]["data"]["interim_dir"]) / "lspo_mentions.parquet"
    interim_lspo.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(interim_lspo, index=False)

    legacy_source_fp = compute_lspo_source_fp_legacy(interim_lspo)
    legacy_subset_identity = compute_subset_identity(
        run_cfg=cfg["run_payload"],
        source_fp=legacy_source_fp,
        sampler_version="v3",
    )
    model_run_id = "full_20260218T111506Z_cli02681429"
    _write_train_artifacts(tmp_path, cfg, model_run_id, subset_cache_key=legacy_subset_identity.subset_tag)
    _apply_fast_mocks(monkeypatch)

    parser = cli.build_parser()
    with pytest.raises(ValueError, match="Subset reproducibility check failed"):
        _run_report(parser, cfg, model_run_id=model_run_id)


def test_cli_run_cluster_test_report_allows_legacy_lspo_compat(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    lspo_df = _toy_lspo_mentions()

    raw_lspo_parquet = Path(cfg["paths_payload"]["data"]["raw_lspo_parquet"])
    raw_lspo_parquet.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(raw_lspo_parquet, index=False)

    interim_lspo = Path(cfg["paths_payload"]["data"]["interim_dir"]) / "lspo_mentions.parquet"
    interim_lspo.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(interim_lspo, index=False)

    legacy_source_fp = compute_lspo_source_fp_legacy(interim_lspo)
    legacy_subset_identity = compute_subset_identity(
        run_cfg=cfg["run_payload"],
        source_fp=legacy_source_fp,
        sampler_version="v3",
    )
    model_run_id = "full_20260218T111506Z_cli02681429"
    _write_train_artifacts(tmp_path, cfg, model_run_id, subset_cache_key=legacy_subset_identity.subset_tag)
    _apply_fast_mocks(monkeypatch)
    _write_legacy_compat_artifacts(tmp_path, cfg, cfg["metrics_dir"] / model_run_id)

    parser = cli.build_parser()
    _run_report(
        parser,
        cfg,
        model_run_id=model_run_id,
        extra=["--allow-legacy-lspo-compat"],
    )

    payload = json.loads(((cfg["metrics_dir"] / model_run_id) / "06_clustering_test_report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["subset_verification_mode"] == "legacy_compat"
    assert payload["subset_cache_key_expected"] == legacy_subset_identity.subset_tag
    assert payload["subset_cache_key_stable_computed"] != payload["subset_cache_key_expected"]
    assert payload["subset_cache_key_legacy_computed"] == legacy_subset_identity.subset_tag


def test_cli_run_cluster_test_report_fails_legacy_compat_on_artifact_mismatch(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    lspo_df = _toy_lspo_mentions()

    raw_lspo_parquet = Path(cfg["paths_payload"]["data"]["raw_lspo_parquet"])
    raw_lspo_parquet.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(raw_lspo_parquet, index=False)

    interim_lspo = Path(cfg["paths_payload"]["data"]["interim_dir"]) / "lspo_mentions.parquet"
    interim_lspo.parent.mkdir(parents=True, exist_ok=True)
    lspo_df.to_parquet(interim_lspo, index=False)

    legacy_source_fp = compute_lspo_source_fp_legacy(interim_lspo)
    legacy_subset_identity = compute_subset_identity(
        run_cfg=cfg["run_payload"],
        source_fp=legacy_source_fp,
        sampler_version="v3",
    )
    model_run_id = "full_20260218T111506Z_cli02681429"
    _write_train_artifacts(tmp_path, cfg, model_run_id, subset_cache_key=legacy_subset_identity.subset_tag)
    _apply_fast_mocks(monkeypatch)
    metrics_dir = cfg["metrics_dir"] / model_run_id
    _write_legacy_compat_artifacts(tmp_path, cfg, metrics_dir)

    pairs_qc_path = metrics_dir / "02_pairs_qc.json"
    pairs_qc = json.loads(pairs_qc_path.read_text(encoding="utf-8"))
    pairs_qc["lspo_pairs"] = int(pairs_qc["lspo_pairs"]) + 1
    pairs_qc_path.write_text(json.dumps(pairs_qc, indent=2), encoding="utf-8")

    parser = cli.build_parser()
    with pytest.raises(ValueError, match="Legacy LSPO compatibility check failed"):
        _run_report(
            parser,
            cfg,
            model_run_id=model_run_id,
            extra=["--allow-legacy-lspo-compat"],
        )
