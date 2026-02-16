from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src import cli


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def _toy_lspo_mentions() -> pd.DataFrame:
    rows = []
    for i in range(8):
        rows.append(
            {
                "mention_id": f"lspo{i}::0",
                "bibcode": f"lb{i}",
                "author_idx": 0,
                "author_raw": f"Smith {i}",
                "title": f"title {i}",
                "abstract": f"abstract {i}",
                "year": 2000 + i,
                "source_type": "lspo",
                "block_key": f"a.blk{i // 2}",
                "orcid": f"o{i // 2}",
            }
        )
    return pd.DataFrame(rows)


def _toy_ads_mentions() -> pd.DataFrame:
    rows = []
    for i in range(8):
        rows.append(
            {
                "mention_id": f"ads{i}::0",
                "bibcode": f"ab{i}",
                "author_idx": 0,
                "author_raw": f"Doe {i}",
                "title": f"title {i}",
                "abstract": f"abstract {i}",
                "year": 2000 + i,
                "source_type": "ads",
                "block_key": f"d.blk{i // 2}",
            }
        )
    return pd.DataFrame(rows)


def _make_configs(
    tmp_path: Path,
    split_balance: dict[str, int] | None = None,
    split_assignment: dict[str, float] | None = None,
    train_seeds: list[int] | None = None,
) -> dict[str, Path]:
    paths_cfg = {
        "project_root": str(tmp_path),
        "data": {
            "raw_lspo_parquet": str(tmp_path / "data/raw/lspo/mock.parquet"),
            "raw_lspo_h5": str(tmp_path / "data/raw/lspo/mock.h5"),
            "raw_ads_publications": str(tmp_path / "data/raw/ads/pubs.jsonl"),
            "raw_ads_references": str(tmp_path / "data/raw/ads/refs.json"),
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
        },
    }
    run_cfg = {
        "stage": "smoke",
        "subset_target_mentions": 6,
        "seed": 11,
        "subset_sampling": {"target_mean_block_size": 4},
        "split_assignment": split_assignment or {"train_ratio": 0.6, "val_ratio": 0.2},
        "pair_building": {"exclude_same_bibcode": True},
        "max_pairs_per_block": 100,
        "split_balance": split_balance
        or {
            "min_neg_val": 2,
            "min_neg_test": 2,
            "max_attempts": 5,
        },
    }
    if train_seeds is not None:
        run_cfg["train_seeds"] = [int(s) for s in train_seeds]
    model_cfg = {
        "name": "mock",
        "representation": {
            "text_model_name": "mock-specter",
            "max_length": 64,
        },
        "training": {
            "input_dim": 818,
            "hidden_dim": 64,
            "output_dim": 16,
            "temperature": 0.25,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "max_epochs": 2,
            "early_stopping_patience": 1,
            "seeds": [1],
            "default_cosine_threshold": 0.35,
        },
    }
    cluster_cfg = {
        "method": "dbscan",
        "eps_mode": "val_sweep",
        "eps": 0.35,
        "eps_fallback": 0.35,
        "eps_sweep_min": 0.2,
        "eps_sweep_max": 0.5,
        "eps_sweep_step": 0.1,
        "eps_min": 0.1,
        "eps_max": 0.9,
        "min_samples": 1,
        "metric": "precomputed",
        "constraints": {"enabled": False},
    }
    gates_cfg = {
        "defaults": {
            "threshold_bounds": {"min": -1.0, "max": 1.0},
            "mention_coverage_min": 0.0,
            "uid_uniqueness_max": 1,
        },
        "stages": {"smoke": {"f1_min": 0.0, "min_neg_val": 0, "min_neg_test": 0}},
    }

    cfg_dir = tmp_path / "cfg"
    return {
        "paths": _write_yaml(cfg_dir / "paths.yaml", paths_cfg),
        "run": _write_yaml(cfg_dir / "run.yaml", run_cfg),
        "model": _write_yaml(cfg_dir / "model.yaml", model_cfg),
        "cluster": _write_yaml(cfg_dir / "cluster.yaml", cluster_cfg),
        "gates": _write_yaml(cfg_dir / "gates.yaml", gates_cfg),
        "metrics_dir": Path(paths_cfg["artifacts"]["metrics_dir"]),
    }


def _apply_fast_mocks(monkeypatch, captured_split: dict[str, int | float] | None = None) -> None:
    lspo_df = _toy_lspo_mentions()
    ads_df = _toy_ads_mentions()

    def _prepare_lspo(*, parquet_path=None, h5_path=None, output_path, **_kwargs):
        _ = parquet_path, h5_path
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lspo_df.to_parquet(p, index=False)
        return lspo_df.copy()

    def _prepare_ads(*, publications_path=None, references_path=None, output_path, **_kwargs):
        _ = publications_path, references_path
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ads_df.to_parquet(p, index=False)
        return ads_df.copy()

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

    def _assign(
        mentions,
        seed=11,
        train_ratio=0.6,
        val_ratio=0.2,
        min_neg_val=0,
        min_neg_test=0,
        max_attempts=1,
        return_meta=False,
        **_kwargs,
    ):
        if captured_split is not None:
            captured_split["seed"] = int(seed)
            captured_split["train_ratio"] = float(train_ratio)
            captured_split["val_ratio"] = float(val_ratio)
            captured_split["min_neg_val"] = int(min_neg_val)
            captured_split["min_neg_test"] = int(min_neg_test)
            captured_split["max_attempts"] = int(max_attempts)
        out = mentions.copy()
        splits = ["train", "val", "test"]
        out["split"] = [splits[i % len(splits)] for i in range(len(out))]
        if "orcid" not in out.columns:
            out["orcid"] = [f"o{i // 2}" for i in range(len(out))]
        meta = {
            "status": "ok",
            "attempts": 1,
            "min_neg_val": int(min_neg_val),
            "min_neg_test": int(min_neg_test),
            "split_label_counts": {
                "train": {"pos": 5, "neg": 2, "labeled_pairs": 7},
                "val": {"pos": 3, "neg": max(1, int(min_neg_val)), "labeled_pairs": 4},
                "test": {"pos": 3, "neg": max(1, int(min_neg_test)), "labeled_pairs": 4},
            },
        }
        return (out, meta) if return_meta else out

    def _pairs(mentions, return_meta=False, exclude_same_bibcode=True, **_kwargs):
        rows = []
        skipped_same_pub = 0
        for block_key, grp in mentions.groupby("block_key", sort=False):
            if len(grp) < 2:
                continue
            a = grp.iloc[0]
            b = grp.iloc[1]
            if exclude_same_bibcode and str(a.get("bibcode")) == str(b.get("bibcode")):
                skipped_same_pub += 1
                continue
            split = str(a.get("split", "inference"))
            label = None
            if "orcid" in grp.columns:
                oa = str(a.get("orcid", "")).strip()
                ob = str(b.get("orcid", "")).strip()
                label = int(bool(oa) and oa == ob)
            rows.append(
                {
                    "pair_id": f"{a['mention_id']}__{b['mention_id']}",
                    "mention_id_1": str(a["mention_id"]),
                    "mention_id_2": str(b["mention_id"]),
                    "block_key": str(block_key),
                    "split": split,
                    "label": label,
                }
            )
        out = pd.DataFrame(rows)
        meta = {
            "exclude_same_bibcode": bool(exclude_same_bibcode),
            "same_publication_pairs_skipped": int(skipped_same_pub),
            "balance_train": bool(_kwargs.get("balance_train", False)),
        }
        return (out, meta) if return_meta else out

    def _train(
        mentions,
        pairs,
        chars2vec,
        text_emb,
        model_config,
        seeds,
        run_id,
        output_dir,
        metrics_output=None,
        **_kwargs,
    ):
        ckpt = Path(output_dir) / f"{run_id}_seed{int(seeds[0])}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("checkpoint", encoding="utf-8")
        manifest = {
            "run_id": run_id,
            "best_seed": int(seeds[0]),
            "best_checkpoint": str(ckpt),
            "best_threshold": 0.25,
            "best_threshold_selection_status": "ok",
            "best_threshold_source": "val_f1_opt",
            "best_val_class_counts": {"pos": 8, "neg": 3},
            "best_test_class_counts": {"pos": 7, "neg": 2},
            "best_val_f1": 0.90,
            "runs": [],
        }
        if metrics_output is not None:
            with Path(metrics_output).open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        return manifest

    def _score(mentions, pairs, output_path=None, **_kwargs):
        out = pairs[["pair_id", "mention_id_1", "mention_id_2", "block_key"]].copy()
        out["cosine_sim"] = np.linspace(0.6, 0.9, num=len(out), dtype=np.float32) if len(out) else np.array([], dtype=np.float32)
        out["distance"] = (1.0 - out["cosine_sim"]).astype(np.float32)
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(output_path, index=False)
        return out

    def _cluster(mentions, pair_scores, output_path=None, **_kwargs):
        out = mentions[["mention_id", "block_key"]].copy()
        out["author_uid"] = out["block_key"].astype(str) + "::0"
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(output_path, index=False)
        return out

    def _export(mentions, clusters, output_path):
        out = mentions[["bibcode", "mention_id"]].merge(
            clusters[["mention_id", "author_uid"]],
            on="mention_id",
            how="left",
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, index=False)
        return out

    monkeypatch.setattr(cli, "prepare_lspo_mentions", _prepare_lspo)
    monkeypatch.setattr(cli, "prepare_ads_mentions", _prepare_ads)
    monkeypatch.setattr(cli, "get_or_create_chars2vec_embeddings", _chars)
    monkeypatch.setattr(cli, "get_or_create_specter_embeddings", _text)
    monkeypatch.setattr(cli, "assign_lspo_splits", _assign)
    monkeypatch.setattr(cli, "build_pairs_within_blocks", _pairs)
    monkeypatch.setattr(cli, "train_nand_across_seeds", _train)
    monkeypatch.setattr(cli, "score_pairs_with_checkpoint", _score)
    monkeypatch.setattr(cli, "cluster_blockwise_dbscan", _cluster)
    monkeypatch.setattr(cli, "build_publication_author_mapping", _export)


def _run_stage(parser: argparse.ArgumentParser, cfg: dict[str, Path], run_id: str, extra: list[str] | None = None) -> None:
    argv = [
        "run-stage",
        "--run-stage",
        "smoke",
        "--paths-config",
        str(cfg["paths"]),
        "--run-config",
        str(cfg["run"]),
        "--model-config",
        str(cfg["model"]),
        "--cluster-config",
        str(cfg["cluster"]),
        "--gates-config",
        str(cfg["gates"]),
        "--run-id",
        run_id,
        "--no-progress",
    ]
    if extra:
        argv.extend(extra)
    args = parser.parse_args(argv)
    args.func(args)


def test_cli_run_stage_smoke_stub(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    _apply_fast_mocks(monkeypatch)

    parser = cli.build_parser()
    run_id = "smoke_test_cli_run"
    _run_stage(parser, cfg, run_id)

    metrics_dir = cfg["metrics_dir"] / run_id
    assert (metrics_dir / "00_context.json").exists()
    assert (metrics_dir / "01_subset_summary.json").exists()
    assert (metrics_dir / "02_split_balance.json").exists()
    assert (metrics_dir / "02_pairs_qc.json").exists()
    assert (metrics_dir / "03_train_manifest.json").exists()
    assert (metrics_dir / "04_cluster_qc.json").exists()
    assert (metrics_dir / "05_stage_metrics_smoke.json").exists()
    assert (metrics_dir / "05_go_no_go_smoke.json").exists()
    stage_metrics = json.loads((metrics_dir / "05_stage_metrics_smoke.json").read_text(encoding="utf-8"))
    assert "ads_cluster_assignments" in stage_metrics["counts"]
    assert "ads_blocks" in stage_metrics["counts"]
    assert "split_balance_status" in stage_metrics
    go_payload = json.loads((metrics_dir / "05_go_no_go_smoke.json").read_text(encoding="utf-8"))
    assert "warnings" in go_payload


def test_cli_run_stage_resume_behavior(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    _apply_fast_mocks(monkeypatch)

    parser = cli.build_parser()
    run_id = "smoke_test_resume"
    _run_stage(parser, cfg, run_id)

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("This function should not run during resume.")

    monkeypatch.setattr(cli, "prepare_lspo_mentions", _should_not_run)
    monkeypatch.setattr(cli, "prepare_ads_mentions", _should_not_run)
    monkeypatch.setattr(cli, "build_stage_subset", _should_not_run)
    monkeypatch.setattr(cli, "train_nand_across_seeds", _should_not_run)
    monkeypatch.setattr(cli, "score_pairs_with_checkpoint", _should_not_run)
    monkeypatch.setattr(cli, "cluster_blockwise_dbscan", _should_not_run)

    _run_stage(parser, cfg, run_id)
    assert (cfg["metrics_dir"] / run_id / "05_go_no_go_smoke.json").exists()


def test_cli_run_stage_uses_split_balance_from_run_config(monkeypatch, tmp_path: Path):
    split_cfg = {"min_neg_val": 7, "min_neg_test": 9, "max_attempts": 13}
    split_assignment = {"train_ratio": 0.55, "val_ratio": 0.25}
    cfg = _make_configs(tmp_path, split_balance=split_cfg, split_assignment=split_assignment)
    captured: dict[str, int | float] = {}
    _apply_fast_mocks(monkeypatch, captured_split=captured)

    parser = cli.build_parser()
    _run_stage(parser, cfg, "smoke_test_split_cfg", extra=["--force"])

    assert captured["min_neg_val"] == 7
    assert captured["min_neg_test"] == 9
    assert captured["max_attempts"] == 13
    assert captured["train_ratio"] == 0.55
    assert captured["val_ratio"] == 0.25


def test_cli_run_stage_uses_train_seeds_from_run_config(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path, train_seeds=[6, 7])
    _apply_fast_mocks(monkeypatch)
    captured: dict[str, list[int]] = {}
    base_train = cli.train_nand_across_seeds

    def _capture_train(*args, **kwargs):
        captured["seeds"] = [int(s) for s in kwargs["seeds"]]
        return base_train(*args, **kwargs)

    monkeypatch.setattr(cli, "train_nand_across_seeds", _capture_train)

    parser = cli.build_parser()
    _run_stage(parser, cfg, "smoke_test_train_seeds", extra=["--force"])

    assert captured["seeds"] == [6, 7]


def test_cli_run_stage_writes_val_sweep_eps_metadata(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    _apply_fast_mocks(monkeypatch)

    parser = cli.build_parser()
    run_id = "smoke_test_eps_sweep"
    _run_stage(parser, cfg, run_id, extra=["--force"])

    meta_path = cfg["metrics_dir"] / run_id / "04_clustering_config_used.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    eps_meta = payload["eps_resolution"]
    assert eps_meta["eps_mode"] == "val_sweep"
    assert eps_meta["sweep_status"] in {"ok", "fallback_no_val_pairs", "fallback_no_valid_candidates"}
    assert "n_valid_candidates" in eps_meta
    assert "boundary_hit" in eps_meta
    assert "boundary_side" in eps_meta
    assert "f1_gap_best_second" in eps_meta
    if eps_meta["sweep_status"] == "ok":
        assert len(eps_meta["sweep_results"]) >= 1
        assert 0.2 <= float(payload["cluster_config_used"]["eps"]) <= 0.5


def test_cli_run_stage_marks_val_sweep_boundary_hit(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    _apply_fast_mocks(monkeypatch)
    state: dict[str, float] = {}

    base_cluster = cli.cluster_blockwise_dbscan

    def _cluster_capture_eps(mentions, pair_scores, cluster_config, output_path=None, **kwargs):
        state["eps"] = float(cluster_config.get("eps", 0.0))
        return base_cluster(mentions, pair_scores, output_path=output_path, **kwargs)

    def _metrics_force_edge(_pairs, _clusters):
        eps = float(state.get("eps", 0.0))
        return {
            "f1": eps,
            "precision": eps,
            "recall": eps,
            "accuracy": eps,
            "n_pairs": int(len(_pairs)),
        }

    monkeypatch.setattr(cli, "cluster_blockwise_dbscan", _cluster_capture_eps)
    monkeypatch.setattr(cli, "_cluster_pairwise_metrics", _metrics_force_edge)

    def _assign_all_val(mentions, return_meta=False, **_kwargs):
        out = mentions.copy()
        out["split"] = "val"
        if "orcid" not in out.columns:
            out["orcid"] = [f"o{i // 2}" for i in range(len(out))]
        meta = {
            "status": "ok",
            "attempts": 1,
            "min_neg_val": 0,
            "min_neg_test": 0,
            "split_label_counts": {
                "train": {"pos": 0, "neg": 0, "labeled_pairs": 0},
                "val": {"pos": 3, "neg": 0, "labeled_pairs": 3},
                "test": {"pos": 0, "neg": 0, "labeled_pairs": 0},
            },
        }
        return (out, meta) if return_meta else out

    monkeypatch.setattr(cli, "assign_lspo_splits", _assign_all_val)

    parser = cli.build_parser()
    run_id = "smoke_test_eps_boundary"
    _run_stage(parser, cfg, run_id, extra=["--force"])

    meta_path = cfg["metrics_dir"] / run_id / "04_clustering_config_used.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    eps_meta = payload["eps_resolution"]
    assert eps_meta["sweep_status"] == "ok"
    assert eps_meta["boundary_hit"] is True
    assert eps_meta["boundary_side"] == "max"
    assert float(eps_meta["selected_eps"]) == 0.5
    assert int(eps_meta["n_valid_candidates"]) >= 1
    assert eps_meta["f1_gap_best_second"] is not None


def test_cli_run_stage_cli_seeds_override_run_config(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path, train_seeds=[6, 7])
    _apply_fast_mocks(monkeypatch)
    captured: dict[str, list[int]] = {}
    base_train = cli.train_nand_across_seeds

    def _capture_train(*args, **kwargs):
        captured["seeds"] = [int(s) for s in kwargs["seeds"]]
        return base_train(*args, **kwargs)

    monkeypatch.setattr(cli, "train_nand_across_seeds", _capture_train)

    parser = cli.build_parser()
    _run_stage(parser, cfg, "smoke_test_train_seeds_override", extra=["--force", "--seeds", "2", "4"])

    assert captured["seeds"] == [2, 4]


def test_cli_run_stage_rebuilds_invalid_shared_subset_cache(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    _apply_fast_mocks(monkeypatch)
    parser = cli.build_parser()

    prime_run = "smoke_test_cache_prime"
    _run_stage(parser, cfg, prime_run, extra=["--force"])
    prime_summary = json.loads((cfg["metrics_dir"] / prime_run / "01_subset_summary.json").read_text(encoding="utf-8"))
    subset_tag = prime_summary["subset_tag"]

    shared_subsets_dir = tmp_path / "data/cache/_shared/subsets"
    lspo_shared = shared_subsets_dir / f"lspo_mentions_{subset_tag}.parquet"
    ads_shared = shared_subsets_dir / f"ads_mentions_{subset_tag}.parquet"
    assert lspo_shared.exists()
    assert ads_shared.exists()

    # Corrupt cache shape to force validator rebuild.
    _toy_lspo_mentions().head(1).to_parquet(lspo_shared, index=False)

    run_id = "smoke_test_cache_rebuild"
    _run_stage(parser, cfg, run_id)

    summary = json.loads((cfg["metrics_dir"] / run_id / "01_subset_summary.json").read_text(encoding="utf-8"))
    assert summary["cache_rebuilt"] is True
    assert summary["cache_valid"] is True
    assert "expected" in str(summary.get("cache_invalid_reason"))


def test_cli_run_stage_aborts_early_on_split_balance_infeasible(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    _apply_fast_mocks(monkeypatch)

    def _assign_infeasible(
        mentions,
        seed=11,
        train_ratio=0.6,
        val_ratio=0.2,
        min_neg_val=0,
        min_neg_test=0,
        max_attempts=1,
        return_meta=False,
        **_kwargs,
    ):
        out = mentions.copy()
        out["split"] = "train"
        if "orcid" not in out.columns:
            out["orcid"] = [f"o{i // 2}" for i in range(len(out))]
        if int(max_attempts) == 1:
            meta = {
                "status": "ok",
                "attempts": 1,
                "min_neg_val": int(min_neg_val),
                "min_neg_test": int(min_neg_test),
                "required_neg_total": int(min_neg_val + min_neg_test),
                "max_possible_neg_total": 1000,
                "split_label_counts": {"train": {"pos": 10, "neg": 10, "labeled_pairs": 20}, "val": {"pos": 0, "neg": 0, "labeled_pairs": 0}, "test": {"pos": 0, "neg": 0, "labeled_pairs": 0}},
            }
        else:
            meta = {
                "status": "split_balance_infeasible",
                "attempts": 1,
                "min_neg_val": int(min_neg_val),
                "min_neg_test": int(min_neg_test),
                "required_neg_total": int(min_neg_val + min_neg_test),
                "max_possible_neg_total": 0,
                "split_label_counts": {"train": {"pos": 10, "neg": 0, "labeled_pairs": 10}, "val": {"pos": 0, "neg": 0, "labeled_pairs": 0}, "test": {"pos": 0, "neg": 0, "labeled_pairs": 0}},
            }
        return (out, meta) if return_meta else out

    def _train_should_not_run(*_args, **_kwargs):
        raise AssertionError("train should not run when split balance is infeasible")

    monkeypatch.setattr(cli, "assign_lspo_splits", _assign_infeasible)
    monkeypatch.setattr(cli, "train_nand_across_seeds", _train_should_not_run)

    parser = cli.build_parser()
    with pytest.raises(RuntimeError, match="split_balance_infeasible"):
        _run_stage(parser, cfg, "smoke_test_infeasible_abort", extra=["--force"])
