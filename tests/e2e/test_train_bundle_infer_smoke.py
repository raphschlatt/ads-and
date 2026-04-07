from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from author_name_disambiguation import cli


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


def _write_dataset(tmp_path: Path, dataset_id: str) -> None:
    ds_dir = tmp_path / "data" / "raw" / "ads" / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)
    pubs = [
        {
            "Bibcode": "bib1",
            "Author": ["Doe J", "Doe J."],
            "Title_en": "Paper 1",
            "Abstract_en": "Abstract 1",
            "Year": 2020,
            "Affiliation": "Inst A",
        },
        {
            "Bibcode": "bib2",
            "Author": ["Doe J"],
            "Title_en": "Paper 2",
            "Abstract_en": "Abstract 2",
            "Year": 2021,
            "Affilliation": "Inst B",
        },
    ]
    with (ds_dir / "publications.jsonl").open("w", encoding="utf-8") as f:
        for row in pubs:
            f.write(json.dumps(row) + "\n")


def _make_configs(tmp_path: Path) -> dict[str, Path]:
    paths_cfg = {
        "project_root": str(tmp_path),
        "data": {
            "raw_lspo_parquet": str(tmp_path / "data/raw/lspo/mock.parquet"),
            "raw_lspo_h5": str(tmp_path / "data/raw/lspo/mock.h5"),
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
        "stage": "smoke",
        "subset_target_mentions": 6,
        "seed": 11,
        "subset_sampling": {"target_mean_block_size": 4},
        "split_assignment": {"train_ratio": 0.6, "val_ratio": 0.2},
        "pair_building": {"exclude_same_bibcode": True},
        "max_pairs_per_block": 100,
        "split_balance": {"min_neg_val": 2, "min_neg_test": 2, "max_attempts": 5},
        "train_seeds": [1],
    }
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
            "precision_mode": "fp32",
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
        "boundary_diagnostics": {
            "diag_min": 0.55,
            "diag_max": 0.70,
            "diag_step": 0.05,
            "delta_f1_threshold": 0.005,
        },
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
        "stages": {
            "smoke": {"f1_min": 0.0, "min_neg_val": 0, "min_neg_test": 0},
            "infer_sources": {"f1_min": 0.0, "min_neg_val": 0, "min_neg_test": 0},
        },
    }

    cfg_dir = tmp_path / "cfg"
    return {
        "paths": _write_yaml(cfg_dir / "paths.yaml", paths_cfg),
        "run": _write_yaml(cfg_dir / "run.yaml", run_cfg),
        "model": _write_yaml(cfg_dir / "model.yaml", model_cfg),
        "cluster": _write_yaml(cfg_dir / "cluster.yaml", cluster_cfg),
        "gates": _write_yaml(cfg_dir / "gates.yaml", gates_cfg),
        "data_root": tmp_path / "data",
        "artifacts_root": tmp_path / "artifacts",
        "raw_lspo_parquet": Path(paths_cfg["data"]["raw_lspo_parquet"]),
        "raw_lspo_h5": Path(paths_cfg["data"]["raw_lspo_h5"]),
    }


def _apply_fast_mocks(monkeypatch) -> None:
    lspo_df = _toy_lspo_mentions()

    def _prepare_lspo(*, parquet_path=None, h5_path=None, output_path, **_kwargs):
        _ = parquet_path, h5_path
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lspo_df.to_parquet(p, index=False)
        return lspo_df.copy()

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

    def _assign(mentions, min_neg_val=0, min_neg_test=0, return_meta=False, **_kwargs):
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

    def _pairs(mentions, return_meta=False, **_kwargs):
        rows = []
        for block_key, grp in mentions.groupby("block_key", sort=False):
            if len(grp) < 2:
                continue
            a = grp.iloc[0]
            b = grp.iloc[1]
            label = None
            if "orcid" in grp.columns:
                label = int(str(a.get("orcid", "")).strip() == str(b.get("orcid", "")).strip())
            rows.append(
                {
                    "pair_id": f"{a['mention_id']}__{b['mention_id']}",
                    "mention_id_1": str(a["mention_id"]),
                    "mention_id_2": str(b["mention_id"]),
                    "block_key": str(block_key),
                    "split": str(a.get("split", "inference")),
                    "label": label,
                }
            )
        out = pd.DataFrame(rows)
        meta = {
            "same_publication_pairs_skipped": 0,
            "pairs_written": int(len(out)),
            "cpu_sharding_mode": str(_kwargs.get("sharding_mode", "auto")),
            "cpu_sharding_enabled": str(_kwargs.get("sharding_mode", "auto")) != "off",
            "cpu_workers_requested": "auto" if _kwargs.get("num_workers") is None else int(_kwargs["num_workers"]),
            "cpu_workers_effective": 4 if _kwargs.get("num_workers") is None else int(_kwargs["num_workers"]),
            "cpu_min_pairs_per_worker": int(_kwargs.get("min_pairs_per_worker", 1)),
            "ram_budget_bytes": int(_kwargs["ram_budget_bytes"]) if _kwargs.get("ram_budget_bytes") is not None else None,
            "total_pairs_est": int(len(out)),
        }
        return (out, meta) if return_meta else out

    def _train(seeds, run_id, output_dir, metrics_output=None, **_kwargs):
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
            "best_test_f1": 0.89,
            "precision_mode": "fp32",
            "runs": [],
        }
        if metrics_output is not None:
            with Path(metrics_output).open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        return manifest

    def _score(mentions, pairs, output_path=None, **_kwargs):
        del mentions
        if isinstance(pairs, (str, Path)):
            pairs = pd.read_parquet(pairs)
        out = pairs[["pair_id", "mention_id_1", "mention_id_2", "block_key"]].copy()
        if len(out):
            out["cosine_sim"] = np.linspace(0.6, 0.9, num=len(out), dtype=np.float32)
            out["distance"] = (1.0 - out["cosine_sim"]).astype(np.float32)
        else:
            out["cosine_sim"] = pd.Series(dtype=np.float32)
            out["distance"] = pd.Series(dtype=np.float32)
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
        if bool(_kwargs.get("return_meta")):
            return out, {
                "cluster_backend_requested": str(_kwargs.get("backend", "auto")),
                "cluster_backend_effective": "sklearn_cpu",
                "cpu_sharding_mode": str(_kwargs.get("sharding_mode", "auto")),
                "cpu_sharding_enabled": str(_kwargs.get("sharding_mode", "auto")) != "off",
                "cpu_workers_requested": "auto" if _kwargs.get("num_workers") is None else int(_kwargs["num_workers"]),
                "cpu_workers_effective": 4 if _kwargs.get("num_workers") is None else int(_kwargs["num_workers"]),
                "ram_budget_bytes": int(_kwargs["ram_budget_bytes"]) if _kwargs.get("ram_budget_bytes") is not None else None,
            }
        return out

    def _encode_mentions(chars2vec, source_text_embeddings, mention_source_index, output_path, norms_output_path=None, **_kwargs):
        del chars2vec, source_text_embeddings
        mention_index = np.load(mention_source_index)
        arr = np.ones((len(mention_index), 4), dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1).astype(np.float32)
        np.save(output_path, arr)
        np.save(norms_output_path, norms)
        return (
            Path(output_path),
            {
                "resolved_device": "cpu",
                "effective_precision_mode": "fp32",
                "mention_encode_seconds": 0.01,
                "mention_embedding_shape": [int(len(arr)), int(arr.shape[1])],
                "mention_embedding_bytes": int(arr.nbytes),
                "mention_norm_bytes": int(norms.nbytes),
                "cuda_oom_fallback_used": False,
            },
        )

    def _score_from_embeddings(mentions, pairs, output_path=None, score_callback=None, **_kwargs):
        out = _score(mentions=mentions, pairs=pairs, output_path=output_path, **_kwargs)
        score_df = out[0] if isinstance(out, tuple) else out
        if score_callback is not None:
            score_callback(
                {
                    "pair_id": score_df["pair_id"].astype(str).to_numpy(copy=False),
                    "mention_id_1": score_df["mention_id_1"].astype(str).to_numpy(copy=False),
                    "mention_id_2": score_df["mention_id_2"].astype(str).to_numpy(copy=False),
                    "block_key": score_df["block_key"].astype(str).to_numpy(copy=False),
                    "distance": score_df["distance"].astype(np.float32).to_numpy(copy=False),
                }
            )
        if _kwargs.get("return_runtime_meta"):
            return score_df, {
                "requested_device": str(_kwargs.get("device", "cpu")),
                "resolved_device": "cpu",
                "fallback_reason": None,
                "torch_version": "2.10.0+cpu",
                "torch_cuda_version": None,
                "torch_cuda_available": False,
                "cuda_probe_error": None,
                "model_to_cuda_error": None,
                "effective_precision_mode": "fp32",
                "pair_scoring_strategy": "preencoded_mentions",
                "mention_storage_device": "cpu",
                "cuda_oom_fallback_used": False,
                "numeric_clamping": {"clamped": False},
            }
        return out

    class _FakeExactGraphClusterAccumulator:
        def __init__(self, *, mentions, cluster_config, backend_requested="auto"):
            del cluster_config
            self._mentions = mentions
            self._backend_requested = backend_requested
            self._pair_rows = 0

        def consume_score_columns(self, score_columns):
            self._pair_rows += int(len(score_columns.get("block_key", [])))

        def finalize(self):
            out = self._mentions[["mention_id", "block_key"]].copy()
            out["author_uid"] = out["block_key"].astype(str) + "::0"
            return out, {
                "cluster_backend_requested": str(self._backend_requested),
                "cluster_backend_effective": "connected_components_cpu",
                "cpu_sharding_mode": "off",
                "cpu_sharding_enabled": False,
                "cpu_workers_requested": 4,
                "cpu_workers_effective": 4,
                "ram_budget_bytes": None,
                "total_pairs_est": int(self._pair_rows),
                "block_p95": 2.0,
                "block_size_histogram": {"1": 1},
                "block_count_by_bucket": {"1": 1},
                "total_seconds_by_bucket": {},
                "dbscan_seconds_by_bucket": {},
                "distance_matrix_seconds_total": 0.0,
                "constraints_seconds_total": 0.0,
                "sanitize_seconds_total": 0.0,
                "dbscan_seconds_total": 0.0,
                "gpu_transfer_seconds_total": 0.0,
                "top_slow_blocks": [],
                "sanitize_totals": {
                    "corrected_blocks": 0,
                    "non_finite_count": 0,
                    "negative_count": 0,
                    "above_max_count": 0,
                    "asymmetry_pairs": 0,
                    "diag_reset_count": 0,
                },
            }

    def _export(mentions, clusters, output_path):
        out = mentions[["bibcode", "author_idx", "mention_id", "source_type"]].merge(
            clusters[["mention_id", "author_uid"]],
            on="mention_id",
            how="left",
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, index=False)
        return out

    monkeypatch.setattr(cli, "prepare_lspo_mentions", _prepare_lspo)
    monkeypatch.setattr(cli, "get_or_create_chars2vec_embeddings", _chars)
    monkeypatch.setattr(cli, "get_or_create_specter_embeddings", _text)
    monkeypatch.setattr(cli, "assign_lspo_splits", _assign)
    monkeypatch.setattr(cli, "build_pairs_within_blocks", _pairs)
    monkeypatch.setattr(cli, "train_nand_across_seeds", _train)
    monkeypatch.setattr(cli, "score_pairs_with_checkpoint", _score)
    monkeypatch.setattr(cli, "cluster_blockwise_dbscan", _cluster)
    monkeypatch.setattr("author_name_disambiguation.source_inference.get_or_create_chars2vec_embeddings", _chars)
    monkeypatch.setattr("author_name_disambiguation.source_inference.get_or_create_specter_embeddings", _text)
    monkeypatch.setattr("author_name_disambiguation.source_inference.build_pairs_within_blocks", _pairs)
    monkeypatch.setattr("author_name_disambiguation.source_inference.score_pairs_with_checkpoint", _score)
    monkeypatch.setattr("author_name_disambiguation.source_inference.cluster_blockwise_dbscan", _cluster)
    monkeypatch.setattr("author_name_disambiguation.source_inference.encode_mentions_to_memmap", _encode_mentions)
    monkeypatch.setattr("author_name_disambiguation.source_inference.score_pairs_from_mention_embeddings", _score_from_embeddings)
    monkeypatch.setattr("author_name_disambiguation.source_inference.ExactGraphClusterAccumulator", _FakeExactGraphClusterAccumulator)


def test_train_bundle_infer_contract_smoke(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    _write_dataset(tmp_path, dataset_id="ads_fixture")
    raw_lspo_path = tmp_path / "data" / "raw" / "lspo" / "mock.parquet"
    raw_lspo_path.parent.mkdir(parents=True, exist_ok=True)
    _toy_lspo_mentions().to_parquet(raw_lspo_path, index=False)
    _apply_fast_mocks(monkeypatch)
    parser = cli.build_parser()

    train_run_id = "smoke_train_fixture"
    args = parser.parse_args(
        [
            "run-train-stage",
            "--run-stage",
            "smoke",
            "--data-root",
            str(cfg["data_root"]),
            "--artifacts-root",
            str(cfg["artifacts_root"]),
            "--raw-lspo-parquet",
            str(cfg["raw_lspo_parquet"]),
            "--raw-lspo-h5",
            str(cfg["raw_lspo_h5"]),
            "--run-config",
            str(cfg["run"]),
            "--model-config",
            str(cfg["model"]),
            "--cluster-config",
            str(cfg["cluster"]),
            "--gates-config",
            str(cfg["gates"]),
            "--run-id",
            train_run_id,
            "--no-progress",
        ]
    )
    args.func(args)

    report_args = parser.parse_args(
        [
            "run-cluster-test-report",
            "--model-run-id",
            train_run_id,
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
    )
    report_args.func(report_args)

    report_json = tmp_path / "artifacts" / "metrics" / train_run_id / "06_clustering_test_report.json"
    assert report_json.exists()
    report_payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert report_payload["status"] == "ok"

    export_args = parser.parse_args(
        [
            "export-model-bundle",
            "--model-run-id",
            train_run_id,
            "--artifacts-root",
            str(cfg["artifacts_root"]),
        ]
    )
    export_args.func(export_args)

    bundle_dir = tmp_path / "artifacts" / "models" / train_run_id / "bundle_v1"
    assert (bundle_dir / "bundle_manifest.json").exists()
    assert (bundle_dir / "checkpoint.pt").exists()

    infer_run_id = "infer_fixture"
    publications_path = tmp_path / "data" / "raw" / "ads" / "ads_fixture" / "publications.jsonl"
    output_root = tmp_path / "artifacts" / "exports" / infer_run_id
    infer_args = parser.parse_args(
        [
            "run-infer-sources",
            "--publications-path",
            str(publications_path),
            "--output-root",
            str(output_root),
            "--dataset-id",
            "ads_fixture",
            "--model-bundle",
            str(bundle_dir),
            "--cluster-config",
            str(cfg["cluster"]),
            "--gates-config",
            str(cfg["gates"]),
            "--no-progress",
        ]
    )
    infer_args.func(infer_args)

    metrics_dir = output_root
    clusters_path = output_root / "mention_clusters.parquet"
    export_path = output_root / "source_author_assignments.parquet"
    mentions_path = output_root / "interim" / "mentions_full.parquet"

    assert (metrics_dir / "05_stage_metrics_infer_sources.json").exists()
    assert (metrics_dir / "05_go_no_go_infer_sources.json").exists()
    assert clusters_path.exists()
    assert export_path.exists()
    assert mentions_path.exists()

    mentions = pd.read_parquet(mentions_path)
    clusters = pd.read_parquet(clusters_path)
    joined = mentions[["mention_id"]].merge(clusters[["mention_id", "author_uid"]], on="mention_id", how="left")
    assert int(joined["author_uid"].notna().sum()) == int(len(mentions))
