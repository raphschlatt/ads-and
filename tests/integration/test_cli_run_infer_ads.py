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
from src.infer_ads_api import InferAdsRequest, run_infer_ads


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


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
    cluster_cfg = {
        "method": "dbscan",
        "eps_mode": "val_sweep",
        "eps": 0.35,
        "eps_fallback": 0.35,
        "eps_min": 0.15,
        "eps_max": 0.85,
        "min_samples": 1,
        "metric": "precomputed",
        "constraints": {"enabled": False},
    }
    model_cfg = {
        "name": "mock-nand",
        "representation": {"text_model_name": "mock-specter", "max_length": 64},
        "training": {"precision_mode": "fp32"},
    }
    run_cfg = {"max_pairs_per_block": 100, "pair_building": {"exclude_same_bibcode": True}}

    cfg_dir = tmp_path / "cfg"
    return {
        "paths": _write_yaml(cfg_dir / "paths.yaml", paths_cfg),
        "cluster": _write_yaml(cfg_dir / "cluster.yaml", cluster_cfg),
        "model": _write_yaml(cfg_dir / "model.yaml", model_cfg),
        "run": _write_yaml(cfg_dir / "run.yaml", run_cfg),
        "metrics_dir": Path(paths_cfg["artifacts"]["metrics_dir"]),
    }


def _write_dataset(tmp_path: Path, dataset_id: str, with_references: bool = True) -> Path:
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

    if with_references:
        refs = [
            {
                "Bibcode": "bib3",
                "Author": ["Doe J"],
                "Title_en": "Paper 3",
                "Abstract_en": "Abstract 3",
                "Year": 2022,
                "Affiliation": "Inst C",
            }
        ]
        with (ds_dir / "references.jsonl").open("w", encoding="utf-8") as f:
            for row in refs:
                f.write(json.dumps(row) + "\n")
    return ds_dir


def _write_model_run_artifacts(tmp_path: Path, cfg: dict[str, Path], model_run_id: str) -> None:
    metrics_dir = cfg["metrics_dir"] / model_run_id
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / model_run_id / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    with (metrics_dir / "03_train_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": model_run_id,
                "best_checkpoint": str(checkpoint_path),
                "best_threshold": 0.25,
                "best_test_f1": 0.91,
                "best_val_f1": 0.92,
                "best_val_class_counts": {"pos": 100, "neg": 80},
                "best_test_class_counts": {"pos": 95, "neg": 75},
            },
            f,
            indent=2,
        )
    with (metrics_dir / "04_clustering_config_used.json").open("w", encoding="utf-8") as f:
        json.dump({"eps_resolution": {"selected_eps": 0.42}}, f, indent=2)
    with (metrics_dir / "00_context.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_config": str(cfg["run"]),
                "model_config": str(cfg["model"]),
            },
            f,
            indent=2,
        )


def _apply_fast_mocks(monkeypatch) -> None:
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
                "cpu_workers_effective": int(_kwargs.get("num_workers", 1)),
            }
        return out

    monkeypatch.setattr(cli, "get_or_create_chars2vec_embeddings", _chars)
    monkeypatch.setattr(cli, "get_or_create_specter_embeddings", _text)
    monkeypatch.setattr(cli, "score_pairs_with_checkpoint", _score)
    monkeypatch.setattr(cli, "cluster_blockwise_dbscan", _cluster)


def _run_infer_ads(
    parser: argparse.ArgumentParser,
    cfg: dict[str, Path],
    dataset_id: str,
    run_id: str,
    *,
    model_run_id: str | None = None,
    model_bundle: str | None = None,
    infer_stage: str | None = None,
    uid_scope: str | None = None,
    uid_namespace: str | None = None,
) -> None:
    argv = [
        "run-infer-ads",
        "--dataset-id",
        dataset_id,
        "--paths-config",
        str(cfg["paths"]),
        "--cluster-config",
        str(cfg["cluster"]),
        "--run-id",
        run_id,
        "--no-progress",
    ]
    if infer_stage is not None:
        argv.extend(["--infer-stage", infer_stage])
    if uid_scope is not None:
        argv.extend(["--uid-scope", uid_scope])
    if uid_namespace is not None:
        argv.extend(["--uid-namespace", uid_namespace])
    if model_run_id is not None:
        argv.extend(["--model-run-id", model_run_id])
    if model_bundle is not None:
        argv.extend(["--model-bundle", model_bundle])
    args = parser.parse_args(argv)
    args.func(args)


def test_cli_run_infer_ads_writes_artifacts(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    dataset_id = "my_ads_2026"
    model_run_id = "full_2026abc"
    _write_dataset(tmp_path, dataset_id=dataset_id, with_references=False)
    _write_model_run_artifacts(tmp_path, cfg, model_run_id=model_run_id)
    _apply_fast_mocks(monkeypatch)

    parser = cli.build_parser()
    run_id = "infer_ads_test"
    _run_infer_ads(parser, cfg, dataset_id=dataset_id, model_run_id=model_run_id, run_id=run_id)

    metrics_dir = cfg["metrics_dir"] / run_id
    assert (metrics_dir / "00_context.json").exists()
    assert (metrics_dir / "00_cache_refs.json").exists()
    assert (metrics_dir / "00_run_consistency.json").exists()
    assert (metrics_dir / "01_input_summary.json").exists()
    assert (metrics_dir / "01_run_consistency.json").exists()
    assert (metrics_dir / "02_preflight_infer.json").exists()
    assert (metrics_dir / "02_run_consistency.json").exists()
    assert (metrics_dir / "03_pairs_qc.json").exists()
    assert (metrics_dir / "04_cluster_qc.json").exists()
    assert (metrics_dir / "04_source_export_qc.json").exists()
    assert (metrics_dir / "04_clustering_config_used.json").exists()
    assert (metrics_dir / "04_run_consistency.json").exists()
    assert (metrics_dir / "05_stage_metrics_infer_ads.json").exists()
    assert (metrics_dir / "05_go_no_go_infer_ads.json").exists()
    assert (metrics_dir / "05_run_consistency.json").exists()

    context = json.loads((metrics_dir / "00_context.json").read_text(encoding="utf-8"))
    assert context["dataset_id"] == dataset_id
    assert context["model_run_id"] == model_run_id
    assert context["selected_eps"] == 0.42
    assert context["pipeline_scope"] == "infer"
    assert context["model_source_type"] == "run_id"
    assert context["infer_stage"] == "full"
    assert context["score_batch_size"] == 8192
    assert context["cpu_sharding_mode"] == "auto"
    assert context["cluster_backend_requested"] == "auto"
    assert context["cluster_backend_effective"] in {"auto", "sklearn_cpu"}
    assert context["uid_scope"] == "dataset"
    assert context["uid_namespace"] == "my_ads_2026"

    summary = json.loads((metrics_dir / "01_input_summary.json").read_text(encoding="utf-8"))
    assert summary["references_present"] is False
    assert summary["subset_ratio"] == 1.0

    stage_metrics = json.loads((metrics_dir / "05_stage_metrics_infer_ads.json").read_text(encoding="utf-8"))
    assert stage_metrics["stage"] == "infer_ads"
    assert stage_metrics["metric_scope"] == "infer"
    assert stage_metrics["counts"]["ads_mentions"] > 0
    assert stage_metrics["infer_stage"] == "full"
    assert stage_metrics["memory_feasible"] in {True, False, None}
    assert stage_metrics["pair_upper_bound"] is not None

    cluster_path = tmp_path / "artifacts" / "clusters" / run_id / "ads_clusters_infer_ads.parquet"
    export_path = tmp_path / "artifacts" / "clusters" / run_id / "publication_authors_infer_ads.parquet"
    pubs_out = tmp_path / "artifacts" / "exports" / run_id / "publications.disambiguated.jsonl"
    assert cluster_path.exists()
    assert export_path.exists()
    assert pubs_out.exists()

    clusters = pd.read_parquet(cluster_path)
    export = pd.read_parquet(export_path)
    assert {"mention_id", "block_key", "author_uid"}.issubset(clusters.columns)
    assert {"mention_id", "author_uid", "author_uid_local"}.issubset(clusters.columns)
    assert {"bibcode", "author_idx", "mention_id", "source_type", "author_uid", "author_uid_local"}.issubset(
        export.columns
    )
    assert clusters["author_uid"].astype(str).str.startswith("my_ads_2026::").all()
    assert (clusters["author_uid_local"].astype(str) == clusters["block_key"].astype(str) + "::0").all()

    first_row = json.loads(pubs_out.read_text(encoding="utf-8").splitlines()[0])
    assert "AuthorUID" in first_row
    assert all(uid is None or str(uid).startswith("my_ads_2026::") for uid in first_row["AuthorUID"])


def test_cli_run_infer_ads_resume_reuses_artifacts(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    dataset_id = "my_ads_2026"
    model_run_id = "full_2026abc"
    _write_dataset(tmp_path, dataset_id=dataset_id, with_references=True)
    _write_model_run_artifacts(tmp_path, cfg, model_run_id=model_run_id)
    _apply_fast_mocks(monkeypatch)
    parser = cli.build_parser()
    run_id = "infer_ads_resume"
    _run_infer_ads(parser, cfg, dataset_id=dataset_id, model_run_id=model_run_id, run_id=run_id)

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("This function should not run during resume.")

    monkeypatch.setattr(cli, "prepare_ads_mentions", _should_not_run)
    monkeypatch.setattr(cli, "build_pairs_within_blocks", _should_not_run)
    monkeypatch.setattr(cli, "score_pairs_with_checkpoint", _should_not_run)
    monkeypatch.setattr(cli, "cluster_blockwise_dbscan", _should_not_run)

    _run_infer_ads(parser, cfg, dataset_id=dataset_id, model_run_id=model_run_id, run_id=run_id)
    assert (cfg["metrics_dir"] / run_id / "05_go_no_go_infer_ads.json").exists()


def test_cli_run_infer_ads_with_mini_stage_builds_subset(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    dataset_id = "my_ads_2026"
    model_run_id = "full_2026abc"
    _write_dataset(tmp_path, dataset_id=dataset_id, with_references=True)
    _write_model_run_artifacts(tmp_path, cfg, model_run_id=model_run_id)
    _apply_fast_mocks(monkeypatch)
    parser = cli.build_parser()
    run_id = "infer_ads_mini"
    _run_infer_ads(
        parser,
        cfg,
        dataset_id=dataset_id,
        model_run_id=model_run_id,
        run_id=run_id,
        infer_stage="mini",
    )

    metrics_dir = cfg["metrics_dir"] / run_id
    context = json.loads((metrics_dir / "00_context.json").read_text(encoding="utf-8"))
    summary = json.loads((metrics_dir / "01_input_summary.json").read_text(encoding="utf-8"))
    stage_metrics = json.loads((metrics_dir / "05_stage_metrics_infer_ads.json").read_text(encoding="utf-8"))

    assert context["infer_stage"] == "mini"
    assert summary["subset_tag"].startswith("infer_mini_")
    assert 0.0 < float(summary["subset_ratio"]) <= 1.0
    assert stage_metrics["infer_stage"] == "mini"


def test_cli_run_infer_ads_memory_policy_fail_aborts_early(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    dataset_id = "my_ads_2026"
    model_run_id = "full_2026abc"
    _write_dataset(tmp_path, dataset_id=dataset_id, with_references=True)
    _write_model_run_artifacts(tmp_path, cfg, model_run_id=model_run_id)
    _apply_fast_mocks(monkeypatch)
    parser = cli.build_parser()

    run_id = "infer_ads_mem_fail"
    args = parser.parse_args(
        [
            "run-infer-ads",
            "--dataset-id",
            dataset_id,
            "--model-run-id",
            model_run_id,
            "--paths-config",
            str(cfg["paths"]),
            "--cluster-config",
            str(cfg["cluster"]),
            "--run-id",
            run_id,
            "--memory-policy",
            "fail",
            "--max-ram-fraction",
            "0.0000001",
            "--no-progress",
        ]
    )
    with pytest.raises(RuntimeError, match="memory_feasible=false"):
        args.func(args)

    metrics_dir = cfg["metrics_dir"] / run_id
    assert (metrics_dir / "02_preflight_infer.json").exists()
    assert not (metrics_dir / "05_stage_metrics_infer_ads.json").exists()


def test_cli_run_infer_ads_supports_model_bundle(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    dataset_id = "my_ads_2026"
    model_run_id = "full_2026abc"
    _write_dataset(tmp_path, dataset_id=dataset_id, with_references=True)
    _write_model_run_artifacts(tmp_path, cfg, model_run_id=model_run_id)
    _apply_fast_mocks(monkeypatch)

    parser = cli.build_parser()
    export_args = parser.parse_args(
        [
            "export-model-bundle",
            "--model-run-id",
            model_run_id,
            "--paths-config",
            str(cfg["paths"]),
        ]
    )
    export_args.func(export_args)

    bundle_manifest = tmp_path / "artifacts" / "models" / model_run_id / "bundle_v1" / "bundle_manifest.json"
    assert bundle_manifest.exists()

    run_id = "infer_ads_bundle"
    _run_infer_ads(
        parser,
        cfg,
        dataset_id=dataset_id,
        run_id=run_id,
        model_bundle=str(bundle_manifest.parent),
    )

    metrics_dir = cfg["metrics_dir"] / run_id
    context = json.loads((metrics_dir / "00_context.json").read_text(encoding="utf-8"))
    assert context["model_source_type"] == "bundle"
    assert context["model_bundle_dir"] == str(bundle_manifest.parent.resolve())
    assert context["selected_eps"] == 0.42
    assert context["uid_scope"] == "dataset"
    assert context["uid_namespace"] == "my_ads_2026"


def test_cli_run_infer_ads_uid_scope_local_keeps_local_ids(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    dataset_id = "my_ads_2026"
    model_run_id = "full_2026abc"
    _write_dataset(tmp_path, dataset_id=dataset_id, with_references=True)
    _write_model_run_artifacts(tmp_path, cfg, model_run_id=model_run_id)
    _apply_fast_mocks(monkeypatch)

    parser = cli.build_parser()
    run_id = "infer_ads_uid_local"
    _run_infer_ads(
        parser,
        cfg,
        dataset_id=dataset_id,
        model_run_id=model_run_id,
        run_id=run_id,
        uid_scope="local",
    )

    cluster_path = tmp_path / "artifacts" / "clusters" / run_id / "ads_clusters_infer_ads.parquet"
    clusters = pd.read_parquet(cluster_path)
    assert {"author_uid", "author_uid_local"}.issubset(clusters.columns)
    assert (clusters["author_uid"].astype(str) == clusters["author_uid_local"].astype(str)).all()
    assert (~clusters["author_uid"].astype(str).str.startswith("my_ads_2026::")).all()


def test_cli_and_api_infer_ads_parity(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    dataset_id = "my_ads_2026"
    model_run_id = "full_2026abc"
    _write_dataset(tmp_path, dataset_id=dataset_id, with_references=True)
    _write_model_run_artifacts(tmp_path, cfg, model_run_id=model_run_id)
    _apply_fast_mocks(monkeypatch)

    parser = cli.build_parser()
    run_id_cli = "infer_ads_cli_parity"
    run_id_api = "infer_ads_api_parity"

    _run_infer_ads(
        parser,
        cfg,
        dataset_id=dataset_id,
        model_run_id=model_run_id,
        run_id=run_id_cli,
    )

    api_result = run_infer_ads(
        InferAdsRequest(
            dataset_id=dataset_id,
            model_run_id=model_run_id,
            paths_config=str(cfg["paths"]),
            cluster_config=str(cfg["cluster"]),
            run_id=run_id_api,
            progress=False,
        )
    )

    cli_metrics_dir = cfg["metrics_dir"] / run_id_cli
    api_metrics_dir = cfg["metrics_dir"] / run_id_api
    cli_stage = json.loads((cli_metrics_dir / "05_stage_metrics_infer_ads.json").read_text(encoding="utf-8"))
    api_stage = json.loads((api_metrics_dir / "05_stage_metrics_infer_ads.json").read_text(encoding="utf-8"))

    assert api_result.run_id == run_id_api
    assert api_result.metrics_dir == api_metrics_dir
    assert cli_stage["metric_scope"] == api_stage["metric_scope"] == "infer"
    assert cli_stage["infer_stage"] == api_stage["infer_stage"] == "full"
    assert cli_stage["counts"]["ads_mentions"] == api_stage["counts"]["ads_mentions"]
    assert cli_stage["counts"]["ads_cluster_assignments"] == api_stage["counts"]["ads_cluster_assignments"]


def test_cli_run_infer_ads_uid_scope_registry_is_stable_across_runs(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    dataset_id = "my_ads_2026"
    model_run_id = "full_2026abc"
    _write_dataset(tmp_path, dataset_id=dataset_id, with_references=True)
    _write_model_run_artifacts(tmp_path, cfg, model_run_id=model_run_id)
    _apply_fast_mocks(monkeypatch)

    parser = cli.build_parser()
    run_id_a = "infer_ads_registry_a"
    run_id_b = "infer_ads_registry_b"
    uid_namespace = "stable_ads"

    _run_infer_ads(
        parser,
        cfg,
        dataset_id=dataset_id,
        model_run_id=model_run_id,
        run_id=run_id_a,
        uid_scope="registry",
        uid_namespace=uid_namespace,
    )
    _run_infer_ads(
        parser,
        cfg,
        dataset_id=dataset_id,
        model_run_id=model_run_id,
        run_id=run_id_b,
        uid_scope="registry",
        uid_namespace=uid_namespace,
    )

    clusters_a = pd.read_parquet(tmp_path / "artifacts" / "clusters" / run_id_a / "ads_clusters_infer_ads.parquet")
    clusters_b = pd.read_parquet(tmp_path / "artifacts" / "clusters" / run_id_b / "ads_clusters_infer_ads.parquet")
    merged = clusters_a[["mention_id", "author_uid"]].merge(
        clusters_b[["mention_id", "author_uid"]],
        on="mention_id",
        suffixes=("_a", "_b"),
        how="inner",
    )
    assert len(merged) == len(clusters_a)
    assert (merged["author_uid_a"].astype(str) == merged["author_uid_b"].astype(str)).all()
    assert merged["author_uid_a"].astype(str).str.startswith("stable_ads::au").all()
    local_to_global_a = clusters_a.groupby("author_uid_local")["author_uid"].nunique()
    local_to_global_b = clusters_b.groupby("author_uid_local")["author_uid"].nunique()
    assert int(local_to_global_a.max()) == 1
    assert int(local_to_global_b.max()) == 1

    stage_a = json.loads((tmp_path / "artifacts" / "metrics" / run_id_a / "05_stage_metrics_infer_ads.json").read_text())
    stage_b = json.loads((tmp_path / "artifacts" / "metrics" / run_id_b / "05_stage_metrics_infer_ads.json").read_text())
    assert stage_a["uid_local_to_global_valid"] is True
    assert stage_b["uid_local_to_global_valid"] is True
    assert stage_a["uid_local_to_global_max_nunique"] == 1
    assert stage_b["uid_local_to_global_max_nunique"] == 1


def test_cli_run_infer_ads_uid_scope_registry_rebuilds_stage_reports_on_resume(monkeypatch, tmp_path: Path):
    cfg = _make_configs(tmp_path)
    dataset_id = "my_ads_2026"
    model_run_id = "full_2026abc"
    _write_dataset(tmp_path, dataset_id=dataset_id, with_references=True)
    _write_model_run_artifacts(tmp_path, cfg, model_run_id=model_run_id)
    _apply_fast_mocks(monkeypatch)

    parser = cli.build_parser()
    run_id = "infer_ads_registry_resume"
    uid_namespace = "stable_ads"

    _run_infer_ads(
        parser,
        cfg,
        dataset_id=dataset_id,
        model_run_id=model_run_id,
        run_id=run_id,
        uid_scope="registry",
        uid_namespace=uid_namespace,
    )

    metrics_dir = cfg["metrics_dir"] / run_id
    stage_path = metrics_dir / "05_stage_metrics_infer_ads.json"
    stage_path.write_text('{"stale": true}', encoding="utf-8")

    _run_infer_ads(
        parser,
        cfg,
        dataset_id=dataset_id,
        model_run_id=model_run_id,
        run_id=run_id,
        uid_scope="registry",
        uid_namespace=uid_namespace,
    )

    rebuilt_stage = json.loads(stage_path.read_text(encoding="utf-8"))
    assert rebuilt_stage.get("stale") is None
    assert rebuilt_stage["stage"] == "infer_ads"
    assert rebuilt_stage["metric_scope"] == "infer"
