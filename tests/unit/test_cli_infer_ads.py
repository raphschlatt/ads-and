from __future__ import annotations

import json
from pathlib import Path

import pytest
import pandas as pd
import yaml

from author_name_disambiguation import cli


def test_resolve_ads_dataset_files_references_optional(tmp_path: Path):
    base_dir = tmp_path / "data" / "raw" / "ads"
    ds_dir = base_dir / "my_ads_2026"
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "publications.jsonl").write_text(
        '{"Bibcode":"b1","Author":["Doe J"],"Title_en":"t","Abstract_en":"a","Year":2020,"Affiliation":"x"}\n',
        encoding="utf-8",
    )
    data_cfg = {"raw_ads_publications": str(base_dir / "legacy.jsonl")}

    resolved = cli._resolve_ads_dataset_files(data_cfg, "my_ads_2026")
    assert resolved["dataset_id"] == "my_ads_2026"
    assert resolved["references_present"] is False
    assert resolved["references_path"] is None
    assert str(resolved["publications_path"]).endswith("publications.jsonl")
    assert resolved["dataset_source_fp"]


def test_resolve_ads_dataset_files_references_present(tmp_path: Path):
    base_dir = tmp_path / "data" / "raw" / "ads"
    ds_dir = base_dir / "my_ads_2026"
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "publications.json").write_text("[]", encoding="utf-8")
    (ds_dir / "references.jsonl").write_text(
        '{"Bibcode":"r1","Author":["Doe J"],"Title_en":"t","Abstract_en":"a","Year":2020,"Affiliation":"x"}\n',
        encoding="utf-8",
    )
    data_cfg = {"raw_ads_publications": str(base_dir / "legacy.jsonl")}

    resolved = cli._resolve_ads_dataset_files(data_cfg, "my_ads_2026")
    assert resolved["references_present"] is True
    assert str(resolved["references_path"]).endswith("references.jsonl")
    assert str(resolved["publications_path"]).endswith("publications.json")


def test_resolve_ads_dataset_files_parquet_candidates(tmp_path: Path):
    base_dir = tmp_path / "data" / "raw" / "ads"
    ds_dir = base_dir / "my_ads_2026"
    ds_dir.mkdir(parents=True, exist_ok=True)

    pubs = pd.DataFrame(
        [
            {
                "Bibcode": "b1",
                "Author": ["Doe J"],
                "Title_en": "t",
                "Abstract_en": "a",
                "Year": 2020,
                "Affiliation": "x",
            }
        ]
    )
    refs = pd.DataFrame(
        [
            {
                "Bibcode": "r1",
                "Author": ["Doe J"],
                "Title_en": "t",
                "Abstract_en": "a",
                "Year": 2020,
                "Affiliation": "x",
            }
        ]
    )
    pubs.to_parquet(ds_dir / "publ_final.parquet", index=False)
    refs.to_parquet(ds_dir / "refs_final.parquet", index=False)
    data_cfg = {"raw_ads_publications": str(base_dir / "legacy.jsonl")}

    resolved = cli._resolve_ads_dataset_files(data_cfg, "my_ads_2026")
    assert resolved["references_present"] is True
    assert str(resolved["publications_path"]).endswith("publ_final.parquet")
    assert str(resolved["references_path"]).endswith("refs_final.parquet")
    assert resolved["dataset_source_fp"]


def test_resolve_ads_dataset_files_rejects_traversal(tmp_path: Path):
    base_dir = tmp_path / "data" / "raw" / "ads"
    base_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = {"raw_ads_publications": str(base_dir / "legacy.jsonl")}
    with pytest.raises(ValueError, match="must not contain path traversal"):
        cli._resolve_ads_dataset_files(data_cfg, "../escape")


def test_resolve_model_run_for_inference_requires_selected_eps(tmp_path: Path):
    metrics_dir = tmp_path / "artifacts" / "metrics" / "model_run_1"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tmp_path / "artifacts" / "checkpoints" / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("checkpoint", encoding="utf-8")

    with (metrics_dir / "03_train_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"best_checkpoint": str(ckpt), "best_threshold": 0.3}, f)
    with (metrics_dir / "04_clustering_config_used.json").open("w", encoding="utf-8") as f:
        json.dump({"eps_resolution": {}}, f)

    paths_cfg = {"artifacts": {"metrics_dir": str(tmp_path / "artifacts" / "metrics")}}
    with pytest.raises(ValueError, match="selected_eps missing"):
        cli._resolve_model_run_for_inference(paths_cfg=paths_cfg, model_run_id="model_run_1")


def test_resolve_model_run_for_inference_loads_run_and_model_cfg(tmp_path: Path):
    metrics_dir = tmp_path / "artifacts" / "metrics" / "model_run_1"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tmp_path / "artifacts" / "checkpoints" / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("checkpoint", encoding="utf-8")

    run_cfg = {"max_pairs_per_block": 123, "pair_building": {"exclude_same_bibcode": True}}
    model_cfg = {"name": "nand", "representation": {"text_model_name": "mock-model", "max_length": 64}}
    run_cfg_path = tmp_path / "cfg" / "run.yaml"
    model_cfg_path = tmp_path / "cfg" / "model.yaml"
    run_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with run_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=False)
    with model_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(model_cfg, f, sort_keys=False)

    with (metrics_dir / "03_train_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"best_checkpoint": str(ckpt), "best_threshold": 0.31}, f)
    with (metrics_dir / "04_clustering_config_used.json").open("w", encoding="utf-8") as f:
        json.dump({"eps_resolution": {"selected_eps": 0.42}}, f)
    with (metrics_dir / "00_context.json").open("w", encoding="utf-8") as f:
        json.dump({"run_config": str(run_cfg_path), "model_config": str(model_cfg_path)}, f)

    paths_cfg = {"artifacts": {"metrics_dir": str(tmp_path / "artifacts" / "metrics")}}
    resolved = cli._resolve_model_run_for_inference(paths_cfg=paths_cfg, model_run_id="model_run_1")
    assert resolved["selected_eps"] == 0.42
    assert resolved["best_threshold"] == 0.31
    assert resolved["run_cfg"]["max_pairs_per_block"] == 123
    assert resolved["model_cfg"]["representation"]["text_model_name"] == "mock-model"


def test_resolve_infer_run_cfg_normalizes_defaults(tmp_path: Path):
    cfg_path = tmp_path / "infer-mini.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "subset_target_mentions": 1234,
                "seed": "7",
                "subset_sampling": {"target_mean_block_size": "3.5"},
                "infer_overrides": {
                    "max_pairs_per_block": "999",
                    "score_batch_size": "4096",
                    "cpu_sharding_mode": "on",
                    "cpu_workers": "4",
                    "cpu_min_pairs_per_worker": "2000000",
                    "cpu_target_ram_fraction": "0.7",
                    "cluster_backend": "sklearn_cpu",
                },
            },
            f,
            sort_keys=False,
        )

    cfg, source = cli._resolve_infer_run_cfg(infer_stage="mini", infer_run_config=str(cfg_path))
    assert source == str(cfg_path)
    assert cfg["stage"] == "mini"
    assert cfg["subset_target_mentions"] == 1234
    assert cfg["seed"] == 7
    assert cfg["subset_sampling"]["target_mean_block_size"] == 3.5
    assert cfg["infer_overrides"]["max_pairs_per_block"] == 999
    assert cfg["infer_overrides"]["score_batch_size"] == 4096
    assert cfg["infer_overrides"]["cpu_sharding_mode"] == "on"
    assert cfg["infer_overrides"]["cpu_workers"] == 4
    assert cfg["infer_overrides"]["cpu_min_pairs_per_worker"] == 2000000
    assert cfg["infer_overrides"]["cpu_target_ram_fraction"] == 0.7
    assert cfg["infer_overrides"]["cluster_backend"] == "sklearn_cpu"


def test_compute_infer_subset_identity_changes_with_cfg():
    cfg_a = {"subset_target_mentions": 1000, "seed": 11, "subset_sampling": {"target_mean_block_size": 4.0}}
    cfg_b = {"subset_target_mentions": 2000, "seed": 11, "subset_sampling": {"target_mean_block_size": 4.0}}
    id_a = cli._compute_infer_subset_identity(dataset_source_fp="abc123", infer_stage="mini", infer_cfg=cfg_a)
    id_b = cli._compute_infer_subset_identity(dataset_source_fp="abc123", infer_stage="mini", infer_cfg=cfg_b)
    assert id_a["subset_tag"] != id_b["subset_tag"]
    assert id_a["sampler_version"] == cli.INFER_SUBSET_CACHE_VERSION


def test_build_infer_preflight_estimates_pair_upper_bound():
    mentions = pd.DataFrame(
        [
            {"mention_id": "m1", "block_key": "a"},
            {"mention_id": "m2", "block_key": "a"},
            {"mention_id": "m3", "block_key": "a"},
            {"mention_id": "m4", "block_key": "b"},
        ]
    )
    preflight = cli._build_infer_preflight(
        mentions=mentions,
        max_pairs_per_block=2,
        score_batch_size=128,
        max_ram_fraction=0.5,
    )
    assert preflight["n_mentions"] == 4
    assert preflight["n_blocks"] == 2
    assert preflight["pair_upper_bound"] == 2
    assert preflight["estimate_bytes"]["total_upper_bound"] > 0
