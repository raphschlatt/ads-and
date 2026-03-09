from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from author_name_disambiguation import cli


def test_resolve_model_run_for_inference_requires_selected_eps(tmp_path: Path):
    metrics_dir = tmp_path / "artifacts" / "metrics" / "model_run_1"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tmp_path / "artifacts" / "checkpoints" / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("checkpoint", encoding="utf-8")

    with (metrics_dir / "03_train_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_checkpoint": str(ckpt), "best_threshold": 0.3}, handle)
    with (metrics_dir / "04_clustering_config_used.json").open("w", encoding="utf-8") as handle:
        json.dump({"eps_resolution": {}}, handle)

    with pytest.raises(ValueError, match="selected_eps missing"):
        cli._resolve_model_run_for_inference(
            artifacts_root=tmp_path / "artifacts",
            model_run_id="model_run_1",
        )


def test_resolve_model_run_for_inference_loads_run_and_model_cfg(tmp_path: Path):
    metrics_dir = tmp_path / "artifacts" / "metrics" / "model_run_1"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tmp_path / "artifacts" / "checkpoints" / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("checkpoint", encoding="utf-8")

    run_cfg_path = tmp_path / "cfg" / "run.yaml"
    model_cfg_path = tmp_path / "cfg" / "model.yaml"
    run_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with run_cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"max_pairs_per_block": 123, "pair_building": {"exclude_same_bibcode": True}}, handle, sort_keys=False)
    with model_cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"name": "nand", "representation": {"text_model_name": "mock-model", "max_length": 64}},
            handle,
            sort_keys=False,
        )

    with (metrics_dir / "03_train_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_checkpoint": str(ckpt), "best_threshold": 0.31}, handle)
    with (metrics_dir / "04_clustering_config_used.json").open("w", encoding="utf-8") as handle:
        json.dump({"eps_resolution": {"selected_eps": 0.42}}, handle)
    with (metrics_dir / "00_context.json").open("w", encoding="utf-8") as handle:
        json.dump({"run_config": str(run_cfg_path), "model_config": str(model_cfg_path)}, handle)

    resolved = cli._resolve_model_run_for_inference(
        artifacts_root=tmp_path / "artifacts",
        model_run_id="model_run_1",
    )

    assert resolved["selected_eps"] == 0.42
    assert resolved["best_threshold"] == 0.31
    assert resolved["run_cfg"]["max_pairs_per_block"] == 123
    assert resolved["model_cfg"]["representation"]["text_model_name"] == "mock-model"
