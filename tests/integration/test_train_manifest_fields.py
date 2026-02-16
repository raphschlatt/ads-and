import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.approaches.nand import train as train_mod
from src.approaches.nand.train import train_nand_across_seeds


def test_train_manifest_contains_new_threshold_and_class_fields(tmp_path: Path, monkeypatch):
    def fake_train_nand_seed(**kwargs):
        seed = int(kwargs["seed"])
        return {
            "seed": seed,
            "checkpoint": str(tmp_path / f"seed{seed}.pt"),
            "threshold": 0.33 + (seed * 0.01),
            "threshold_selection_status": "ok",
            "threshold_source": "val_f1_opt",
            "train_class_counts": {"pos": 20, "neg": 12},
            "val_class_counts": {"pos": 12, "neg": 15},
            "test_class_counts": {"pos": 10, "neg": 11},
            "val_stats": {"f1": 0.80 + (seed * 0.01), "precision": 0.8, "recall": 0.8, "accuracy": 0.8},
            "test_metrics": {"f1": 0.79, "precision": 0.8, "recall": 0.78, "accuracy": 0.79},
        }

    monkeypatch.setattr(train_mod, "train_nand_seed", fake_train_nand_seed)

    manifest_path = tmp_path / "manifest.json"
    manifest = train_nand_across_seeds(
        mentions=pd.DataFrame({"mention_id": ["m1"]}),
        pairs=pd.DataFrame({"mention_id_1": [], "mention_id_2": [], "split": [], "label": []}),
        chars2vec=np.zeros((1, 50), dtype=np.float32),
        text_emb=np.zeros((1, 768), dtype=np.float32),
        model_config={},
        seeds=[1, 2],
        run_id="smoke_test",
        output_dir=tmp_path,
        metrics_output=manifest_path,
        device="cpu",
    )

    assert manifest["best_threshold_selection_status"] == "ok"
    assert manifest["best_threshold_source"] == "val_f1_opt"
    assert manifest["best_train_class_counts"]["neg"] == 12
    assert manifest["best_val_class_counts"]["neg"] == 15
    assert manifest["best_test_class_counts"]["neg"] == 11
    assert manifest["best_test_f1"] == 0.79
    assert manifest["best_test_metrics"]["precision"] == 0.8
    assert manifest["precision_mode"] == "fp32"

    assert manifest_path.exists()
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["best_threshold_selection_status"] == "ok"
    assert on_disk["best_test_f1"] == 0.79
    assert on_disk["precision_mode"] == "fp32"
