from __future__ import annotations

from pathlib import Path

import pytest

from author_name_disambiguation.common.cache_ops import hash_checkpoint_model_state

try:  # pragma: no cover
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False

@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_hash_checkpoint_model_state_is_run_id_invariant(tmp_path: Path):
    import torch

    state_dict = {
        "layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "layer.bias": torch.tensor([0.1, -0.2], dtype=torch.float32),
    }
    model_config = {"input_dim": 2, "hidden_dim": 2}

    ckpt_a = {
        "state_dict": state_dict,
        "model_config": model_config,
        "run_id": "run_a",
        "threshold": 0.35,
    }
    ckpt_b = {
        "state_dict": state_dict,
        "model_config": model_config,
        "run_id": "run_b",
        "threshold": 0.91,
    }

    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    torch.save(ckpt_a, path_a)
    torch.save(ckpt_b, path_b)

    assert hash_checkpoint_model_state(path_a) == hash_checkpoint_model_state(path_b)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_hash_checkpoint_model_state_changes_when_weights_change(tmp_path: Path):
    import torch

    ckpt_a = {
        "state_dict": {"layer.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32)},
        "model_config": {"input_dim": 2},
    }
    ckpt_b = {
        "state_dict": {"layer.weight": torch.tensor([[1.0, 3.0]], dtype=torch.float32)},
        "model_config": {"input_dim": 2},
    }

    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    torch.save(ckpt_a, path_a)
    torch.save(ckpt_b, path_b)

    assert hash_checkpoint_model_state(path_a) != hash_checkpoint_model_state(path_b)
