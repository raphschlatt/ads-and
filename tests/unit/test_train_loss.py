import numpy as np
import pandas as pd
import pytest

from author_name_disambiguation.approaches.nand.train import _combined_pair_loss, train_nand_seed


def test_combined_pair_loss_includes_info_nce_and_negative_margin_terms():
    torch = pytest.importorskip("torch")

    z1 = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    z2 = torch.tensor(
        [
            [1.0, 0.0],   # positive match
            [0.0, 1.0],   # positive match
            [1.0, 0.0],   # hard negative (high cosine)
            [1.0, 0.0],   # hard negative (high cosine)
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([1, 1, 0, 0], dtype=torch.int64)

    total, parts = _combined_pair_loss(
        torch=torch,
        z1=z1,
        z2=z2,
        labels=y,
        temperature=0.25,
        infonce_weight=1.0,
        negative_loss_weight=1.0,
        negative_margin=0.65,
    )

    assert float(parts["info_nce"].item()) >= 0.0
    assert float(parts["neg_margin"].item()) > 0.0
    assert float(total.item()) >= float(parts["info_nce"].item())


def test_train_nand_seed_requires_negatives_when_enabled(tmp_path):
    pytest.importorskip("torch")

    mentions = pd.DataFrame(
        [
            {"mention_id": "m1", "block_key": "b", "orcid": "o1"},
            {"mention_id": "m2", "block_key": "b", "orcid": "o1"},
        ]
    )
    pairs = pd.DataFrame(
        [
            {
                "pair_id": "m1__m2",
                "mention_id_1": "m1",
                "mention_id_2": "m2",
                "block_key": "b",
                "split": "train",
                "label": 1,
            }
        ]
    )

    chars = np.zeros((2, 50), dtype=np.float32)
    text = np.zeros((2, 768), dtype=np.float32)
    model_cfg = {
        "input_dim": 818,
        "hidden_dim": 16,
        "output_dim": 8,
        "batch_size": 2,
        "max_epochs": 1,
        "early_stopping_patience": 1,
        "temperature": 0.25,
        "require_hard_negatives": True,
    }

    with pytest.raises(ValueError, match="No negative train pairs found"):
        train_nand_seed(
            mentions=mentions,
            pairs=pairs,
            chars2vec=chars,
            text_emb=text,
            model_config=model_cfg,
            seed=1,
            run_id="t",
            output_dir=tmp_path,
            device="cpu",
        )
