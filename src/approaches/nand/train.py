from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.approaches.nand.modeling import create_encoder, info_nce_loss


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for NAND training.") from exc
    return torch, DataLoader, TensorDataset


def build_feature_matrix(chars2vec: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    if len(chars2vec) != len(text_emb):
        raise ValueError(f"Embedding length mismatch: chars={len(chars2vec)}, text={len(text_emb)}")
    feats = np.concatenate([chars2vec.astype(np.float32), text_emb.astype(np.float32)], axis=1)
    return feats


def _build_index(mentions: pd.DataFrame) -> Dict[str, int]:
    return {str(m): i for i, m in enumerate(mentions["mention_id"].tolist())}


def _pairs_to_index_arrays(pairs: pd.DataFrame, mention_index: Dict[str, int], labeled_only: bool = True):
    arr_1, arr_2, labels = [], [], []
    for row in pairs.itertuples(index=False):
        m1 = str(row.mention_id_1)
        m2 = str(row.mention_id_2)
        if m1 not in mention_index or m2 not in mention_index:
            continue
        label = getattr(row, "label", None)
        if labeled_only and (label is None or pd.isna(label)):
            continue
        arr_1.append(mention_index[m1])
        arr_2.append(mention_index[m2])
        labels.append(int(label) if label is not None and not pd.isna(label) else -1)

    if not arr_1:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.asarray(arr_1, dtype=np.int64), np.asarray(arr_2, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def _compute_best_threshold(sim: np.ndarray, labels: np.ndarray) -> tuple[float, Dict[str, float]]:
    thresholds = np.linspace(-1.0, 1.0, num=2001)
    best_f1 = -1.0
    best_thr = 0.0
    best_stats: Dict[str, float] = {}

    for thr in thresholds:
        pred = (sim >= thr).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_stats = {
                "f1": float(f1),
                "precision": float(precision_score(labels, pred, zero_division=0)),
                "recall": float(recall_score(labels, pred, zero_division=0)),
                "accuracy": float(accuracy_score(labels, pred)),
            }

    return best_thr, best_stats


def _score_pairs(
    model,
    features: np.ndarray,
    pair_idx_1: np.ndarray,
    pair_idx_2: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    torch, _, _ = _require_torch()
    device = next(model.parameters()).device
    sims = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(pair_idx_1), batch_size):
            end = min(start + batch_size, len(pair_idx_1))
            i1 = pair_idx_1[start:end]
            i2 = pair_idx_2[start:end]
            x1 = torch.from_numpy(features[i1]).to(device)
            x2 = torch.from_numpy(features[i2]).to(device)
            z1 = model(x1)
            z2 = model(x2)
            s = torch.nn.functional.cosine_similarity(z1, z2, dim=1)
            sims.append(s.detach().cpu().numpy())

    return np.concatenate(sims, axis=0) if sims else np.array([], dtype=np.float32)


def train_nand_seed(
    mentions: pd.DataFrame,
    pairs: pd.DataFrame,
    chars2vec: np.ndarray,
    text_emb: np.ndarray,
    model_config: Dict[str, Any],
    seed: int,
    run_id: str,
    output_dir: str | Path,
    device: str = "auto",
) -> Dict[str, Any]:
    torch, DataLoader, TensorDataset = _require_torch()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    features = build_feature_matrix(chars2vec=chars2vec, text_emb=text_emb)
    mindex = _build_index(mentions)

    train_pairs = pairs[(pairs["split"] == "train") & (pairs["label"] == 1)]
    val_pairs = pairs[(pairs["split"] == "val") & pairs["label"].notna()]
    test_pairs = pairs[(pairs["split"] == "test") & pairs["label"].notna()]

    train_i1, train_i2, _ = _pairs_to_index_arrays(train_pairs, mindex, labeled_only=True)
    val_i1, val_i2, val_y = _pairs_to_index_arrays(val_pairs, mindex, labeled_only=True)
    test_i1, test_i2, test_y = _pairs_to_index_arrays(test_pairs, mindex, labeled_only=True)

    if len(train_i1) == 0:
        raise ValueError("No positive train pairs found. Check pair building and split assignment.")

    dataset = TensorDataset(torch.from_numpy(train_i1), torch.from_numpy(train_i2))
    loader = DataLoader(
        dataset,
        batch_size=int(model_config.get("batch_size", 2048)),
        shuffle=True,
        drop_last=False,
    )

    model = create_encoder(model_config)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(model_config.get("learning_rate", 1e-5)))
    max_epochs = int(model_config.get("max_epochs", 250))
    patience = int(model_config.get("early_stopping_patience", 25))
    temperature = float(model_config.get("temperature", 0.25))

    best_val_loss = float("inf")
    best_state = None
    stale = 0

    train_history: List[float] = []
    val_history: List[float] = []

    for _epoch in range(max_epochs):
        model.train()
        batch_losses = []

        for b_i1, b_i2 in loader:
            b_i1 = b_i1.numpy()
            b_i2 = b_i2.numpy()
            x1 = torch.from_numpy(features[b_i1]).to(device)
            x2 = torch.from_numpy(features[b_i2]).to(device)

            z1 = model(x1)
            z2 = model(x2)
            loss = info_nce_loss(z1, z2, temperature=temperature)

            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        train_history.append(train_loss)

        # Val loss on positive pairs only.
        model.eval()
        val_pos = val_pairs[val_pairs["label"] == 1]
        v_i1, v_i2, _ = _pairs_to_index_arrays(val_pos, mindex, labeled_only=True)
        v_losses = []
        if len(v_i1) > 0:
            with torch.no_grad():
                for start in range(0, len(v_i1), int(model_config.get("batch_size", 2048))):
                    end = min(start + int(model_config.get("batch_size", 2048)), len(v_i1))
                    x1 = torch.from_numpy(features[v_i1[start:end]]).to(device)
                    x2 = torch.from_numpy(features[v_i2[start:end]]).to(device)
                    z1 = model(x1)
                    z2 = model(x2)
                    vloss = info_nce_loss(z1, z2, temperature=temperature)
                    v_losses.append(float(vloss.detach().cpu().item()))

        val_loss = float(np.mean(v_losses)) if v_losses else train_loss
        val_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if len(val_i1) == 0:
        threshold = 0.0
        val_stats = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    else:
        val_sim = _score_pairs(model, features, val_i1, val_i2)
        threshold, val_stats = _compute_best_threshold(val_sim, val_y)

    test_metrics = {"f1": None, "precision": None, "recall": None, "accuracy": None}
    if len(test_i1) > 0:
        test_sim = _score_pairs(model, features, test_i1, test_i2)
        pred = (test_sim >= threshold).astype(int)
        test_metrics = {
            "f1": float(f1_score(test_y, pred, zero_division=0)),
            "precision": float(precision_score(test_y, pred, zero_division=0)),
            "recall": float(recall_score(test_y, pred, zero_division=0)),
            "accuracy": float(accuracy_score(test_y, pred)),
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{run_id}_seed{seed}.pt"

    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": model_config,
        "threshold": float(threshold),
        "seed": int(seed),
        "run_id": run_id,
        "val_stats": val_stats,
        "test_metrics": test_metrics,
        "train_loss_history": train_history,
        "val_loss_history": val_history,
    }
    torch.save(checkpoint, ckpt_path)

    return {
        "seed": int(seed),
        "checkpoint": str(ckpt_path),
        "threshold": float(threshold),
        "val_stats": val_stats,
        "test_metrics": test_metrics,
    }


def train_nand_across_seeds(
    mentions: pd.DataFrame,
    pairs: pd.DataFrame,
    chars2vec: np.ndarray,
    text_emb: np.ndarray,
    model_config: Dict[str, Any],
    seeds: List[int],
    run_id: str,
    output_dir: str | Path,
    metrics_output: str | Path | None = None,
    device: str = "auto",
) -> Dict[str, Any]:
    runs = []
    for seed in seeds:
        result = train_nand_seed(
            mentions=mentions,
            pairs=pairs,
            chars2vec=chars2vec,
            text_emb=text_emb,
            model_config=model_config,
            seed=int(seed),
            run_id=run_id,
            output_dir=output_dir,
            device=device,
        )
        runs.append(result)

    best = max(runs, key=lambda r: r["val_stats"].get("f1", 0.0) if r.get("val_stats") else 0.0)
    manifest = {
        "run_id": run_id,
        "runs": runs,
        "best_seed": best["seed"],
        "best_checkpoint": best["checkpoint"],
        "best_threshold": best["threshold"],
        "best_val_f1": best["val_stats"].get("f1"),
    }

    if metrics_output is not None:
        p = Path(metrics_output)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return manifest
