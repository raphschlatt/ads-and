from __future__ import annotations

from contextlib import nullcontext
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from author_name_disambiguation.approaches.nand.modeling import create_encoder, info_nce_loss


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for NAND training.") from exc
    return torch, DataLoader, TensorDataset


def _resolve_device(torch, device: str) -> str:
    if device != "auto":
        return device
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _ = torch.cuda.current_device()
        _ = torch.empty(1, device="cuda")
        return "cuda"
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"CUDA appears unavailable in this session ({exc!r}); falling back to CPU.",
            RuntimeWarning,
        )
        return "cpu"


def _resolve_effective_precision_mode(torch, precision_mode: str, device: str) -> str:
    mode = str(precision_mode or "fp32").strip().lower()
    if mode not in {"fp32", "amp_bf16"}:
        warnings.warn(f"Unknown precision_mode={precision_mode!r}; falling back to fp32.", RuntimeWarning)
        return "fp32"
    if mode == "fp32":
        return "fp32"
    if not str(device).startswith("cuda"):
        warnings.warn("precision_mode=amp_bf16 requested on non-CUDA device; falling back to fp32.", RuntimeWarning)
        return "fp32"
    is_supported = True
    try:
        if hasattr(torch.cuda, "is_bf16_supported"):
            is_supported = bool(torch.cuda.is_bf16_supported())
    except Exception:
        is_supported = False
    if not is_supported:
        warnings.warn("CUDA BF16 is not supported in this environment; falling back to fp32.", RuntimeWarning)
        return "fp32"
    return "amp_bf16"


def _autocast_context(torch, precision_mode: str):
    if str(precision_mode) == "amp_bf16":
        if hasattr(torch, "autocast"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()
    return nullcontext()


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


def _label_class_counts(labels: np.ndarray) -> Dict[str, int]:
    if labels is None or len(labels) == 0:
        return {"pos": 0, "neg": 0}
    return {"pos": int((labels == 1).sum()), "neg": int((labels == 0).sum())}


def _combined_pair_loss(
    *,
    torch,
    z1,
    z2,
    labels,
    temperature: float,
    infonce_weight: float,
    negative_loss_weight: float,
    negative_margin: float,
):
    labels = labels.to(z1.device).long()
    pos_mask = labels == 1
    neg_mask = labels == 0

    info_loss = torch.zeros((), device=z1.device)
    neg_loss = torch.zeros((), device=z1.device)

    if bool(pos_mask.any()):
        info_loss = info_nce_loss(z1[pos_mask], z2[pos_mask], temperature=temperature)
    if bool(neg_mask.any()):
        cos_neg = torch.nn.functional.cosine_similarity(z1[neg_mask], z2[neg_mask], dim=1)
        neg_loss = torch.nn.functional.relu(cos_neg - float(negative_margin)).mean()

    total = float(infonce_weight) * info_loss + float(negative_loss_weight) * neg_loss
    return total, {"info_nce": info_loss, "neg_margin": neg_loss}


def _compute_best_threshold(
    sim: np.ndarray,
    labels: np.ndarray,
    default_threshold: float = 0.35,
) -> tuple[float, Dict[str, float], str, str]:
    counts = _label_class_counts(labels)
    has_pos = counts["pos"] > 0
    has_neg = counts["neg"] > 0

    if not has_pos and not has_neg:
        return float(default_threshold), {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}, "fallback_no_labels", "fallback_default"
    if not has_pos:
        pred = (sim >= default_threshold).astype(int)
        stats = {
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
            "accuracy": float(accuracy_score(labels, pred)),
        }
        return float(default_threshold), stats, "fallback_no_positives", "fallback_default"
    if not has_neg:
        pred = (sim >= default_threshold).astype(int)
        stats = {
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
            "accuracy": float(accuracy_score(labels, pred)),
        }
        return float(default_threshold), stats, "fallback_no_negatives", "fallback_default"

    thresholds = np.linspace(-1.0, 1.0, num=2001)
    best_key = (-1.0, -1.0)
    best_thr = float(default_threshold)
    best_stats: Dict[str, float] = {}

    for thr in thresholds:
        pred = (sim >= thr).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        edge_margin = min(float(thr + 1.0), float(1.0 - thr))
        key = (float(f1), edge_margin)
        if key > best_key:
            best_key = key
            best_thr = float(thr)
            best_stats = {
                "f1": float(f1),
                "precision": float(precision_score(labels, pred, zero_division=0)),
                "recall": float(recall_score(labels, pred, zero_division=0)),
                "accuracy": float(accuracy_score(labels, pred)),
            }

    return best_thr, best_stats, "ok", "val_f1_opt"


def _score_pairs(
    model,
    features: np.ndarray,
    pair_idx_1: np.ndarray,
    pair_idx_2: np.ndarray,
    batch_size: int = 8192,
    precision_mode: str = "fp32",
    show_progress: bool = False,
) -> np.ndarray:
    torch, _, _ = _require_torch()
    device = next(model.parameters()).device
    sims = []

    model.eval()
    starts = range(0, len(pair_idx_1), batch_size)
    if show_progress:
        try:
            from tqdm.auto import tqdm

            total = (len(pair_idx_1) + batch_size - 1) // batch_size
            starts = tqdm(starts, total=total, desc="Score pairs", leave=False)
        except Exception:
            pass

    with torch.no_grad():
        for start in starts:
            end = min(start + batch_size, len(pair_idx_1))
            i1 = pair_idx_1[start:end]
            i2 = pair_idx_2[start:end]
            x1 = torch.from_numpy(features[i1]).to(device)
            x2 = torch.from_numpy(features[i2]).to(device)
            with _autocast_context(torch, precision_mode):
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
    precision_mode: str = "fp32",
    show_progress: bool = False,
) -> Dict[str, Any]:
    torch, DataLoader, TensorDataset = _require_torch()

    requested_device = device
    device = _resolve_device(torch, device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    features = build_feature_matrix(chars2vec=chars2vec, text_emb=text_emb)
    mindex = _build_index(mentions)

    train_pairs = pairs[(pairs["split"] == "train") & pairs["label"].notna()]
    val_pairs = pairs[(pairs["split"] == "val") & pairs["label"].notna()]
    test_pairs = pairs[(pairs["split"] == "test") & pairs["label"].notna()]

    train_i1, train_i2, train_y = _pairs_to_index_arrays(train_pairs, mindex, labeled_only=True)
    val_i1, val_i2, val_y = _pairs_to_index_arrays(val_pairs, mindex, labeled_only=True)
    test_i1, test_i2, test_y = _pairs_to_index_arrays(test_pairs, mindex, labeled_only=True)
    train_class_counts = _label_class_counts(train_y)
    val_class_counts = _label_class_counts(val_y)
    test_class_counts = _label_class_counts(test_y)

    if train_class_counts["pos"] == 0:
        raise ValueError("No positive train pairs found. Check pair building and split assignment.")
    require_hard_negatives = bool(model_config.get("require_hard_negatives", True))
    if require_hard_negatives and train_class_counts["neg"] == 0:
        raise ValueError(
            "No negative train pairs found while require_hard_negatives=true. "
            f"Train class counts: {train_class_counts}"
        )

    dataset = TensorDataset(
        torch.from_numpy(train_i1),
        torch.from_numpy(train_i2),
        torch.from_numpy(train_y.astype(np.int64)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(model_config.get("batch_size", 2048)),
        shuffle=True,
        drop_last=False,
    )

    model = create_encoder(model_config)
    try:
        model.to(device)
    except Exception as exc:
        if requested_device == "auto" and str(device).startswith("cuda"):
            warnings.warn(
                f"Moving training model to CUDA failed ({exc!r}); falling back to CPU.",
                RuntimeWarning,
            )
            device = "cpu"
            model.to(device)
        else:
            raise

    effective_precision_mode = _resolve_effective_precision_mode(
        torch=torch,
        precision_mode=precision_mode or str(model_config.get("precision_mode", "fp32")),
        device=device,
    )

    opt = torch.optim.Adam(model.parameters(), lr=float(model_config.get("learning_rate", 1e-5)))
    max_epochs = int(model_config.get("max_epochs", 250))
    patience = int(model_config.get("early_stopping_patience", 25))
    temperature = float(model_config.get("temperature", 0.25))
    infonce_weight = float(model_config.get("infonce_weight", 1.0))
    negative_loss_weight = float(model_config.get("negative_loss_weight", 1.0))
    negative_margin = float(model_config.get("negative_margin", 0.65))

    best_val_loss = float("inf")
    best_state = None
    stale = 0

    train_history: List[float] = []
    val_history: List[float] = []

    epochs = range(max_epochs)
    if show_progress:
        try:
            from tqdm.auto import tqdm

            epochs = tqdm(epochs, total=max_epochs, desc=f"Train seed {seed}", leave=False)
        except Exception:
            pass

    for _epoch in epochs:
        model.train()
        batch_losses = []

        for b_i1, b_i2, b_y in loader:
            b_i1 = b_i1.numpy()
            b_i2 = b_i2.numpy()
            x1 = torch.from_numpy(features[b_i1]).to(device)
            x2 = torch.from_numpy(features[b_i2]).to(device)
            y = b_y.to(device)

            with _autocast_context(torch, effective_precision_mode):
                z1 = model(x1)
                z2 = model(x2)
                loss, _ = _combined_pair_loss(
                    torch=torch,
                    z1=z1,
                    z2=z2,
                    labels=y,
                    temperature=temperature,
                    infonce_weight=infonce_weight,
                    negative_loss_weight=negative_loss_weight,
                    negative_margin=negative_margin,
                )

            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        train_history.append(train_loss)

        # Validation uses the same combined objective as train.
        model.eval()
        v_i1, v_i2, v_y = val_i1, val_i2, val_y
        v_losses = []
        if len(v_i1) > 0:
            with torch.no_grad():
                for start in range(0, len(v_i1), int(model_config.get("batch_size", 2048))):
                    end = min(start + int(model_config.get("batch_size", 2048)), len(v_i1))
                    x1 = torch.from_numpy(features[v_i1[start:end]]).to(device)
                    x2 = torch.from_numpy(features[v_i2[start:end]]).to(device)
                    y = torch.from_numpy(v_y[start:end].astype(np.int64)).to(device)
                    with _autocast_context(torch, effective_precision_mode):
                        z1 = model(x1)
                        z2 = model(x2)
                        vloss, _ = _combined_pair_loss(
                            torch=torch,
                            z1=z1,
                            z2=z2,
                            labels=y,
                            temperature=temperature,
                            infonce_weight=infonce_weight,
                            negative_loss_weight=negative_loss_weight,
                            negative_margin=negative_margin,
                        )
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

    default_threshold = float(model_config.get("default_cosine_threshold", 0.35))
    if len(val_i1) == 0:
        threshold = float(default_threshold)
        val_stats = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
        threshold_selection_status = "fallback_no_labels"
        threshold_source = "fallback_default"
    else:
        val_sim = _score_pairs(
            model,
            features,
            val_i1,
            val_i2,
            precision_mode=effective_precision_mode,
            show_progress=show_progress,
        )
        threshold, val_stats, threshold_selection_status, threshold_source = _compute_best_threshold(
            val_sim,
            val_y,
            default_threshold=default_threshold,
        )

    test_metrics = {"f1": None, "precision": None, "recall": None, "accuracy": None}
    if len(test_i1) > 0:
        test_sim = _score_pairs(
            model,
            features,
            test_i1,
            test_i2,
            precision_mode=effective_precision_mode,
            show_progress=show_progress,
        )
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
        "threshold_selection_status": threshold_selection_status,
        "threshold_source": threshold_source,
        "val_class_counts": val_class_counts,
        "test_class_counts": test_class_counts,
        "train_class_counts": train_class_counts,
        "seed": int(seed),
        "run_id": run_id,
        "precision_mode": effective_precision_mode,
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
        "threshold_selection_status": threshold_selection_status,
        "threshold_source": threshold_source,
        "train_class_counts": train_class_counts,
        "val_class_counts": val_class_counts,
        "test_class_counts": test_class_counts,
        "precision_mode": effective_precision_mode,
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
    precision_mode: str = "fp32",
    show_progress: bool = False,
) -> Dict[str, Any]:
    runs = []
    seed_iter = seeds
    if show_progress:
        try:
            from tqdm.auto import tqdm

            seed_iter = tqdm(seeds, total=len(seeds), desc="Seeds", leave=False)
        except Exception:
            pass

    for seed in seed_iter:
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
            precision_mode=precision_mode,
            show_progress=show_progress,
        )
        runs.append(result)

    best = max(runs, key=lambda r: r["val_stats"].get("f1", 0.0) if r.get("val_stats") else 0.0)
    manifest = {
        "run_id": run_id,
        "runs": runs,
        "best_seed": best["seed"],
        "best_checkpoint": best["checkpoint"],
        "best_threshold": best["threshold"],
        "best_threshold_selection_status": best.get("threshold_selection_status"),
        "best_threshold_source": best.get("threshold_source"),
        "best_train_class_counts": best.get("train_class_counts"),
        "best_val_class_counts": best.get("val_class_counts"),
        "best_test_class_counts": best.get("test_class_counts"),
        "best_test_f1": (best.get("test_metrics") or {}).get("f1"),
        "best_test_metrics": best.get("test_metrics"),
        "best_val_f1": best["val_stats"].get("f1"),
        "precision_mode": best.get("precision_mode", str(precision_mode or "fp32")),
    }

    if metrics_output is not None:
        p = Path(metrics_output)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return manifest
