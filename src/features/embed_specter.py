from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _hash_stub_embedding(text: str, dim: int = 768) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(h[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _to_text(title: str, abstract: str) -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if title and abstract:
        return f"{title} [SEP] {abstract}"
    return title or abstract


def _has_valid_precomputed(values: Iterable) -> bool:
    for item in values:
        if isinstance(item, list) and len(item) == 768:
            return True
    return False


def _stack_precomputed(values: Iterable, texts: list[str]) -> np.ndarray:
    out = []
    for idx, item in enumerate(values):
        if isinstance(item, list) and len(item) == 768:
            out.append(np.asarray(item, dtype=np.float32))
        else:
            out.append(_hash_stub_embedding(texts[idx], dim=768))
    return np.vstack(out).astype(np.float32)


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


def generate_specter_embeddings(
    mentions: pd.DataFrame,
    model_name: str = "allenai/specter",
    batch_size: int = 16,
    max_length: int = 256,
    device: str = "auto",
    prefer_precomputed: bool = True,
    use_stub_if_missing: bool = False,
    show_progress: bool = False,
) -> np.ndarray:
    titles = mentions["title"].fillna("").astype(str).tolist()
    abstracts = mentions["abstract"].fillna("").astype(str).tolist()
    texts = [_to_text(t, a) for t, a in zip(titles, abstracts)]

    if prefer_precomputed and "precomputed_embedding" in mentions.columns:
        pre = mentions["precomputed_embedding"].tolist()
        if _has_valid_precomputed(pre):
            return _stack_precomputed(pre, texts)

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        if not use_stub_if_missing:
            raise RuntimeError(
                "SPECTER embedding generation requires `torch` and `transformers`, or precomputed 768-dim embeddings."
            ) from exc
        return np.vstack([_hash_stub_embedding(t, dim=768) for t in texts]).astype(np.float32)

    requested_device = device
    device = _resolve_device(torch, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    try:
        model.to(device)
    except Exception as exc:
        if requested_device == "auto" and str(device).startswith("cuda"):
            warnings.warn(
                f"Moving SPECTER model to CUDA failed ({exc!r}); falling back to CPU.",
                RuntimeWarning,
            )
            device = "cpu"
            model.to(device)
        else:
            raise
    model.eval()

    vectors: list[np.ndarray] = []
    starts = range(0, len(texts), batch_size)
    if show_progress:
        try:
            from tqdm.auto import tqdm

            total = (len(texts) + batch_size - 1) // batch_size
            starts = tqdm(starts, total=total, desc="SPECTER batches", leave=False)
        except Exception:
            pass
    with torch.no_grad():
        for start in starts:
            chunk = texts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
            vectors.append(cls)

    return np.vstack(vectors).astype(np.float32)


def get_or_create_specter_embeddings(
    mentions: pd.DataFrame,
    output_path: str | Path,
    force_recompute: bool = False,
    model_name: str = "allenai/specter",
    batch_size: int = 16,
    max_length: int = 256,
    device: str = "auto",
    prefer_precomputed: bool = True,
    use_stub_if_missing: bool = False,
    show_progress: bool = False,
) -> np.ndarray:
    output = Path(output_path)
    if output.exists() and not force_recompute:
        return np.load(output)

    emb = generate_specter_embeddings(
        mentions=mentions,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        prefer_precomputed=prefer_precomputed,
        use_stub_if_missing=use_stub_if_missing,
        show_progress=show_progress,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, emb)
    return emb
