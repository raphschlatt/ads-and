from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping


def resolve_shared_cache_root(data_cfg: Mapping[str, Any]) -> Path:
    explicit = data_cfg.get("shared_cache_dir")
    if explicit:
        return Path(str(explicit)) / "_shared"
    processed_dir = data_cfg.get("processed_dir")
    if processed_dir:
        return Path(str(processed_dir)).parent / "cache" / "_shared"
    subset_cache_dir = Path(str(data_cfg["subset_cache_dir"]))
    return subset_cache_dir.parent.parent / "cache" / "_shared"


def stable_hash(payload: Any, length: int = 12) -> str:
    if isinstance(payload, bytes):
        blob = payload
    elif isinstance(payload, str):
        blob = payload.encode("utf-8")
    else:
        blob = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[: int(length)]


def hash_file(path: str | Path, length: int = 12) -> str:
    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[: int(length)]


def hash_checkpoint_model_state(
    checkpoint_path: str | Path,
    *,
    score_pipeline_version: str = "v2",
    length: int = 12,
) -> str:
    """Build a run-id-invariant hash from checkpoint model contents.

    Hash payload:
    - score pipeline version marker
    - serialized model config
    - ordered state_dict tensors (name, shape, dtype, bytes)
    """
    try:
        import numpy as np
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Torch and NumPy are required to hash checkpoint model state.") from exc

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint missing 'state_dict': {checkpoint_path}")

    model_cfg = ckpt.get("model_config", {})
    cfg_blob = json.dumps(model_cfg, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")

    h = hashlib.sha1()
    h.update(str(score_pipeline_version).encode("utf-8"))
    h.update(b"|")
    h.update(cfg_blob)

    for key in sorted(state.keys()):
        value = state[key]
        if hasattr(value, "detach"):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        arr = np.ascontiguousarray(arr)
        h.update(b"|")
        h.update(str(key).encode("utf-8"))
        h.update(b"|")
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(b"|")
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(b"|")
        h.update(arr.tobytes(order="C"))

    return h.hexdigest()[: int(length)]


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        raise IsADirectoryError(f"Expected file path, got directory: {path}")


def link_or_copy(shared_path: str | Path, run_path: str | Path) -> str:
    src = Path(shared_path)
    dst = Path(run_path)
    if not src.exists():
        raise FileNotFoundError(f"Shared source missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        if dst.exists() and os.path.samefile(src, dst):
            return "existing"
    except Exception:
        pass

    if dst.exists() or dst.is_symlink():
        _remove_existing(dst)

    try:
        os.link(src, dst)
        return "hardlink"
    except Exception:
        pass

    try:
        dst.symlink_to(src)
        return "symlink"
    except Exception:
        pass

    shutil.copy2(src, dst)
    return "copy"
